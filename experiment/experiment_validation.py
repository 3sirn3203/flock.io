import os
import json
import argparse
from typing import Dict
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
from torch.utils.data import random_split, ConcatDataset

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template
from plot_loss import plot_loss_from_trainer


def load_config(config_path: str = "config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: int = 0.1
    target_modules: list = ["q_proj", "v_proj"]
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    learning_rate: float = 2e-4
    warmup_steps_ratio: float = 0.1
    bf16: bool = True
    optim: str = "adam_torch"


def train_lora(model_id: str, context_length: int, data: str, augmented_data: str,
               validation_ratio: float, validation_strategy: str, seed: int,
               training_args: Dict):

    # Define training arguments
    per_device_train_batch_size = training_args.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 8)
    num_train_epochs = training_args.get("num_train_epochs", 1)
    lora_rank = training_args.get("lora_rank", 8)
    lora_alpha = training_args.get("lora_alpha", 16)
    lora_dropout = training_args.get("lora_dropout", 0.1)
    target_modules = training_args.get("target_modules", ["q_proj", "v_proj"])
    early_stopping_patience = training_args.get("early_stopping_patience", 3)
    early_stopping_threshold = training_args.get("early_stopping_threshold", 0.001)
    learning_rate = training_args.get("learning_rate", 2e-4)
    warmup_steps_ratio = training_args.get("warmup_steps_ratio", 0.1)
    bf16 = training_args.get("bf16", True)
    optim = training_args.get("optim", "adam_torch")

    assert model_id in model2template, f"model_id {model_id} not supported"

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load main dataset
    data_path = f"data/{data}"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    full_dataset = SFTDataset(
        file=data_path,
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    
    # Split main dataset into train and validation sets
    total_size = len(full_dataset)
    val_size = int(total_size * validation_ratio)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)  # Use passed seed
    )
    
    # Handle augmented data if provided
    augmented_dataset = None
    if augmented_data is not None:
        augmented_data_path = f"data/{augmented_data}"
        if not os.path.exists(augmented_data_path):
            print(f"Warning: Augmented data file {augmented_data_path} does not exist. Skipping augmented data.")
        else:
            print(f"Loading augmented data from: {augmented_data_path}")
            augmented_dataset = SFTDataset(
                file=augmented_data_path,
                tokenizer=tokenizer,
                max_seq_length=context_length,
                template=model2template[model_id],
            )
    
    # Configure validation strategy
    print(f"Using validation strategy: {validation_strategy}")
    
    if validation_strategy == "original_only":
        # Only use original data for validation
        final_val_dataset = val_dataset
        if augmented_dataset is not None:
            train_dataset = ConcatDataset([train_dataset, augmented_dataset])
            print(f"Train: original train + augmented, Validation: original only")
    
    elif validation_strategy == "mixed":
        # Use original + portion of augmented for validation
        if augmented_dataset is not None:
            # Split augmented data (80% to train, 20% to validation)
            aug_total = len(augmented_dataset)
            aug_val_size = int(aug_total * validation_ratio)
            aug_train_size = aug_total - aug_val_size
            
            aug_train_dataset, aug_val_dataset = random_split(
                augmented_dataset,
                [aug_train_size, aug_val_size],
                generator=torch.Generator().manual_seed(seed + 1)  # Different seed
            )
            
            train_dataset = ConcatDataset([train_dataset, aug_train_dataset])
            final_val_dataset = ConcatDataset([val_dataset, aug_val_dataset])
            print(f"Train: original train + augmented train, Validation: original + augmented")
        else:
            final_val_dataset = val_dataset
            print(f"No augmented data provided, using original validation only")
    
    elif validation_strategy == "augmented_only":
        # Use only augmented data for validation (if available)
        if augmented_dataset is not None:
            # Split augmented data (80% to train, 20% to validation)
            aug_total = len(augmented_dataset)
            aug_val_size = int(aug_total * validation_ratio)
            aug_train_size = aug_total - aug_val_size
            
            aug_train_dataset, aug_val_dataset = random_split(
                augmented_dataset,
                [aug_train_size, aug_val_size],
                generator=torch.Generator().manual_seed(seed + 1)
            )
            
            train_dataset = ConcatDataset([train_dataset, aug_train_dataset])
            final_val_dataset = aug_val_dataset
            print(f"Train: original train + augmented train, Validation: augmented only")
        else:
            final_val_dataset = val_dataset
            print(f"No augmented data provided, falling back to original validation")
    
    else:
        raise ValueError(f"Unknown validation_strategy: {validation_strategy}. "
                        f"Choose from: 'original_only', 'mixed', 'augmented_only'")
    
    print(f"Final dataset split: Train={len(train_dataset)}, Validation={len(final_val_dataset)}")

    # Calculate warmup steps based on the number of training examples
    num_examples = len(train_dataset)  # Use train dataset size
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    warmup_steps = int(num_examples * num_train_epochs * warmup_steps_ratio / batch_size)

    training_config = SFTConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,  # Same batch size for eval
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        bf16=bf16,
        logging_steps=20,
        output_dir="outputs",
        optim=optim,
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        max_seq_length=context_length,
        
        # Evaluation settings
        evaluation_strategy="steps",  # Evaluate every eval_steps
        eval_steps=20,  # Same as logging_steps
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # Use final train dataset
        eval_dataset=final_val_dataset,     # Use final validation dataset
        args=training_config,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # Plot training and validation losses
    print("Generating loss plots...")
    plot_loss_from_trainer(trainer, output_dir="outputs")

    # upload lora weights and tokenizer
    print("Training Completed.")


if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Path to the configuration JSON file (default: config.json)"
    )
    args = parser.parse_args()
    
    # Load configuration from specified config file
    config = load_config(args.config)

    # Unpack config file
    model_id = config.get("model_id", "Qwen/Qwen1.5-0.5B")
    context_length = config.get("context_length", 4096)
    data = config.get("data", "demo_data.jsonl")
    augmented_data = config.get("augmented_data", None)
    validation_ratio = config.get("validation_ratio", 0.2)
    validation_strategy = config.get("validation_strategy", "original_only")
    seed = config.get("seed", 42)
    training_args = config.get("training_args", {})

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id,
        context_length=context_length,
        validation_ratio=validation_ratio,
        validation_strategy=validation_strategy,
        data=data,
        augmented_data=augmented_data,
        seed=seed,
        training_args=training_args,
    )
