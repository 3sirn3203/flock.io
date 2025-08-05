import os
from typing import Dict
from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template


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

    # Load dataset
    dataset = SFTDataset(
        file=f"data/{data}",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    
    # Load augmented dataset if provided and concatenate
    if augmented_data is not None:
        augmented_dataset = SFTDataset(
            file=f"data/{augmented_data}",
            tokenizer=tokenizer,
            max_seq_length=context_length,
            template=model2template[model_id],
        )
        # Concatenate datasets
        dataset = ConcatDataset([dataset, augmented_dataset])

    # Calculate warmup steps based on the number of training examples
    num_examples = len(dataset)  # Use dataset size
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
        max_seq_length=context_length
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
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

    # upload lora weights and tokenizer
    print("Training Completed.")


if __name__ == "__main__":
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    # Set model ID and context length
    model_id = "Qwen/Qwen1.5-0.5B"
    context_length = 2048
    data = "demo_data.jsonl"
    augmented_data = None

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, 
        context_length=context_length, 
        data=data,
        augmented_data=augmented_data,
        training_args=training_args.__dict__
    )
