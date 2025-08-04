import os
import json
from typing import Dict
from dataclasses import dataclass

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template


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
    warmup_steps: float = 0.1
    bf16: bool = True
    optim: str = "adam_torch"


def train_lora(model_id: str, context_length: int, training_args: Dict, data_file_name: str):
    # Define training arguments
    per_device_train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    num_train_epochs = training_args.num_train_epochs
    lora_rank = training_args.lora_rank
    lora_alpha = training_args.lora_alpha
    lora_dropout = training_args.lora_dropout
    target_modules = training_args.target_modules
    early_stopping_patience = training_args.early_stopping_patience
    early_stopping_threshold = training_args.early_stopping_threshold
    learning_rate = training_args.learning_rate
    warmup_steps = training_args.warmup_steps
    bf16 = training_args.bf16
    optim = training_args.optim
    
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

    training_args = SFTConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
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
    data_path = f"data/{data_file_name}"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    dataset = SFTDataset(
        file=data_path,
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
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
    # Load configuration from config.json
    config = load_config("config.json")

    # Set model ID and context length from config
    model_id = config.get("model_id", "Qwen/Qwen1.5-0.5B")
    context_length = config.get("context_length", 4096)
    data_file_name = config.get("data_file_name", "demo_data.jsonl")
    training_args = config.get("training_args", {})

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id,
        context_length=context_length,
        training_args=training_args,
        data_file_name=data_file_name
    )
