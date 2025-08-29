import os
from dotenv import load_dotenv
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
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: int = 0.1
    target_modules: list = ["q_proj", "v_proj"]
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    optim: str = "adam_torch"
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    warmup_steps_ratio: float = 0.1
    bf16: bool = True
    evals_per_epoch: int = 2


def train_lora(model_id: str, context_length: int, data: str, augmented_data: str,
               training_args: Dict):

    root_dir = os.path.dirname(os.path.realpath(__file__))

    # hyperparameter 설정을 불러옵니다.
    lora_rank = training_args.get("lora_rank", 8)
    lora_alpha = training_args.get("lora_alpha", 16)
    lora_dropout = training_args.get("lora_dropout", 0.1)
    target_modules = training_args.get("target_modules", ["q_proj", "v_proj"])
    
    per_device_train_batch_size = training_args.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 8)
    num_train_epochs = training_args.get("num_train_epochs", 1)
    learning_rate = training_args.get("learning_rate", 2e-4)
    optim = training_args.get("optim", "adam_torch")
    early_stopping_patience = training_args.get("early_stopping_patience", 3)
    early_stopping_threshold = training_args.get("early_stopping_threshold", 0.001)
    warmup_steps_ratio = training_args.get("warmup_steps_ratio", 0.1)
    bf16 = training_args.get("bf16", True)
    evals_per_epoch = training_args.get("evals_per_epoch", 2)

    assert model_id in model2template, f"model_id {model_id} not supported"

    """
    LoRa 설정
        config 파일을 통해 다음의 hyperparameter를 설정합니다.
        - lora_rank
        - lora_alpha
        - lora_dropout
        - target_modules
    """
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
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

    # 대회에서 기본 제공된 데이터셋을 불러옵니다.
    data_path = f"{root_dir}/{data}"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} does not exist.")
    
    dataset = SFTDataset(
        file=data_path,
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )
    dataset_size = len(dataset)
    
    # augmented data가 None이 아닌 경우 불러온 후 train set에 concat 합니다.
    if augmented_data is not None:
        augmented_data_path = f"{root_dir}/{augmented_data}"
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
            dataset = ConcatDataset([dataset, augmented_dataset])
            dataset_size = len(dataset)

    print(f"Final dataset size: {dataset_size}")

    # 전체 훈련 셈플의 개수를 바탕으로 warmup_steps를 계산합니다.
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    warmup_steps = int(dataset_size * num_train_epochs * warmup_steps_ratio / batch_size)
    print(f"Warmup steps: {warmup_steps}")

    # Logging strategy를 설정합니다.
    steps_per_epoch = max(1, (dataset_size + batch_size - 1) // batch_size)
    interval = max(1, steps_per_epoch // evals_per_epoch)

    training_config = SFTConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        optim=optim,
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
        warmup_steps=warmup_steps,
        bf16=bf16,
        logging_steps=interval,
        output_dir="outputs",
        remove_unused_columns=False,
        max_seq_length=context_length
    )

    # Trainer를 정의합니다.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_config,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # 모델 훈련 및 저장을 수행합니다.
    trainer.train()
    trainer.save_model("outputs")

    # Checkpoint 폴더를 삭제합니다.
    os.system("rm -rf outputs/checkpoint-*")

    print("Training Completed.")


if __name__ == "__main__":

     # .env 파일을 읽어서 환경변수로 등록합니다.
    load_dotenv()

    # Define training arguments for LoRA fine-tuning
    training_args = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
        "learning_rate": 2e-4,
        "warmup_steps_ratio": 0.1,
        "bf16": True,
        "optim": "adam_torch"
    }
    
    # Set model ID and context length
    model_id = "Qwen/Qwen2.5-0.5B"
    context_length = 4096
    data = "demo_data.jsonl"
    augmented_data = None

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, 
        context_length=context_length, 
        data=data,
        augmented_data=augmented_data,
        training_args=training_args
    )
