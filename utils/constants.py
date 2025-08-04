qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "system": "You are a helpful assistant.",
}

model2template = {
    "Qwen/Qwen2.5-0.5B": qwen_template,
    "Qwen/Qwen2.5-1.5B": qwen_template,
    "Qwen/Qwen2.5-3B": qwen_template,
    "Qwen/Qwen2.5-7B": qwen_template,
    "Qwen/Qwen2.5-0.5B-Instruct": qwen_template,
    "Qwen/Qwen2.5-1.5B-Instruct": qwen_template,
    "Qwen/Qwen2.5-3B-Instruct": qwen_template,
    "Qwen/Qwen2.5-7B-Instruct": qwen_template,
}

model2size = {
    "Qwen/Qwen2.5-0.5B": 494_000_000,
    "Qwen/Qwen2.5-1.5B": 1_544_000_000,
    "Qwen/Qwen2.5-3B": 3_090_000_000,
    "Qwen/Qwen2.5-7B": 7_615_000_000,
    "Qwen/Qwen2.5-0.5B-Instruct": 494_000_000,
    "Qwen/Qwen2.5-1.5B-Instruct": 1_544_000_000,
    "Qwen/Qwen2.5-3B-Instruct": 3_090_000_000,
    "Qwen/Qwen2.5-7B-Instruct": 7_615_000_000,
}

model2base_model = {
    "Qwen/Qwen2.5-0.5B": "qwen2.5",
    "Qwen/Qwen2.5-1.5B": "qwen2.5",
    "Qwen/Qwen2.5-3B": "qwen2.5",
    "Qwen/Qwen2.5-7B": "qwen2.5",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2.5",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2.5",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5",
}
