from transformers import pipeline
import os
import json
from datetime import datetime
from pathlib import Path

# Use an English base model for demo (gpt2). If you want to test the local checkpoint,
# set model_dir = "output\\checkpoint-3" instead.
model_dir = "gpt2"
print(f"Loading model from: {model_dir}")

device = 0
from transformers import GenerationConfig
generator = pipeline("text-generation", model=model_dir, device=device)

prompts = [
    "你好，今天天气怎么样？",
    "请给我写一段关于机器学习入门的简短介绍。",
    "帮我把这句话翻译成英文：我喜欢学习人工智能。",
]

for i, p in enumerate(prompts, 1):
    print(f"\n--- Prompt {i} ---")
    print("User:", p)
    gen_cfg = GenerationConfig(max_new_tokens=80, do_sample=False)
    out = generator(p, generation_config=gen_cfg)
    generated = out[0]["generated_text"]
    print("Model:", generated)
    # save dialog to data/dialogs.jsonl
    out_file = Path("data/dialogs.jsonl")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "prompt": p,
        "response": generated,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
