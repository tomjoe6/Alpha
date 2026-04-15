import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument(
        "--adapters",
        default="output_lora,output_lora_v2,output_lora_v3",
        help="Comma-separated adapter dirs. Use 'none' to evaluate base model only.",
    )
    p.add_argument("--prompts_file", default=None, help="Optional jsonl file with {'prompt': '...'}")
    p.add_argument("--output_json", default="reports/adapter_compare.json")
    p.add_argument("--output_md", default="reports/adapter_compare.md")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--device", type=int, default=-1)
    return p.parse_args()


def load_prompts(path: str | None):
    if not path:
        return [
            "Explain machine learning in two sentences.",
            "How can I start learning Python as a beginner?",
            "Why does my chatbot keep repeating itself?",
            "Give me a short 3-step study plan for AI.",
            "What is the difference between training and inference?",
        ]

    prompts = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(rec.get("prompt", "")).strip()
            if text:
                prompts.append(text)
    return prompts


def get_device(device_id: int):
    if device_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def build_prompt(user_text: str):
    system_prompt = "You are a helpful, concise assistant that always replies in English."
    return f"{system_prompt}\nUser: {user_text}\nAssistant:"


def generate_reply(tokenizer, model, prompt: str, max_new_tokens: int):
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_model(base_model: str, adapter_dir: str | None, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    device = get_device(args.device)

    adapter_items = [x.strip() for x in args.adapters.split(",") if x.strip()]
    if not adapter_items:
        adapter_items = ["none"]

    results = []
    for item in adapter_items:
        adapter_dir = None if item.lower() == "none" else item
        run_name = "base_model" if adapter_dir is None else adapter_dir

        print(f"[INFO] Evaluating: {run_name}")
        tokenizer, model = load_model(args.base_model, adapter_dir, device)

        run_rows = []
        for p in prompts:
            prompt = build_prompt(p)
            response = generate_reply(tokenizer, model, prompt, args.max_new_tokens)
            run_rows.append({"prompt": p, "response": response})

        results.append({"run": run_name, "rows": run_rows})

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    output_md = Path(args.output_md)
    lines = ["# Adapter Comparison Report", ""]
    for run in results:
        lines.append(f"## {run['run']}")
        lines.append("")
        for i, row in enumerate(run["rows"], start=1):
            lines.append(f"{i}. Prompt: {row['prompt']}")
            lines.append(f"   Response: {row['response']}")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[DONE] JSON report: {output_json}")
    print(f"[DONE] Markdown report: {output_md}")


if __name__ == "__main__":
    main()
