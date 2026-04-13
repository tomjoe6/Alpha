import argparse
from difflib import SequenceMatcher
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="microsoft/DialoGPT-small")
    p.add_argument("--adapter_dir", default=None, help="LoRA adapter directory, e.g. output_lora")
    p.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for the first GPU")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling for more diverse replies")
    p.add_argument("--repetition_penalty", type=float, default=1.15, help="Penalty to reduce repeated text")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Avoid repeating n-grams of this size")
    p.add_argument("--history_turns", type=int, default=3, help="How many turns of history to keep")
    p.add_argument("--anti_copy_retry", type=int, default=1, help="Retry count when reply looks like copied user text")
    p.add_argument("--log_file", default="data/dialogs.jsonl")
    p.add_argument(
        "--system_prompt",
        default="You are a helpful, concise assistant that always replies in English. Do not mirror or copy user wording/casing.",
    )
    p.add_argument("--demo", action="store_true", help="Run built-in examples and exit")
    return p.parse_args()


def load_model_and_tokenizer(model_name_or_path: str, adapter_dir: str | None, device_id: int):
    print(f"Loading model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if adapter_dir:
        print(f"Loading LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

        device = torch.device(f"cuda:{device_id}") if device_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

    model.eval()
    return tokenizer, model


def build_prompt(system_prompt: str, history: list[tuple[str, str]], user_text: str, history_turns: int):
    turns = history[-history_turns:]
    parts = [system_prompt.strip(), ""]
    for user, assistant in turns:
        parts.append(f"User: {user}\nAssistant: {assistant}")
    parts.append(f"User: {user_text}\nAssistant:")
    return "\n".join(parts)


def generate_reply(tokenizer, model, prompt: str, args):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_cfg_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        gen_cfg_kwargs["temperature"] = args.temperature
        gen_cfg_kwargs["top_p"] = args.top_p
    gen_cfg = GenerationConfig(**gen_cfg_kwargs)

    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_cfg)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not reply:
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

    # Keep only the assistant segment and trim obvious repetitive degeneration.
    reply = re.split(r"\n(?:User|Assistant):", reply, maxsplit=1)[0].strip()
    reply = re.sub(r"(?:(?:\b:D\b|\bD:\b|:D|D:)[\s:]*){6,}", ":D", reply)
    if len(reply) > 30 and len(set(reply)) / max(len(reply), 1) < 0.12:
        reply = "I generated a repetitive response. Please ask again and I will answer more clearly."
    return reply


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_copy_like(user_text: str, reply: str) -> bool:
    u = _normalize_for_compare(user_text)
    r = _normalize_for_compare(reply)
    if not u or not r:
        return False
    if u == r:
        return True

    u_tokens = u.split()
    r_tokens = r.split()
    overlap = len(set(u_tokens) & set(r_tokens)) / max(len(set(u_tokens) | set(r_tokens)), 1)
    sim = SequenceMatcher(None, u, r).ratio()

    # Short high-overlap outputs are usually mirrored or lightly edited echoes.
    if (sim >= 0.72 and len(r_tokens) <= 14) or overlap >= 0.82:
        return True
    return False


def generate_non_copy_reply(tokenizer, model, user_text: str, prompt: str, args):
    normalized_user = _normalize_for_compare(user_text)
    if "do not copy me" in normalized_user or "dont copy me" in normalized_user:
        return "Understood. I will not mirror your wording and will answer in my own words."
    if normalized_user in {"no", "nope", "nah"}:
        return "Understood. Tell me what you want me to do instead."

    response = generate_reply(tokenizer, model, prompt, args)
    if not is_copy_like(user_text, response):
        return response

    anti_copy_prompt = (
        prompt
        + "\nInstruction: Do NOT copy, mirror, or restate the user's wording. "
        + "Respond with original phrasing and useful content.\nAssistant:"
    )
    retries = max(0, int(args.anti_copy_retry))
    for _ in range(retries):
        response = generate_reply(tokenizer, model, anti_copy_prompt, args)
        if not is_copy_like(user_text, response):
            return response

    return "Understood. I will reply in original wording instead of mirroring your phrasing."


def log_dialog(log_file: str, prompt: str, response: str):
    out_file = Path(log_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_demo(tokenizer, model, args):
    prompts = [
        "Hello, how is the weather today?",
        "Write a short introduction to machine learning for a beginner.",
        "Translate this sentence into English: I like studying artificial intelligence.",
    ]
    history: list[tuple[str, str]] = []
    for i, user_text in enumerate(prompts, 1):
        print(f"\n--- Prompt {i} ---")
        print("User:", user_text)
        prompt = build_prompt(args.system_prompt, history, user_text, args.history_turns)
        response = generate_non_copy_reply(tokenizer, model, user_text, prompt, args)
        print("Assistant:", response)
        history.append((user_text, response))
        log_dialog(args.log_file, user_text, response)


def run_chat(tokenizer, model, args):
    print("Type exit / quit / q to end the conversation.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            user_text = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExited.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Exited.")
            break

        prompt = build_prompt(args.system_prompt, history, user_text, args.history_turns)
        response = generate_non_copy_reply(tokenizer, model, user_text, prompt, args)
        print("Assistant:", response)
        history.append((user_text, response))
        log_dialog(args.log_file, user_text, response)


def main():
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, args.adapter_dir, args.device)
    if args.demo:
        run_demo(tokenizer, model, args)
    else:
        run_chat(tokenizer, model, args)


if __name__ == "__main__":
    main()
