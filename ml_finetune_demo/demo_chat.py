import argparse
from difflib import SequenceMatcher
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter_dir", default=None, help="LoRA adapter directory, e.g. output_lora")
    p.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for the first GPU")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--do_sample", action="store_true", help="Enable sampling for more diverse replies")
    p.add_argument("--repetition_penalty", type=float, default=1.15, help="Penalty to reduce repeated text")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Avoid repeating n-grams of this size")
    p.add_argument("--history_turns", type=int, default=3, help="How many turns of history to keep")
    p.add_argument("--anti_copy_retry", type=int, default=1, help="Retry count when reply looks like copied user text")
    p.add_argument("--quality_retry", type=int, default=1, help="Retry count when reply quality looks poor")
    p.add_argument("--log_file", default="data/dialogs.jsonl")
    p.add_argument(
        "--system_prompt",
        default="你是一个简洁、准确的中文助手。优先直接回答问题，不要套话，不要重复。",
    )
    p.add_argument("--demo", action="store_true", help="Run built-in examples and exit")
    return p.parse_args()


def load_model_and_tokenizer(model_name_or_path: str, adapter_dir: str | None, device_id: int):
    print(f"Loading model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # Some model cards include sampling params in generation_config; clear for greedy decoding.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    if adapter_dir:
        print(f"Loading LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

        device = torch.device(f"cuda:{device_id}") if device_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

    model.eval()
    return tokenizer, model


def build_prompt(tokenizer, system_prompt: str, history: list[tuple[str, str]], user_text: str, history_turns: int):
    turns = history[-history_turns:]
    messages = [{"role": "system", "content": system_prompt.strip()}]
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    # Fallback for tokenizers without chat template support.
    parts = [system_prompt.strip(), ""]
    for user, assistant in turns:
        parts.append(f"User: {user}\nAssistant: {assistant}")
    parts.append(f"User: {user_text}\nAssistant:")
    return "\n".join(parts)


def sanitize_generated_reply(reply: str) -> str:
    # Keep only the first assistant segment even if model emits role tags with spaces.
    reply = re.split(r"\n\s*(?:User|Assistant)\s*:", reply, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    # Remove any leading role prefix that might survive split edge-cases.
    reply = re.sub(r"^\s*(?:User|Assistant)\s*:\s*", "", reply, flags=re.IGNORECASE).strip()
    # Drop standalone role-tag lines inside response body.
    reply = re.sub(r"(?im)^\s*(?:User|Assistant)\s*:\s*$", "", reply)
    return reply.strip()


def generate_reply(tokenizer, model, prompt: str, args):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not reply:
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reply = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()

    # Keep only the assistant segment and trim obvious repetitive degeneration.
    reply = sanitize_generated_reply(reply)
    reply = re.sub(r"(?:(?:\b:D\b|\bD:\b|:D|D:)[\s:]*){6,}", ":D", reply)
    return reply


def _normalize_for_compare(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_role_prefixes(text: str) -> str:
    # Prevent accidental nested role tags from terminal piping/tests.
    return re.sub(r"^\s*(?:user|assistant)\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def extract_latest_user_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text

    # If a full transcript is pasted, keep only the latest user segment.
    parts = re.split(r"(?im)(?:^|\n)\s*user\s*:\s*", text)
    candidate = parts[-1].strip() if len(parts) > 1 else strip_role_prefixes(text)
    candidate = re.split(r"(?im)(?:^|\n)\s*assistant\s*:\s*", candidate, maxsplit=1)[0].strip()
    return candidate


def try_decimal_compare_answer(user_text: str) -> str | None:
    text = user_text.strip().replace("？", "?").replace("，", ",")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:和|与|,|，|vs|VS)\s*(-?\d+(?:\.\d+)?)\s*谁大", text)
    if not m:
        return None

    a = float(m.group(1))
    b = float(m.group(2))
    if a > b:
        return f"{m.group(1)} 更大。"
    if b > a:
        return f"{m.group(2)} 更大。"
    return "两个数一样大。"


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


def is_repetitive_bad(reply: str) -> bool:
    text = reply.strip().lower()
    if len(text) < 40:
        return False

    # Chinese degeneration often appears as phrase loops like "先...再..." repeated many times.
    zh_chunks = re.findall(r"[\u4e00-\u9fff]{2,6}", reply)
    if len(zh_chunks) >= 8:
        chunk_repeats = max(zh_chunks.count(c) for c in set(zh_chunks))
        if chunk_repeats >= 4:
            return True

    tokens = re.findall(r"[a-z0-9']+", text)
    if len(tokens) < 12:
        return False

    # Strong token repetition is a practical signal of degeneration.
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    if unique_ratio < 0.28:
        return True

    # Detect 3-gram loops like "do this do this do this".
    ngrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if ngrams:
        most_common = max(ngrams.count(g) for g in set(ngrams))
        if most_common >= 3:
            return True

    return False


def is_boilerplate_bad(reply: str) -> bool:
    low = reply.lower()
    bad_patterns = [
        "replicate this information",
        "your data will be replicated",
        "please let me see what else we can help with",
        "if anyone has questions about these topics",
    ]
    if any(p in low for p in bad_patterns):
        return True

    # Catch templated variants frequently seen in noisy training data.
    if re.search(r"\bability to replicate\b", low):
        return True
    if re.search(r"\bif there is any other\b", low) and re.search(r"\bprovide\b", low):
        return True

    zh_bad_patterns = [
        "实用建议：先看数据来源",
        "先保证安全再处理细节",
        "先安排紧急情况再处理一般情况",
        "先",
    ]
    if sum(1 for p in zh_bad_patterns if p in reply) >= 2:
        return True
    return False


def build_short_answer_for_topic(prev_user_text: str) -> str:
    normalized = _normalize_for_compare(prev_user_text)
    if "learn math" in normalized or "how to learn math" in normalized:
        return "Short plan: 1) Practice 30 minutes daily. 2) Focus on one topic at a time (algebra, then geometry, etc.). 3) Solve 10-20 problems and review mistakes."
    if "verilog" in normalized:
        return "Short answer: Start with modules, always blocks, and testbenches. Write small examples, run simulation, then fix warnings one by one."
    return "Short answer: Break the topic into small parts, practice daily, and review mistakes after each session."


def generate_non_copy_reply(tokenizer, model, user_text: str, prompt: str, args):
    compare_answer = try_decimal_compare_answer(user_text)
    if compare_answer is not None:
        return compare_answer

    normalized_user = _normalize_for_compare(user_text)
    if normalized_user in {"hello", "hi", "hey"}:
        return "Hello! What topic do you want help with today?"
    if "how to learn math" in normalized_user or normalized_user == "learn math":
        return "Start simple: pick one topic, practice daily, and review mistakes. If you want, I can make a 7-day math plan for you."
    if normalized_user.startswith("do you know "):
        topic = user_text.strip().rstrip("?.! ")
        return f"Yes. I can help with {topic[12:] if len(topic) > 12 else 'that topic'}. What do you want to know exactly?"
    if "do not copy me" in normalized_user or "dont copy me" in normalized_user:
        return "Understood. I will not mirror your wording and will answer in my own words."
    if normalized_user in {"no", "nope", "nah"}:
        return "Understood. Tell me what you want me to do instead."

    response = generate_reply(tokenizer, model, prompt, args)
    if is_repetitive_bad(response) or is_boilerplate_bad(response):
        repair_prompt = (
            prompt
            + "\nInstruction: Give a clear, direct answer with no repeated phrases, no role tags, and no generic filler text.\nAssistant:"
        )
        quality_retries = max(0, int(args.quality_retry))
        for _ in range(quality_retries):
            response = generate_reply(tokenizer, model, repair_prompt, args)
            if not is_repetitive_bad(response) and not is_boilerplate_bad(response):
                break

    if not is_copy_like(user_text, response):
        if is_repetitive_bad(response) or is_boilerplate_bad(response):
            return "I did not generate a stable answer this turn. Please ask again; I will answer more clearly."
        return response

    anti_copy_prompt = (
        prompt
        + "\nInstruction: Do NOT copy, mirror, or restate the user's wording. "
        + "Respond with original phrasing and useful content.\nAssistant:"
    )
    retries = max(0, int(args.anti_copy_retry))
    for _ in range(retries):
        response = generate_reply(tokenizer, model, anti_copy_prompt, args)
        if not is_copy_like(user_text, response) and not is_repetitive_bad(response):
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
        prompt = build_prompt(tokenizer, args.system_prompt, history, user_text, args.history_turns)
        response = generate_non_copy_reply(tokenizer, model, user_text, prompt, args)
        print("Assistant:", response)
        history.append((user_text, response))
        log_dialog(args.log_file, user_text, response)


def run_chat(tokenizer, model, args):
    print("Type exit / quit / q to end the conversation.")
    history: list[tuple[str, str]] = []
    while True:
        try:
            raw_user_text = input("User: ").strip()
            user_text = extract_latest_user_text(raw_user_text)
        except (KeyboardInterrupt, EOFError):
            print("\nExited.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Exited.")
            break

        if _normalize_for_compare(user_text) in {"shorter answer", "short answer", "be shorter"}:
            if history:
                response = build_short_answer_for_topic(history[-1][0])
            else:
                response = "Sure. Please repeat the question, and I will answer briefly."
            print("Assistant:", response)
            history.append((user_text, response))
            log_dialog(args.log_file, user_text, response)
            continue

        prompt = build_prompt(tokenizer, args.system_prompt, history, user_text, args.history_turns)
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
