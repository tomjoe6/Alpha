import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_english(text: str, min_alpha_ratio: float = 0.55) -> bool:
    if not text:
        return False
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in text)
    if letters == 0:
        return False
    return ascii_letters / max(letters, 1) >= min_alpha_ratio


def is_high_quality(prompt: str, response: str, min_words: int = 4, max_words: int = 120) -> bool:
    p = normalize_text(prompt)
    r = normalize_text(response)

    if len(p) < 3 or len(r) < 8:
        return False

    if not looks_english(p) or not looks_english(r):
        return False

    p_words = p.split()
    r_words = r.split()
    if len(r_words) < min_words or len(r_words) > max_words:
        return False

    low_r = r.lower()
    low_p = p.lower()

    # Remove low-information and repetitive outputs.
    bad_responses = {"hello", "hi", "ok", "okay", "thanks", "thank you", "yes", "no"}
    if low_r in bad_responses:
        return False

    blocked_substrings = [
        "as an ai language model",
        "i cannot assist with",
        "i can't assist with",
        "i do not have personal",
    ]
    if any(s in low_r for s in blocked_substrings):
        return False

    # Avoid direct echoing.
    if low_r == low_p:
        return False

    # Avoid obvious repetition loops.
    tokens = [t.lower() for t in r_words]
    if len(tokens) >= 6:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.45:
            return False

    return True


def iter_daily_dialog(max_samples: int):
    ds = load_dataset("daily_dialog", split="train")
    count = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        dialog = row.get("dialog", [])
        if not isinstance(dialog, list) or len(dialog) < 2:
            continue
        for i in range(len(dialog) - 1):
            prompt = dialog[i]
            response = dialog[i + 1]
            yield "daily_dialog", prompt, response
            count += 1
            if max_samples > 0 and count >= max_samples:
                return


def iter_ultrachat(max_samples: int):
    # UltraChat format may vary by revision; this parser handles common message schema.
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    count = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        msgs = row.get("messages", [])
        if not isinstance(msgs, list):
            continue
        for i in range(len(msgs) - 1):
            a = msgs[i]
            b = msgs[i + 1]
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            if a.get("role") != "user" or b.get("role") != "assistant":
                continue
            prompt = a.get("content", "")
            response = b.get("content", "")
            yield "ultrachat_200k", prompt, response
            count += 1
            if max_samples > 0 and count >= max_samples:
                return


def iter_oasst1(max_samples: int):
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    rows = []
    for row in ds:
        if not isinstance(row, dict):
            continue
        if row.get("lang") != "en":
            continue
        rows.append(row)

    by_id = {}
    for row in rows:
        mid = row.get("message_id")
        if mid:
            by_id[mid] = row

    count = 0
    for row in rows:
        if row.get("role") != "assistant":
            continue
        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        if parent.get("role") != "prompter":
            continue

        prompt = parent.get("text", "")
        response = row.get("text", "")
        yield "oasst1", prompt, response
        count += 1
        if max_samples > 0 and count >= max_samples:
            return


def _extract_hh_pair(chosen_text: str):
    if not chosen_text:
        return None
    matches = re.findall(r"Human:\s*(.*?)\n\nAssistant:\s*(.*?)(?=\n\nHuman:|$)", chosen_text, flags=re.S)
    if not matches:
        return None
    prompt, response = matches[-1]
    return prompt.strip(), response.strip()


def iter_hh_rlhf(max_samples: int):
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    count = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        pair = _extract_hh_pair(str(row.get("chosen", "")))
        if not pair:
            continue
        prompt, response = pair
        yield "hh_rlhf", prompt, response
        count += 1
        if max_samples > 0 and count >= max_samples:
            return


def iter_dolly(max_samples: int):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    count = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        instruction = str(row.get("instruction", "")).strip()
        context = str(row.get("context", "")).strip()
        response = str(row.get("response", "")).strip()
        if not instruction or not response:
            continue
        prompt = instruction if not context else f"{instruction}\nContext: {context}"
        yield "dolly_15k", prompt, response
        count += 1
        if max_samples > 0 and count >= max_samples:
            return


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/train_from_hf.jsonl")
    parser.add_argument(
        "--sources",
        default="daily_dialog,ultrachat_200k,oasst1,hh_rlhf,dolly_15k",
        help="Comma-separated sources: daily_dialog, ultrachat_200k, oasst1, hh_rlhf, dolly_15k",
    )
    parser.add_argument("--max_per_source", type=int, default=2000)
    parser.add_argument("--min_response_words", type=int, default=4)
    parser.add_argument("--max_response_words", type=int, default=120)
    args = parser.parse_args()

    source_names = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows = []
    seen = set()

    iterators = {
        "daily_dialog": iter_daily_dialog,
        "ultrachat_200k": iter_ultrachat,
        "oasst1": iter_oasst1,
        "hh_rlhf": iter_hh_rlhf,
        "dolly_15k": iter_dolly,
    }

    for source_name in source_names:
        if source_name not in iterators:
            print(f"[WARN] Unknown source: {source_name}")
            continue

        print(f"[INFO] Loading source: {source_name}")
        try:
            iterator = iterators[source_name](args.max_per_source)
            accepted = 0
            total = 0
            for source, prompt, response in iterator:
                total += 1
                prompt = normalize_text(str(prompt))
                response = normalize_text(str(response))

                if not is_high_quality(
                    prompt,
                    response,
                    min_words=args.min_response_words,
                    max_words=args.max_response_words,
                ):
                    continue

                key = (prompt.lower(), response.lower())
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        "prompt": f"User: {prompt}\nAssistant:",
                        "response": f" {response}",
                        "source": source,
                    }
                )
                accepted += 1
            print(f"[INFO] {source_name}: accepted {accepted} / scanned {total}")
        except Exception as exc:
            print(f"[WARN] Failed to load {source_name}: {exc}")

    output_path = Path(args.output)
    write_jsonl(output_path, rows)
    print(f"[DONE] Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    main()
