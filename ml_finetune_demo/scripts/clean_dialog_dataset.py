import argparse
import json
import re
import shutil
from pathlib import Path


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_role_prefix(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(?:user|human|assistant|gpt|用户|助手)\s*:\s*", "", text, flags=re.I)
    return text


def looks_english(text: str, min_alpha_ratio: float = 0.55) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    ascii_letters = sum(("a" <= ch.lower() <= "z") for ch in text)
    if letters == 0:
        return False
    return ascii_letters / max(letters, 1) >= min_alpha_ratio


def chinese_char_count(text: str) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text)


def looks_chinese(text: str, min_chinese_chars: int = 2) -> bool:
    return chinese_char_count(text) >= min_chinese_chars


def too_repetitive(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if len(tokens) < 18:
        return False
    uniq_ratio = len(set(tokens)) / max(len(tokens), 1)
    return uniq_ratio < 0.35


def too_repetitive_cn(text: str) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < 16:
        return False
    uniq_ratio = len(set(chars)) / max(len(chars), 1)
    return uniq_ratio < 0.4


def shorten_text(text: str, max_chars: int) -> str:
    text = normalize_text(strip_role_prefix(text))
    if len(text) <= max_chars:
        return text

    pieces = re.split(r"(?<=[。！？!?；;\.])\s*", text)
    kept = []
    total = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        next_total = total + len(piece)
        if kept:
            next_total += 1
        if next_total > max_chars:
            break
        kept.append(piece)
        total = next_total

    if kept:
        return "".join(kept).strip()

    return text[:max_chars].rstrip("，,。！？!?；;：:、 ")


def is_unusable(prompt: str, response: str, language: str = "auto") -> bool:
    p = normalize_text(strip_role_prefix(prompt))
    r = normalize_text(strip_role_prefix(response))

    if len(p) < 5 or len(r) < 8:
        return True

    p_cn = looks_chinese(p)
    r_cn = looks_chinese(r)
    p_en = looks_english(p)
    r_en = looks_english(r)

    if language == "zh":
        if not (p_cn and r_cn):
            return True
        if too_repetitive_cn(r):
            return True
    elif language == "en":
        if not (p_en and r_en):
            return True
        if too_repetitive(r):
            return True
    else:
        if not ((p_cn and r_cn) or (p_en and r_en)):
            return True
        if (p_cn and r_cn and too_repetitive_cn(r)) or (p_en and r_en and too_repetitive(r)):
            return True

    if "user:" in r.lower() or "assistant:" in r.lower():
        return True

    bad_snippets = [
        "(list alternatives here)",
        "ability to replicate",
        "your data will be replicated",
        "please let me see what else we can help with",
        "if anyone has questions about these topics",
        "my understanding you have to use your own language",
        "can we talk about how much of an improvement",
    ]
    low = r.lower()
    if any(s in low for s in bad_snippets):
        return True

    return False


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Clean low-quality dialogue training samples")
    p.add_argument("--input", default="data/train_from_hf_zh.jsonl")
    p.add_argument("--output", default="data/train_from_hf_zh_clean.jsonl")
    p.add_argument("--inplace", action="store_true", help="Rewrite input file in place and keep .bak backup")
    p.add_argument("--summary", default="data/train_from_hf_zh_clean_summary.json")
    p.add_argument("--language", default="auto", choices=["auto", "en", "zh"])
    p.add_argument("--max_prompt_chars", type=int, default=120)
    p.add_argument("--max_response_chars", type=int, default=180)
    args = p.parse_args()

    in_path = Path(args.input)
    rows = read_jsonl(in_path)

    cleaned = []
    seen = set()
    removed = 0
    truncated = 0
    for row in rows:
        prompt = strip_role_prefix(str(row.get("prompt", "")))
        response = strip_role_prefix(str(row.get("response", "")))
        if is_unusable(prompt, response, language=args.language):
            removed += 1
            continue
        clean_prompt = shorten_text(prompt, args.max_prompt_chars)
        clean_response = shorten_text(response, args.max_response_chars)
        if clean_prompt != normalize_text(prompt) or clean_response != normalize_text(response):
            truncated += 1
        key = (normalize_text(clean_prompt).lower(), normalize_text(clean_response).lower())
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        row_out = dict(row)
        row_out["prompt"] = clean_prompt
        row_out["response"] = " " + clean_response.lstrip()
        cleaned.append(row_out)

    out_path = Path(args.output)
    if args.inplace:
        backup_path = in_path.with_suffix(in_path.suffix + ".bak")
        shutil.copy2(in_path, backup_path)
        out_path = in_path

    write_jsonl(out_path, cleaned)

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "total": len(rows),
        "kept": len(cleaned),
        "removed": removed,
        "truncated": truncated,
        "kept_ratio": round(len(cleaned) / max(len(rows), 1), 4),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
