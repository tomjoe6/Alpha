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


def chinese_char_count(text: str) -> int:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text)


def looks_chinese(text: str, min_chinese_chars: int = 2) -> bool:
    if not text:
        return False
    if chinese_char_count(text) >= min_chinese_chars:
        return True
    # Allow very short Chinese prompts/responses with at least one CJK character.
    return chinese_char_count(text) >= 1 and not looks_english(text)


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


def is_high_quality_zh(prompt: str, response: str, min_chars: int = 8, max_chars: int = 500) -> bool:
    p = normalize_text(prompt)
    r = normalize_text(response)

    if len(p) < 2 or len(r) < 2:
        return False
    if not looks_chinese(p) or not looks_chinese(r):
        return False

    if len(p) > max_chars or len(r) > max_chars:
        return False
    if len(r) < min_chars:
        return False

    low_r = r.lower()
    low_p = p.lower()
    bad_responses = {"你好", "谢谢", "谢谢你", "是", "不是", "好", "可以", "行"}
    if low_r in bad_responses:
        return False
    if low_r == low_p:
        return False

    # Avoid Chinese spam/repetition.
    chars = [ch for ch in r if not ch.isspace()]
    if len(chars) >= 12:
        unique_ratio = len(set(chars)) / len(chars)
        if unique_ratio < 0.35:
            return False

    blocked_substrings = [
        "作为一个ai",
        "作为一个人工智能",
        "我不能",
        "我无法",
        "请提供更多信息",
        "如果你有任何问题",
    ]
    if any(s in low_r for s in blocked_substrings):
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


def extract_pairs_from_row(row: dict):
    # Common field pairs.
    field_pairs = [
        ("prompt", "response"),
        ("instruction", "output"),
        ("instruction", "response"),
        ("question", "answer"),
        ("query", "answer"),
        ("input", "output"),
        ("prompt", "completion"),
        ("instruction", "answer"),
        ("问", "答"),
        ("问题", "回答"),
        ("问题", "答案"),
        ("指令", "回复"),
        ("输入", "输出"),
        ("人类", "助手"),
    ]

    for a, b in field_pairs:
        if a in row and b in row and row.get(a) is not None and row.get(b) is not None:
            p = str(row.get(a, "")).strip()
            r = str(row.get(b, "")).strip()
            if p and r:
                return [(p, r)]

    # Conversation-like structures.
    for list_key in ("messages", "conversations", "conversation", "dialog", "dialogs", "chat", "turns"):
        msgs = row.get(list_key)
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue

        pairs = []
        for i in range(len(msgs) - 1):
            a = msgs[i]
            b = msgs[i + 1]
            if isinstance(a, dict) and isinstance(b, dict):
                ra = str(a.get("role", a.get("from", ""))).lower().strip()
                rb = str(b.get("role", b.get("from", ""))).lower().strip()
                ac = str(a.get("content", a.get("value", a.get("text", "")))).strip()
                bc = str(b.get("content", b.get("value", b.get("text", "")))).strip()
                if not ac or not bc:
                    continue
                if ra in {"user", "human", "prompter", "question", "input", "user1", "用户"} and rb in {"assistant", "gpt", "bot", "answer", "output", "assistant1", "助手"}:
                    pairs.append((ac, bc))
            elif isinstance(a, str) and isinstance(b, str):
                if a.strip() and b.strip():
                    pairs.append((a.strip(), b.strip()))
        if pairs:
            return pairs

    # Single text field with role markers or separators.
    for text_key in ("text", "content", "dialogue", "conversation_text"):
        text = row.get(text_key)
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        # Very lightweight split for common formats.
        if "assistant:" in text.lower() and "user:" in text.lower():
            blocks = re.split(r"(?i)(?:^|\n)\s*(?:user|human|用户)\s*:\s*", text)
            if len(blocks) >= 2:
                # Use last user/assistant pair if present.
                last = blocks[-1]
                pair = _extract_last_assistant_block(last)
                if pair:
                    p, r = pair
                    return [(p.strip(), r.strip())]

    return []


def _extract_last_assistant_block(text: str):
    low = text.lower()
    idx = low.rfind("assistant:")
    if idx == -1:
        return None
    p = text[:idx].strip()
    r = text[idx + len("assistant:") :].strip()
    if p and r:
        return p, r
    return None


def iter_dataset_auto(repo_id: str, max_samples: int, language: str):
    ds = load_dataset(repo_id, split="train")
    count = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        for prompt, response in extract_pairs_from_row(row):
            yield repo_id, prompt, response
            count += 1
            if max_samples > 0 and count >= max_samples:
                return


def iter_dataset_auto_zh(repo_id: str, max_samples: int):
    ds = load_dataset(repo_id, split="train")
    count = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        for prompt, response in extract_pairs_from_row(row):
            prompt = normalize_text(str(prompt))
            response = normalize_text(str(response))
            if not looks_chinese(prompt) or not looks_chinese(response):
                continue
            yield repo_id, prompt, response
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
    parser.add_argument("--language", default="zh", choices=["en", "zh"])
    parser.add_argument(
        "--sources",
        default=None,
        help="Comma-separated sources. If omitted, a language-specific default list is used.",
    )
    parser.add_argument("--max_per_source", type=int, default=2000)
    parser.add_argument("--min_response_words", type=int, default=4)
    parser.add_argument("--max_response_words", type=int, default=120)
    args = parser.parse_args()

    default_sources = {
        "en": "daily_dialog,ultrachat_200k,oasst1,hh_rlhf,dolly_15k",
        "zh": "svjack/GLM-Open-Dialogue-Chinese-Dataset-v2,ticoAg/Chinese-medical-dialogue,DialogueCharacter/chinese_general_instruction_with_reward_score,DialogueCharacter/chinese_dialogue_instruction_with_reward_score_judged_by_13B_baichuan2,Nexdata/100000_Instruction_Following_Evaluation_SFT_for_Chinese_LLM_Text_Data,lucky2me/any-clm_text_conversation_common_zh",
    }
    source_list = args.sources or default_sources[args.language]
    source_names = [s.strip() for s in source_list.split(",") if s.strip()]
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
        print(f"[INFO] Loading source: {source_name}")
        try:
            if source_name in iterators:
                iterator = iterators[source_name](args.max_per_source)
            else:
                iterator = iter_dataset_auto_zh(source_name, args.max_per_source)
            accepted = 0
            total = 0
            for source, prompt, response in iterator:
                total += 1
                prompt = normalize_text(str(prompt))
                response = normalize_text(str(response))

                if args.language == "zh":
                    if not is_high_quality_zh(prompt, response):
                        continue
                else:
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
