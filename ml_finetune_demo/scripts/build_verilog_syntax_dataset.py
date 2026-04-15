import argparse
import json
import random
import re
from pathlib import Path


HEADER_BLOCK_RE = re.compile(r"/\*(.*?)\*/", re.S)
META_LINE_RE = re.compile(r"^\s*:(?P<key>[a-zA-Z0-9_]+)\s*:\s*(?P<val>.*)$")


def parse_svtest_metadata(text: str) -> dict:
    m = HEADER_BLOCK_RE.search(text)
    if not m:
        return {}

    meta = {}
    for line in m.group(1).splitlines():
        mm = META_LINE_RE.match(line.strip())
        if mm:
            key = mm.group("key").strip()
            val = mm.group("val").strip()
            meta[key] = val
    return meta


def infer_label_from_metadata(meta: dict) -> tuple[int, str]:
    reason = meta.get("should_fail_because", "").strip()
    if reason:
        return 0, reason
    return 1, ""


def clean_code(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Build labeled syntax dataset from sv-tests")
    p.add_argument("--svtests_dir", default="third_party/sv-tests/tests")
    p.add_argument("--output_train", default="data/verilog_syntax_train.jsonl")
    p.add_argument("--output_valid", default="data/verilog_syntax_valid.jsonl")
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_chars", type=int, default=12000)
    p.add_argument("--min_chars", type=int, default=30)
    p.add_argument("--summary", default="data/verilog_syntax_summary.json")
    args = p.parse_args()

    root = Path(args.svtests_dir)
    if not root.exists():
        raise FileNotFoundError(f"sv-tests directory not found: {root}")

    files = []
    for ext in ("*.sv", "*.v"):
        files.extend(root.rglob(ext))

    rows = []
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        code = clean_code(raw, max_chars=args.max_chars)
        if len(code) < args.min_chars:
            continue

        meta = parse_svtest_metadata(raw)
        label, fail_reason = infer_label_from_metadata(meta)
        rows.append(
            {
                "task": "verilog_syntax_check",
                "input": code,
                "label": label,
                "source": "sv-tests",
                "file": str(fp.as_posix()),
                "name": meta.get("name", ""),
                "tags": meta.get("tags", ""),
                "fail_reason": fail_reason,
            }
        )

    if not rows:
        raise RuntimeError("No dataset rows generated. Check input path and file formats.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_valid = int(len(rows) * args.valid_ratio)
    n_valid = max(1, n_valid)
    valid = rows[:n_valid]
    train = rows[n_valid:]

    write_jsonl(Path(args.output_train), train)
    write_jsonl(Path(args.output_valid), valid)

    summary = {
        "total": len(rows),
        "train": len(train),
        "valid": len(valid),
        "valid_ratio": args.valid_ratio,
        "positive_valid_syntax": sum(1 for r in rows if r["label"] == 1),
        "negative_invalid_syntax": sum(1 for r in rows if r["label"] == 0),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[DONE] train -> {args.output_train}")
    print(f"[DONE] valid -> {args.output_valid}")


if __name__ == "__main__":
    main()
