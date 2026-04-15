import argparse
import json
import random
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
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


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def mutate_remove_semicolon(code: str) -> str | None:
    lines = code.splitlines()
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.endswith(";") and not s.startswith("//"):
            lines[i] = ln.rstrip().rstrip(";")
            return "\n".join(lines)
    return None


def mutate_break_endmodule(code: str) -> str | None:
    if "endmodule" not in code:
        return None
    return code.replace("endmodule", "endmodulee", 1)


def mutate_unbalanced_paren(code: str) -> str | None:
    m = re.search(r"\([^\)]*\)", code)
    if not m:
        return None
    a, b = m.span()
    frag = code[a:b]
    if len(frag) < 2:
        return None
    return code[:a] + frag[:-1] + code[b:]


def mutate_missing_end(code: str) -> str | None:
    if "\nend\n" in code:
        return code.replace("\nend\n", "\n", 1)
    if code.strip().endswith("\nend"):
        return code[: -len("\nend")]
    return None


MUTATORS = [
    ("remove_semicolon", mutate_remove_semicolon),
    ("break_endmodule", mutate_break_endmodule),
    ("unbalanced_paren", mutate_unbalanced_paren),
    ("missing_end", mutate_missing_end),
]


def main():
    p = argparse.ArgumentParser(description="Augment Verilog syntax dataset with synthetic invalid samples")
    p.add_argument("--input_train", default="data/verilog_syntax_train.jsonl")
    p.add_argument("--input_valid", default="data/verilog_syntax_valid.jsonl")
    p.add_argument("--output_train", default="data/verilog_syntax_train_aug.jsonl")
    p.add_argument("--output_valid", default="data/verilog_syntax_valid_aug.jsonl")
    p.add_argument("--neg_per_positive", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary", default="data/verilog_syntax_aug_summary.json")
    args = p.parse_args()

    rng = random.Random(args.seed)
    train = read_jsonl(Path(args.input_train))
    valid = read_jsonl(Path(args.input_valid))

    positives = [r for r in train if int(r.get("label", 1)) == 1]
    seen_inputs = {str(r.get("input", "")) for r in train}
    synthetic = []

    for row in positives:
        generated = 0
        mutator_order = MUTATORS[:]
        rng.shuffle(mutator_order)
        for m_name, mut in mutator_order:
            if generated >= args.neg_per_positive:
                break
            new_code = mut(str(row.get("input", "")))
            if not new_code:
                continue
            if new_code in seen_inputs:
                continue
            seen_inputs.add(new_code)
            out = dict(row)
            out["input"] = new_code
            out["label"] = 0
            out["source"] = "synthetic_mutation"
            out["fail_reason"] = f"synthetic:{m_name}"
            synthetic.append(out)
            generated += 1

    out_train = train + synthetic
    rng.shuffle(out_train)
    write_jsonl(Path(args.output_train), out_train)
    write_jsonl(Path(args.output_valid), valid)

    summary = {
        "base_train": len(train),
        "base_valid": len(valid),
        "base_positive_train": sum(1 for r in train if int(r.get("label", 1)) == 1),
        "base_negative_train": sum(1 for r in train if int(r.get("label", 1)) == 0),
        "synthetic_added": len(synthetic),
        "aug_train": len(out_train),
        "aug_positive_train": sum(1 for r in out_train if int(r.get("label", 1)) == 1),
        "aug_negative_train": sum(1 for r in out_train if int(r.get("label", 1)) == 0),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] train -> {args.output_train}")
    print(f"[DONE] valid -> {args.output_valid}")


if __name__ == "__main__":
    main()
