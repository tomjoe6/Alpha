import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_tool(tool: str, code_path: Path, timeout_sec: int) -> tuple[bool, str]:
    if tool == "iverilog":
        cmd = ["iverilog", "-g2012", "-t", "null", str(code_path)]
    elif tool == "verilator":
        cmd = ["verilator", "--lint-only", str(code_path)]
    elif tool == "verible":
        cmd = ["verible-verilog-syntax", str(code_path)]
    else:
        raise ValueError(f"Unsupported tool: {tool}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        ok = proc.returncode == 0
        msg = (proc.stdout or "") + (proc.stderr or "")
        return ok, msg.strip()
    except Exception as exc:
        return False, str(exc)


def verify_row(row: dict, tool: str, timeout_sec: int) -> tuple[int, str]:
    code = str(row.get("input", ""))
    suffix = ".sv"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = Path(f.name)

    try:
        ok, msg = run_tool(tool, tmp, timeout_sec)
        return (1 if ok else 0), msg
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


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


def main():
    p = argparse.ArgumentParser(description="Re-verify Verilog syntax labels with an HDL tool")
    p.add_argument("--input", default="data/verilog_syntax_train.jsonl")
    p.add_argument("--output", default="data/verilog_syntax_train_relabeled.jsonl")
    p.add_argument("--tool", default="iverilog", choices=["iverilog", "verilator", "verible"])
    p.add_argument("--timeout_sec", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=0, help="0 means all")
    p.add_argument("--summary", default="data/verilog_syntax_relabel_summary.json")
    args = p.parse_args()

    tool_bin = {
        "iverilog": "iverilog",
        "verilator": "verilator",
        "verible": "verible-verilog-syntax",
    }[args.tool]

    if shutil.which(tool_bin) is None:
        summary = {
            "status": "skipped",
            "reason": f"Tool not found in PATH: {tool_bin}",
            "input": args.input,
            "output": args.output,
        }
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    rows = read_jsonl(Path(args.input))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    relabeled = []
    changed = 0
    for i, row in enumerate(rows, 1):
        new_label, tool_msg = verify_row(row, args.tool, args.timeout_sec)
        old_label = int(row.get("label", 1))
        if new_label != old_label:
            changed += 1
        row_out = dict(row)
        row_out["label_tool"] = args.tool
        row_out["label_old"] = old_label
        row_out["label"] = new_label
        row_out["tool_message"] = tool_msg[:1200]
        relabeled.append(row_out)

        if i % 100 == 0:
            print(f"[INFO] verified {i}/{len(rows)}")

    write_jsonl(Path(args.output), relabeled)
    summary = {
        "status": "ok",
        "tool": args.tool,
        "total": len(relabeled),
        "changed_labels": changed,
        "kept_labels": len(relabeled) - changed,
        "positive": sum(1 for r in relabeled if int(r.get("label", 1)) == 1),
        "negative": sum(1 for r in relabeled if int(r.get("label", 1)) == 0),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] relabeled dataset -> {args.output}")


if __name__ == "__main__":
    main()
