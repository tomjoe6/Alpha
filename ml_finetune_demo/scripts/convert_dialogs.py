import json
from pathlib import Path

SRC = Path("data/dialogs.jsonl")
DST = Path("data/train_from_dialogs.jsonl")
DST.parent.mkdir(parents=True, exist_ok=True)

def clean(prompt, response):
    if not isinstance(prompt, str):
        prompt = str(prompt)
    if not isinstance(response, str):
        response = str(response)
    # remove exact prompt prefix in response if present
    if response.startswith(prompt):
        response = response[len(prompt):]
    response = response.strip()
    prompt = prompt.strip()
    return prompt, response

seen = set()
written = 0
with SRC.open("r", encoding="utf-8") as f_in, DST.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        p, r = clean(rec.get("prompt", ""), rec.get("response", ""))
        if not r or len(r) < 3:
            continue
        key = (p, r)
        if key in seen:
            continue
        seen.add(key)
        out = {"prompt": f"{p}\nAssistant:", "response": " " + r}
        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        written += 1

print(f"Converted {written} examples to {DST}")
