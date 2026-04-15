import argparse
import json
import random
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_role_prefix(text: str) -> str:
    text = normalize_text(text)
    return re.sub(r"^(?:user|human|assistant|gpt|用户|助手)\s*:\s*", "", text, flags=re.I)


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


PROMPT_TEMPLATES = [
    "{topic}时应该怎么做？",
    "如果遇到{topic}，最基本的处理是什么？",
    "关于{topic}，有什么常识建议？",
    "请给出{topic}的简短建议。",
    "面对{topic}，先做哪一步？",
    "说说{topic}的正确做法。",
    "日常碰到{topic}，通常怎么处理？",
    "你会如何建议处理{topic}？",
    "请用一句话回答：{topic}。",
    "跟{topic}有关的简单常识是什么？",
    "{topic}的时候，最稳妥的做法是什么？",
    "{topic}时最应该注意什么？",
    "碰到{topic}，第一步该做什么？",
    "如果想处理好{topic}，要记住什么？",
    "关于{topic}，最实用的一条建议是什么？",
    "{topic}这种情况，一般怎么应对？",
    "如果出现{topic}，应该先避免什么？",
    "给我一个关于{topic}的生活常识。",
    "请直接回答：{topic}要怎么处理？",
    "{topic}时，怎样更稳妥？",
    "如何看待{topic}这件事？",
    "遇到{topic}，通常先做哪件事？",
    "请说说{topic}的基础原则。",
    "{topic}的简单处理办法是什么？",
    "如果必须面对{topic}，最重要的一点是什么？",
    "{topic}时，什么做法比较合理？",
    "请回答与{topic}相关的常识。",
    "面对{topic}，怎样做更安全？",
    "关于{topic}，你会怎么提醒别人？",
    "请给出{topic}的要点。",
]

RESPONSE_PREFIXES = [
    "",
    "常见做法是：",
    "一般建议：",
    "实用建议：",
    "可以这样处理：",
]

PROMPT_PREFIXES = [
    "",
    "请简要回答：",
    "请直接回答：",
    "用常识回答：",
    "给出一个实用建议：",
    "请从日常经验出发回答：",
]

PROMPT_SUFFIXES = [
    "",
    "请尽量简洁。",
    "一句话说明即可。",
    "给新手一个建议。",
    "优先说最关键的一点。",
]

RESPONSE_SUFFIXES = [
    "",
    "这样通常更安全。",
    "这是比较稳妥的做法。",
    "先保证安全再处理细节。",
    "必要时请联系专业人士。",
]


FACTS = [
    ("闻到煤气味", "先开窗通风、关闭阀门，远离火源并联系专业人员。"),
    ("看到电线裸露", "先断电，别直接触碰，联系电工处理。"),
    ("发生火灾", "先报警、尽快撤离，并用湿毛巾捂住口鼻。"),
    ("车在路上抛锚", "先停到安全处并开启双闪。"),
    ("被烫伤", "先用流动凉水冲洗降温。"),
    ("食物变质", "不要吃，直接丢掉。"),
    ("过马路", "先看红绿灯，走斑马线。"),
    ("下雨天走路", "慢一点，注意防滑。"),
    ("头晕想晕倒", "先坐下或躺下，补水并求助。"),
    ("收到可疑链接", "不要乱点，先核实来源。"),
    ("睡眠不足", "容易疲劳，注意力也会下降。"),
    ("发烧", "先休息补水，必要时就医。"),
    ("感冒咳嗽", "多休息，注意保暖。"),
    ("眼睛疲劳", "先休息，尽量远离屏幕。"),
    ("口渴", "及时喝水。"),
    ("久坐", "起来活动一下。"),
    ("运动后出汗", "擦干并换上干衣服。"),
    ("伤口出血", "先按压止血并清洁伤口。"),
    ("吃得太快", "容易噎着，应该慢一点。"),
    ("肚子痛", "先观察，严重就及时就医。"),
    ("食物要保鲜", "放冰箱并密封保存。"),
    ("房间闷", "先开窗通风。"),
    ("地面湿滑", "先擦干再走。"),
    ("电池没电", "及时充电。"),
    ("洗完衣服", "晾干再收起来。"),
    ("垃圾分类", "按当地规则处理。"),
    ("节约用水", "关紧水龙头。"),
    ("节约用电", "不用时关灯断电。"),
    ("卫生间有异味", "清洁并通风。"),
    ("锅里溢出", "先关火再处理。"),
    ("冰块放久了", "会慢慢融化。"),
    ("热水放一会儿", "会逐渐变凉。"),
    ("水加热", "会慢慢变热，继续加热会沸腾。"),
    ("太阳出来", "会带来光和热。"),
    ("磁铁靠近铁", "通常会吸住它。"),
    ("光被物体挡住", "就会形成影子。"),
    ("听到雷声", "一般说明附近可能有雷电。"),
    ("声音传播", "需要空气等介质。"),
    ("白天比晚上亮", "因为有太阳光。"),
    ("电器工作", "通常需要电源接通。"),
    ("玻璃", "通常是透明的。"),
    ("一天", "有24小时。"),
    ("一周", "有7天。"),
    ("一年", "有12个月。"),
    ("预约迟到", "先提前通知对方。"),
    ("起床后", "先看看今天的安排。"),
    ("睡前", "先准备好第二天要用的东西。"),
    ("出门前", "检查钥匙、手机和钱包。"),
    ("赶时间", "尽量提前出发。"),
    ("做计划", "要给任务留一点缓冲。"),
    ("约定时间", "准时更好。"),
    ("密码", "最好设置得复杂一些。"),
    ("重要文件", "要记得备份。"),
    ("软件提示更新", "先确认来源再更新。"),
    ("充电器发热", "先停用并检查。"),
    ("Wi-Fi 不稳", "先重启路由器试试。"),
    ("收到陌生邮件", "不要轻易点附件。"),
    ("电脑变慢", "先清理后台程序。"),
    ("手机存储满", "删除不需要的文件。"),
    ("打字前", "先确认输入法。"),
    ("发消息前", "先检查收件人。"),
    ("做笔记", "重点要记下来。"),
    ("遇到不会的问题", "先查资料再提问。"),
    ("开会", "先准备好要点。"),
    ("截止日期快到了", "先做最重要的部分。"),
    ("团队合作", "先分工再沟通。"),
    ("说服别人", "先听对方想法。"),
    ("汇报工作", "先讲结论。"),
    ("需要请假", "提前说明比较好。"),
    ("写邮件", "标题要清楚。"),
    ("复习", "分批次进行更有效。"),
    ("坐公交", "先看站牌。"),
    ("坐地铁", "先看路线图。"),
    ("打车", "先确认目的地。"),
    ("开车", "不要酒后驾驶。"),
    ("停车", "按规定停放。"),
    ("迷路", "先看地图。"),
    ("行李太重", "先减负再赶路。"),
    ("赶飞机", "最好提前到机场。"),
    ("上高铁", "核对车次和座位。"),
    ("排队", "按顺序等候。"),
    ("买东西前", "先比价格。"),
    ("收到发票", "妥善保管。"),
    ("预算有限", "先买必需品。"),
    ("发现质量问题", "保留凭证。"),
    ("网购付款", "确认链接正规。"),
    ("借钱给人", "先约定还款方式。"),
    ("办卡或签约", "先看清条款。"),
    ("看到打折", "也要看是不是真的划算。"),
    ("现金不够", "先确认支付方式。"),
    ("存钱", "先养成记录习惯。"),
    ("别人帮忙", "先说谢谢。"),
    ("吵架时", "先冷静下来。"),
    ("沟通误会", "先确认事实。"),
    ("说话前", "先想清楚再开口。"),
    ("提问", "背景说清楚更容易得到帮助。"),
    ("承诺别人", "要尽量守信。"),
    ("别人难过", "先安慰对方。"),
    ("送礼物", "量力而行就好。"),
    ("初次见面", "礼貌问候最稳妥。"),
    ("想拒绝别人", "可以委婉表达。"),
]


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


def clean_base_rows(rows, max_prompt_chars: int, max_response_chars: int):
    cleaned = []
    seen = set()
    truncated = 0

    for row in rows:
        prompt = strip_role_prefix(str(row.get("prompt", "")))
        response = strip_role_prefix(str(row.get("response", "")))
        prompt = shorten_text(prompt, max_prompt_chars)
        response = shorten_text(response, max_response_chars)
        if not prompt or not response:
            continue
        key = (prompt.lower(), response.lower())
        if key in seen:
            continue
        seen.add(key)
        if prompt != strip_role_prefix(str(row.get("prompt", ""))) or response != strip_role_prefix(str(row.get("response", ""))):
            truncated += 1
        cleaned.append({"prompt": f"User: {prompt}\nAssistant:", "response": f" {response}", "source": row.get("source", "hf_zh")})
    return cleaned, truncated, seen


def build_common_sense_rows(seen, target_count: int, seed: int = 42):
    rng = random.Random(seed)
    rows = []
    max_attempts = max(target_count * 40, 20000)
    attempts = 0
    no_progress_rounds = 0
    last_size = 0

    while len(rows) < target_count and attempts < max_attempts and no_progress_rounds < 8:
        topic, answer = rng.choice(FACTS)
        template = rng.choice(PROMPT_TEMPLATES)
        prompt_prefix = rng.choice(PROMPT_PREFIXES)
        prompt_suffix = rng.choice(PROMPT_SUFFIXES)
        resp_prefix = rng.choice(RESPONSE_PREFIXES)
        resp_suffix = rng.choice(RESPONSE_SUFFIXES)

        core_prompt = template.format(topic=topic)
        prompt_text = f"{prompt_prefix}{core_prompt}{prompt_suffix}".strip()
        response_text = f"{resp_prefix}{answer}{resp_suffix}".strip()

        key = (prompt_text.lower(), response_text.lower())
        if key not in seen:
            seen.add(key)
            rows.append(
                {
                    "prompt": f"User: {prompt_text}\nAssistant:",
                    "response": f" {response_text}",
                    "source": "synthetic_common_sense_zh",
                }
            )

        attempts += 1

        if attempts % 1000 == 0:
            if len(rows) == last_size:
                no_progress_rounds += 1
            else:
                no_progress_rounds = 0
                last_size = len(rows)

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build a shorter Chinese dialogue dataset with extra common-sense samples")
    parser.add_argument("--base_input", default="data/train_from_hf_zh.jsonl")
    parser.add_argument("--output", default="data/train_from_hf_zh_common.jsonl")
    parser.add_argument("--summary", default="data/train_from_hf_zh_common_summary.json")
    parser.add_argument("--target_total", type=int, default=6000)
    parser.add_argument("--max_prompt_chars", type=int, default=120)
    parser.add_argument("--max_response_chars", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_rows = read_jsonl(Path(args.base_input))
    cleaned_base, truncated, seen = clean_base_rows(base_rows, args.max_prompt_chars, args.max_response_chars)

    synthetic_target = max(args.target_total - len(cleaned_base), 0)
    synthetic_rows = build_common_sense_rows(seen, synthetic_target, seed=args.seed)

    merged = cleaned_base + synthetic_rows
    rng = random.Random(args.seed)
    rng.shuffle(merged)

    out_path = Path(args.output)
    write_jsonl(out_path, merged)

    summary = {
        "base_input": str(args.base_input),
        "output": str(out_path),
        "base_total": len(base_rows),
        "base_kept": len(cleaned_base),
        "base_truncated": truncated,
        "synthetic_added": len(synthetic_rows),
        "final_total": len(merged),
        "target_total": args.target_total,
        "sources": {
            "hf_zh": len(cleaned_base),
            "synthetic_common_sense_zh": len(synthetic_rows),
        },
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()