微调入门示例（单卡练手）

这套目录的目标是：先把“数据 → 训练 → 对话推理 → 记录样本”这条链路跑通。

## 1. 准备环境

```powershell
conda activate cjt
cd Alpha\ml_finetune_demo
python -m pip install -r requirements.txt
```

说明：`torch` 建议直接用 conda 安装对应 CUDA 版本。

## 2. 先跑最小训练

```powershell
python train.py --model_name gpt2 --train_file data/train.jsonl --output_dir output --num_train_epochs 1 --per_device_train_batch_size 1
```

这个脚本适合练习“基础训练流程”。

## 3. 跑 LoRA 微调

```powershell
python lora_train.py --model_name_or_path gpt2 --train_file data/train_from_dialogs.jsonl --output_dir output_lora --num_train_epochs 1 --per_device_train_batch_size 1
```

这个脚本更接近对话模型常见做法：只训练小量参数，成本更低。

## 4. 运行交互式对话

```powershell
python demo_chat.py --model_name_or_path gpt2
```

如果要加载 LoRA 适配器：

```powershell
python demo_chat.py --model_name_or_path gpt2 --adapter_dir output_lora
```

如果要直接测试某个本地 checkpoint：

```powershell
python demo_chat.py --model_name_or_path output\checkpoint-3
```

## 5. 调试建议

- 先用 `--demo` 跑固定样例，确认模型能出字。
- 再进入交互模式，观察多轮上下文是否生效。
- 如果在 CPU 上运行，使用 `--device -1`。
- 如果输出太随机，先不要开 `--do_sample`。
- 如果想看更像“聊天”的效果，可以把 `--history_turns` 调大一点。

## 6. 数据格式

- `train.py` 读取 `data/train.jsonl`，要求字段是 `text`。
- `lora_train.py` 读取 `prompt` / `response`。
- `demo_chat.py` 会把你输入的对话记录追加保存到 `data/dialogs.jsonl`。

说明：这套工程是“对话模型调试练习台”，适合学习数据、训练和推理流程；不是完整的大模型训练框架。
