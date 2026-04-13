English + Chinese Bilingual Guide / 中英双语说明

This project is a dialogue fine-tuning sandbox.
这个项目是一个对话微调练习沙盒。

Goal: run the full path from data to training to chat testing.
目标：把数据、训练、对话测试这条完整链路跑通。

## 1) Environment / 环境

```powershell
conda activate cjt
cd Alpha\ml_finetune_demo
python -m pip install -r requirements.txt
```

If you have a GPU, install a CUDA-matched torch build.
如果有 GPU，建议安装与 CUDA 匹配的 torch。

## 2) Two Training Routes / 两条训练路线

Route A (recommended): LoRA fine-tuning.
路线 A（推荐）：LoRA 微调。

```powershell
python lora_train.py --model_name_or_path microsoft/DialoGPT-small --train_file data/train_from_dialogs.jsonl --output_dir output_lora --num_train_epochs 2 --per_device_train_batch_size 2
```

Route B: full fine-tuning.
路线 B：全量微调。

```powershell
python train.py --model_name microsoft/DialoGPT-small --train_file data/train.jsonl --output_dir output --num_train_epochs 2 --per_device_train_batch_size 2
```

## 3) Chat Inference / 对话推理

Base model only:
只用基座模型：

```powershell
python demo_chat.py --model_name_or_path microsoft/DialoGPT-small --device -1
```

Base model + LoRA adapter:
基座模型 + LoRA 适配器：

```powershell
python demo_chat.py --model_name_or_path microsoft/DialoGPT-small --adapter_dir output_lora --device -1
```

## 4) Which Data File Is Used? / 训练数据文件怎么选？

- `data/train_from_dialogs.jsonl`
  - For `lora_train.py`.
  - 给 `lora_train.py` 用（推荐主数据）。

- `data/train.jsonl`
  - For `train.py`.
  - 给 `train.py` 用（全量训练）。

- `data/dialogs.jsonl`
  - Raw chat logs from `demo_chat.py`; do not train on it directly.
  - `demo_chat.py` 产生的原始日志，不要直接用于训练。

- `data/en_train.jsonl`
  - Legacy tiny sample file.
  - 早期小样本示例文件，可选参考。

## 5) Why `train_from_hf.jsonl` Exists? / 为什么新建了 `train_from_hf.jsonl`？

`train_from_dialogs.jsonl` is your local curated dataset.
`train_from_dialogs.jsonl` 是你本地整理的数据。

`train_from_hf.jsonl` is auto-built from open Hugging Face datasets using quality filters.
`train_from_hf.jsonl` 是从 Hugging Face 开源数据集自动抓取并清洗得到的。

They are alternatives for training input.
它们是两种可选训练输入。

Use one dataset at a time for clean experiments.
建议一次只用一个数据集训练，方便对比实验结果。

## 6) Auto-collect High-quality Data / 自动抓取高质量对话数据

```powershell
python scripts/build_dialog_dataset.py --output data/train_from_hf.jsonl --sources daily_dialog,ultrachat_200k,oasst1,hh_rlhf,dolly_15k --max_per_source 1500
```

Then train:
然后训练：

```powershell
python lora_train.py --model_name_or_path microsoft/DialoGPT-small --train_file data/train_from_hf.jsonl --output_dir output_lora_hf --num_train_epochs 2 --per_device_train_batch_size 2
```

## 7) How to Compare `output_lora`, `v2`, `v3`? / 怎么比较 `output_lora`、`v2`、`v3`？

Rule: compare on the same prompt set and same decoding settings.
原则：用同一套测试问题、同一组生成参数比较。

Auto compare script:
自动对比脚本：

```powershell
python scripts/compare_adapters.py --base_model microsoft/DialoGPT-small --adapters output_lora,output_lora_v2,output_lora_v3 --output_json reports/adapter_compare.json --output_md reports/adapter_compare.md --device -1
```

Outputs:
输出文件：

- `reports/adapter_compare.json`
- `reports/adapter_compare.md`

Pick the best run using these criteria:
按以下标准选择最佳版本：

- Relevance to prompt / 回答相关性
- Fluency and grammar / 流畅度和语法
- Less repetition / 少复读
- Stable multi-turn behavior / 多轮稳定性

## 8) Command-level Principles / 终端命令背后原理

- `python scripts/build_dialog_dataset.py ...`
  - Data ETL: load -> clean -> deduplicate -> write JSONL.
  - 数据 ETL：加载 -> 清洗 -> 去重 -> 写入 JSONL。

- `python lora_train.py ...`
  - Train only adapter parameters; base weights mostly frozen.
  - 只训练适配器参数，基座权重基本不动。

- `python train.py ...`
  - Full fine-tuning updates many base model parameters.
  - 全量微调会更新大量基座参数。

- `python demo_chat.py ...`
  - Autoregressive generation for interactive chat.
  - 自回归生成，用于交互对话。

## 9) Practical Target / 实用目标

With only 20 to 30 samples, quality improvement is limited.
只有 20 到 30 条样本时，效果提升会很有限。

A practical beginner target is 300 to 1000 high-quality English pairs.
新手阶段建议目标：至少 3000 到 100000 条高质量英文问答对。
