微调入门示例（单卡练手）

步骤：

1. 激活 conda 环境（你已创建 `cjt`）：

```powershell
conda activate cjt
```

2. 进入示例目录并安装 Python 依赖：

```powershell
cd Alpha\ml_finetune_demo
python -m pip install -r requirements.txt
```

（注意：`torch` 已通过 conda 安装；若没有请用 conda 安装对应 CUDA 版本的 torch）

3. 运行微调（示例基于 `gpt2`，小规模练手）：

```powershell
python train.py --model_name gpt2 --train_file data/train.jsonl --output_dir output --num_train_epochs 1 --per_device_train_batch_size 1
```

说明：这是最小化流程示例，后续可替换模型为更合适的基座，并加入 PEFT/LoRA、accelerate、DeepSpeed 等优化。
