import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="gpt2")
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", default="output_lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading dataset...")
    ds = load_dataset("json", data_files={"train": args.train_file})

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        texts = []
        prompts = examples.get("prompt")
        responses = examples.get("response")
        for p, r in zip(prompts, responses):
            # format: prompt + response
            texts.append(str(p) + str(r))
        return tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_seq_length)

    tokenized = ds["train"].map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=100,
        fp16=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
