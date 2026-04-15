import argparse
import torch
import re
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
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", default="output_lora")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--system_prompt", default="You are a helpful, concise assistant.")
    p.add_argument("--drop_non_english", type=bool, default=False, help="Skip samples containing obvious non-English text")
    return p.parse_args()


def infer_lora_target_modules(model):
    candidate_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["query_key_value"],
        ["c_attn"],
    ]

    available_leaf_names = set()
    for name, module in model.named_modules():
        if not name:
            continue
        if hasattr(module, "weight"):
            available_leaf_names.add(name.split(".")[-1])

    for group in candidate_groups:
        if all(g in available_leaf_names for g in group):
            return group
    for group in candidate_groups:
        if any(g in available_leaf_names for g in group):
            return [g for g in group if g in available_leaf_names]
    return ["c_attn"]


def main():
    args = parse_args()

    print("Loading dataset...")
    ds = load_dataset("json", data_files={"train": args.train_file})

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def looks_english(text: str) -> bool:
        return not any("\u4e00" <= ch <= "\u9fff" for ch in text)

    def normalize_prompt_response(prompt_text: str, response_text: str) -> tuple[str, str]:
        prompt_text = prompt_text.strip()
        response_text = response_text.strip()

        # If prompt already contains role tags, extract the latest user part.
        if "User:" in prompt_text:
            parts = prompt_text.split("User:")
            prompt_text = parts[-1].strip()
        prompt_text = prompt_text.replace("Assistant:", "").strip()

        # Remove accidental role prefixes from response.
        response_text = re.sub(r"^Assistant:\\s*", "", response_text)
        response_text = re.sub(r"^User:\\s*", "", response_text)
        return prompt_text, response_text

    def preprocess(examples):
        texts = []
        prompts = examples.get("prompt")
        responses = examples.get("response")
        for p, r in zip(prompts, responses):
            prompt_text, response_text = normalize_prompt_response(str(p), str(r))
            if args.drop_non_english and (not looks_english(prompt_text) or not looks_english(response_text)):
                continue
            formatted = f"{args.system_prompt}\nUser: {prompt_text}\nAssistant: {response_text}"
            texts.append(formatted)
        # Return empty tokenization if all samples filtered (prevents IndexError)
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        return tokenizer(texts, truncation=True, padding="max_length", max_length=args.max_seq_length)

    tokenized = ds["train"].map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    target_modules = infer_lora_target_modules(model)
    print(f"LoRA target modules: {target_modules}")

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
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
        fp16=torch.cuda.is_available(),
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
