"""
train.py — SFT fine-tuning script for ICD-10 medical coding

Usage:
    python src/train.py --model_id google/medgemma-4b-it --output_dir ./output
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch


def format_prompt(example):
    return {
        "text": f"""You are a clinical coding assistant. Assign the correct ICD-10 code(s) for the following clinical note.

Clinical note: {example['clinical_note']}

ICD-10 Code(s): {example['icd_codes']}"""
    }


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # Load and format your dataset here
    # dataset = load_dataset("your_dataset")
    # dataset = dataset.map(format_prompt)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        # train_dataset=dataset["train"],
    )

    trainer.train()

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    args = parser.parse_args()
    main(args)
