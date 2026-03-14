# train.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os

# ---------------------------
# Configuration
# ---------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"  # or "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR = "./model"
DATA_FILE = "data/privacy_commands.json"  # your training data in JSONL format

# ---------------------------
# Load tokenizer and model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token  # for padding

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ---------------------------
# Prepare dataset
# ---------------------------
def load_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    # Expect each line: {"instruction": "...", "output": "..."}
    return [json.loads(line) for line in lines]

def format_example(example):
    # Format as a conversation for instruction tuning
    prompt = f"User: {example['instruction']}\nAssistant: {example['output']}"
    return prompt

raw_data = load_data(DATA_FILE)
formatted_texts = [format_example(ex) for ex in raw_data]

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

dataset = Dataset.from_dict({"text": formatted_texts})
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ---------------------------
# Training arguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,  # we'll push manually
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ---------------------------
# Train and save
# ---------------------------
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")