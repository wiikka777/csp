import os
import gc
import random
import pandas as pd
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# ======================================================
# 1. 基础配置
# ======================================================
model_name = "Qwen/Qwen1.5-1.8B"
output_dir = "Qwen1.8B_LCU_SFT"
batch_size = 2
num_epochs = 4   # 你已经验证过最佳区间 4~6
learning_rate = 2e-5

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"Loading base model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# ======================================================
# 2. LoRA 配置
# ======================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
print("LoRA injected successfully.")

# ======================================================
# 3. 加载领域数据（必须已存在 jsonl 文件）
# ======================================================
domain_files = {
    "staytime": "staytime_prediction.jsonl",
    "top_comment": "kuaiComt_top_comment_prediction_data.jsonl",
    "interaction": "user_comment_interaction_prediction.jsonl"
}

datasets = []
min_size = float('inf')

print("\nLoading domain datasets...")
for name, file in domain_files.items():
    ds = load_dataset("json", data_files=file, split="train")
    min_size = min(min_size, len(ds))
    datasets.append(ds)
    print(f"[OK] {name}: {len(ds)} samples")

# ======================================================
# 4. 加载 Alpaca-GPT4 通用数据
# ======================================================
alpaca = load_dataset("json", data_files="alpaca_data.json", split="train")
print(f"[OK] Alpaca: {len(alpaca)} samples")

def format_data(ex):
    prompt = ex.get("instruction","") + "\n" + ex.get("input","")
    return {"prompt": prompt.strip(), "completion": ex.get("output","")}

alpaca = alpaca.map(format_data, remove_columns=alpaca.column_names)

# 取比例 1:1:1:3
datasets = [ds.shuffle(seed=42).select(range(min_size)) for ds in datasets]
alpaca = alpaca.shuffle(seed=42).select(range(min_size * 3))

combined = concatenate_datasets(datasets + [alpaca])
print(f"\n[✓] Total SFT Training Samples = {len(combined)}")

# ======================================================
# 5. 划分训练/验证集
# ======================================================
dataset = combined.train_test_split(test_size=0.05, seed=42)
print(f"Train = {len(dataset['train'])}, Validation = {len(dataset['test'])}")

# ======================================================
# 6. Tokenization
# ======================================================
def tokenize(examples):
    texts = []
    for p, c in zip(examples["prompt"], examples["completion"]):
        text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=["prompt", "completion"])

# ======================================================
# 7. Trainer 参数
# ======================================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=8,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    evaluation_strategy="steps",
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# ======================================================
# 8. 开始训练
# ======================================================
print("\n🔥 Start Fine-tuning...\n")
trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n✅ Training Completed Successfully!")
print(f"Fine-tuned model saved at: {output_dir}")
