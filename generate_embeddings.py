import os
import torch
import pandas as pd
import csv
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel # 确保 LoRA 加载
import numpy as np # 新增

# ======================================================
# 1. 配置
# ======================================================
FINETUNED_MODEL_PATH = "finetune/tuneQwen7B_LCU_SFT"
BASE_MODEL = "Qwen/Qwen2-7B"
VIDEO_METADATA_PATH = "LCU-main/rec_datasets/KuaiComt/photo_table_final.csv"
COMMENT_METADATA_PATH = "LCU-main/rec_datasets/KuaiComt/comment_table_final.csv"

REQUIRED_VIDEO_ID_PATH = "LCU-main/rec_datasets/WM_KuaiComt/REQUIRED_video_ids.json"
REQUIRED_COMMENT_ID_PATH = "LCU-main/rec_datasets/WM_KuaiComt/REQUIRED_comment_ids.json"

output_dir = "finetune/embeddings"
os.makedirs(output_dir, exist_ok=True)
device = "cuda"

VIDEO_ID_COL = 'photo_id'
VIDEO_TEXT_COL = 'caption'
COMMENT_ID_COL = 'comment_id' 
COMMENT_TEXT_COL = 'comment_content'

# 用于记录生成 NaN 的 ID
NAN_LOG_FILE = os.path.join(output_dir, "nan_embedding_ids.csv")

# ======================================================
# 2. 加载模型和 tokenizer (使用 LoRA)
# ======================================================
print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print(f"Loading LoRA adapter from: {FINETUNED_MODEL_PATH}")
model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
model.eval()
print("LoRA adapter loaded.")


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
print("Model and tokenizer loaded.\n")

# ======================================================
# 3. 加载必需的 ID
# ======================================================
print(f"Loading REQUIRED video IDs from {REQUIRED_VIDEO_ID_PATH}...")
with open(REQUIRED_VIDEO_ID_PATH, 'r') as f:
    required_video_ids = set(int(v) for v in json.load(f))
print(f"Loaded {len(required_video_ids)} required video IDs (short IDs).")

print(f"Loading REQUIRED comment IDs from {REQUIRED_COMMENT_ID_PATH}...")
with open(REQUIRED_COMMENT_ID_PATH, 'r') as f:
    required_comment_ids = set(int(v) for v in json.load(f)) 
print(f"Loaded {len(required_comment_ids)} required comment IDs (short IDs).")

# ======================================================
# 4. 嵌入生成函数 (添加 NaN 清洗)
# ======================================================
def create_video_prompt(text):
    return f"Video Title: {str(text) if pd.notna(text) else ''}"

def create_comment_prompt(text):
    return f"Comment: {str(text) if pd.notna(text) else ''}"

def encode_text(text, prompt_func, log_type, max_len=128):
    """编码文本并执行 Last Token Pooling。在返回前检查 NaN/Inf。"""
    prompt = prompt_func(text)
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_len, 
        padding=True 
    ).to(device)

    with torch.no_grad():
        try:
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
        except TypeError:
            # Fallback for models without output_hidden_states in standard outputs
            hidden = model.transformer(**inputs).last_hidden_state

    mask = inputs['attention_mask']
    last_token_idx = mask.sum(dim=1) - 1
    
    hidden_size = model.config.hidden_size
    
    # 如果序列长度为 0 (通常不会发生，但作为安全检查)
    if last_token_idx.item() < 0:
        return torch.zeros(hidden_size).cpu()

    embedding = hidden[0, last_token_idx.item(), :]
    embedding_cpu = embedding.cpu()

    # ----------------------------------------------------
    # ✅ 核心修正：生成时检测并处理 NaN/Inf
    # ----------------------------------------------------
    if not torch.isfinite(embedding_cpu).all():
        # 如果检测到 NaN 或 Inf，记录并返回一个零向量
        nan_mask = ~torch.isfinite(embedding_cpu)
        print(f"\n[ATTENTION] {log_type} embedding failed (NaN/Inf detected). Replacing with zeros.")
        
        # 记录到 NaN log 文件
        with open(NAN_LOG_FILE, 'a') as f:
            f.write(f"{log_type}: {text}\n")
            
        return torch.zeros(hidden_size).cpu()

    return embedding_cpu

# ======================================================
# 5. 生成视频嵌入 (已修复)
# ======================================================
'''
video_embeddings = {}
print("\nGenerating video embeddings...")

# 写入 NaN log 文件的头部
with open(NAN_LOG_FILE, 'w') as f:
    f.write("--- NaN/Inf Embedding Log ---\n")

# 读取 video_table_final.csv
video_df = pd.read_csv(
    VIDEO_METADATA_PATH, 
    sep='\t',
    usecols=[VIDEO_TEXT_COL],
    engine='python', 
    on_bad_lines='skip',
    dtype={VIDEO_TEXT_COL: str}
)
print(f"Loaded {len(video_df)} total videos from video_table_final.")

valid_short_video_ids = [idx for idx in required_video_ids if 0 <= idx < len(video_df)]
print(f"Found {len(valid_short_video_ids)} videos to embed (using short ID as row index).")

for short_id in tqdm(valid_short_video_ids, desc="Processing videos"):
    text = video_df.iloc[short_id][VIDEO_TEXT_COL]
    
    # 传递类型用于日志记录
    video_embeddings[short_id] = encode_text(text, create_video_prompt, log_type=f"Video ID {short_id}")

save_path_video = os.path.join(output_dir, "video_embeddings_qwen7b_tiny.pt")
torch.save(video_embeddings, save_path_video)
print(f"\n✅ Video embeddings saved at {save_path_video} (Total: {len(video_embeddings)})")
'''
# ======================================================
# 6. 生成评论嵌入 (已修复)
# ======================================================

comment_embeddings = {}
print("\nGenerating comment embeddings...")

# 读取 comment_table_final.csv
comment_df = pd.read_csv(
    COMMENT_METADATA_PATH, 
    sep='\t',
    usecols=[COMMENT_TEXT_COL],
    engine='python', 
    on_bad_lines='skip',
    dtype={COMMENT_TEXT_COL: str}
)
print(f"Loaded {len(comment_df)} total comments from comment_table_final.")

valid_short_ids = [idx for idx in required_comment_ids if 0 <= idx < len(comment_df)]
print(f"Found {len(valid_short_ids)} comments to embed (using short ID as row index).")

for short_id in tqdm(valid_short_ids, desc="Processing comments"):
    text = comment_df.iloc[short_id][COMMENT_TEXT_COL]
    
    # 传递类型用于日志记录
    comment_embeddings[short_id] = encode_text(text, create_comment_prompt, log_type=f"Comment ID {short_id}")

save_path_comment = os.path.join(output_dir, "comment_embeddings_qwen7b_tiny.pt")
torch.save(comment_embeddings, save_path_comment)
print(f"\n✅ Comment embeddings saved at {save_path_comment} (Total: {len(comment_embeddings)})")

print(f"\nAll NaN/Inf instances logged in: {NAN_LOG_FILE}")
