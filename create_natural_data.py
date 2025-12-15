import pandas as pd
import ast
import json
import os

# ======================================================
# 1. 配置
# ======================================================
FULL_INTERACT_LOG = "/user/zhuohang.yu/u24922/LCU-main_backup/rec_datasets/WM_KuaiComt/KuaiComt_subset.csv"
# 命名新的中立数据集
NEUTRAL_INTERACT_LOG_OUT = "/user/zhuohang.yu/u24922/LCU-main_backup/rec_datasets/WM_KuaiComt/KuaiComt_NEUTRAL_subset.csv"

# ID文件保持不变，因为ID来源不依赖于显式反馈字段
REQUIRED_COMMENT_IDS_OUT = "/user/zhuohang.yu/u24922/LCU-main_backup/rec_datasets/WM_KuaiComt/REQUIRED_comment_ids_NEUTRAL.json"
REQUIRED_VIDEO_IDS_OUT = "/user/zhuohang.yu/u24922/LCU-main_backup/rec_datasets/WM_KuaiComt/REQUIRED_video_ids_NEUTRAL.json"

# (关键!) 采样比例设为 50% (0.5)
SAMPLE_FRACTION = 0.5 
RANDOM_SEED = 42

EXPLICIT_FEEDBACK_COLS = [
    'is_like', 
    'is_follow', 
    'is_comment', 
    'is_forward', 
    'is_hate'
]

print(f"Using sample fraction: {SAMPLE_FRACTION * 100}%")

# ======================================================
# 2. 采样交互日志并移除显式反馈
# ======================================================
print(f"Loading full interaction log from {FULL_INTERACT_LOG}...")
try:
    # 你的 DEBUG 确认了文件是逗号分隔的
    interact_df = pd.read_csv(FULL_INTERACT_LOG, sep=',')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

print(f"Full log loaded: {len(interact_df)} rows.")

# 随机采样
neutral_interact_df = interact_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED)
print(f"Sampled neutral log: {len(neutral_interact_df)} rows.")

# --- 移除显式反馈列 ---
cols_to_drop = [col for col in EXPLICIT_FEEDBACK_COLS if col in neutral_interact_df.columns]

if cols_to_drop:
    neutral_interact_df.drop(columns=cols_to_drop, inplace=True)
    print(f"✅ Removed explicit feedback columns: {cols_to_drop}")
else:
    print("Warning: Explicit feedback columns not found in DataFrame.")

# ======================================================
# 3. 收集所有必需的 "短 ID" (作为整数)
# ======================================================
print("Collecting all required *SHORT* comment IDs...")
required_comment_ids = set()

# 1. 提取 comment0_id ... comment5_id
short_id_cols = [c for c in neutral_interact_df.columns if c.startswith("comment") and c.endswith("_id")]
print(f"Found short ID columns: {short_id_cols}")

for col in short_id_cols:
    vals = neutral_interact_df[col].dropna()
    vals = vals[vals.apply(lambda x: str(x).replace('.','',1).isdigit())]
    required_comment_ids.update(vals.astype(int).tolist())

# 2. 提取 sampled_comments_reindexed 中的列表
def extract_ids_from_list(lst_str):
    try:
        if pd.isna(lst_str):
            return
        id_list = ast.literal_eval(str(lst_str)) 
        if isinstance(id_list, list):
            required_comment_ids.update(int(item) for item in id_list if str(item).isdigit())
    except (ValueError, SyntaxError, TypeError):
        pass

if 'sampled_comments_reindexed' in neutral_interact_df.columns:
    neutral_interact_df['sampled_comments_reindexed'].apply(extract_ids_from_list)

print(f"Total unique *SHORT* comment IDs required: {len(required_comment_ids)}")

# ======================================================
# 4. 收集所有必需的 "视频 ID" (作为整数)
# ======================================================
print("Collecting all required video IDs...")
required_video_ids = set(neutral_interact_df['video_id'].dropna().astype(int))
print(f"Total unique video IDs required: {len(required_video_ids)}")

# ======================================================
# 5. 保存输出
# ======================================================
# 保存 NEUTRAL 日志
neutral_interact_df.to_csv(NEUTRAL_INTERACT_LOG_OUT, index=False, sep=',') 
print(f"✅ Saved NEUTRAL interaction log to: {NEUTRAL_INTERACT_LOG_OUT}")

# 保存 ID 列表 (JSON)
with open(REQUIRED_COMMENT_IDS_OUT, 'w') as f:
    json.dump(list(required_comment_ids), f)
print(f"✅ Saved REQUIRED comment IDs to: {REQUIRED_COMMENT_IDS_OUT}")

with open(REQUIRED_VIDEO_IDS_OUT, 'w') as f:
    json.dump(list(required_video_ids), f)
print(f"✅ Saved REQUIRED video IDs to: {REQUIRED_VIDEO_IDS_OUT}")
