import pandas as pd
import ast
import json
import random
import os

# ======================================================
# 1. 配置
# ======================================================
FULL_INTERACT_LOG = "LCU-main/rec_datasets/WM_KuaiComt/KuaiComt_subset.csv"
TINY_INTERACT_LOG_OUT = "LCU-main/rec_datasets/WM_KuaiComt/KuaiComt_TINY_subset.csv"
REQUIRED_COMMENT_IDS_OUT = "LCU-main/rec_datasets/WM_KuaiComt/REQUIRED_comment_ids.json"
REQUIRED_VIDEO_IDS_OUT = "LCU-main/rec_datasets/WM_KuaiComt/REQUIRED_video_ids.json"

# (关键!) 采样比例设为 50% (0.5)，这样数据量既够训练，又不会让嵌入生成跑几天
SAMPLE_FRACTION = 0.5 
print(f"Using sample fraction: {SAMPLE_FRACTION * 100}%")

# ======================================================
# 2. 采样交互日志
# ======================================================
print(f"Loading full interaction log from {FULL_INTERACT_LOG}...")
# (修正!) 你的 DEBUG 确认了文件是逗号分隔的
try:
    interact_df = pd.read_csv(FULL_INTERACT_LOG, sep=',')
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

print(f"Full log loaded: {len(interact_df)} rows.")

# 随机采样
tiny_interact_df = interact_df.sample(frac=SAMPLE_FRACTION, random_state=42)
print(f"Sampled tiny log: {len(tiny_interact_df)} rows.")

# ======================================================
# 3. 收集所有必需的 "短 ID" (作为整数)
# ======================================================
print("Collecting all required *SHORT* comment IDs from the TINY log...")
required_comment_ids = set()

# 1. 提取 comment0_id ... comment5_id
short_id_cols = [c for c in tiny_interact_df.columns if c.startswith("comment") and c.endswith("_id")]
print(f"Found short ID columns: {short_id_cols}")

for col in short_id_cols:
    # 确保转换为整数
    vals = tiny_interact_df[col].dropna()
    # 过滤掉非数字字符 (以防万一)
    vals = vals[vals.apply(lambda x: str(x).replace('.','',1).isdigit())]
    required_comment_ids.update(vals.astype(int).tolist())

# 2. 提取 sampled_comments_reindexed 中的列表
def extract_ids_from_list(lst_str):
    try:
        if pd.isna(lst_str):
            return
        id_list = ast.literal_eval(str(lst_str)) 
        if isinstance(id_list, list):
            # 将列表中的每一项转换为整数并加入集合
            required_comment_ids.update(int(item) for item in id_list if str(item).isdigit())
    except (ValueError, SyntaxError):
        pass

if 'sampled_comments_reindexed' in tiny_interact_df.columns:
    tiny_interact_df['sampled_comments_reindexed'].apply(extract_ids_from_list)

print(f"Total unique *SHORT* comment IDs required: {len(required_comment_ids)}")

# ======================================================
# 4. 收集所有必需的 "视频 ID" (作为整数)
# ======================================================
print("Collecting all required video IDs...")
# (关键!) 确保 video_id 也被当作整数处理，避免后续比较出错
required_video_ids = set(tiny_interact_df['video_id'].dropna().astype(int))
print(f"Total unique video IDs required: {len(required_video_ids)}")

# ======================================================
# 5. 保存输出
# ======================================================
# 保存 TINY 日志
tiny_interact_df.to_csv(TINY_INTERACT_LOG_OUT, index=False, sep=',') 
print(f"✅ Saved TINY interaction log to: {TINY_INTERACT_LOG_OUT}")

# 保存 ID 列表 (JSON)
with open(REQUIRED_COMMENT_IDS_OUT, 'w') as f:
    json.dump(list(required_comment_ids), f)
print(f"✅ Saved REQUIRED comment IDs to: {REQUIRED_COMMENT_IDS_OUT}")

with open(REQUIRED_VIDEO_IDS_OUT, 'w') as f:
    json.dump(list(required_video_ids), f)
print(f"✅ Saved REQUIRED video IDs to: {REQUIRED_VIDEO_IDS_OUT}")