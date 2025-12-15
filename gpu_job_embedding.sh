#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM 资源请求配置 (必须在脚本顶部)
# -----------------------------------------------------------------------------
#SBATCH --job-name=Qwen_LoRA_SFT
#SBATCH --partition=scc-gpu           # 指定 GPU 计算分区
#SBATCH --nodes=1                     # 请求 1 个节点
#SBATCH --ntasks=1                    # 每个节点运行 1 个任务
#SBATCH --cpus-per-task=16             # 请求 8 个 CPU 核心
#SBATCH --time=20:00:00                # 运行时间限制
#SBATCH --mem=128G                     # 内存限制：16GB
#SBATCH --output=slurm-%j.out         # 标准输出文件 (文件名包含作业ID)
#SBATCH --error=slurm-%j.err          # 错误输出文件

# 关键修正：使用 GRES 语法请求 1 块 A100 GPU
# 这解决了 'Invalid feature specification' 错误，并消除了 GPU 类型未指定的警告
#SBATCH --gpus=4
#SBATCH --constraint=40gb_vram

# =================================================================
# 1. 路径和变量定义
# =================================================================
# 明确定义项目路径 (手动输入完整路径，消除环境变量依赖)
PROJECT_DIR_PATH="/projects/scc/UGOE/UXEI/UMIN/scc_umin_ag_xiaoming_fu/umin_kurs_datascismartcity2526/dir.project"

# 明确定义 VENV Python 解释器路径 (解决 'source activate' 错误)
VENV_PYTHON="$HOME/hpc_gpu_venv/bin/python"
# =================================================================
# 2. 软件环境加载 (必须在运行 Python 之前完成)
# =================================================================
module purge

# 2.1 依赖模块：加载 GCC 编译器 (python/3.11.9 的依赖)
module load gcc/13.2.0

# 2.2 Python 模块
module load python/3.11.9

# 2.3 CUDA 模块：确保与 PyTorch cu118 版本一致
module load cuda/11.8.0

# =================================================================
# 3. 运行您的 GPU 应用程序 (使用 VENV 的绝对路径)
# =================================================================

# 切换到项目目录 (您的 Python 脚本和数据文件应位于此目录)
cd $PROJECT_DIR_PATH

echo "Starting job on compute node: $(hostname)"
echo "CUDA Version loaded: $(which nvcc)"
echo "Python Version used: $(which python)"
echo "-------------------------------------"
ABSOLUTE_SCRIPT_PATH="/user/zhuohang.yu/u24922/LCU-main/finetune/generate_embeddings.py"
# 使用 VENV 的绝对路径运行您的 Python 脚本
echo "Starting training script: generate_embeddings.py"
$VENV_PYTHON $ABSOLUTE_SCRIPT_PATH

# 确保脚本返回一个成功退出码
if [ $? -eq 0 ]; then
    echo "✅ Job completed successfully."
else
    echo "❌ Job failed. Check slurm-${SLURM_JOB_ID}.err"
fi
