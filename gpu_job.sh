#!/bin/bash
# -----------------------------------------------------------------------------
# SLURM 资源请求配置,跑LCU
# -----------------------------------------------------------------------------
#SBATCH --job-name=InternVL_DCN_PyTorch # 更新作业名称以反映实际运行的 Python 脚本
#SBATCH --partition=scc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # 增加 CPU 核心数，以匹配 128G 内存
#SBATCH --time=05:00:00                 # 运行时间限制：5 小时
#SBATCH --mem=128G                      # 🔴 关键修正：增大内存请求至 128GB (解决 mmap/RAM 限制)
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gpus=A100:1                   # 关键修正：使用 GRES 语法请求 1 块 A100 GPU

# =================================================================
# 1. 路径和变量定义
# =================================================================
# 明确定义项目路径 (用于 cd)
PROJECT_DIR_PATH="/projects/scc/UGOE/UXEI/UMIN/scc_umin_ag_xiaoming_fu/umin_kurs_datascismartcity2526/dir.project"

# 明确定义 VENV Python 解释器路径 (解决 ModuleNotFoundError)
VENV_PYTHON="$HOME/hpc_gpu_venv/bin/python"

# 明确定义要运行的主 Python 脚本的绝对路径
MAIN_SCRIPT="/user/zhuohang.yu/u24922/LCU-main/src/main.py"

# 明确定义 Python 脚本的输出目录 (用于 --fout 参数)
OUTPUT_DIR="../rec_datasets/WM_KuaiComt/DCN_WLR_0.001_0.1_test1.7b_40_2_61"

# =================================================================
# 2. 软件环境加载
# =================================================================
module purge
module load gcc/13.2.0
module load python/3.11.9
module load cuda/11.8.0 

# =================================================================
# 3. 运行您的 PyTorch 应用程序 (直接调用 VENV Python)
# =================================================================

# 切换到项目目录 (用于处理相对路径和日志输出)
cd $PROJECT_DIR_PATH

echo "Starting job on compute node: $(hostname)"
echo "CUDA Version loaded: $(which nvcc)"
echo "Python Interpreter: $VENV_PYTHON"
echo "-------------------------------------"

# 🔴 核心修正：使用 VENV Python 解释器运行 main.py 并传递所有参数
CUDA_VISIBLE_DEVICES=0 $VENV_PYTHON $MAIN_SCRIPT \
    --fout $OUTPUT_DIR \
    --dat_name KuaiComt \
    --model_name DCN \
    --label_name WLR \
    --randseed 61 \
    --load_to_eval 0 \
    --epoch_num 1 \
    --label1_name user_clicked \
    --label2_name comments_score \
    --lambda1 0.001 \
    --lambda2 0.1

# 检查 Python 脚本的退出码
if [ $? -eq 0 ]; then
    echo "✅ Job completed successfully."
else
    echo "❌ Job failed. Check slurm-${SLURM_JOB_ID}.err"
fi
