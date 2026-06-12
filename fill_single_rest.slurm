#!/usr/bin/env bash
#SBATCH --job-name=single_rest
#SBATCH --array=0-9%2
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=/data/projects/nullanet/experiments/minoo/choptin/slurm_screen/single_rest_%A_%a.log
#SBATCH --time=24:00:00
source /data/nfs/home/minoo/miniconda3/etc/profile.d/conda.sh
export HF_HUB_CACHE=/models
export TRANSFORMERS_CACHE=/models
export HUGGINGFACE_HUB_CACHE=/models
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/projects/nullanet/experiments/minoo/choptin
#          0     1     2          3          4      5      6     7     8          9
MODELS=( qwen  qwen  qwen       qwen       qwen3  qwen3  qwen3 qwen3 qwen3      qwen3)
DATAS=(  aime  aime  humaneval  humaneval  math   math   aime  aime  humaneval  humaneval)
NAMES=(  a2_A  a2_C  a2_A       a2_C       a2_A   a2_C   a2_A  a2_C  a2_A       a2_C)
DIRS=( qwen2.5-7b qwen2.5-7b qwen2.5-7b qwen2.5-7b qwen3-4b qwen3-4b qwen3-4b qwen3-4b qwen3-4b qwen3-4b)
ENVS=( minoo_env minoo_env minoo_env minoo_env minoo_qwen3 minoo_qwen3 minoo_qwen3 minoo_qwen3 minoo_qwen3 minoo_qwen3)
EXTRA=("--resampler systematic" "--resampler chopthin" "--resampler systematic" "--resampler chopthin" "--resampler systematic" "--resampler chopthin" "--resampler systematic" "--resampler chopthin" "--resampler systematic" "--resampler chopthin")
i=$SLURM_ARRAY_TASK_ID
conda activate ${ENVS[$i]}
ds=${DATAS[$i]}
if [ "$ds" = "aime" ]; then PROBS=$(seq -s, 0 89); elif [ "$ds" = "humaneval" ]; then PROBS=$(seq -s, 0 163); else PROBS=$(seq -s, 0 499); fi
echo "env=${ENVS[$i]} model=${MODELS[$i]} ds=$ds arm=${NAMES[$i]} tok=4096 -> runs/${ds}_${DIRS[$i]}_n32_s42/${NAMES[$i]}"
python3 run_baseline.py --dataset $ds --problem_idx_list "$PROBS" --seed 42 --model ${MODELS[$i]} \
  --n_particles 32 --max_new_tokens 4096 --block_size 64 --ramp_T 100 --ess_threshold 0.5 \
  --boxed_check_window_tokens 256 --eta 5.8284271247461903 --device cuda:0 \
  --out_dir "runs/${ds}_${DIRS[$i]}_n32_s42/${NAMES[$i]}" --temperature 0.5 --trigger ess ${EXTRA[$i]}
echo "=== single_rest ${MODELS[$i]} ${ds} ${NAMES[$i]} DONE ==="
