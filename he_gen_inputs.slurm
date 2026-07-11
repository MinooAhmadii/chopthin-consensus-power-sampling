#!/bin/bash
#SBATCH --job-name=he_inputgen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=03:00:00
#SBATCH --output=he_inputgen_%j.out
set -e

# --- edit for your cluster/environment ---
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate <env-with-transformers+torch>
# export HF_HUB_CACHE=/path/to/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Step 1: each base model synthesizes its own test inputs (inputs only, no gold tests).
for m in qwen_math qwen qwen3; do
  echo "=== input-gen $m ==="
  python3 -u he_gen_inputs.py "$m"
done
echo INPUTGEN_DONE

# Step 2: run the 32 candidates on those inputs, cluster by behavior, majority-vote.
echo "=== behavior-majority selection (all 164, model-generated inputs) ==="
python3 -u he_behavior_select.py --runs-dir runs
echo ALL_DONE
