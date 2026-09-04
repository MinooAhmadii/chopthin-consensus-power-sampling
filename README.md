# Chopthin-Consensus Power Sampling (CCPS)

**Diversity-preserving resampling and consensus selection for training-free LLM reasoning.**
Published at the COLM 2026 Workshop on Efficient Reasoning.

<p align="center">
  <img src="figures/ccps_fig1.png" alt="Systematic vs Chopthin resampling on the same six particles" width="900">
</p>

*Six particles, two resampling events. Circle size = weight, color = founding ancestor, ✕ = deleted, R = surviving lineages. **Left:** standard (systematic) resampling resets every weight to 1/N, kills the low-weight ★ particle at the first event, and ends with 2 lineages and the wrong majority answer. **Right:** Chopthin keeps unequal weights within a bounded ratio, so ★ survives, 4 lineages remain, and the vote is correct.*

## What this is

Power sampling draws answers from a sharpened version of the base model, p(y | x)^α with α > 1, and recovers much of the reasoning gain of RL post-training with no training. Running it as sequential Monte Carlo ([Power-SMC](https://github.com/ArminAzizi98/Power-SMC)) makes it fast: N particles decode in parallel and are resampled when their weights become uneven.

CCPS changes two things in that pipeline:

1. **Chopthin resampling** (Gandy & Lau, 2016) replaces systematic resampling. Instead of forcing all weights to be equal, it bounds the ratio between the largest and smallest weight by η, thins light particles instead of deleting them, and carries the unequal weights forward. It is unbiased, returns exactly N particles, and guarantees an ESS floor of about N/2 at η = 3+√8.
2. **Semantic-majority selection** replaces the final weight draw. Identical trajectories are merged, equivalent answers are clustered, and the answer supported by the most *distinct* trajectories is returned. For code, programs are clustered by their behavior on model-generated test inputs; no ground-truth tests are used.

## Results

Final-answer accuracy (%). N = 32, α = 2, no fine-tuning.

| Model | Method | MATH500 | GSM8K | AIME | GPQA | HumanEval |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-Math-7B | Power-SMC | 77.0 | 89.5 | 15.6 | 30.3 | 58.5 |
| | **CCPS** | **81.4** | **90.8** | **16.7** | **33.8** | **61.6** |
| Qwen2.5-7B | Power-SMC | 74.2 | 91.0 | 10.4 | 29.3 | 73.2 |
| | **CCPS** | **76.0** | **91.4** | **12.2** | **30.3** | **76.8** |
| Qwen3-4B | Power-SMC | 79.0 | 90.1 | 15.6 | 29.8 | **71.3** |
| | **CCPS** | **81.8** | **92.1** | 15.6 | **40.4** | 70.7 |

Chopthin raises **oracle coverage** (the fraction of problems where at least one of the 32 particles is correct) in 13 of 15 settings. The selector turns that coverage into accuracy.

<p align="center">
  <img src="figures/oracle_coverage.png" alt="Oracle coverage, Chopthin vs systematic, 15 model x benchmark cells" width="900">
</p>

## Install

```bash
git clone https://github.com/MinooAhmadii/chopthin-consensus-power-sampling
cd chopthin-consensus-power-sampling
pip install -r requirements.txt
```

Python 3.10+ and one GPU with at least 48 GB for a 7B model at N = 32. Models download from Hugging Face on first run.

## Run

Both arms use the same command; only `--resampler` differs. These flags reproduce the paper setting (N = 32, α = 2, ESS trigger 0.5, block 64, α-ramp over the first 100 tokens, up to 4096 new tokens).

```bash
# CCPS arm: Chopthin with carried weights
python3 run_baseline.py --dataset math --model qwen_math --n_particles 32 \
  --max_new_tokens 4096 --temperature 0.5 --ramp_T 100 --seed 42 \
  --resampler chopthin --eta 5.8284271247461903 \
  --problem_idx_list "$(seq -s, 0 499)" --out_dir runs/math/chopthin

# Power-SMC baseline: systematic resampling (same command, one flag)
python3 run_baseline.py --dataset math --model qwen_math --n_particles 32 \
  --max_new_tokens 4096 --temperature 0.5 --ramp_T 100 --seed 42 \
  --resampler systematic \
  --problem_idx_list "$(seq -s, 0 499)" --out_dir runs/math/systematic
```

- `--temperature` sets α = 1/temperature, so 0.5 means α = 2. There is no `--alpha` flag.
- `--eta 5.8284271247461903` is 3+√8, the value whose ESS floor is about N/2. `--resampler chopthin_reset` is the ablation that uses Chopthin's offspring counts but resets the weights.
- `--seed` is required. The same seed gives both arms identical token streams until their first differing resampling event, so comparisons are paired.
- Datasets (`--dataset`): `math` (MATH500, 500), `gsm8k` (1319), `aime` (2022–2024, 90), `gpqa` (Diamond, 198), `humaneval` (164). All files are in `data/`.
- Models (`--model`): `qwen_math` (Qwen2.5-Math-7B), `qwen` (Qwen2.5-7B), `qwen3` (Qwen3-4B), `phi` (Phi-3.5-mini). Add your own to `model_map` in `run_baseline.py`.

Each run folder gets `per_question.jsonl` (selected answer and outcome per problem), `per_run/p*.json` (all N particles: token ids, weights, chosen index), and `config.json` (every resolved setting).

### Selection and metrics

```bash
# Oracle coverage (best-of-N) and selected accuracy, Chopthin vs systematic.
# Edit the run-folder list at the bottom of the script for your own runs.
CCPS_RUNS=runs python3 oracle_all.py all

# HumanEval: behavior-majority selection, no ground-truth tests
python3 he_gen_inputs.py qwen_math               # step 1: the base model writes test inputs
python3 he_behavior_select.py --runs-dir runs    # step 2: cluster programs by behavior, vote
```

`he_gen_inputs.slurm` runs both HumanEval steps on SLURM. The math/GPQA semantic-majority selector (merge, cluster with the task grader, vote by distinct trajectories) will be added to this repository.

`python3 figures/figure1_oracle_ceiling.py` regenerates the coverage figure above from the Table 3 counts.

## Layout

| Path | What |
|---|---|
| `chopthin.py` | the Chopthin resampler (pure Python, no dependencies) |
| `smc_samp_utils.py` | SMC loop; calls `chopthin()` when `--resampler chopthin` |
| `run_baseline.py` | entry point for all benchmarks |
| `power_samp_utils.py`, `constants.py` | prompt templates and model utilities |
| `oracle_all.py` | oracle coverage and selected accuracy per arm |
| `he_gen_inputs.py`, `he_behavior_select.py`, `he_gen_inputs.slurm` | behavior-majority selector for code |
| `grader_utils/` | answer graders (math, GSM8K, GPQA) and the HumanEval sandbox |
| `data/` | MATH500, GSM8K, AIME 2022–2024, GPQA Diamond, HumanEval |
| `figures/` | figure script and images used in this README |
| `README_powersmc.md` | the original Power-SMC readme (engine details, single-prompt API) |

## Citation

```bibtex
@inproceedings{ahmadi2026ccps,
  title     = {Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding},
  author    = {Ahmadi, Minoo and Azizi, Seyedarmin and Baghaei Potraghloo, Erfan and Kamal, Mehdi and Pedram, Massoud},
  booktitle = {COLM 2026 Workshop on Efficient Reasoning},
  year      = {2026}
}
```

## Acknowledgments

This repository is built on **Power-SMC** and its codebase: Azizi, Baghaei Potraghloo, Ahmadi, Kundu, and Pedram, *Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning* ([arXiv:2602.10273](https://arxiv.org/abs/2602.10273), [code](https://github.com/ArminAzizi98/Power-SMC)). The SMC engine, KV-cache handling, graders, and benchmark loaders come from there; CCPS adds the resampler and the selector. Power-SMC in turn builds on [Reasoning with Sampling](https://github.com/aakaran/reasoning-with-sampling) by Karan & Du. The Chopthin algorithm is from Gandy & Lau, *The chopthin algorithm for resampling*, IEEE Transactions on Signal Processing, 2016.
