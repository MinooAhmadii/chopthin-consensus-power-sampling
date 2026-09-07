<h1 align="center">Chopthin-Consensus Power Sampling</h1>
<h3 align="center">A Diversity-Preserving Approach to LLM Decoding</h3>

<p align="center">
  Minoo Ahmadi, Seyedarmin Azizi, Erfan Baghaei Potraghloo, Mehdi Kamal, Massoud Pedram<br>
  University of Southern California
</p>

<p align="center">
  <a href="https://openreview.net/pdf?id=yfR1TAzTjx"><img src="https://img.shields.io/badge/Paper-OpenReview-b31b1b" alt="Paper on OpenReview"></a>
  <a href="https://github.com/MinooAhmadii/chopthin-consensus-power-sampling/releases/tag/v0.1.0"><img src="https://img.shields.io/badge/Release-v0.1.0-6f42c1" alt="Release v0.1.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2e7d32" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab" alt="Python 3.10+">
  <a href="https://github.com/MinooAhmadii/chopthin-consensus-power-sampling/actions/workflows/tests.yml"><img src="https://github.com/MinooAhmadii/chopthin-consensus-power-sampling/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
</p>

Official code for the paper, accepted at the COLM 2026 Workshop on Efficient Reasoning.

<p align="center">
  <img src="figures/ccps_fig1.svg" alt="Systematic vs Chopthin resampling on the same six particles" width="900">
</p>

*Six particles, two resampling events (circle size = weight, color = founding ancestor, ✕ = deleted, R = surviving lineages). Systematic resampling deletes the low-weight ★ particle and ends with 2 lineages and the wrong majority answer; Chopthin keeps it, 4 lineages remain, and the vote is correct.*

## What it does

Power-SMC decodes N particles in parallel from the sharpened distribution p(y | x)^α and resamples them when their weights become uneven. CCPS changes two things: **Chopthin resampling** bounds the ratio between the largest and smallest weight by η instead of equalizing all weights, so distinct low-weight trajectories survive; **semantic-majority selection** returns the answer supported by the most distinct trajectories instead of a weighted draw (for code, programs are grouped by their behavior on inputs the base model wrote itself). Both selectors run post hoc on the saved particles.

## Install

```bash
git clone https://github.com/MinooAhmadii/chopthin-consensus-power-sampling
cd chopthin-consensus-power-sampling
pip install -r requirements.txt
```

Python 3.10+, one 48 GB GPU for a 7B model at N = 32. The paper used PyTorch 2.5.1 with Transformers 4.45.0 (Qwen2.5) and 4.51.3 (Qwen3-4B).

GPQA Diamond is gated and not included: run `hf auth login` and `python data/build_gpqa_diamond.py` to rebuild the exact file (checksum-verified). Sources and checksums of the other benchmarks are in [`data/README.md`](data/README.md).

## Run

```bash
python scripts/run.py --dataset math --model qwen_math --resampler systematic --seed 42 --out_dir runs/math/qwen_math/systematic
python scripts/run.py --dataset math --model qwen_math --resampler chopthin   --seed 42 --out_dir runs/math/qwen_math/chopthin
```

Defaults are the paper's settings (N = 32, α = 2, ESS trigger 0.5, block 64, α ramp 100 tokens, 4096 new tokens, η = 3+√8). `--dataset` is `math`, `gsm8k`, `aime`, `gpqa` or `humaneval`; `--model` is `qwen_math`, `qwen`, `qwen3` or any Hugging Face id; `--problems` takes `all`, `0,1,2` or `0-99`; `--resume` restarts an interrupted run. HumanEval uses the paper's prompt per model (`--he_pipeline stub|cot`). Every particle is saved under `per_run/`.

## Select and score

```bash
python scripts/select_majority.py --dataset math runs/math/qwen_math/systematic runs/math/qwen_math/chopthin
CCPS_ALLOW_CODE_EXEC=1 python scripts/he_behavior_select.py --run runs/humaneval/qwen_math/chopthin --inputs data/he_inputs/he_inputs_qwen_math.json
python scripts/oracle_coverage.py --dataset math runs/math/qwen_math/systematic runs/math/qwen_math/chopthin
```

Each prints the weight-draw accuracy, the selector's accuracy and the oracle coverage (add `--pool` to treat several seeds as one cell). Checked against the paper's saved runs: all 15 cells of Table 2 and the oracle coverage of Figure 2 reproduce exactly. HumanEval grading executes model-written code, so it is off unless `CCPS_ALLOW_CODE_EXEC=1` is set; run it in an isolated environment.

`ccps/` holds the library (resampler, SMC loop, prompts, graders), `scripts/` the entry points, `tests/` offline tests (`python -m pytest tests`).

## Citation

```bibtex
@inproceedings{ahmadi2026ccps,
  title     = {Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding},
  author    = {Ahmadi, Minoo and Azizi, Seyedarmin and Baghaei Potraghloo, Erfan and Kamal, Mehdi and Pedram, Massoud},
  booktitle = {COLM 2026 Workshop on Efficient Reasoning},
  year      = {2026}
}
```

## License

MIT. Built on [Power-SMC](https://github.com/ArminAzizi98/Power-SMC) (MIT); file-level provenance in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
