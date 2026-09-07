# Chopthin-Consensus Power Sampling (CCPS)

Code for *Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding*, accepted at the [COLM 2026 Workshop on Efficient Reasoning](https://wdlctc.github.io/efficient-reasoning-2026/) (October 9, 2026; non-archival). Paper: [OpenReview](https://openreview.net/forum?id=yfR1TAzTjx).

<p align="center">
  <img src="figures/ccps_fig1.svg" alt="Systematic vs Chopthin resampling on the same six particles" width="900">
</p>

*Six particles, two resampling events. Circle size = weight, color = founding ancestor, ✕ = deleted, R = surviving lineages. **Left:** systematic resampling resets every weight to 1/N, deletes the low-weight ★ particle at the first event, and ends with 2 lineages and the wrong majority answer. **Right:** Chopthin keeps unequal weights within a bounded ratio, so ★ survives, 4 lineages remain, and the vote is correct.*

## What it does

Power sampling draws answers from a sharpened version of the base model, p(y | x)^α with α > 1, and recovers much of the gain of RL post-training without any training. Power-SMC runs it as sequential Monte Carlo: N particles decode in parallel and are resampled when their weights become uneven.

CCPS changes two things in that pipeline:

1. **Chopthin resampling** replaces systematic resampling. Instead of forcing all weights to be equal, it bounds the ratio between the largest and smallest weight by η, keeps light particles with probability proportional to their weight, and carries the unequal weights forward. It returns exactly N particles, is unbiased, and guarantees an ESS floor of about N/2 at η = 3+√8.
2. **Semantic-majority selection** replaces the final weight draw. Identical trajectories are merged, equivalent answers are clustered, and the answer supported by the most distinct trajectories is returned. For code (HumanEval) the clusters are formed by running the candidate programs on inputs the base model wrote itself.

Both steps are post hoc with respect to the sampler: `run.py` saves every particle, and the selectors read the saved runs.

## Results

Chopthin raises oracle coverage (at least one of the 32 particles is correct) in 13 of 15 model–benchmark settings, and CCPS matches or beats Power-SMC's final-answer accuracy in 14 of 15. Full tables are in the paper; every number in them can be regenerated from the saved runs with the scripts below.

<p align="center">
  <img src="figures/oracle_coverage.svg" alt="Oracle coverage, Chopthin vs systematic, 15 model x benchmark cells" width="900">
</p>

## Install

```bash
git clone https://github.com/MinooAhmadii/chopthin-consensus-power-sampling
cd chopthin-consensus-power-sampling
pip install -r requirements.txt
```

Python 3.10+ and one GPU with at least 48 GB for a 7B model at N = 32. Models download from Hugging Face on first run. The paper's runs used PyTorch 2.5.1 and Transformers 4.45.0 for the Qwen2.5 models and 4.51.3 for Qwen3-4B (Qwen3 needs ≥ 4.51, which is what `requirements.txt` pins).

## Data

MATH500, GSM8K, AIME 2022–2024 and HumanEval are in `data/` with their sources, licenses and checksums listed in [`data/README.md`](data/README.md). **GPQA Diamond is not included**: its license asks users not to publish the questions. Rebuild the exact file the paper used from the gated release with your own Hugging Face access:

```bash
hf auth login                          # or: huggingface-cli login
python data/build_gpqa_diamond.py      # -> data/gpqa_diamond.json, checksum-verified
```

## Security

Two graders evaluate model output:

- **HumanEval** runs model-written Python (`grader_utils/he_execute.py`, adapted from OpenAI's human-eval sandbox: separate process, timeout, destructive calls disabled). That is still code execution. It is off by default; run it inside an isolated environment (container or VM, no network, nothing valuable on disk) and set `CCPS_ALLOW_CODE_EXEC=1`. `run.py --dataset humaneval`, `he_behavior_select.py` and `oracle_coverage.py --dataset humaneval` refuse to start without it.
- **MATH500** compares answers with SymPy, whose parser evaluates Python. `grader_utils/math_grader.py` refuses, before parsing, strings that do not look like a plain math answer: quotes, dunders, statement separators, any word other than a few math functions, and power towers or large exponents whose evaluation would run away.

## Run

The defaults are the paper's settings: N = 32, α = 2, ESS trigger 0.5, block 64, α ramped over the first 100 tokens, up to 4096 new tokens, η = 3+√8.

```bash
# Power-SMC arm (systematic resampling)
python run.py --dataset math --model qwen_math --resampler systematic --seed 42 --out_dir runs/math/qwen_math/systematic

# Chopthin arm
python run.py --dataset math --model qwen_math --resampler chopthin --seed 42 --out_dir runs/math/qwen_math/chopthin
```

- `--dataset`: `math` (MATH500), `gsm8k`, `aime` (2022–2024), `gpqa` (Diamond), `humaneval`.
- `--model`: `qwen_math` (Qwen2.5-Math-7B), `qwen` (Qwen2.5-7B), `qwen3` (Qwen3-4B), or any Hugging Face model id.
- `--problems`: `all` (default), a list `0,1,2`, or a range `0-99`. `--resume` skips problems whose per-run file already exists, so a sharded or interrupted run can be restarted.
- `--temperature` sets α = 1/temperature; 0.5 means α = 2.
- `--resampler chopthin_reset` is the ablation of Section 6.2 (Chopthin's offspring counts with the weights reset to 1/N).
- `--he_pipeline` (HumanEval only): `stub` feeds the raw code stub and cuts the completion at the first canonical stop word; `cot` adds a chain-of-thought instruction and takes the last ```python block as a standalone program. The default is the paper's choice per model: `cot` for `qwen`, `stub` for `qwen_math` and `qwen3` (under `cot` the math model reasons past the token budget before writing code). The protocol is recorded in the run's `config.json`.

Each run folder gets `per_run/p*.json` (all N particles: token ids, final weights, chosen index), `per_question.jsonl` (weight-drawn answer and outcome per problem), and `config.json` (every setting, the model id, library versions and the git commit).

## Selection and coverage

All three scripts read run folders written by `run.py` and take the tokenizer from each run's `config.json` (`--tokenizer` overrides it).

```bash
# Semantic-majority selection (MATH500, GSM8K, AIME, GPQA): the CCPS rows of Table 2 are the
# "majority" column of the Chopthin arm. Also prints the weight draw, argmax and oracle.
python select_majority.py --dataset math runs/math/qwen_math/systematic runs/math/qwen_math/chopthin

# Several seeds as one cell (the paper's Qwen2.5-7B AIME cell is seeds 42, 43, 44 pooled)
python select_majority.py --dataset aime --pool runs/aime/qwen/s42/chopthin runs/aime/qwen/s43/chopthin runs/aime/qwen/s44/chopthin

# HumanEval: behavior-majority selection without ground-truth tests
python he_gen_inputs.py qwen_math          # optional: the base model writes test inputs -> data/he_inputs/
CCPS_ALLOW_CODE_EXEC=1 python he_behavior_select.py --run runs/humaneval/qwen_math/systematic runs/humaneval/qwen_math/chopthin \
                                                    --inputs data/he_inputs/he_inputs_qwen_math.json

# Oracle coverage (Figure 2) and weight-draw accuracy of two runs
python oracle_coverage.py --dataset math runs/math/qwen_math/systematic runs/math/qwen_math/chopthin
```

The HumanEval test inputs used in the paper are in `data/he_inputs/` (one file per base model, shared by both arms and all selectors), so `he_gen_inputs.py` only matters for a new model. `he_behavior_select.py` decodes programs with the protocol recorded in the run's `config.json` (`--protocol stub|cot` for runs made before it was recorded).

## Reproducing the paper

Table 2 has 15 cells (3 models × 5 benchmarks), each with a systematic arm and a Chopthin arm at seed 42; the Qwen2.5-7B AIME cell additionally uses seeds 43 and 44 and reports the three pooled (n = 270). Everything else is the default.

```bash
for model in qwen_math qwen qwen3; do
  for ds in math gsm8k aime gpqa humaneval; do
    for arm in systematic chopthin; do
      python run.py --dataset $ds --model $model --resampler $arm --seed 42 --out_dir runs/$ds/$model/$arm
    done
  done
done
# then select_majority.py / he_behavior_select.py on each pair, oracle_coverage.py for Figure 2
```

The scripts in this repository were checked against the paper's saved runs: `select_majority.py` reproduces all 12 discrete-answer cells of Table 2 (both arms) and the oracle counts of Figure 2 exactly, and `he_behavior_select.py` reproduces the three HumanEval cells (61.6 / 76.8 / 70.7) and their oracles.

`python figures/oracle_coverage.py` redraws the coverage figure above from the embedded counts.

## Tests

```bash
pip install pytest
python -m pytest tests          # chopthin properties, math grader, HumanEval cot contract
```

The HumanEval contract test executes HumanEval's own reference solutions in the sandbox and opts in by itself; nothing else runs model output. The same tests run in CI (`.github/workflows/tests.yml`).

## Layout

| Path | What |
|---|---|
| `chopthin.py` | the Chopthin resampler (pure Python) |
| `smc.py` | SMC decoding loop; calls `chopthin()` when `--resampler chopthin` |
| `run.py` | entry point |
| `prompts.py`, `he_protocols.py` | prompt templates; HumanEval prompt/extraction protocols (`stub`, `cot`) |
| `select_majority.py` | semantic-majority selector (MATH500, GSM8K, AIME, GPQA) |
| `he_gen_inputs.py`, `he_behavior_select.py` | behavior-majority selector for code |
| `oracle_coverage.py` | oracle coverage of saved runs |
| `grader_utils/` | answer extraction and grading; HumanEval sandbox |
| `data/` | benchmark files, GPQA rebuild script, HumanEval test inputs ([`data/README.md`](data/README.md)) |
| `figures/` | figure script and the images above |
| `tests/` | offline tests |

## Citation

```bibtex
@inproceedings{ahmadi2026ccps,
  title     = {Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding},
  author    = {Ahmadi, Minoo and Azizi, Seyedarmin and Baghaei Potraghloo, Erfan and Kamal, Mehdi and Pedram, Massoud},
  booktitle = {COLM 2026 Workshop on Efficient Reasoning},
  year      = {2026}
}
```

`CITATION.cff` carries the same entry.

## License and acknowledgments

MIT License (see `LICENSE`). The SMC decoding loop, KV-cache handling, graders and benchmark files come from [Power-SMC](https://github.com/ArminAzizi98/Power-SMC) (Azizi, Baghaei Potraghloo, Ahmadi, Kundu, and Pedram, [arXiv:2602.10273](https://arxiv.org/abs/2602.10273)), MIT-licensed, which in turn builds on [Reasoning with Sampling](https://github.com/aakaran/reasoning-with-sampling) by Karan & Du; the math grader is OpenAI's PRM800K grader and the code sandbox is OpenAI's human-eval. File-level provenance and notices are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The Chopthin algorithm is from Gandy & Lau, *The chopthin algorithm for resampling*, IEEE Transactions on Signal Processing, 2016.
