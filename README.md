# Power-SMC + Chopthin

A variant of **Power-SMC** (Azizi et al., *Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning*, arXiv:2602.10273) that swaps the default **systematic resampler** for the **chopthin resampler** (Gandy & Lau, 2016) to preserve particle diversity during sequence-level SMC power sampling.

**Why.** Systematic resampling equalizes particle weights to `1/N` and prunes low-weight particles, collapsing the population onto a few dominant trajectories. **Chopthin** instead *bounds* the ratio between the largest and smallest output weight by `η`, which provably guarantees an ESS floor (`≥ N/2` at `η = 3+√8`): a low-probability-but-possibly-correct trajectory is *thinned, not annihilated*, so it survives long enough to develop. Measured by best-of-N **oracle (pass@N)** accuracy, chopthin keeps strictly more correct trajectories alive than the baseline.

> Base method, full install notes, and benchmark details: see **[`README_powersmc.md`](README_powersmc.md)** (the original Power-SMC readme).

## The whole diff from baseline Power-SMC
1. **`chopthin.py`** — the chopthin resampler (standalone, no internal deps).
2. It is wired into the SMC loop in **`smc_samp_utils.py`** (which `import`s `chopthin`) and selected at runtime by **`--resampler chopthin --eta <η>`** in `run_baseline.py`.

That's it: **one new file + one resampler swap.** Everything else is the baseline.

## How to run it (you do **not** run `chopthin.py` directly)
`chopthin.py` is a **library, not an entry point.** The only thing you execute is **`run_baseline.py`**, and `chopthin.py` is reached *only* when you pass `--resampler chopthin`:

```
run_baseline.py                       ← the script you run
  └─ smc_samp_utils.py                 ← SMC kernel  (from chopthin import chopthin)
       └─ on each resampling step, if --resampler chopthin:
            _chopthin_resample()  →  chopthin()     ← code in chopthin.py
```

- **`--resampler systematic`** → the baseline path (built-in `systematic_resample`); `chopthin.py` is imported but **never called**.
- **`--resampler chopthin`** → activates `chopthin.py`.

The baseline and chopthin runs are **the same command**, differing only in that one flag (see below).

## Setup
```bash
pip install -r requirements.txt        # or: uv pip install -r requirements.txt
# models download from HuggingFace; set HF_HUB_CACHE / HF_HUB_OFFLINE=1 as needed
```

## Run — chopthin vs. systematic (same entry point, one flag)
```bash
# chopthin (this variant)
python3 run_baseline.py --dataset aime --model qwen --n_particles 32 \
  --max_new_tokens 4096 --temperature 0.5 --seed 42 --trigger ess \
  --resampler chopthin --eta 5.8284271247461903 \
  --problem_idx_list "$(seq -s, 0 89)" --out_dir runs/aime_chopthin

# baseline (systematic): identical command, just  --resampler systematic   (eta ignored)
```
- **α is set by `--temperature` (α = 1/temperature):** `--temperature 0.5` ⇒ α = 2. There is no `--alpha` flag.
- `--seed` is **required**.
- `--eta 5.8284271247461903` = `3+√8`, the canonical value giving an ESS floor of `N/2`.
- Datasets: `math`, `aime`, `gsm8k`, `humaneval`, `gpqa` (provided in `data/`).

## Change the model
Edit `model_map` in **`run_baseline.py`** and add a line:
```python
model_map = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "qwen3": "Qwen/Qwen3-4B",
    "mymodel": "org/your-model-id",      # <- add yours
}
```
then pass `--model mymodel`. (Models without GQA, e.g. Phi-3.5-mini, have a large KV cache at `N=32×4096`; use `--device auto` with ≥3 GPUs.)

## Outputs & the oracle metric
Each `--out_dir` gets:
- `per_question.jsonl` — selected-answer outcome per problem
- `per_run/p*.json` — all `N` particles per problem (token ids, weights, chosen index)
- `config.json` — the full resolved config (every hyperparameter + the exact CLI args)

Compute **oracle / pass@N** — a problem counts correct if ≥1 of the `N` particles is correct, the key diversity metric — with **`oracle_all.py`** (edit the folder/dataset list at the bottom, then `python3 oracle_all.py`).

## Test-free HumanEval selection (behavior-majority)
For code, a majority over answer *strings* is undefined, so we vote over program **behavior** instead — the code analogue of the semantic-majority selector used on the math/QA benchmarks. It uses **no ground-truth tests**, so it engages on all 164 HumanEval problems and is deployable on unseen problems.

Two steps, run from the repo root:
```bash
# 1) each base model synthesizes its own test INPUTS (inputs only, no gold outputs)
python3 he_gen_inputs.py qwen_math      # -> stage1_inputs_qwen_math.json
python3 he_gen_inputs.py qwen
python3 he_gen_inputs.py qwen3

# 2) run the 32 candidates on those inputs, cluster by identical behavior, majority-vote
python3 he_behavior_select.py --runs-dir runs
```
`he_behavior_select.py` prints, per (model, arm): the `smc` floor, the **behavior-majority** accuracy, and the `oracle` ceiling, plus how many problems it engaged on. HumanEval's held-out unit tests are used **only to score** the selected program — the selector never sees them. `he_gen_inputs.slurm` runs both steps on SLURM.

> Candidate programs are decoded from the saved `per_run/p*.json` particles (see Outputs above); the behavioral executor is the sandbox in `grader_utils/he_execute.py`.

## Figures
`figures/figure1_oracle_ceiling.py` regenerates **Figure 1** (oracle coverage, chopthin vs. systematic, all 15 model×benchmark cells). The oracle counts are embedded in the script; only matplotlib is required.
```bash
python3 figures/figure1_oracle_ceiling.py     # -> oracle_ceiling_wide.pdf / .png
```

## Layout
| Path | What |
| ---- | ---- |
| `chopthin.py` | the chopthin resampler (the contribution) |
| `run_baseline.py` | entry point |
| `smc_samp_utils.py`, `power_samp_utils.py`, `constants.py` | SMC engine |
| `grader_utils/` | answer graders (math / code / GPQA) + HumanEval sandbox (`he_execute.py`) |
| `data/` | benchmark datasets (MATH500, AIME, GSM8K, HumanEval, GPQA) |
| `oracle_all.py` | oracle / pass@N metric |
| `he_gen_inputs.py` | test-free HumanEval, step 1: base model generates its own test inputs |
| `he_behavior_select.py` | test-free HumanEval, step 2: behavior-majority code selector |
| `he_gen_inputs.slurm` | SLURM launcher for both HumanEval steps |
| `figures/figure1_oracle_ceiling.py` | regenerates Figure 1 (oracle coverage, all 15 cells) |
| `README_powersmc.md` | original Power-SMC readme (base method) |
