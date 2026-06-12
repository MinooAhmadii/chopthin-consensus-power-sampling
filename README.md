# Power-SMC + Chopthin

A variant of **Power-SMC** (Azizi et al., *Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning*, arXiv:2602.10273) that swaps the default **systematic resampler** for the **chopthin resampler** (Gandy & Lau, 2016) to preserve particle diversity during sequence-level SMC power sampling.

**Why.** Systematic resampling equalizes particle weights to `1/N` and prunes low-weight particles, collapsing the population onto a few dominant trajectories. **Chopthin** instead *bounds* the ratio between the largest and smallest output weight by `η`, which provably guarantees an ESS floor (`≥ N/2` at `η = 3+√8`): a low-probability-but-possibly-correct trajectory is *thinned, not annihilated*, so it survives long enough to develop. Measured by best-of-N **oracle (pass@N)** accuracy, chopthin keeps strictly more correct trajectories alive than the baseline.

> Base method, full install notes, and benchmark details: see **[`README_powersmc.md`](README_powersmc.md)** (the original Power-SMC readme).

## The whole diff from baseline Power-SMC
1. **`chopthin.py`** — the chopthin resampler (standalone, no internal deps).
2. It is wired into the SMC loop in **`smc_samp_utils.py`** (which `import`s `chopthin`) and selected at runtime by **`--resampler chopthin --eta <η>`** in `run_baseline.py`.

That's it: **one new file + one resampler swap.** Everything else is the baseline.

## Setup
```bash
pip install -r requirements.txt        # or: uv pip install -r requirements.txt
# models download from HuggingFace; set HF_HUB_CACHE / HF_HUB_OFFLINE=1 as needed
```

## Run — chopthin vs. systematic (same entry point, one flag)
```bash
# chopthin (this variant)
python3 run_baseline.py --dataset aime --model qwen --n_particles 32 \
  --max_new_tokens 4096 --alpha 2 --temperature 0.5 --trigger ess \
  --resampler chopthin --eta 5.8284271247461903 \
  --problem_idx_list "$(seq -s, 0 89)" --out_dir runs/aime_chopthin

# baseline (systematic): identical command, just  --resampler systematic   (eta ignored)
```
- `--eta 5.8284271247461903` = `3+√8`, the canonical value giving an ESS floor of `N/2`.
- Datasets: `math`, `aime`, `gsm8k`, `humaneval`, `gpqa` (provided in `data/`).
- `fill_single_rest.slurm` is an example multi-model SLURM launcher showing the full invocation.

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

## Layout
| Path | What |
| ---- | ---- |
| `chopthin.py` | the chopthin resampler (the contribution) |
| `run_baseline.py` | entry point |
| `smc_samp_utils.py`, `power_samp_utils.py`, `constants.py` | SMC engine |
| `grader_utils/` | answer graders (math / code / GPQA) |
| `data/` | benchmark datasets (MATH500, AIME, GSM8K, HumanEval, GPQA) |
| `oracle_all.py` | oracle / pass@N metric |
| `fill_single_rest.slurm` | example multi-model SLURM launcher |
| `README_powersmc.md` | original Power-SMC readme (base method) |
