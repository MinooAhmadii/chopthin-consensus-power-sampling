# Benchmark files

Every record has the fields `run.py` reads (`prompt`/`problem`, `answer`, `id`) plus the metadata of its source. The files are the exact ones used for the paper; the checksums let you verify a copy.

| File | Records | Source | License | sha256 |
|---|---|---|---|---|
| `MATH500.json` | 500 | PRM800K `math_splits/test.jsonl` (the 500-problem MATH subset of Lightman et al., 2023), as shipped by Reasoning-with-Sampling and Power-SMC | MIT (OpenAI PRM800K; problems from Hendrycks et al. MATH, MIT) | `838cd5ffc217ee85…` |
| `gsm8k.jsonl` | 1319 | GSM8K test split (Cobbe et al., 2021), `openai/gsm8k` on Hugging Face | MIT | `f677cc2d55e35ab3…` |
| `aime_combined.jsonl` | 90 | AIME 2022, 2023, 2024 (I and II), 90 problems, from the `AI-MO/aimo-validation-aime` dataset on Hugging Face; the `url` field of each record points to the AoPS wiki page of the problem | problems are copyright MAA; redistributed here for research use as in prior work | `7fc0fde0c2d34989…` |
| `HumanEval.jsonl` | 164 | OpenAI HumanEval `HumanEval.jsonl.gz` (Chen et al., 2021) | MIT | `09a2a07b794cc062…` |

## GPQA Diamond (not included)

GPQA (Rein et al., 2023) is distributed under a gated license that asks users not to publish the questions, so `gpqa_diamond.json` is **not** in this repository. Rebuild it locally:

```bash
hf auth login                     # (or huggingface-cli login) an account that has accepted the GPQA terms
python data/build_gpqa_diamond.py # -> data/gpqa_diamond.json (198 records)
```

The script downloads `gpqa_diamond.csv` from `Idavidrein/gpqa` (pinned revision), orders the four choices exactly as in the paper's runs (a fixed per-record shuffle, so nothing about the questions is stored here), and checks the result against the checksum of the file the paper used (`b18c9dad2621abd9…`).

## HumanEval behavioral test inputs

`he_inputs/he_inputs_<model>.json` holds the call expressions each base model wrote for `he_behavior_select.py` (Section 4.3 of the paper). They were generated once with `he_gen_inputs.py` and are shared by both resampling arms and all selectors.

| File | sha256 |
|---|---|
| `he_inputs/he_inputs_qwen_math.json` | `d7e28982e2254e10…` |
| `he_inputs/he_inputs_qwen.json` | `53a7c385cee60edb…` |
| `he_inputs/he_inputs_qwen3.json` | `36d51127326d99fe…` |

Full checksums:

```
838cd5ffc217ee852f460a5c649ea4825f777e1b99c590b38fc500c6561e1e06  data/MATH500.json
f677cc2d55e35ab3f73a8617293175340a6483378f33b6d184b5228285a74907  data/gsm8k.jsonl
7fc0fde0c2d349891e33daebb2eaeae6f67fe303db4d5c11937dbb6ca0a7f511  data/aime_combined.jsonl
09a2a07b794cc062d0ae0fada3779b5db9da735d94a3e7b7a6af6516708fed15  data/HumanEval.jsonl
d7e28982e2254e10a145294dccdef0b1c5aaaaa61de439c5ab21bdbe3d1609ee  data/he_inputs/he_inputs_qwen_math.json
53a7c385cee60edb56399b9a574b202acc605b6b50577fe87f73a10ccb4555d9  data/he_inputs/he_inputs_qwen.json
36d51127326d99fe48ef91220a654cafeaa0491c257a7e79caec613b288accb5  data/he_inputs/he_inputs_qwen3.json
```
