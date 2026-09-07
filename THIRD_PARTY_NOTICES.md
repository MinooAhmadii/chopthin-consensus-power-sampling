# Third-party code and data

CCPS is released under the MIT License (see `LICENSE`). The files below contain or derive from
work by others; their notices are preserved here and, where the file is a close copy, in the
file header.

| File | Origin | License | What CCPS changed |
|---|---|---|---|
| `ccps/smc.py` | Power-SMC, `smc_samp_utils.py` (Azizi et al., 2026), https://github.com/ArminAzizi98/Power-SMC | MIT, Copyright (c) 2026 Seyedarmin Azizi | Chopthin resampling branch, unequal-weight carry-over, unused options removed, renamed |
| `ccps/prompts.py` | Power-SMC, `power_samp_utils.py` / `constants.py`; the math prompt follows Reasoning-with-Sampling (Karan & Du, 2025), https://github.com/aakaran/reasoning-with-sampling | MIT (Power-SMC); the upstream Reasoning-with-Sampling repository publishes no license file | Reduced to the three prompt templates used in the paper; HumanEval prompts moved to `he_protocols.py` |
| `ccps/graders/math_grader.py`, `ccps/graders/math_normalize.py` | OpenAI PRM800K grader (`grading/grader.py`, `grading/math_normalize.py`), https://github.com/openai/prm800k; `math_normalize.py` is itself adapted from the Hendrycks et al. MATH release | MIT, Copyright (c) 2023 OpenAI; MATH: MIT | Input guard before SymPy parsing (`_looks_like_math`) |
| `ccps/graders/he_execute.py` | OpenAI human-eval `execution.py`, https://github.com/openai/human-eval | MIT, Copyright (c) 2021 OpenAI | `exec` enabled behind the `CCPS_ALLOW_CODE_EXEC=1` opt-in |
| `ccps/graders/answers.py` | Written for CCPS (three-tier answer extraction, grading dispatch); the MATH check calls the PRM800K grader above | MIT (CCPS) | — |
| `ccps/he_protocols.py` | Written for CCPS. The HumanEval stop-word list is the standard one also used by Zhou et al. (2026) | MIT (CCPS) | — |
| `ccps/chopthin.py` | Written for CCPS; implements Gandy & Lau, *The chopthin algorithm for resampling*, IEEE TSP 2016 | MIT (CCPS) | — |
| `data/MATH500.json`, `data/gsm8k.jsonl`, `data/HumanEval.jsonl`, `data/aime_combined.jsonl` | See `data/README.md` | See `data/README.md` | Format only |

Reasoning-with-Sampling (Karan & Du) reached this repository through Power-SMC, which is
MIT-licensed by a co-author of CCPS. The only code in this repository whose provenance leads
only to the unlicensed upstream (a HumanEval code-extraction regex) has been removed; the
graders it shipped are OpenAI's MIT-licensed files listed above.
