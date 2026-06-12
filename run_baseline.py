"""Power-SMC baseline runner — built on Armin Azizi's official implementation.

This script is a thin wrapper around Armin's `smc_power_sample_memopt` that:
- uses his exact SMCSamplingConfig defaults (max_new_tokens=4096 by default in v3-parity
  mode; stop_on_boxed=True, n_particles=32, temperature=0.25, alpha=4.0, ess_threshold=0.5,
  block_size=64)
- supports MATH-500, AIME, GPQA via --dataset flag (3-tier parser per dataset)
- mirrors v3 island runner's instrumentation: 4-way outcome, per_question.jsonl,
  startup assertions, full resolved-config dump, metadata stamping

Usage:
    python run_baseline.py --dataset math --problem_idx_list 0,1,2,3,4 \
        --seed 42 --out_dir experiments/baseline_smoke
"""

import argparse
import collections
import datetime as _dt
import json
import os
import random
import re as _re_b
import subprocess as _sp
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import PROMPT, COT
from grader_utils.math_grader import grade_answer
from grader_utils.parse_utils import parse_answer
from power_samp_utils import format_prompt
from smc_samp_utils import SMCSamplingConfig, smc_power_sample_memopt


# =========================================================================
# v3-parity: 3-tier per-dataset parser + 4-way outcome classification
# (copied verbatim from run_island_prototype_v3.py — keep in sync if either changes)
# =========================================================================

_TIER_SPECS = {
    "math": [
        (_re_b.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", _re_b.DOTALL), "full"),
        (_re_b.compile(r"(?:answer\s*(?:is|:)|\\boxed|=)\s*([^.\n]{1,200})", _re_b.IGNORECASE), 300),
        (_re_b.compile(r"([^\s.,;:!?]+)\s*(?:\.|$)", _re_b.DOTALL), 300),
    ],
    "aime": [
        (_re_b.compile(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}"), "full"),
        (_re_b.compile(r"answer\s+is\s*[:\s]*([0-9]{1,3})", _re_b.IGNORECASE), 300),
        (_re_b.compile(r"\b([0-9]{1,3})\b"), "full"),
    ],
    "gpqa": [
        (_re_b.compile(r"\\boxed\s*\{\s*([A-Da-d])\s*\}"), "full"),
        (_re_b.compile(r"answer\s+is\s*[:\s]*\(?\s*([A-Da-d])\s*\)?", _re_b.IGNORECASE), 300),
        (_re_b.compile(r"(?<![(\w])\b([A-D])\b(?![)\w])"), 200),
    ],
    "gsm8k": [
        # primary: boxed-number
        (_re_b.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}"), "full"),
        # fallback_1: "answer is N" or "= N" in last 300 chars
        (_re_b.compile(r"(?:answer\s*(?:is|:)|=)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", _re_b.IGNORECASE), 300),
        # fallback_2: last number in last 300 chars
        (_re_b.compile(r"(-?\d[\d,]*(?:\.\d+)?)"), 300),
    ],
    "humaneval": [
        # placeholder — humaneval bypasses regex tiers via extract_answer_3tier special case.
        # If you see "primary" for a humaneval problem, it means extract_code returned non-None.
    ],
}
_TIER_NAMES = ("primary", "fallback_1", "fallback_2")


def extract_answer_3tier(text, dataset, problem=None):
    """3-tier extraction returning (extracted_str_or_None, extraction_path).

    For humaneval, the 3-tier regex framework doesn't apply — code extraction
    needs the entry_point. We use Armin's extract_code from grader_utils.he_grader.
    Pass `problem` dict (with entry_point field) for humaneval.
    """
    text = text or ""
    if dataset == "humaneval":
        if problem is None:
            return None, "none"
        from grader_utils.he_grader import extract_code as _extract_code
        try:
            code = _extract_code(text, problem.get("entry_point", ""))
            if code and len(code.strip()) > 0:
                return code, "primary"
        except Exception:
            pass
        return None, "none"

    if dataset not in _TIER_SPECS:
        raise ValueError(f"Unknown dataset for parser: {dataset!r}")
    for tier_idx, (pat, scope) in enumerate(_TIER_SPECS[dataset]):
        hay = text if scope == "full" else text[-int(scope):]
        matches = pat.findall(hay)
        if matches:
            extracted = matches[-1].strip() if isinstance(matches[-1], str) else matches[-1]
            if dataset == "gpqa":
                extracted = extracted.upper()
            return extracted, _TIER_NAMES[tier_idx]
    return None, "none"


def grade_answer_3tier(extracted, gold, dataset, problem=None):
    """Grade the extracted string against gold per-dataset.

    For humaneval, pass the `problem` dict so we can call check_correctness.
    """
    if extracted is None:
        return False
    if dataset == "math":
        return bool(grade_answer(extracted, gold))
    if dataset == "aime":
        try:
            return int(extracted) == int(gold)
        except (TypeError, ValueError):
            return False
    if dataset == "gpqa":
        return str(extracted).strip().upper() == str(gold).strip().upper()
    if dataset == "gsm8k":
        try:
            # Strip commas, compare as floats (handle integer and decimal answers)
            ext_clean = str(extracted).replace(",", "").strip()
            gold_clean = str(gold).replace(",", "").strip()
            return abs(float(ext_clean) - float(gold_clean)) < 1e-6
        except (TypeError, ValueError):
            return False
    if dataset == "humaneval":
        if problem is None:
            return False
        try:
            from grader_utils.he_check import check_correctness
            result = check_correctness(problem, extracted, timeout=3.0)
            return bool(result.get("passed", False))
        except Exception:
            return False
    raise ValueError(f"Unknown dataset for grader: {dataset!r}")


def classify_outcome(extracted, gold, dataset, response_length_tokens, max_new_tokens, problem=None):
    """Return one of: correct | wrong_with_answer | no_answer | truncated.
    'truncated' takes precedence over 'no_answer'."""
    hit_cap = response_length_tokens >= max_new_tokens
    if extracted is None and hit_cap:
        return "truncated", hit_cap
    if extracted is None:
        return "no_answer", hit_cap
    if grade_answer_3tier(extracted, gold, dataset, problem=problem):
        return "correct", hit_cap
    return "wrong_with_answer", hit_cap


def _load_metadata(dataset, data_dir="data"):
    """Per-dataset side-table idx -> metadata. Mirrors v3."""
    if dataset == "math":
        meta_path = os.path.join(data_dir, "MATH500_with_level.jsonl")
        main_path = os.path.join(data_dir, "MATH500.json")
        if not (os.path.exists(meta_path) and os.path.exists(main_path)):
            return {}
        with open(main_path) as f:
            main = json.load(f)
        with open(meta_path) as f:
            hf_records = [json.loads(l) for l in f if l.strip()]
        id2hf = {r["unique_id"]: r for r in hf_records}
        out = {}
        for i, p in enumerate(main):
            uid = p.get("id") or p.get("unique_id")
            h = id2hf.get(uid)
            if h:
                lvl_raw = h.get("level")
                lvl_int = None
                if isinstance(lvl_raw, int):
                    lvl_int = lvl_raw
                elif isinstance(lvl_raw, str):
                    m = _re_b.search(r"\d+", lvl_raw)
                    if m: lvl_int = int(m.group())
                out[i] = {"level": lvl_int, "subject": h.get("subject", ""), "unique_id": uid}
        return out
    elif dataset == "aime":
        path = os.path.join(data_dir, "aime_combined.jsonl")
        if not os.path.exists(path): return {}
        out = {}
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                r = json.loads(line)
                out[i] = {"year": r.get("year"), "exam": r.get("exam"),
                          "problem_number": r.get("problem_number"), "url": r.get("url")}
        return out
    elif dataset == "gpqa":
        for path in [os.path.join(data_dir, "gpqa_diamond.json"), os.path.join(data_dir, "GPQA.json")]:
            if os.path.exists(path):
                with open(path) as f:
                    items = json.load(f)
                return {i: {"high_level_domain": r.get("high_level_domain", ""),
                            "subdomain": r.get("subdomain", ""), "id": r.get("id", "")}
                        for i, r in enumerate(items)}
        return {}
    elif dataset == "gsm8k":
        path = os.path.join(data_dir, "gsm8k.jsonl")
        if not os.path.exists(path): return {}
        out = {}
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                r = json.loads(line)
                out[i] = {"id": r.get("id", f"gsm8k_{i}"),
                          "source": r.get("source", "")}
        return out
    elif dataset == "humaneval":
        path = os.path.join(data_dir, "HumanEval.jsonl")
        if not os.path.exists(path): return {}
        out = {}
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                r = json.loads(line)
                out[i] = {"task_id": r.get("task_id", ""),
                          "entry_point": r.get("entry_point", ""),
                          "id": r.get("id", r.get("task_id", f"he_{i}"))}
        return out
    return {}


@torch.no_grad()
def per_token_logp_under_base(model, full_token_ids, prompt_len: int):
    """Run a single forward pass on the full sequence under the BASE model
    (no tempering) and return per-token log-prob of EACH generated token.

    Returns (per_token_logp_list, sum_logp_float). Length of list = generated tokens.
    Useful for diagnosing where the chosen sequence was uncertain under the base.
    """
    if full_token_ids.dim() == 1:
        full_token_ids = full_token_ids.unsqueeze(0)
    full_token_ids = full_token_ids.to(model.device)
    out = model(full_token_ids)
    logits = out.logits[:, :-1, :]                     # predict token t+1 from logits at t
    log_probs = F.log_softmax(logits, dim=-1)
    targets = full_token_ids[:, 1:]                    # the actual tokens
    per_tok = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # Generated tokens only — everything from prompt_len onward in the target sequence
    # full_token_ids[:, 1:] aligns to logits[:, :-1], so index prompt_len-1 onward gives gen tokens
    gen_per_tok = per_tok[0, prompt_len - 1:].cpu().tolist()
    return gen_per_tok, float(sum(gen_per_tok))


@torch.no_grad()
def teacher_forced_gold_logp(model, tokenizer, input_text: str, gold_text: str, prompt_len: int):
    """Compute cumulative log-prob the BASE model assigns to (prompt + gold_text).

    Returns (per_token_gold_logp_list, sum_logp_float). Length = number of gold tokens.
    """
    full_text = input_text + gold_text
    ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)
    out = model(ids)
    logits = out.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    per_tok = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    gold_per_tok = per_tok[0, prompt_len - 1:].cpu().tolist()
    return gold_per_tok, float(sum(gold_per_tok))


def set_all_seeds(seed: int) -> None:
    """Seed the global RNGs (Python random, NumPy, torch, CUDA). NOTE: the SMC core's
    internal token-sampling/resampling generators are seeded SEPARATELY via the explicit
    `seed=` argument passed to smc_power_sample_memopt in the main loop — this function
    does NOT reach them. Both are kept in sync by deriving from the same per-problem seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(
        description="Vanilla Power-SMC baseline (v3-parity) — Armin's official SMC core + per-dataset 3-tier parser, 4-way outcome, per_question.jsonl streaming log, assertions, full config dump."
    )
    # Data + selection
    p.add_argument("--dataset", type=str, default="math",
                   choices=["math", "aime", "gpqa", "gsm8k", "humaneval"],
                   help="Which benchmark. Drives loader + 3-tier parser regex set + grader.")
    p.add_argument("--data", type=str, default=None,
                   help="Optional explicit path to dataset file. If None, auto-picked by --dataset.")
    p.add_argument("--problem_idx_list", type=str, required=True,
                   help="Comma-separated 0-indexed problem indices")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    # Model + decoder
    p.add_argument("--model", type=str, default="qwen_math",
                   help="One of Armin's model keys; default qwen_math = Qwen2.5-Math-7B")
    p.add_argument("--device", type=str, default="cuda:0")
    # SMC config — defaulted to the v3-parity recipe (N=32, max=4096)
    p.add_argument("--n_particles", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="v3-parity: default 4096 (was 3072). v3 asserts >= 4096.")
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--ess_threshold", type=float, default=0.5)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--stop_on_boxed", action="store_true", default=True)
    p.add_argument("--boxed_check_window_tokens", type=int, default=256)
    # Expose alpha-ramp length so baseline can be matched to island runs.
    # SMCSamplingConfig's own default is 400; island runs use 200. Setting
    # this explicitly removes the baseline-vs-island ramp asymmetry.
    p.add_argument("--ramp_T", type=int, default=None,
                   help="Override alpha_ramp_tokens. None = use SMCSamplingConfig default (400). "
                        "Set to 200 to match island v3/v4.")
    # ---- Resampling-scheme experiment flags (P3) ----
    p.add_argument("--resampler", type=str, default="systematic",
                   choices=["systematic", "chopthin", "chopthin_reset"],
                   help="Resampling scheme. 'chopthin' keeps unequal weights (Arm C); "
                        "'chopthin_reset' uses chopthin offspring but resets weights (Arm C').")
    p.add_argument("--trigger", type=str, default="ess",
                   choices=["ess", "robust_ratio", "always"],
                   help="When to resample. ess: ESS<kappa*N (baseline). robust_ratio: "
                        "(q_hi-q_lo) of log-weights > threshold (Arm B). always: every block (Arm D2).")
    p.add_argument("--eta", type=float, default=5.8284271247461903,
                   help="chopthin weight-ratio bound (>=4). Default 3+sqrt(8) ~ 0.5N ESS floor.")
    p.add_argument("--robust_ratio_threshold", type=float, default=2.0,
                   help="log-space (q_hi - q_lo) gap that triggers robust_ratio resampling. CALIBRATE (P4).")
    p.add_argument("--robust_ratio_q", type=str, default="0.9,0.1",
                   help="hi,lo quantiles for the robust_ratio trigger (default 0.9,0.1).")
    p.add_argument("--adaptive_temperature", action="store_true", default=False,
                   help="Use per-step optimal proposal tau_t = 1/alpha_t (the adaptive-tau arm).")
    args = p.parse_args()

    # ---- v3-parity: startup assertions ----
    assert args.n_particles > 0, "n_particles must be positive"
    assert args.max_new_tokens >= 1024, (
        f"max_new_tokens must be >= 1024 (got {args.max_new_tokens}). "
        "Override with --max_new_tokens N if you genuinely want smaller."
    )
    assert args.dataset in {"math", "aime", "gpqa", "gsm8k", "humaneval"}, f"unknown dataset {args.dataset!r}"

    # ---- Output dir + full config dump ----
    os.makedirs(os.path.join(args.out_dir, "per_run"), exist_ok=True)

    # Try git commit (best effort)
    git_commit = None
    try:
        git_commit = _sp.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=_sp.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        git_commit = None

    config_payload = {
        "model": "Qwen/Qwen2.5-Math-7B",
        "dataset": args.dataset,
        "N_total": args.n_particles,
        "n_islands": 1,            # baseline is flat
        "n_per_island": args.n_particles,
        "inter_island_mode": "n/a (flat baseline)",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "alpha": 1.0 / args.temperature,
        "adaptive_temperature": args.adaptive_temperature,
        "resampler": args.resampler,
        "trigger": args.trigger,
        "eta": args.eta,
        "robust_ratio_threshold": args.robust_ratio_threshold,
        "robust_ratio_quantiles": args.robust_ratio_q,
        "alpha_ramp_tokens": (args.ramp_T if args.ramp_T is not None
                              else "n/a (uses SMCSamplingConfig default)"),
        "ess_threshold": args.ess_threshold,
        "block_size": args.block_size,
        "stop_on_boxed": args.stop_on_boxed,
        "boxed_check_window_tokens": args.boxed_check_window_tokens,
        "seed": args.seed,
        "parser_rules_enabled": True,
        "__patch_version__": "baseline_v3_parity",
        "__patches_applied__": [
            "3-tier per-dataset parser (P-B from v3)",
            "4-way outcome field (P-C from v3)",
            "per_question.jsonl streaming log (P-C from v3)",
            "startup assertions (P-E from v3)",
            "full resolved-settings config.json (P-D from v3)",
            "per-problem metadata stamping (level/year/subject)",
        ],
        "start_time_iso": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit,
        "_cli_args": vars(args),
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config_payload, f, indent=2, default=str)

    # Print the resolved settings as one visible block (v3 P-E parity)
    print("=" * 70)
    print("BASELINE (v3-parity) RESOLVED SETTINGS")
    print("=" * 70)
    for k, v in config_payload.items():
        if k.startswith("_") or k == "__patches_applied__":
            continue
        print(f"  {k:>30}: {v}")
    print("=" * 70)

    # ---- Model + data ----
    set_all_seeds(args.seed)

    model_map = {
        "qwen": "Qwen/Qwen2.5-7B",
        "qwen_math": "Qwen/Qwen2.5-Math-7B",
        "phi": "microsoft/Phi-3.5-mini-instruct",
        "qwen3": "Qwen/Qwen3-4B",
    }
    model_id = model_map[args.model]
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    hf_model.eval()
    # ---- Free-win provenance fields (added 2026-05-26) ----
    n_model_params = int(sum(p.numel() for p in hf_model.parameters()))
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info["gpu_total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            gpu_info["gpu_cc"] = f"{props.major}.{props.minor}"
        except Exception:
            pass
    import transformers as _transformers
    env_info = {
        "n_model_params": n_model_params,
        "torch_version": torch.__version__,
        "transformers_version": _transformers.__version__,
        "cuda_version": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodename": os.environ.get("SLURMD_NODENAME", os.environ.get("HOSTNAME", "")),
        **gpu_info,
    }
    print(f"--- env / hw info ---")
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    print(f"---")
    cfg_path = os.path.join(args.out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            existing_cfg = json.load(f)
        existing_cfg["env"] = env_info
        with open(cfg_path, "w") as f:
            json.dump(existing_cfg, f, indent=2, default=str)
    run_t0 = time.time()


    # ---- Dataset loading (dataset-aware) ----
    if args.data is None:
        default_data_paths = {
            "math": "data/MATH500.json",
            "aime": "data/aime_combined.jsonl",
            "gpqa": "data/gpqa_diamond.json",  # will fallback to GPQA.json if missing
            "gsm8k": "data/gsm8k.jsonl",
            "humaneval": "data/HumanEval.jsonl",
        }
        data_path = default_data_paths[args.dataset]
        if args.dataset == "gpqa" and not os.path.exists(data_path):
            data_path = "data/GPQA.json"
    else:
        data_path = args.data

    if args.dataset in ("aime", "gsm8k", "humaneval"):
        with open(data_path) as f:
            dataset = [json.loads(l) for l in f if l.strip()]
    else:
        with open(data_path) as f:
            dataset = json.load(f)

    # ---- v3-parity: per-problem metadata side-table ----
    metadata_by_idx = _load_metadata(args.dataset, data_dir="data")

    problem_idx_list = [int(x) for x in args.problem_idx_list.split(",")]
    print(f"Running {len(problem_idx_list)} problems on seed {args.seed} (dataset={args.dataset}).")
    n_with_meta = sum(1 for i in problem_idx_list if i in metadata_by_idx)
    print(f"Metadata coverage: {n_with_meta}/{len(problem_idx_list)} problems have metadata stamped.")

    # ---- SMC config (Armin's defaults) ----
    alpha = 1.0 / args.temperature
    cfg_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        alpha=alpha,
        n_particles=args.n_particles,
        ess_threshold=args.ess_threshold,
        temperature=args.temperature,
        block_size=args.block_size,
        stop_on_boxed=args.stop_on_boxed,
        boxed_check_window_tokens=args.boxed_check_window_tokens,
        adaptive_temperature=args.adaptive_temperature,
        resampler=args.resampler,
        trigger=args.trigger,
        eta=args.eta,
        robust_ratio_threshold=args.robust_ratio_threshold,
        robust_ratio_quantiles=tuple(float(x) for x in args.robust_ratio_q.split(",")),
    )
    if args.ramp_T is not None:
        cfg_kwargs["alpha_ramp_tokens"] = args.ramp_T
    cfg = SMCSamplingConfig(**cfg_kwargs)

    # ---- Main loop ----
    summaries = []
    # v3 P-C parity: open per_question.jsonl, line-buffered
    jsonl_path = os.path.join(args.out_dir, "per_question.jsonl")
    jsonl_fp = open(jsonl_path, "w", buffering=1)

    for idx_pos, prob_idx in enumerate(problem_idx_list):
        # Per-problem seed = base + offset so seeds are deterministic per (seed, prob_idx)
        prob_seed = args.seed * 1_000_000 + prob_idx
        set_all_seeds(prob_seed)

        data = dataset[prob_idx]
        # Dataset-aware prompt + gold extraction
        if args.dataset == "math":
            question = data["prompt"]
            gold = data["answer"]
            problem_id = data.get("id") or f"math_{prob_idx}"
        elif args.dataset == "aime":
            question = data["problem"]
            gold = str(data["answer"])
            problem_id = data.get("id") or f"aime_{prob_idx}"
        elif args.dataset == "gpqa":
            question = data["prompt"]
            gold = data["answer"]
            problem_id = data.get("id") or f"gpqa_{prob_idx}"
        elif args.dataset == "gsm8k":
            question = data["problem"]
            gold = str(data["answer"])
            problem_id = data.get("id") or f"gsm8k_{prob_idx}"
        elif args.dataset == "humaneval":
            question = data["prompt"]            # function signature + docstring
            gold = data.get("canonical_solution", "")  # gold code, used for reference only
            problem_id = data.get("task_id") or data.get("id") or f"he_{prob_idx}"
        else:
            raise ValueError(args.dataset)

        if args.dataset == "gpqa":
            # GPQA needs a multi-choice instruction, not the math template
            input_text = (
                "Answer the following multiple-choice question by selecting the correct option.\n\n"
                + question
                + "\nThink step by step, then put ONLY the letter (A, B, C, or D) of your chosen "
                  "answer inside \\boxed{}. Example: \\boxed{B}"
            )
        elif args.dataset == "humaneval":
            # HumanEval: the prompt IS the function signature + docstring. Append a code-completion ask.
            input_text = (
                question
                + "\n\n# Complete the function above. Output ONLY the function body in a python code block (```python ... ```)."
            )
        else:
            # math, aime, gsm8k all use Armin's math-style prompt
            input_text = format_prompt(question, args.model, tokenizer, cot=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(hf_model.device)
        prompt_len = input_ids.size(1)

        # Memory tracking (added 2026-05-26)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.time()
        smc_out = smc_power_sample_memopt(hf_model, tokenizer, input_ids, cfg, seed=prob_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wallclock = time.time() - t0

        peak_memory_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                          if torch.cuda.is_available() else None)

        chosen_ids = smc_out["chosen_sequence"][prompt_len:].cpu()
        completion = tokenizer.decode(chosen_ids, skip_special_tokens=True)

        # v3 P-B parity: 3-tier extractor + per-dataset grader (problem dict passed for humaneval)
        extracted, extraction_path = extract_answer_3tier(completion, args.dataset, problem=data)
        response_length_tokens = int(len(chosen_ids))
        outcome, hit_token_cap = classify_outcome(
            extracted, gold, args.dataset, response_length_tokens, args.max_new_tokens, problem=data,
        )
        correct = (outcome == "correct")
        # Backward-compat: parsed = extracted if 3-tier matched, else Armin's fallback
        parsed = extracted if extracted is not None else parse_answer(completion)

        stats = smc_out["stats"]

        # ---- Extra probability diagnostics (added 2026-05-25) ----
        # 1) Per-token log-prob of the winning particle's sequence under the BASE model.
        winner_per_token_logp, winner_logp_sum = per_token_logp_under_base(
            hf_model, smc_out["chosen_sequence"], prompt_len
        )
        # 2) Teacher-forced log-prob of the gold answer under the BASE model.
        gold_per_token_logp, gold_logp_sum = teacher_forced_gold_logp(
            hf_model, tokenizer, input_text, gold, prompt_len
        )
        # 3) SMC re-weighting contribution per particle (free derived quantity).
        #    log_w_final - cum_logp_final = the amount of weight SMC added beyond
        #    what the base model already assigned to each particle's path.
        log_w_final_list = smc_out["log_w"].detach().cpu().tolist()
        cum_logp_final_list = list(stats.get("cum_logp_final", []))
        if len(log_w_final_list) == len(cum_logp_final_list):
            smc_lift_per_particle = [w - p for w, p in zip(log_w_final_list, cum_logp_final_list)]
        else:
            smc_lift_per_particle = []

        # ---- Save everything ----
        meta = metadata_by_idx.get(prob_idx, {})
        summary = {
            "problem_idx": int(prob_idx),
            "problem_id": problem_id,
            "seed": int(args.seed),
            "prob_seed": int(prob_seed),
            "dataset": args.dataset,
            "gold_answer": gold,
            "parsed_answer": parsed,
            "extracted_answer": extracted,
            "extraction_path": extraction_path,
            "outcome": outcome,                # v3 P-C parity: 4-way
            "correct": int(correct),           # backward-compat
            "max_tokens_used": response_length_tokens,
            "hit_token_cap": bool(hit_token_cap),
            "wall_clock_seconds": wallclock,
            "response_length_tokens": response_length_tokens,
            "peak_memory_gb": peak_memory_gb,
            "flops_estimate": (
                2 * args.n_particles * response_length_tokens * n_model_params
            ) if response_length_tokens and n_model_params else None,
            "completion_first_400": completion[:400],
            "completion": completion,
            "metadata": meta,
            # Per-particle data
            "log_w_final": log_w_final_list,
            "chosen_idx": int(smc_out["chosen_idx"]),
            "all_particle_token_ids": [
                seq.detach().cpu().tolist() if hasattr(seq, "detach") else list(seq)
                for seq in smc_out["sequences"]
            ],
            # ---- Probability diagnostics ----
            "winner_per_token_logp": winner_per_token_logp,
            "winner_logp_sum": winner_logp_sum,
            "gold_per_token_logp": gold_per_token_logp,
            "gold_logp_sum": gold_logp_sum,
            "smc_lift_per_particle": smc_lift_per_particle,
            # SMC stats (lists / scalars only — no tensors)
            "n_resamples_total": int(stats.get("resample_count", 0)),
            "ess_history": list(stats.get("ess_history", [])),
            "mean_logw_history": list(stats.get("mean_logw_history", [])),
            "max_logw_history": list(stats.get("max_logw_history", [])),
            "unique_ancestors_history": list(stats.get("unique_ancestors_history", [])),
            "cum_logp_final": list(stats.get("cum_logp_final", [])),
            # Provenance
            "smc_config": {
                "max_new_tokens": cfg.max_new_tokens,
                "alpha": cfg.alpha,
                "n_particles": cfg.n_particles,
                "ess_threshold": cfg.ess_threshold,
                "temperature": cfg.temperature,
                "block_size": cfg.block_size,
                "stop_on_boxed": cfg.stop_on_boxed,
                "boxed_check_window_tokens": cfg.boxed_check_window_tokens,
            },
            "source": "armin_official_smc_power_sample_memopt",
        }
        summaries.append(summary)

        out_path = os.path.join(args.out_dir, "per_run", f"p{prob_idx:03d}_s{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        # ---- v3 P-C parity: append per_question.jsonl line ----
        jsonl_line = {
            "problem_idx": int(prob_idx),
            "problem_id": problem_id,
            "dataset": args.dataset,
            "ground_truth": gold,
            "final_answer": extracted,
            "outcome": outcome,
            "extraction_path": extraction_path,
            "max_tokens_used": response_length_tokens,
            "hit_token_cap": bool(hit_token_cap),
            "final_log_w_chosen": float(log_w_final_list[smc_out["chosen_idx"]])
                                  if log_w_final_list and 0 <= smc_out["chosen_idx"] < len(log_w_final_list) else None,
            "wall_time_seconds": float(wallclock),
            "peak_memory_gb": peak_memory_gb,
            "flops_estimate": (
                2 * args.n_particles * response_length_tokens * n_model_params
            ) if response_length_tokens and n_model_params else None,
            **meta,
        }
        jsonl_fp.write(json.dumps(jsonl_line, default=str) + "\n")

        print(f"[{idx_pos + 1}/{len(problem_idx_list)}] p{prob_idx} "
              f"outcome={outcome} t={wallclock:.1f}s resamples={summary['n_resamples_total']} "
              f"len={summary['response_length_tokens']} path={extraction_path} "
              f"baseline_v3parity")

    jsonl_fp.close()

    # Aggregate (v3 P-C parity: outcome distribution)
    n = len(summaries)
    c = sum(s["correct"] for s in summaries)
    outcome_counts = collections.Counter(s["outcome"] for s in summaries)
    total_wallclock_seconds = time.time() - run_t0
    throughput = n / max(total_wallclock_seconds, 1)
    print(f"\n=== Done (baseline v3-parity). Accuracy: {c}/{n} = {c / max(n, 1):.4f} ===")
    print(f"Outcome counts: {dict(outcome_counts)}")
    print(f"Total wall-clock: {total_wallclock_seconds:.1f}s   throughput: {throughput:.3f} problems/sec")
    with open(os.path.join(args.out_dir, "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
