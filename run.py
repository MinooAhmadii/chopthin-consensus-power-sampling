"""Run one benchmark with one resampler and save every particle.

    python run.py --dataset math --model qwen_math --resampler chopthin --seed 42 --out_dir runs/math/chopthin

Defaults are the paper's settings: N = 32, alpha = 2, ESS trigger 0.5, block 64, alpha ramp
over the first 100 tokens, up to 4096 new tokens, eta = 3 + sqrt(8).

Output, per problem: per_run/p<idx>_s<seed>.json with all N final sequences and their
weights, and one line in per_question.jsonl with the selected (weight-drawn) answer.
"""

import argparse
import collections
import datetime
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from grader_utils.answers import DATASETS, classify_outcome, extract_answer
from prompts import build_prompt
from smc import SMCConfig, run_smc

MODELS = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen3": "Qwen/Qwen3-4B",
}

DATA_FILES = {
    "math": "data/MATH500.json",
    "gsm8k": "data/gsm8k.jsonl",
    "aime": "data/aime_combined.jsonl",
    "gpqa": "data/gpqa_diamond.json",
    "humaneval": "data/HumanEval.jsonl",
}


def load_problems(path):
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def question_and_gold(item, dataset, idx):
    """Return (question text, gold answer, problem id) for one dataset item."""
    if dataset in ("math", "gpqa"):
        return item["prompt"], item["answer"], item.get("id") or f"{dataset}_{idx}"
    if dataset in ("aime", "gsm8k"):
        return item["problem"], str(item["answer"]), item.get("id") or f"{dataset}_{idx}"
    if dataset == "humaneval":
        # The prompt is the function signature + docstring; the gold is only kept for reference.
        return item["prompt"], item.get("canonical_solution", ""), item.get("task_id") or f"humaneval_{idx}"
    raise ValueError(dataset)


def parse_problems(spec, n):
    """'all', a comma list '0,1,2', or a range '0-99' (inclusive)."""
    if spec == "all":
        return list(range(n))
    if "-" in spec and "," not in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="math", choices=DATASETS)
    p.add_argument("--data", default=None, help="dataset file; default is the one in data/")
    p.add_argument("--model", default="qwen_math",
                   help="qwen_math | qwen | qwen3, or any Hugging Face model id")
    p.add_argument("--resampler", default="systematic", choices=["systematic", "chopthin", "chopthin_reset"])
    p.add_argument("--eta", type=float, default=5.8284271247461903, help="chopthin weight-ratio bound (>= 4)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--problems", default="all", help="'all', '0,1,2', or '0-99'")
    p.add_argument("--n_particles", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.5, help="proposal temperature; alpha = 1/temperature")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--ess_threshold", type=float, default=0.5)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--ramp_tokens", type=int, default=100, help="alpha ramps from 1 to its final value over this many tokens")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    assert args.n_particles > 0

    os.makedirs(os.path.join(args.out_dir, "per_run"), exist_ok=True)
    model_id = MODELS.get(args.model, args.model)

    cfg = SMCConfig(
        n_particles=args.n_particles,
        alpha=1.0 / args.temperature,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        alpha_ramp_tokens=args.ramp_tokens,
        block_size=args.block_size,
        ess_threshold=args.ess_threshold,
        resampler=args.resampler,
        eta=args.eta,
    )

    settings = {
        "model": model_id,
        "dataset": args.dataset,
        "resampler": args.resampler,
        "eta": args.eta,
        "seed": args.seed,
        "problems": args.problems,
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "git_commit": git_commit(),
        "start_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(settings, f, indent=2, default=str)
    for k, v in settings.items():
        print(f"  {k:>22}: {v}")

    set_all_seeds(args.seed)
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(args.device)
    model.eval()

    problems = load_problems(args.data or DATA_FILES[args.dataset])
    indices = parse_problems(args.problems, len(problems))
    print(f"Running {len(indices)} problems, seed {args.seed}, dataset {args.dataset}.")

    outcomes = []
    jsonl = open(os.path.join(args.out_dir, "per_question.jsonl"), "w", buffering=1)
    run_t0 = time.time()

    for pos, idx in enumerate(indices):
        # Per-problem seed, so every (seed, problem) pair is reproducible on its own.
        prob_seed = args.seed * 1_000_000 + idx
        set_all_seeds(prob_seed)

        item = problems[idx]
        question, gold, problem_id = question_and_gold(item, args.dataset, idx)
        input_text = build_prompt(question, args.dataset, args.model, tokenizer)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        prompt_len = input_ids.size(1)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        out = run_smc(model, tokenizer, input_ids, cfg, seed=prob_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wallclock = time.time() - t0
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None

        chosen_ids = out["chosen_sequence"][prompt_len:].cpu()
        completion = tokenizer.decode(chosen_ids, skip_special_tokens=True)
        extracted, extraction_path = extract_answer(completion, args.dataset, problem=item)
        response_length = int(len(chosen_ids))
        outcome, hit_cap = classify_outcome(
            extracted, gold, args.dataset, response_length, args.max_new_tokens, problem=item,
        )
        outcomes.append(outcome)
        stats = out["stats"]

        record = {
            "problem_idx": int(idx),
            "problem_id": problem_id,
            "seed": int(args.seed),
            "prob_seed": int(prob_seed),
            "dataset": args.dataset,
            "gold_answer": gold,
            "extracted_answer": extracted,
            "extraction_path": extraction_path,
            "outcome": outcome,
            "correct": int(outcome == "correct"),
            "response_length_tokens": response_length,
            "hit_token_cap": bool(hit_cap),
            "wall_clock_seconds": wallclock,
            "peak_memory_gb": peak_memory_gb,
            "completion": completion,
            # all N particles
            "chosen_idx": int(out["chosen_idx"]),
            "log_w_final": out["log_w"].detach().cpu().tolist(),
            "all_particle_token_ids": [seq.detach().cpu().tolist() for seq in out["sequences"]],
            # resampling statistics
            "n_resamples": int(stats["resample_count"]),
            "ess_history": list(stats["ess_history"]),
            "ess_after_history": list(stats["ess_after_history"]),
            "unique_ancestors_history": list(stats["unique_ancestors_history"]),
        }
        with open(os.path.join(args.out_dir, "per_run", f"p{idx:03d}_s{args.seed}.json"), "w") as f:
            json.dump(record, f, indent=2)

        jsonl.write(json.dumps({
            "problem_idx": int(idx),
            "problem_id": problem_id,
            "gold_answer": gold,
            "final_answer": extracted,
            "outcome": outcome,
            "response_length_tokens": response_length,
            "hit_token_cap": bool(hit_cap),
            "wall_clock_seconds": float(wallclock),
        }, default=str) + "\n")

        print(f"[{pos + 1}/{len(indices)}] p{idx} outcome={outcome} t={wallclock:.1f}s "
              f"resamples={record['n_resamples']} len={response_length}")

    jsonl.close()
    n = len(outcomes)
    c = sum(o == "correct" for o in outcomes)
    print(f"\nAccuracy: {c}/{n} = {c / max(n, 1):.4f}")
    print(f"Outcomes: {dict(collections.Counter(outcomes))}")
    print(f"Total wall-clock: {time.time() - run_t0:.1f}s")


if __name__ == "__main__":
    main()
