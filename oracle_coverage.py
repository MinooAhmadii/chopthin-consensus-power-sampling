#!/usr/bin/env python3
"""Oracle coverage of saved runs: the fraction of problems where at least one of the N final
particles is correct (paper Table 3). Also prints each run's selected-answer accuracy.

    python oracle_coverage.py --dataset math runs/math/systematic runs/math/chopthin

Each run directory must come from run.py. The tokenizer named in its config.json decodes the
particles, and HumanEval programs are extracted with the protocol recorded there
(he_protocols.py); --tokenizer and --protocol override both for older runs. With two runs,
the coverage difference (second minus first) is printed as well.
"""
import argparse
import glob
import json
import os

from transformers import AutoTokenizer

from grader_utils.answers import DATASETS, extract_answer, is_correct
from he_protocols import PROTOCOLS, get_protocol


def load_humaneval(path):
    with open(path) as f:
        return {r["task_id"]: r for r in (json.loads(l) for l in f if l.strip())}


def analyze(run_dir, dataset, humaneval=None, protocol=None, tokenizer=None):
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    tok = AutoTokenizer.from_pretrained(tokenizer or cfg["model"])
    proto = get_protocol(protocol or cfg.get("he_pipeline") or "legacy") if dataset == "humaneval" else None
    n = selected = 0
    covered = set()
    for path in sorted(glob.glob(os.path.join(run_dir, "per_run", "*.json"))):
        with open(path) as f:
            d = json.load(f)
        gold, seqs, ci = d["gold_answer"], d["all_particle_token_ids"], d["chosen_idx"]
        prompt_len = max(0, len(seqs[ci]) - d.get("response_length_tokens", 0))
        problem = humaneval.get(d["problem_id"]) if dataset == "humaneval" else None
        n += 1
        selected += int(bool(d["correct"]))
        for seq in seqs:
            text = tok.decode(seq[prompt_len:], skip_special_tokens=True)
            if proto:
                answer, _ = proto.extract(text, problem)
                grade_problem = proto.grade_problem(problem)
            else:
                answer, _ = extract_answer(text, dataset, problem=problem)
                grade_problem = problem
            if is_correct(answer, gold, dataset, problem=grade_problem):
                covered.add(d["problem_idx"])
                break
    return dict(n=n, selected=selected, covered=covered)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data", default="data/HumanEval.jsonl", help="HumanEval problems (needed to grade code)")
    ap.add_argument("runs", nargs="+", help="run directories from run.py")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default=None,
                    help="HumanEval only: override the run's he_pipeline")
    ap.add_argument("--tokenizer", default=None, help="override the tokenizer named in config.json")
    args = ap.parse_args()

    humaneval = load_humaneval(args.data) if args.dataset == "humaneval" else None
    results = []
    for run_dir in args.runs:
        r = analyze(run_dir, args.dataset, humaneval, args.protocol, args.tokenizer)
        results.append(r)
        print(f"{run_dir}: n={r['n']}  selected={100 * r['selected'] / max(r['n'], 1):.1f}%  "
              f"oracle={100 * len(r['covered']) / max(r['n'], 1):.1f}%")
    if len(results) == 2:
        a, b = results
        delta = 100 * (len(b["covered"]) - len(a["covered"])) / max(a["n"], 1)
        print(f"oracle difference (second - first): {delta:+.1f} points  "
              f"(only second: {len(b['covered'] - a['covered'])}, only first: {len(a['covered'] - b['covered'])})")


if __name__ == "__main__":
    main()
