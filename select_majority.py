#!/usr/bin/env python3
"""Semantic-majority selection (paper Section 4.3 and Appendix D) over saved runs.

For each problem the script decodes the N final particles of a run, parses each answer with
the benchmark's extractor, merges particles with identical token sequences into distinct
trajectories (one vote each, carrying their pooled weight), clusters the parsed answers by
grade-equivalence (greedy match to each cluster's representative, with a string-equality
shortcut), and returns the cluster with the most distinct trajectories; ties go to the larger
pooled weight. Particles whose answer does not parse do not vote; if nothing parses the
problem counts as wrong. The gold answer only scores the selected cluster, it never selects.

    python select_majority.py --dataset math runs/math/systematic runs/math/chopthin
    python select_majority.py --dataset aime --pool runs/aime_s42/chopthin runs/aime_s43/chopthin

Prints, per run (or per pooled group with --pool): weight-draw accuracy (as run), accuracy of
the single highest-weight particle, semantic-majority accuracy, oracle coverage, and the mean
number of distinct trajectories per problem. The paper's CCPS rows are the semantic-majority
accuracy of the Chopthin arm. HumanEval uses he_behavior_select.py instead.
"""
import argparse
import glob
import json
import math
import os

from transformers import AutoTokenizer

from grader_utils.answers import extract_answer, is_correct

DATASETS = ("math", "gsm8k", "aime", "gpqa")


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


def equivalent(a, b, dataset):
    """Grade-equivalence of two parsed answers (the same check that scores against the gold)."""
    if a == b:
        return True
    try:
        return bool(is_correct(a, b, dataset))
    except Exception:
        return False


def select(distinct, dataset):
    """Majority cluster over distinct trajectories, or None if no answer parsed."""
    clusters = []                                   # rep, correct, votes, mass
    for t in distinct:
        if not t["answer"]:
            continue
        for c in clusters:
            if equivalent(t["answer"], c["rep"], dataset):
                c["votes"] += t["votes"]
                c["mass"] += t["mass"]
                break
        else:
            clusters.append(dict(rep=t["answer"], correct=t["correct"], votes=t["votes"], mass=t["mass"]))
    if not clusters:
        return None
    return max(clusters, key=lambda c: (c["votes"], c["mass"]))


def analyze(run_dirs, dataset, tokenizer=None):
    with open(os.path.join(run_dirs[0], "config.json")) as f:
        tok = AutoTokenizer.from_pretrained(tokenizer or json.load(f)["model"])
    files = []
    for rd in run_dirs:
        files += sorted(glob.glob(os.path.join(rd, "per_run", "*.json")))
    st = dict(n=0, weight_draw=0, argmax=0, majority=0, oracle=0, distinct=0)
    for path in files:
        with open(path) as f:
            d = json.load(f)
        gold, seqs, ci = d["gold_answer"], d["all_particle_token_ids"], d["chosen_idx"]
        prompt_len = max(0, len(seqs[ci]) - d.get("response_length_tokens", 0))
        w = softmax(d["log_w_final"])
        particles = []
        for i, seq in enumerate(seqs):
            text = tok.decode(seq[prompt_len:], skip_special_tokens=True)
            answer, _ = extract_answer(text, dataset)
            particles.append(dict(answer=answer, correct=bool(is_correct(answer, gold, dataset)),
                                  w=w[i], key=tuple(seq)))
        st["n"] += 1
        st["weight_draw"] += int(bool(d["correct"]))
        st["argmax"] += int(max(particles, key=lambda p: p["w"])["correct"])
        st["oracle"] += int(any(p["correct"] for p in particles))
        by_seq = {}                                  # identical token sequences -> one trajectory
        for p in particles:
            t = by_seq.get(p["key"])
            if t is None:
                by_seq[p["key"]] = dict(answer=p["answer"], correct=p["correct"], mass=p["w"], votes=1)
            else:
                t["mass"] += p["w"]
        distinct = list(by_seq.values())
        st["distinct"] += len(distinct)
        best = select(distinct, dataset)
        st["majority"] += int(best["correct"]) if best else 0
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("runs", nargs="+", help="run directories from run.py")
    ap.add_argument("--pool", action="store_true",
                    help="treat all given runs as one cell (e.g. several seeds of the same arm)")
    ap.add_argument("--tokenizer", default=None, help="override the tokenizer named in config.json")
    ap.add_argument("--out", default=None, help="write the counts as JSON")
    args = ap.parse_args()

    groups = [args.runs] if args.pool else [[r] for r in args.runs]
    results = {}
    for group in groups:
        st = analyze(group, args.dataset, args.tokenizer)
        name = " + ".join(group)
        results[name] = st
        n = max(st["n"], 1)
        print(f"{name}: n={st['n']}  weight_draw={100 * st['weight_draw'] / n:.1f}%  "
              f"argmax={100 * st['argmax'] / n:.1f}%  majority={100 * st['majority'] / n:.1f}%  "
              f"oracle={100 * st['oracle'] / n:.1f}%  distinct/problem={st['distinct'] / n:.1f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
