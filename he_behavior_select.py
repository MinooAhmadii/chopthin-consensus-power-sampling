#!/usr/bin/env python3
"""
Test-free HumanEval selection, step 2 of 2: behavior-majority voting.

This is the CODE analogue of semantic-majority selection. For each problem it:
  1. decodes the N candidate programs (from the saved per_run/p*.json particles) and
     deduplicates identical ones into distinct programs;
  2. runs every distinct program on the self-generated inputs from he_gen_inputs.py
     (stage1_inputs_<key>.json) inside the sandboxed executor (grader_utils/he_execute);
  3. clusters programs by identical output behavior and returns the majority cluster
     (ties broken by pooled particle weight);
  4. scores that representative on HumanEval's held-out tests -- used ONLY to measure
     accuracy, never by the selector.

No ground-truth tests enter selection, so it engages on all 164 problems and is fully
deployable on unseen problems. For each (model, arm) it prints:
    smc floor | BEHAVIOR-MAJORITY | oracle ceiling,  plus engagement counts.

Run from the repo root (so `run_baseline` and `grader_utils` import cleanly):
    python3 he_behavior_select.py --runs-dir runs --inputs-dir .
"""
import os
import json
import glob
import math
import argparse
import multiprocessing
from collections import defaultdict

from transformers import AutoTokenizer
from run_baseline import extract_answer_3tier
from grader_utils.he_execute import (check_correctness, reliability_guard, swallow_io,
                                      time_limit, create_tempdir)

# (display name, tokenizer, run-folder under --runs-dir, inputs key from he_gen_inputs.py).
# Adjust the run-folder names to your own runs/<folder>/{a2_A,a2_C}/per_run/*.json layout.
# (Qwen2.5-7B shares a byte-identical tokenizer with Qwen2.5-Math-7B, hence the reuse.)
CONFIGS = [
    ("Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct", "humaneval_n32_s42",           "qwen_math"),
    ("Qwen2.5-7B",      "Qwen/Qwen2.5-Math-7B-Instruct", "humaneval_qwen2.5-7b_n32_s42", "qwen"),
    ("Qwen3-4B",        "Qwen/Qwen3-4B",                 "humaneval_qwen3-4b_n32_s42",   "qwen3"),
]
ARMS = [("a2_A", "systematic"), ("a2_C", "chopthin")]

_TOKS = {}


def get_tok(name):
    if name not in _TOKS:
        _TOKS[name] = AutoTokenizer.from_pretrained(name)
    return _TOKS[name]


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def _behavior_worker(prompt, completion, exprs, result):
    """Run `completion` on each input expr in a fresh sandbox; append the tuple of repr'd outputs."""
    with create_tempdir():
        import os
        import shutil
        rmtree, rmdir, chdir = shutil.rmtree, os.rmdir, os.chdir
        reliability_guard()
        driver = ("\n__OUTS=[]\nfor __e in " + repr(exprs) + ":\n"
                  "    try:\n        __OUTS.append(repr(eval(__e)))\n"
                  "    except Exception as __x:\n        __OUTS.append('ERR:'+type(__x).__name__)\n")
        try:
            g = {}
            with swallow_io():
                with time_limit(4.0):
                    exec(prompt + completion + driver, g)
            result.append(list(g.get("__OUTS", [])))
        except BaseException:
            result.append(None)
        shutil.rmtree, os.rmdir, os.chdir = rmtree, rmdir, chdir


_MGR = None


def behavior(prob, completion, exprs):
    """Behavior signature = tuple of `completion`'s outputs on `exprs`, or None if it won't run."""
    global _MGR
    if completion is None or not exprs:
        return None
    if _MGR is None:
        _MGR = multiprocessing.Manager()
    res = _MGR.list()
    p = multiprocessing.Process(target=_behavior_worker,
                                args=(prob["prompt"], completion, exprs, res))
    p.start()
    p.join(timeout=6.0)
    if p.is_alive():
        p.kill()
        p.join()
    if not res or res[0] is None:
        return None
    return tuple(res[0])


def passes_hidden(prob, completion):
    """Held-out HumanEval unit tests -- used ONLY to score accuracy, never by the selector."""
    if completion is None:
        return False
    try:
        return check_correctness(dict(prob), completion, 6.0).get("passed", False)
    except Exception:
        return False


def analyze(model, tokname, run, key, runs_dir, inputs_dir, he):
    tok = get_tok(tokname)
    inputs = json.load(open(os.path.join(inputs_dir, f"stage1_inputs_{key}.json")))
    res = {}
    for arm, label in ARMS:
        st = dict(n=0, smc=0, behav=0, oracle=0, has_in=0, engaged=0)
        for f in sorted(glob.glob(f"{runs_dir}/{run}/{arm}/per_run/*.json")):
            d = json.load(open(f))
            st["n"] += 1
            seqs = d["all_particle_token_ids"]
            ci = d["chosen_idx"]
            w = softmax(d.get("log_w_final", [0.0] * len(seqs)))
            rlen = d.get("response_length_tokens", 0)
            pl = max(0, len(seqs[ci]) - rlen)                    # prompt-length offset
            pid = d.get("problem_id") or d.get("metadata", {}).get("task_id")
            prob = he.get(pid)
            codes = [extract_answer_3tier(tok.decode(s[pl:], skip_special_tokens=True),
                                          "humaneval", problem=prob)[0] for s in seqs]
            distinct = defaultdict(list)                          # program string -> particle indices
            for i, c in enumerate(codes):
                distinct[c].append(i)

            if d["correct"]:                                     # native smc-selected outcome
                st["smc"] += 1

            hidc = {}

            def hid(c):                                          # memoized held-out-test score
                if c is None:
                    return False
                if c in hidc:
                    return hidc[c]
                r = passes_hidden(prob, c)
                hidc[c] = r
                return r

            if any(hid(c) for c in distinct if c is not None):   # best-of-N ceiling
                st["oracle"] += 1

            exprs = inputs.get(pid, [])
            if exprs:
                st["has_in"] += 1
            behav_ok = d["correct"]                              # fallback = smc draw if unusable
            if exprs:
                clusters = defaultdict(list)                     # behavior signature -> programs
                for c in distinct:
                    if c is None:
                        continue
                    sg = behavior(prob, c, exprs)
                    if sg is not None:
                        clusters[sg].append(c)
                if clusters:
                    mass = lambda cs: sum(w[i] for c in cs for i in distinct[c])
                    best = max(clusters, key=lambda sg: (len(clusters[sg]), mass(clusters[sg])))
                    rep = max(clusters[best], key=lambda c: sum(w[i] for i in distinct[c]))
                    behav_ok = hid(rep)
                    st["engaged"] += 1
            if behav_ok:
                st["behav"] += 1

        res[label] = st
        n = st["n"] or 1
        print(f"[{model:16s} {label:10s}] n={st['n']}  smc={100 * st['smc'] / n:4.1f}  "
              f"BEHAV-MAJ={100 * st['behav'] / n:4.1f}  oracle={100 * st['oracle'] / n:4.1f}   "
              f"(has_inputs={st['has_in']}, engaged={st['engaged']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="Test-free behavior-majority HumanEval selection.")
    ap.add_argument("--runs-dir", default="runs",
                    help="root of runs/<folder>/{a2_A,a2_C}/per_run/*.json")
    ap.add_argument("--inputs-dir", default=".", help="dir holding stage1_inputs_<key>.json")
    ap.add_argument("--data", default="data/HumanEval.jsonl", help="path to HumanEval.jsonl")
    ap.add_argument("--out", default="behavior_majority_results.json")
    args = ap.parse_args()

    he = {}
    for line in open(args.data):
        r = json.loads(line)
        he[r["task_id"]] = r

    print("Test-free behavior-majority selection (model-generated inputs, all 164)\n")
    allres = {}
    for model, tokname, run, key in CONFIGS:
        allres[model] = analyze(model, tokname, run, key, args.runs_dir, args.inputs_dir, he)
    json.dump(allres, open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
