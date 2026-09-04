#!/usr/bin/env python3
"""
Test-free HumanEval selection, step 2 of 2: behavior-majority voting.

This is the CODE analogue of semantic-majority selection. For each problem it:
  1. decodes the N candidate programs (from the saved per_run/p*.json particles) with the
     run's HumanEval protocol (he_protocols.py: legacy / cot / stub) and deduplicates
     identical ones into distinct programs;
  2. runs every distinct program on the self-generated inputs from he_gen_inputs.py
     (data/he_inputs/stage1_inputs_<key>.json) inside the sandboxed executor
     (grader_utils/he_execute);
  3. clusters programs by identical output behavior and returns the majority cluster
     (ties broken by pooled particle weight);
  4. scores that representative on HumanEval's held-out tests -- used ONLY to measure
     accuracy, never by the selector.

No ground-truth tests enter selection, so it engages on all 164 problems and is fully
deployable on unseen problems. For each (model, arm) it prints:
    smc floor (weight draw) | BEHAVIOR-MAJORITY | oracle ceiling,  plus engagement counts.

The default CONFIGS reproduce the paper's HumanEval column (Table 1). Run from the repo
root so `run_baseline`, `he_protocols` and `grader_utils` import cleanly:
    python3 he_behavior_select.py --runs-dir runs
    python3 he_behavior_select.py --run humaneval_stub_qwen_math_n32_s42 --key qwen_math --protocol stub
"""
import os
import json
import glob
import math
import argparse
import multiprocessing
from collections import defaultdict
from typing import Optional, Sequence, Tuple

from transformers import AutoTokenizer
from grader_utils.he_execute import (check_correctness, reliability_guard, swallow_io,
                                      time_limit, create_tempdir)
from he_protocols import HEProtocol, PROTOCOLS, get_protocol

# (display name, tokenizer, run-folder under --runs-dir, inputs key, protocol) -- paper Table 1.
# Each run folder holds {a2_A,a2_C}/per_run/*.json (systematic / chopthin, same seed).
# Qwen2.5-7B shares a byte-identical tokenizer with Qwen2.5-Math-7B, hence the reuse.
CONFIGS = [
    ("Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct", "humaneval_stub_qwen_math_n32_s42", "qwen_math", "stub"),
    ("Qwen2.5-7B",      "Qwen/Qwen2.5-Math-7B-Instruct", "humaneval_cot_qwen2.5-7b_n32_s42", "qwen",      "cot"),
    ("Qwen3-4B",        "Qwen/Qwen3-4B",                 "humaneval_stub_qwen3-4b_n32_s42",  "qwen3",     "stub"),
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
    s = sum(es) or 1.0
    return [e / s for e in es]


def _behavior_worker(program, exprs, result):
    """Run `program` on each input expr in a fresh sandbox; append the list of repr'd outputs."""
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
                    exec(program + driver, g)
            result.append(list(g.get("__OUTS", [])))
        except BaseException:
            result.append(None)
        shutil.rmtree, os.rmdir, os.chdir = rmtree, rmdir, chdir


_MGR = None


def behavior(program: Optional[str], exprs: Sequence[str]) -> Optional[Tuple[str, ...]]:
    """Behavior signature = tuple of `program`'s outputs on `exprs`, or None if it won't run."""
    global _MGR
    if program is None or not exprs:
        return None
    if _MGR is None:
        _MGR = multiprocessing.Manager()
    res = _MGR.list()
    p = multiprocessing.Process(target=_behavior_worker, args=(program, exprs, res))
    p.start()
    p.join(timeout=6.0)
    if p.is_alive():
        p.kill()
        p.join()
    if not res or res[0] is None:
        return None
    return tuple(res[0])


def passes_hidden(grade_prob: dict, code: Optional[str]) -> bool:
    """Held-out HumanEval unit tests -- used ONLY to score accuracy, never by the selector."""
    if code is None:
        return False
    try:
        return check_correctness(dict(grade_prob), code, 6.0).get("passed", False)
    except Exception:
        return False


def analyze(model, tokname, run, key, protocol: HEProtocol, runs_dir, inputs_dir, he):
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
            if prob is None:
                continue
            grade_prob = protocol.grade_problem(prob)
            codes = [protocol.extract(tok.decode(s[pl:], skip_special_tokens=True), prob)
                     for s in seqs]
            distinct = defaultdict(list)                          # program string -> particle indices
            for i, c in enumerate(codes):
                distinct[c].append(i)

            if d["correct"]:                                     # native weight-draw outcome
                st["smc"] += 1

            hidc = {}

            def hid(c):                                          # memoized held-out-test score
                if c is None:
                    return False
                if c in hidc:
                    return hidc[c]
                r = passes_hidden(grade_prob, c)
                hidc[c] = r
                return r

            if any(hid(c) for c in distinct if c is not None):   # best-of-N ceiling
                st["oracle"] += 1

            exprs = inputs.get(pid, [])
            if exprs:
                st["has_in"] += 1
            behav_ok = d["correct"]                              # fallback = weight draw if unusable
            if exprs:
                clusters = defaultdict(list)                     # behavior signature -> programs
                for c in distinct:
                    if c is None:
                        continue
                    sg = behavior(protocol.program(prob, c), exprs)
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
        print(f"[{model:16s} {protocol.name:6s} {label:10s}] n={st['n']}  smc={100 * st['smc'] / n:4.1f}  "
              f"BEHAV-MAJ={100 * st['behav'] / n:4.1f}  oracle={100 * st['oracle'] / n:4.1f}   "
              f"(has_inputs={st['has_in']}, engaged={st['engaged']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="Test-free behavior-majority HumanEval selection.")
    ap.add_argument("--runs-dir", default="runs",
                    help="root of runs/<folder>/{a2_A,a2_C}/per_run/*.json")
    ap.add_argument("--inputs-dir", default="data/he_inputs",
                    help="dir holding stage1_inputs_<key>.json (from he_gen_inputs.py)")
    ap.add_argument("--data", default="data/HumanEval.jsonl", help="path to HumanEval.jsonl")
    ap.add_argument("--out", default="behavior_majority_results.json")
    single = ap.add_argument_group("single run (overrides the paper CONFIGS)")
    single.add_argument("--run", help="run folder under --runs-dir")
    single.add_argument("--key", choices=["qwen_math", "qwen", "qwen3"], help="inputs key / base model")
    single.add_argument("--protocol", choices=sorted(PROTOCOLS),
                        help="HumanEval protocol the run was generated with (--he_pipeline)")
    single.add_argument("--tokenizer", default=None, help="default: the CONFIGS tokenizer for --key")
    single.add_argument("--name", default=None, help="display name (default: --run)")
    args = ap.parse_args()

    if args.run or args.key or args.protocol:
        if not (args.run and args.key and args.protocol):
            ap.error("--run, --key and --protocol must be given together")
        tokname = args.tokenizer or next(t for _, t, _, k, _ in CONFIGS if k == args.key)
        configs = [(args.name or args.run, tokname, args.run, args.key, args.protocol)]
    else:
        configs = CONFIGS

    he = {}
    for line in open(args.data):
        r = json.loads(line)
        he[r["task_id"]] = r

    print("Test-free behavior-majority selection (model-generated inputs, all 164)\n")
    allres = {}
    for model, tokname, run, key, proto in configs:
        allres[model] = analyze(model, tokname, run, key, get_protocol(proto),
                                args.runs_dir, args.inputs_dir, he)
        allres[model]["protocol"] = proto
    json.dump(allres, open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
