#!/usr/bin/env python3
"""
Stage 1 -- fully test-free behavior-majority on ALL 164 HumanEval problems.

Same clustering/majority as Stage 0, but the inputs come from the base model's own generated
test calls (stage1_inputs_<key>.json), not the doctests. So it needs no ground-truth tests and
engages on every problem (not just the 76 with visible doctests). Reports, per (model, arm):
smc floor | BEHAVIOR-MAJORITY (Stage 1) | oracle ceiling, with engagement.
"""
import json, os, glob, math, multiprocessing
from collections import defaultdict
from transformers import AutoTokenizer
from run_baseline import extract_answer_3tier
from grader_utils.he_execute import (check_correctness, reliability_guard, swallow_io,
                                      time_limit, create_tempdir)

BASE = "/data/projects/nullanet/experiments/minoo/choptin"; ROOT = BASE + "/runs"
HE = {}
for _l in open(BASE + "/data/HumanEval.jsonl"):
    _r = json.loads(_l); HE[_r["task_id"]] = _r

CONFIGS = [
    ("Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct", "humaneval_n32_s42",          "qwen_math"),
    ("Qwen2.5-7B",      "Qwen/Qwen2.5-Math-7B-Instruct", "humaneval_qwen2.5-7b_n32_s42","qwen"),
    ("Qwen3-4B",        "Qwen/Qwen3-4B",                 "humaneval_qwen3-4b_n32_s42",  "qwen3"),
]
_TOKS = {}
def get_tok(n):
    if n not in _TOKS: _TOKS[n] = AutoTokenizer.from_pretrained(n)
    return _TOKS[n]
def softmax(xs):
    m = max(xs); es = [math.exp(x - m) for x in xs]; s = sum(es); return [e / s for e in es]

def _behavior_worker(prompt, completion, exprs, result):
    with create_tempdir():
        import os, shutil
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
    global _MGR
    if completion is None or not exprs:
        return None
    if _MGR is None:
        _MGR = multiprocessing.Manager()
    res = _MGR.list()
    p = multiprocessing.Process(target=_behavior_worker, args=(prob["prompt"], completion, exprs, res))
    p.start(); p.join(timeout=6.0)
    if p.is_alive():
        p.kill(); p.join()
    if not res or res[0] is None:
        return None
    return tuple(res[0])

def passes_hidden(prob, completion):
    if completion is None:
        return False
    try:
        return check_correctness(dict(prob), completion, 6.0).get("passed", False)
    except Exception:
        return False

def analyze(model, tokname, run, key):
    tok = get_tok(tokname)
    INPUTS = json.load(open(f"{BASE}/stage1_inputs_{key}.json"))
    res = {}
    for arm, label in [("a2_A", "systematic"), ("a2_C", "chopthin")]:
        st = dict(n=0, smc=0, behav=0, oracle=0, has_in=0, engaged=0)
        for f in sorted(glob.glob(f"{ROOT}/{run}/{arm}/per_run/*.json")):
            d = json.load(open(f)); st["n"] += 1
            seqs = d["all_particle_token_ids"]; ci = d["chosen_idx"]
            w = softmax(d.get("log_w_final", [0.0] * len(seqs)))
            rlen = d.get("response_length_tokens", 0); pl = max(0, len(seqs[ci]) - rlen)
            pid = d.get("problem_id") or d.get("metadata", {}).get("task_id"); prob = HE.get(pid)
            codes = [extract_answer_3tier(tok.decode(s[pl:], skip_special_tokens=True),
                                          "humaneval", problem=prob)[0] for s in seqs]
            distinct = defaultdict(list)
            for i, c in enumerate(codes):
                distinct[c].append(i)
            if d["correct"]:
                st["smc"] += 1
            hidc = {}
            def hid(c):
                if c is None: return False
                if c in hidc: return hidc[c]
                r = passes_hidden(prob, c); hidc[c] = r; return r
            if any(hid(c) for c in distinct if c is not None):
                st["oracle"] += 1
            exprs = INPUTS.get(pid, [])
            if exprs: st["has_in"] += 1
            behav_ok = d["correct"]                    # fallback = smc when no usable behavior
            if exprs:
                clusters = defaultdict(list)
                for c in distinct:
                    if c is None: continue
                    sg = behavior(prob, c, exprs)
                    if sg is not None:
                        clusters[sg].append(c)
                if clusters:
                    mass = lambda cs: sum(w[i] for c in cs for i in distinct[c])
                    best = max(clusters, key=lambda sg: (len(clusters[sg]), mass(clusters[sg])))
                    rep = max(clusters[best], key=lambda c: sum(w[i] for i in distinct[c]))
                    behav_ok = hid(rep); st["engaged"] += 1
            if behav_ok: st["behav"] += 1
        res[label] = st
        n = st["n"]
        print(f"[{model:16s} {label:10s}] n={n}  smc={100*st['smc']/n:4.1f}  "
              f"BEHAV-MAJ(S1)={100*st['behav']/n:4.1f}  oracle={100*st['oracle']/n:4.1f}   "
              f"(has_inputs={st['has_in']}, engaged={st['engaged']})", flush=True)
    return res

if __name__ == "__main__":
    print("Stage 1: fully test-free behavior-majority (model-generated inputs, all 164)\n")
    ALL = {}
    for model, tokname, run, key in CONFIGS:
        ALL[model] = analyze(model, tokname, run, key)
    json.dump(ALL, open(BASE + "/stage1_behavior_majority_results.json", "w"), indent=1)
    print("\nsaved stage1_behavior_majority_results.json")
