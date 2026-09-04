#!/usr/bin/env python3
"""HumanEval selection without ground-truth tests, step 2 of 2: behavior-majority voting.

For each problem it decodes the N candidate programs saved by run.py, merges identical
programs, runs each distinct program on the model-written inputs from he_gen_inputs.py in
the sandbox, groups programs by identical output behavior, and returns the largest group
(ties broken by pooled particle weight). The held-out HumanEval tests are used only to score
the result, never by the selector. Problems with no usable inputs fall back to the run's
weight-drawn particle.

    python he_behavior_select.py --run runs/humaneval/systematic runs/humaneval/chopthin \
                                 --inputs data/he_inputs/he_inputs_qwen_math.json

Programs are decoded with the run's HumanEval protocol (he_protocols.py: stub / cot /
legacy), read from the run's config.json; --protocol and --tokenizer override it for runs
made before run.py recorded them.

Prints, per run: weight-draw accuracy (as run), behavior-majority accuracy, oracle coverage.
"""
import argparse
import glob
import json
import math
import multiprocessing
import os
from collections import defaultdict

from transformers import AutoTokenizer

from grader_utils.he_execute import (check_correctness, create_tempdir, reliability_guard,
                                     swallow_io, time_limit)
from he_protocols import PROTOCOLS, get_protocol


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def _behavior_worker(program, exprs, result):
    """Run `program` on each input expression in a fresh sandbox; append its outputs."""
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


def behavior(program, exprs):
    """Behavior signature = tuple of the program's outputs on `exprs`, or None if it won't run."""
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


def passes_hidden(grade_prob, code):
    """Held-out HumanEval tests, used only to score accuracy."""
    if code is None:
        return False
    try:
        return check_correctness(dict(grade_prob), code, 6.0).get("passed", False)
    except Exception:
        return False


def run_settings(run_dir, protocol=None, tokenizer=None):
    """(protocol, tokenizer) for a run: from its config.json unless overridden."""
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    proto = get_protocol(protocol or cfg.get("he_pipeline") or "legacy")
    tok = AutoTokenizer.from_pretrained(tokenizer or cfg["model"])
    return proto, tok


def analyze(run_dir, inputs, problems, protocol=None, tokenizer=None):
    proto, tok = run_settings(run_dir, protocol, tokenizer)
    st = dict(n=0, weight_draw=0, behavior_majority=0, oracle=0, engaged=0)
    for path in sorted(glob.glob(os.path.join(run_dir, "per_run", "*.json"))):
        with open(path) as f:
            d = json.load(f)
        st["n"] += 1
        seqs, ci = d["all_particle_token_ids"], d["chosen_idx"]
        w = softmax(d.get("log_w_final", [0.0] * len(seqs)))
        prompt_len = max(0, len(seqs[ci]) - d.get("response_length_tokens", 0))
        prob = problems.get(d["problem_id"])
        grade_prob = proto.grade_problem(prob)
        codes = [proto.extract(tok.decode(s[prompt_len:], skip_special_tokens=True), prob)[0]
                 for s in seqs]
        distinct = defaultdict(list)                          # program text -> particle indices
        for i, c in enumerate(codes):
            distinct[c].append(i)

        if d["correct"]:                                      # the run's own weight draw
            st["weight_draw"] += 1

        hidden = {}

        def hid(c):                                           # memoized held-out-test result
            if c is None:
                return False
            if c not in hidden:
                hidden[c] = passes_hidden(grade_prob, c)
            return hidden[c]

        if any(hid(c) for c in distinct if c is not None):
            st["oracle"] += 1

        exprs = inputs.get(d["problem_id"], [])
        ok = d["correct"]                                     # fallback: the weight draw
        if exprs:
            clusters = defaultdict(list)                      # behavior signature -> programs
            for c in distinct:
                if c is None:
                    continue
                sg = behavior(proto.program(prob, c), exprs)
                if sg is not None:
                    clusters[sg].append(c)
            if clusters:
                mass = lambda cs: sum(w[i] for c in cs for i in distinct[c])
                best = max(clusters, key=lambda sg: (len(clusters[sg]), mass(clusters[sg])))
                rep = max(clusters[best], key=lambda c: sum(w[i] for i in distinct[c]))
                ok = hid(rep)
                st["engaged"] += 1
        if ok:
            st["behavior_majority"] += 1

    n = max(st["n"], 1)
    st["protocol"] = proto.name
    print(f"{run_dir} [{proto.name}]: n={st['n']}  weight_draw={100 * st['weight_draw'] / n:.1f}%  "
          f"behavior_majority={100 * st['behavior_majority'] / n:.1f}%  "
          f"oracle={100 * st['oracle'] / n:.1f}%  (selector engaged on {st['engaged']})", flush=True)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run directories from run.py")
    ap.add_argument("--inputs", required=True, help="he_inputs_<model>.json from he_gen_inputs.py")
    ap.add_argument("--data", default="data/HumanEval.jsonl")
    ap.add_argument("--out", default="behavior_majority_results.json")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default=None,
                    help="override the run's he_pipeline (needed for runs made before run.py recorded it)")
    ap.add_argument("--tokenizer", default=None,
                    help="override the tokenizer named in the run's config.json")
    args = ap.parse_args()

    with open(args.inputs) as f:
        inputs = json.load(f)
    with open(args.data) as f:
        problems = {r["task_id"]: r for r in (json.loads(l) for l in f if l.strip())}

    results = {run_dir: analyze(run_dir, inputs, problems, args.protocol, args.tokenizer)
               for run_dir in args.run}
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
