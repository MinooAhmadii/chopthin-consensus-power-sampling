#!/usr/bin/env python3
"""Verify the HumanEval `cot` grading contract on real problems, offline (no model).

For each of the first N problems we synthesize the output the cot pipeline expects (a
markdown block holding the canonical solution, stub included) and confirm that
  - he_cot_extract_code returns a standalone program, and
  - grading it with the CotProtocol problem dict (prompt="") PASSES.
For information it also grades with the ORIGINAL prompt, which makes the sandbox prepend
the stub a second time. Python tolerates the redefinition for well-formed programs, so
this usually passes too; the contract still uses prompt="" so the graded source is exactly
the extracted program.

Run from anywhere:  python3 tests/test_he_cot_contract.py   (also works under pytest)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run_baseline import he_cot_extract_code            # noqa: E402
from he_protocols import get_protocol                   # noqa: E402
from grader_utils.he_check import check_correctness     # noqa: E402

N = 12


def load_problems():
    with open(os.path.join(ROOT, "data", "HumanEval.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def run(n=N, verbose=True):
    cot = get_protocol("cot")
    probs = load_problems()[:n]
    ok_empty = ok_orig = 0
    for p in probs:
        response = f"Reasoning here.\n\n```python\n{p['prompt'] + p['canonical_solution']}\n```"
        code, path = he_cot_extract_code(response, p["prompt"])
        r1 = check_correctness(cot.grade_problem(p), code, timeout=6.0).get("passed", False)
        r2 = check_correctness(p, code, timeout=6.0).get("passed", False)
        ok_empty += bool(r1)
        ok_orig += bool(r2)
        if verbose:
            flag = "" if r1 else "   <-- UNEXPECTED FAIL"
            print(f"{p['task_id']:16s} path={path:16s} prompt='' -> {str(r1):5s} | "
                  f"orig-prompt -> {str(r2):5s}{flag}")
    return ok_empty, ok_orig, len(probs)


def test_cot_contract():
    ok_empty, _, n = run(verbose=False)
    assert ok_empty == n, f"cot contract broken: {ok_empty}/{n} passed with prompt=''"


if __name__ == "__main__":
    ok_empty, ok_orig, n = run()
    print(f"\n  with prompt='' (cot contract)  : {ok_empty}/{n} passed")
    print(f"  with original prompt (wrong)   : {ok_orig}/{n} passed\n")
    if ok_empty == n:
        note = "" if ok_orig == n else "; the original prompt fails on some, so prompt='' is required"
        print(f"RESULT: contract OK -- every extracted program passes with prompt=''{note}.")
    else:
        print("RESULT: PROBLEM -- prompt='' did not pass everything. Do not launch.")
        sys.exit(1)
