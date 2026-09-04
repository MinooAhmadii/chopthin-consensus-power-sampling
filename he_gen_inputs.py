#!/usr/bin/env python3
"""
Test-free HumanEval selection, step 1 of 2: generate the test INPUTS.

The base model synthesizes its own example call-inputs for each HumanEval problem
(inputs only -- no ground-truth outputs, no gold tests). These inputs let the
behavior-majority selector (he_behavior_select.py) run on all 164 problems while
staying fully self-contained: the code analogue of semantic-majority voting.

Usage
-----
    python3 he_gen_inputs.py <qwen_math|qwen|qwen3> [--data data/HumanEval.jsonl] [--out .]

Writes  stage1_inputs_<key>.json  =  { task_id: ["entry(args)", ...] },
which he_behavior_select.py consumes.
"""
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# key -> HuggingFace model id (mirrors model_map in run_baseline.py).
MODELS = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen":      "Qwen/Qwen2.5-7B",
    "qwen3":     "Qwen/Qwen3-4B",
}


def extract_calls(text, entry, cap=8):
    """Pull balanced-paren ``entry(...)`` call expressions out of generated text."""
    calls, i = [], 0
    while True:
        j = text.find(entry + "(", i)
        if j < 0:
            break
        depth, end = 0, -1
        for p in range(j + len(entry), min(len(text), j + len(entry) + 300)):
            c = text[p]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = p
                    break
        if end < 0:
            break
        calls.append(text[j:end + 1])
        i = end + 1
    seen, out = set(), []
    for c in calls:
        c = c.strip()
        if c not in seen and 3 < len(c) < 200:
            seen.add(c)
            out.append(c)
    return out[:cap]


def main():
    ap = argparse.ArgumentParser(description="Generate self-contained HumanEval test inputs.")
    ap.add_argument("key", choices=list(MODELS), help="which base model to sample inputs from")
    ap.add_argument("--data", default="data/HumanEval.jsonl", help="path to HumanEval.jsonl")
    ap.add_argument("--out", default="data/he_inputs",
                    help="directory to write stage1_inputs_<key>.json")
    args = ap.parse_args()

    mpath = MODELS[args.key]
    tasks = [json.loads(l) for l in open(args.data)]

    tok = AutoTokenizer.from_pretrained(mpath)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mpath, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    out = {}
    for idx, t in enumerate(tasks):
        entry, prompt = t["entry_point"], t["prompt"]
        seed = f">>> {entry}("
        # Complete the signature with `pass`, then ask for diverse example calls (inputs only).
        gen_prompt = (prompt + f"    pass\n\n"
                      f"# Diverse example calls to test {entry} (inputs only):\n" + seed)
        ids = tok(gen_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=256, do_sample=True,
                               temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
        text = tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        out[t["task_id"]] = extract_calls(seed + text, entry)
        if idx % 40 == 0:
            print(f"[{args.key}] {idx}/{len(tasks)}  n_calls={len(out[t['task_id']])}  "
                  f"e.g. {out[t['task_id']][:2]}", flush=True)

    dest = os.path.join(args.out, f"stage1_inputs_{args.key}.json")
    json.dump(out, open(dest, "w"))
    tot = sum(len(v) for v in out.values())
    cov = sum(1 for v in out.values() if v)
    print(f"[{args.key}] saved {dest}  mean_calls/problem={tot / len(out):.1f}  "
          f"problems_with_>=1_input={cov}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
