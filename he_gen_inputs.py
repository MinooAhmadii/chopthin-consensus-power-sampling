#!/usr/bin/env python3
"""HumanEval selection without ground-truth tests, step 1 of 2: write the test inputs.

The base model writes example calls for each HumanEval function (inputs only, no expected
outputs, no gold tests). he_behavior_select.py then runs every candidate program on these
inputs and votes by behavior. Generate the inputs once per base model and use them for both
resampling arms.

    python he_gen_inputs.py qwen_math [--data data/HumanEval.jsonl] [--out he_inputs_qwen_math.json]

Writes { task_id: ["entry(args)", ...] }.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import MODELS


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="qwen_math | qwen | qwen3, or any Hugging Face model id")
    ap.add_argument("--data", default="data/HumanEval.jsonl")
    ap.add_argument("--out", default=None, help="output file; default he_inputs_<model>.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model_id = MODELS.get(args.model, args.model)
    out_path = args.out or f"he_inputs_{args.model.replace('/', '_')}.json"
    with open(args.data) as f:
        tasks = [json.loads(l) for l in f if l.strip()]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(args.device).eval()

    out = {}
    for idx, t in enumerate(tasks):
        entry, prompt = t["entry_point"], t["prompt"]
        seed = f">>> {entry}("
        # Complete the signature with `pass`, then ask for diverse example calls (inputs only).
        gen_prompt = (prompt + "    pass\n\n"
                      f"# Diverse example calls to test {entry} (inputs only):\n" + seed)
        ids = tok(gen_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=256, do_sample=True,
                               temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
        text = tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        out[t["task_id"]] = extract_calls(seed + text, entry)
        if idx % 40 == 0:
            print(f"[{args.model}] {idx}/{len(tasks)}  n_calls={len(out[t['task_id']])}  "
                  f"e.g. {out[t['task_id']][:2]}", flush=True)

    with open(out_path, "w") as f:
        json.dump(out, f)
    total = sum(len(v) for v in out.values())
    with_inputs = sum(1 for v in out.values() if v)
    print(f"saved {out_path}  mean calls/problem={total / len(out):.1f}  "
          f"problems with >=1 input={with_inputs}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
