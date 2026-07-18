#!/usr/bin/env python3
"""
Stage 1 input generation. The base model synthesizes its own test-INPUT calls per HumanEval
problem (no ground truth, no gold tests). These inputs let the behavior-majority selector run on
ALL 164 problems and be fully self-contained (the code analogue of semantic majority).

Usage:  python3 stage1_gen_inputs.py <qwen_math|qwen|qwen3>
Saves:  stage1_inputs_<key>.json = { task_id: [ "entry(args)", ... ] }
"""
import sys, json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/data/projects/nullanet/experiments/minoo/choptin"
MODELS = {"qwen_math": "Qwen/Qwen2.5-Math-7B", "qwen": "Qwen/Qwen2.5-7B", "qwen3": "Qwen/Qwen3-4B"}
key = sys.argv[1]; mpath = MODELS[key]
tasks = [json.loads(l) for l in open(BASE + "/data/HumanEval.jsonl")]

tok = AutoTokenizer.from_pretrained(mpath)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.bfloat16, device_map="cuda").eval()

def extract_calls(text, entry):
    """Extract balanced-paren  entry(...)  call expressions from generated text."""
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
                    end = p; break
        if end < 0:
            break
        calls.append(text[j:end + 1]); i = end + 1
    seen, out = set(), []
    for c in calls:
        c = c.strip()
        if c not in seen and 3 < len(c) < 200:
            seen.add(c); out.append(c)
    return out[:8]

out = {}
for idx, t in enumerate(tasks):
    entry, prompt = t["entry_point"], t["prompt"]
    seed = f">>> {entry}("
    gen_prompt = prompt + f"    pass\n\n# Diverse example calls to test {entry} (inputs only):\n" + seed
    ids = tok(gen_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=0.8,
                           top_p=0.95, pad_token_id=tok.pad_token_id)
    text = tok.decode(g[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    calls = extract_calls(seed + text, entry)
    out[t["task_id"]] = calls
    if idx % 40 == 0:
        print(f"[{key}] {idx}/164  n_calls={len(calls)}  e.g. {calls[:2]}", flush=True)

json.dump(out, open(BASE + f"/stage1_inputs_{key}.json", "w"))
tot = sum(len(v) for v in out.values()); cov = sum(1 for v in out.values() if v)
print(f"[{key}] saved stage1_inputs_{key}.json  mean_calls/problem={tot/len(out):.1f}  "
      f"problems_with_>=1_input={cov}/164", flush=True)
