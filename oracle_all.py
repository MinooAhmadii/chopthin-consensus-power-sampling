#!/usr/bin/env python3
"""Oracle vs selected accuracy across experiments: is the bottleneck COVERAGE or SELECTION?
oracle = fraction of problems where >=1 particle is correct (best-of-N).
gap = oracle - accuracy = problems solvable-but-not-selected (selector waste)."""
import json, os, glob, sys
from transformers import AutoTokenizer
from run_baseline import extract_answer_3tier, grade_answer_3tier
from he_protocols import get_protocol

# Root folder that holds the run directories (each with per_run/*.json). Override with
#   CCPS_RUNS=/path/to/runs python3 oracle_all.py [math|gsm8k|he|all]
# The tokenizer is global: set CCPS_TOKENIZER=Qwen/Qwen3-4B for Qwen3 runs.
ROOT = os.environ.get("CCPS_RUNS", "runs")
MODEL = os.environ.get("CCPS_TOKENIZER", "Qwen/Qwen2.5-Math-7B")
tok = AutoTokenizer.from_pretrained(MODEL)

HE_PROBS = {}
def load_he():
    if HE_PROBS: return
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "HumanEval.jsonl")
    for line in open(p):
        r = json.loads(line)
        HE_PROBS[r["task_id"]] = r

def grade_particle(text, gold, dataset, prob_meta, he_protocol=None):
    if dataset == "humaneval":
        # HumanEval runs must be decoded with the protocol they were generated under
        # (run_baseline.py --he_pipeline); see he_protocols.py.
        ext = he_protocol.extract(text, prob_meta)
        return grade_answer_3tier(ext, gold, "humaneval", problem=he_protocol.grade_problem(prob_meta))
    ext, _ = extract_answer_3tier(text, dataset)
    return grade_answer_3tier(ext, gold, dataset)

def analyze(arm_dir, dataset, limit=None, he_protocol="legacy"):
    proto = get_protocol(he_protocol) if dataset == "humaneval" else None
    files = sorted(glob.glob(os.path.join(ROOT, arm_dir, "per_run", "*.json")))
    if limit: files = files[:limit]
    n=sel=oracle=0
    chop_extra=0
    oracle_set=set(); sel_set=set()
    for f in files:
        d = json.load(open(f))
        gold = d["gold_answer"]
        seqs = d["all_particle_token_ids"]
        ci = d["chosen_idx"]
        rlen = d.get("response_length_tokens", 0)
        pl = max(0, len(seqs[ci]) - rlen)
        prob_meta = None
        if dataset == "humaneval":
            load_he()
            prob_meta = HE_PROBS.get(d.get("problem_id")) or HE_PROBS.get(d.get("metadata",{}).get("task_id"))
        any_ok=False
        for seq in seqs:
            text = tok.decode(seq[pl:], skip_special_tokens=True)
            if grade_particle(text, gold, dataset, prob_meta, proto):
                any_ok=True
                if dataset != "humaneval":   # for speed, stop early on non-code
                    break
        n+=1
        s=bool(d["correct"]); sel+=int(s)
        oracle+=int(any_ok)
        if s: sel_set.add(d["problem_idx"])
        if any_ok: oracle_set.add(d["problem_idx"])
    return dict(n=n, sel=sel, oracle=oracle, sel_set=sel_set, oracle_set=oracle_set)

def report(label, bd, cd, dataset, limit=None, he_protocol="legacy"):
    b=analyze(bd,dataset,limit,he_protocol); c=analyze(cd,dataset,limit,he_protocol)
    ba=100*b['sel']/b['n']; bo=100*b['oracle']/b['n']
    ca=100*c['sel']/c['n']; co=100*c['oracle']/c['n']
    # chop oracle gain over base; coverage exclusive
    chop_only = len(c['oracle_set'] - b['oracle_set'])
    base_only = len(b['oracle_set'] - c['oracle_set'])
    print(f"{label:<24} base: acc={ba:4.1f}% orc={bo:4.1f}% gap={bo-ba:4.1f} | "
          f"chop: acc={ca:4.1f}% orc={co:4.1f}% gap={co-ca:4.1f} | "
          f"orcΔ={co-bo:+4.1f} (chop-only {chop_only}, base-only {base_only})")
    return dict(b=b,c=c)

if __name__=="__main__":
    which = sys.argv[1] if len(sys.argv)>1 else "math"
    print(f"{'comparison':<24} {'BASELINE':^28} | {'CHOPTHIN':^28} | oracle delta")
    print("-"*120)
    if which in ("math","all"):
        report("MATH N=8 a2","math500_n8_s42/a2_A","math500_n8_s42/a2_C","math")
        report("MATH N=16 a2","math500_n16_s42/a2_A","math500_n16_s42/a2_C","math")
        report("MATH N=32 a2","full_math500_n32_s42/a2_A","full_math500_n32_s42/a2_C","math")
        report("MATH N=32 a4","full_math500_n32_s42/a4_A","full_math500_n32_s42/a4_C","math")
        report("MATH N=64 a2","full_math500_n64_s42/a2_A","full_math500_n64_s42/a2_C","math")
    if which in ("gsm8k","all"):
        report("GSM8K N=32 a2","gsm8k_full_n32_s42/a2_A","gsm8k_full_n32_s42/a2_C","gsm8k")
    if which in ("he","all"):
        # Paper Table 1 HumanEval runs: stub for Math-7B and Qwen3-4B, cot for Qwen2.5-7B.
        # (Qwen3-4B needs CCPS_TOKENIZER=Qwen/Qwen3-4B.)
        report("HEval Math-7B stub","humaneval_stub_qwen_math_n32_s42/a2_A","humaneval_stub_qwen_math_n32_s42/a2_C","humaneval",he_protocol="stub")
        report("HEval Qwen2.5-7B cot","humaneval_cot_qwen2.5-7b_n32_s42/a2_A","humaneval_cot_qwen2.5-7b_n32_s42/a2_C","humaneval",he_protocol="cot")
        report("HEval Qwen3-4B stub","humaneval_stub_qwen3-4b_n32_s42/a2_A","humaneval_stub_qwen3-4b_n32_s42/a2_C","humaneval",he_protocol="stub")
