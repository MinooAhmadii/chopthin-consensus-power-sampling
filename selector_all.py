#!/usr/bin/env python3
"""Selector ladder across ALL complete chopthin cells (every model x dataset).

Distinct-trajectory voting: particles with identical terminal token sequences (true
resampling duplicates) collapse to ONE vote; genuinely diverged paths each vote once.

Free / CPU rungs (no GPU, no extra model), per discrete-answer dataset:
  smc     = as-run weight-proportional multinomial draw   (= reported selected acc; sanity check)
  argmax  = single highest-weight particle                (removes sampling noise)
  wmaj    = weighted semantic majority (max softmax(log_w) mass per answer cluster)
  maj     = unweighted semantic majority (max distinct-trajectory votes per cluster)
  oracle  = >=1 particle correct                          (= reported oracle; sanity check)

Datasets covered here: math, gsm8k, aime, gpqa  (discrete answer -> majority applies).
HumanEval is CODE: majority is ill-defined, so its selector is the UNIT-TEST one in
`he_select.py` (run that separately). Reuses run_baseline graders so smc/oracle reproduce
the reported headline numbers, and picks the right tokenizer per model.

Run on kaveh:
    cd /data/projects/nullanet/experiments/minoo/choptin
    conda activate minoo_env
    export HF_HUB_CACHE=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
    python3 selector_all.py            # all groups
    python3 selector_all.py math7b     # one group: math7b | qwen25-7b | qwen3-4b
"""
import json, os, glob, math, sys
from transformers import AutoTokenizer
from run_baseline import extract_answer_3tier, grade_answer_3tier

ROOT = "/data/projects/nullanet/experiments/minoo/choptin/runs"

# Tokenizer per model (used only to DECODE token_ids -> text). Math-7B & Qwen2.5-7B share a
# tokenizer; Qwen3-4B has its own (this is why decoding the wrong one corrupts the oracle).
TOKZ = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "qwen":      "Qwen/Qwen2.5-7B",
    "qwen3":     "Qwen/Qwen3-4B",
}
_tok = {}
def get_tok(model):
    if model not in _tok:
        _tok[model] = AutoTokenizer.from_pretrained(TOKZ[model])
    return _tok[model]

# Registry of COMPLETE cells: (group, model, dataset, label, run-folder). Arms a2_A/a2_C added below.
CELLS = [
    # ---- Qwen2.5-Math-7B (primary; full N-sweep) ----
    ("math7b", "qwen_math", "math",  "MATH  N=8",   "math500_n8_s42"),
    ("math7b", "qwen_math", "math",  "MATH  N=16",  "math500_n16_s42"),
    ("math7b", "qwen_math", "math",  "MATH  N=32",  "full_math500_n32_s42"),
    ("math7b", "qwen_math", "math",  "MATH  N=64",  "full_math500_n64_s42"),
    ("math7b", "qwen_math", "gsm8k", "GSM8K N=32",  "gsm8k_full_n32_s42"),
    ("math7b", "qwen_math", "aime",  "AIME  N=32",  "aime_n32_s42"),
    ("math7b", "qwen_math", "aime",  "AIME  N=64",  "aime_n64_shard_s42"),
    ("math7b", "qwen_math", "aime",  "AIME  N=128", "aime_n128_shard_s42"),
    ("math7b", "qwen_math", "gpqa",  "GPQA  N=32",  "gpqa_qwen_math_n32_s42"),
    # ---- Qwen2.5-7B ----
    ("qwen25-7b", "qwen", "math",  "MATH  N=32", "math500_qwen2.5-7b_n32_s42"),
    ("qwen25-7b", "qwen", "gsm8k", "GSM8K N=32", "gsm8k_qwen2.5-7b_n32_s42"),   # done 2026-06-13
    ("qwen25-7b", "qwen", "gpqa",  "GPQA  N=32", "gpqa_qwen2.5-7b_n32_s42"),    # done 2026-06-13
    ("qwen25-7b", "qwen", "aime",  "AIME  N=32", "aime_qwen2.5-7b_n32_s42"),    # seed 42 (s43/s44 also exist)
    # ---- Qwen3-4B ----
    ("qwen3-4b", "qwen3", "math",  "MATH  N=32", "math_qwen3-4b_n32_s42"),
    ("qwen3-4b", "qwen3", "gsm8k", "GSM8K N=32", "gsm8k_qwen3-4b_n32_s42"),   # done 2026-06-14
    ("qwen3-4b", "qwen3", "aime",  "AIME  N=32", "aime_qwen3-4b_n32_s42"),
]
ARMS = [("systematic", "a2_A"), ("chopthin", "a2_C")]


def softmax(xs):
    m = max(xs); es = [math.exp(x - m) for x in xs]; s = sum(es) or 1.0
    return [e / s for e in es]


def grade(text, gold, ds):
    """(correct, extracted-answer-string) for a discrete-answer dataset."""
    ext, _ = extract_answer_3tier(text, ds)
    return bool(grade_answer_3tier(ext, gold, ds)), (ext or "")


def eq(a, b, ds):
    """Semantic (grade-)equivalence of two non-empty answer strings (e.g. 0.5 == 1/2)."""
    if a == b:
        return True
    try:
        return bool(grade_answer_3tier(a, b, ds))
    except Exception:
        return False


def clusters_of(distinct, ds):
    """Greedy grade-equivalence clustering of distinct trajectories (non-empty answers)."""
    cl = []
    for d in distinct:
        if not d["ans"]:
            continue
        for c in cl:
            if eq(d["ans"], c["rep"], ds):
                c["votes"] += d["votes"]; c["mass"] += d["mass"]; break
        else:
            cl.append({"rep": d["ans"], "correct": d["correct"], "votes": d["votes"], "mass": d["mass"]})
    return cl


def analyze(model, ds, arm_dir):
    tok = get_tok(model)
    files = sorted(glob.glob(os.path.join(ROOT, arm_dir, "per_run", "*.json")))
    rung = {k: 0 for k in ("smc", "argmax", "wmaj", "maj", "oracle")}
    n = 0; udist = 0
    for f in files:
        r = json.load(open(f))
        gold = r["gold_answer"]; seqs = r["all_particle_token_ids"]
        ci = r["chosen_idx"]; rlen = r.get("response_length_tokens", 0)
        pl = max(0, len(seqs[ci]) - rlen)            # prompt-token offset (same trick as oracle_all)
        w = softmax(r["log_w_final"])
        parts = []
        for i, seq in enumerate(seqs):
            text = tok.decode(seq[pl:], skip_special_tokens=True)
            ok, ext = grade(text, gold, ds)
            parts.append(dict(ans=ext, correct=ok, w=w[i], key=tuple(seq)))
        n += 1
        rung["oracle"] += int(any(p["correct"] for p in parts))
        rung["smc"] += int(bool(r["correct"]))
        rung["argmax"] += int(max(parts, key=lambda p: p["w"])["correct"])
        # distinct trajectories: dedup by token sequence; sum weight, 1 vote each
        byseq = {}
        for p in parts:
            d = byseq.get(p["key"])
            if d is None:
                byseq[p["key"]] = dict(ans=p["ans"], correct=p["correct"], mass=p["w"], votes=1)
            else:
                d["mass"] += p["w"]
        distinct = list(byseq.values()); udist += len(distinct)
        cl = clusters_of(distinct, ds)
        if cl:
            rung["wmaj"] += int(max(cl, key=lambda c: c["mass"])["correct"])
            rung["maj"] += int(max(cl, key=lambda c: (c["votes"], c["mass"]))["correct"])
    return n, rung, udist / max(n, 1)


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    hdr = f"{'group':<10} {'cell':<11} {'arm':<11} {'smc':>6} {'argmax':>7} {'wmaj':>6} {'maj':>6} {'oracle':>7} {'uniqTraj':>9}"
    out = [hdr, "-" * len(hdr)]
    print(hdr); print("-" * len(hdr))
    last_group = None
    for group, model, ds, label, folder in CELLS:
        if want and group != want:
            continue
        if last_group is not None and group != last_group:
            print(); out.append("")
        last_group = group
        for arm, sub in ARMS:
            try:
                n, rg, ud = analyze(model, ds, f"{folder}/{sub}")
                pct = {k: 100 * v / n for k, v in rg.items()}
                row = (f"{group:<10} {label:<11} {arm:<11} {pct['smc']:6.1f} {pct['argmax']:7.1f} "
                       f"{pct['wmaj']:6.1f} {pct['maj']:6.1f} {pct['oracle']:7.1f} {ud:9.1f}")
            except Exception as e:
                row = f"{group:<10} {label:<11} {arm:<11}  ERROR: {type(e).__name__}: {e}"
            print(row, flush=True); out.append(row)
    with open("selector_all_results.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n(saved -> selector_all_results.txt)")


if __name__ == "__main__":
    main()
