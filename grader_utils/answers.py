"""Answer extraction and grading, one rule set per benchmark.

extract_answer: pull the final answer out of a completion (boxed expression first,
then an "answer is" pattern, then a tail fallback). For code, extract the program.
is_correct: compare an extracted answer with the gold answer under that benchmark's grader.
"""

import re

from grader_utils.he_execute import check_correctness
from grader_utils.math_grader import grade_answer

_TIER_SPECS = {
    "math": [
        (re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL), "full"),
        (re.compile(r"(?:answer\s*(?:is|:)|\\boxed|=)\s*([^.\n]{1,200})", re.IGNORECASE), 300),
        (re.compile(r"([^\s.,;:!?]+)\s*(?:\.|$)", re.DOTALL), 300),
    ],
    "aime": [
        (re.compile(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}"), "full"),
        (re.compile(r"answer\s+is\s*[:\s]*([0-9]{1,3})", re.IGNORECASE), 300),
        (re.compile(r"\b([0-9]{1,3})\b"), "full"),
    ],
    "gpqa": [
        (re.compile(r"\\boxed\s*\{\s*([A-Da-d])\s*\}"), "full"),
        (re.compile(r"answer\s+is\s*[:\s]*\(?\s*([A-Da-d])\s*\)?", re.IGNORECASE), 300),
        (re.compile(r"(?<![(\w])\b([A-D])\b(?![)\w])"), 200),
    ],
    "gsm8k": [
        (re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}"), "full"),
        (re.compile(r"(?:answer\s*(?:is|:)|=)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE), 300),
        (re.compile(r"(-?\d[\d,]*(?:\.\d+)?)"), 300),
    ],
}
_TIER_NAMES = ("primary", "fallback_1", "fallback_2")

DATASETS = ("math", "gsm8k", "aime", "gpqa", "humaneval")


def extract_answer(text, dataset, problem=None):
    """Return (answer_string_or_None, which_rule_matched).

    For humaneval pass the problem dict (needs its entry_point); the answer is the program.
    """
    text = text or ""
    if dataset == "humaneval":
        if problem is None:
            return None, "none"
        from grader_utils.he_grader import extract_code
        try:
            code = extract_code(text, problem.get("entry_point", ""))
            if code and len(code.strip()) > 0:
                return code, "primary"
        except Exception:
            pass
        return None, "none"

    if dataset not in _TIER_SPECS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    for tier_idx, (pat, scope) in enumerate(_TIER_SPECS[dataset]):
        hay = text if scope == "full" else text[-int(scope):]
        matches = pat.findall(hay)
        if matches:
            extracted = matches[-1].strip() if isinstance(matches[-1], str) else matches[-1]
            if dataset == "gpqa":
                extracted = extracted.upper()
            return extracted, _TIER_NAMES[tier_idx]
    return None, "none"


def is_correct(extracted, gold, dataset, problem=None):
    """Grade an extracted answer against the gold answer.

    For humaneval pass the problem dict; the program is run against the held-out tests.
    """
    if extracted is None:
        return False
    if dataset == "math":
        return bool(grade_answer(extracted, gold))
    if dataset == "aime":
        try:
            return int(extracted) == int(gold)
        except (TypeError, ValueError):
            return False
    if dataset == "gpqa":
        return str(extracted).strip().upper() == str(gold).strip().upper()
    if dataset == "gsm8k":
        try:
            ext_clean = str(extracted).replace(",", "").strip()
            gold_clean = str(gold).replace(",", "").strip()
            return abs(float(ext_clean) - float(gold_clean)) < 1e-6
        except (TypeError, ValueError):
            return False
    if dataset == "humaneval":
        if problem is None:
            return False
        try:
            result = check_correctness(problem, extracted, timeout=3.0)
        except Exception as e:  # per-problem sandbox failure: report it, do not kill the run
            print(f"[grade] check_correctness failed: {type(e).__name__}: {e}", flush=True)
            return False
        return bool(result.get("passed", False))
    raise ValueError(f"Unknown dataset: {dataset!r}")


def classify_outcome(extracted, gold, dataset, response_length_tokens, max_new_tokens, problem=None):
    """One of: correct | wrong_with_answer | no_answer | truncated."""
    hit_cap = response_length_tokens >= max_new_tokens
    if extracted is None and hit_cap:
        return "truncated", hit_cap
    if extracted is None:
        return "no_answer", hit_cap
    if is_correct(extracted, gold, dataset, problem=problem):
        return "correct", hit_cap
    return "wrong_with_answer", hit_cap
