#!/usr/bin/env python3
"""HumanEval prompt/extraction contracts, shared by run.py, he_behavior_select.py and
oracle_coverage.py.

A protocol fixes (i) the prompt sent to the model, (ii) how a completion becomes a candidate
program, (iii) what source is executed for a behavioral signature, and (iv) which problem
dict the sandbox grader receives. The sampler, sandbox, clustering and vote do not depend
on it.

    legacy  completion-ask suffix; the extractor returns a function BODY; the sandbox
            prepends the problem stub.
    cot     chain-of-thought instruction; the last ```python block is a STANDALONE program
            (stub included), so it is graded with prompt="" to avoid a second copy of the
            signature.
    stub    the raw code stub, nothing appended; the completion cut at the first canonical
            stop word is a BODY; the sandbox prepends the problem stub.

The paper uses `cot` for Qwen2.5-7B and `stub` for Qwen2.5-Math-7B and Qwen3-4B
(PAPER_PIPELINE): under `cot` the math model reasons past the token budget before writing
code. run.py records the protocol in config.json as "he_pipeline"; the selection and
coverage scripts read it from there.
"""
import re
import textwrap
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from grader_utils.answers import extract_answer

# Protocol used for each paper model when --he_pipeline is not given.
PAPER_PIPELINE = {"qwen": "cot", "qwen_math": "stub", "qwen3": "stub"}

LEGACY_SUFFIX = (
    "\n\n# Complete the function above. Output ONLY the function body in a python "
    "code block (```python ... ```)."
)

COT_SYSTEM = (
    "You are an expert Python programmer. "
    "You will be given a function signature and docstring. "
    "Think step by step about the problem, then provide your solution.\n\n"
    "IMPORTANT: After your reasoning, write your final implementation "
    "inside a single markdown code block like:\n"
    "```python\n<your code>\n```\n\n"
    "Your code block must contain the COMPLETE function definition "
    "Do NOT include any test cases, example calls, or print statements "
    "outside the function body. VERY IMPORTANT: YOU MUST STOP IMMEDIATELY AFTER "
    "GENERATING THE CODE BLOCK. NO FURTHER EXPLANATION OR VERIFICATION IS NEEDED."
)

# Canonical HumanEval stop-word list (the same set as Zhou et al., Entropy-Cut MH). A base
# model continues a code stub with code; the completion is cut at the first stop word.
STOP_WORDS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\nassert",
              "\nimport", "\nfrom", "\n```", "if __name__"]


def cot_build_prompt(stub: str) -> str:
    return (
        f"{COT_SYSTEM}\n\n"
        f"## Problem\n\n"
        f"Complete the following Python function:\n\n"
        f"```python\n{stub}```\n"
    )


def stub_truncate(completion: str) -> str:
    """Cut the continuation at the earliest stop word (the standard HumanEval rule)."""
    cut = len(completion)
    for sw in STOP_WORDS:
        i = completion.find(sw)
        if 0 <= i < cut:
            cut = i
    return completion[:cut]


def _func_name(stub: str) -> Optional[str]:
    m = re.search(r"def\s+(\w+)\s*\(", stub)
    return m.group(1) if m else None


def _as_body(raw: str) -> str:
    """Normalize a body-only snippet to exactly one 4-space indent level."""
    return textwrap.indent(textwrap.dedent(raw).strip("\n"), "    ")


def cot_extract_code(response: str, stub: str) -> Tuple[str, str]:
    """Return (standalone_program, extraction_path). Never returns None."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    func_name = _func_name(stub)
    if blocks:
        raw = blocks[-1].rstrip()          # LAST block: reasoning models show drafts first
        if func_name and f"def {func_name}" in raw:
            return raw.strip() + "\n", "cot_block_full"
        # the block held only a body -> re-indent and prepend the signature
        return stub + _as_body(raw) + "\n", "cot_block_body"
    if func_name:
        m = re.search(
            rf"(def {re.escape(func_name)}\(.*?(?:\n(?=def |\nclass |\Z)|$))",
            response, re.DOTALL)
        if m:
            return m.group(1).rstrip() + "\n", "cot_def_regex"
    return stub + _as_body(response) + "\n", "cot_fallback"


class HEProtocol(ABC):
    """One HumanEval prompt/extraction contract."""

    name: str = ""

    @abstractmethod
    def prompt(self, stub: str) -> str:
        """Full prompt for one problem, given its signature + docstring stub."""

    @abstractmethod
    def extract(self, completion: str, prob: dict) -> Tuple[Optional[str], str]:
        """(candidate program or None, extraction path) from a decoded completion."""

    def program(self, prob: dict, code: str) -> str:
        """Source executed for the behavioral signature. Default: stub + body."""
        return prob["prompt"] + code

    def grade_problem(self, prob: dict) -> dict:
        """Problem dict handed to the sandbox grader. Default: the original problem."""
        return dict(prob)


class LegacyProtocol(HEProtocol):
    name = "legacy"

    def prompt(self, stub: str) -> str:
        return stub + LEGACY_SUFFIX

    def extract(self, completion: str, prob: dict) -> Tuple[Optional[str], str]:
        return extract_answer(completion, "humaneval", problem=prob)


class StubProtocol(HEProtocol):
    name = "stub"

    def prompt(self, stub: str) -> str:
        return stub

    def extract(self, completion: str, prob: dict) -> Tuple[Optional[str], str]:
        return stub_truncate(completion), "stub_stopword"


class CotProtocol(HEProtocol):
    name = "cot"

    def prompt(self, stub: str) -> str:
        return cot_build_prompt(stub)

    def extract(self, completion: str, prob: dict) -> Tuple[Optional[str], str]:
        return cot_extract_code(completion, prob["prompt"])

    def program(self, prob: dict, code: str) -> str:
        return code  # standalone program: the stub is already inside

    def grade_problem(self, prob: dict) -> dict:
        return {**prob, "prompt": ""}


PROTOCOLS: Dict[str, HEProtocol] = {c.name: c() for c in (LegacyProtocol, CotProtocol, StubProtocol)}


def get_protocol(name: str) -> HEProtocol:
    try:
        return PROTOCOLS[name]
    except KeyError:
        raise ValueError(f"unknown HumanEval protocol {name!r}; choose from {sorted(PROTOCOLS)}")
