#!/usr/bin/env python3
"""HumanEval prompt/extraction contracts, shared by the runner, the selector and the oracle.

Three protocols exist. They differ only in
  (i)   how a raw completion becomes a candidate program (`extract`),
  (ii)  what source is executed to obtain a behavioral signature (`program`), and
  (iii) which problem dict the sandbox grader receives (`grade_problem`).
Everything else -- sampler, sandbox, clustering, majority vote -- is protocol-independent.

    legacy  completion-ask prompt; the 3-tier extractor returns a function BODY;
            the sandbox prepends the problem stub.
    cot     chain-of-thought prompt; the last ```python block is a STANDALONE program
            (stub included), so it is graded with prompt="" to avoid a duplicate signature.
    stub    the raw code stub, nothing appended; the completion cut at the first canonical
            stop word is a BODY; the sandbox prepends the problem stub.

Paper (Table 1): `cot` for Qwen2.5-7B, `stub` for Qwen2.5-Math-7B and Qwen3-4B.
Select the protocol at generation time with `run_baseline.py --he_pipeline <name>` and use
the same name when selecting (`he_behavior_select.py --protocol <name>`).
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

from run_baseline import extract_answer_3tier, he_cot_extract_code, he_stub_truncate


class HEProtocol(ABC):
    """One HumanEval prompt/extraction contract."""

    name: str = ""

    @abstractmethod
    def extract(self, completion: str, prob: dict) -> Optional[str]:
        """Turn a decoded completion into the candidate program string (None = unusable)."""

    def program(self, prob: dict, code: str) -> str:
        """Source executed for the behavioral signature. Default: stub + body."""
        return prob["prompt"] + code

    def grade_problem(self, prob: dict) -> dict:
        """Problem dict handed to the sandbox grader. Default: the original problem."""
        return dict(prob)


class LegacyProtocol(HEProtocol):
    name = "legacy"

    def extract(self, completion: str, prob: dict) -> Optional[str]:
        return extract_answer_3tier(completion, "humaneval", problem=prob)[0]


class StubProtocol(HEProtocol):
    name = "stub"

    def extract(self, completion: str, prob: dict) -> Optional[str]:
        return he_stub_truncate(completion)


class CotProtocol(HEProtocol):
    name = "cot"

    def extract(self, completion: str, prob: dict) -> Optional[str]:
        return he_cot_extract_code(completion, prob["prompt"])[0]

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
