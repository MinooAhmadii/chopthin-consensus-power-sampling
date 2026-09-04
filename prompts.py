"""Prompt templates, one per benchmark. Kept byte-identical to the paper's runs.

HumanEval prompts depend on the protocol (legacy / cot / stub) and live in he_protocols.py.
"""

from he_protocols import get_protocol

MATH_PREFIX = "Can you solve the following math problem? "
MATH_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{{}}."

GPQA_PREFIX = "Answer the following multiple-choice question by selecting the correct option.\n\n"
GPQA_SUFFIX = (
    "\nThink step by step, then put ONLY the letter (A, B, C, or D) of your chosen "
    "answer inside \\boxed{}. Example: \\boxed{B}"
)

# Models prompted through their chat template; the rest get the plain text.
CHAT_MODELS = {"qwen3": {"enable_thinking": False}}


def build_prompt(question: str, dataset: str, model: str, tokenizer, he_pipeline: str = "legacy") -> str:
    """Return the full prompt string for one problem."""
    if dataset == "gpqa":
        return GPQA_PREFIX + question + GPQA_SUFFIX
    if dataset == "humaneval":
        return get_protocol(he_pipeline).prompt(question)
    text = MATH_PREFIX + question + MATH_SUFFIX   # math, gsm8k, aime
    if model in CHAT_MODELS:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, **CHAT_MODELS[model],
        )
    return text
