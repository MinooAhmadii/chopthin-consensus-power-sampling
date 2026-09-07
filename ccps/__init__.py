"""Chopthin-Consensus Power Sampling: library code (sampler, resampler, prompts, graders)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DATA_DIR = os.path.join(ROOT, "data")

MODELS = {
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen3": "Qwen/Qwen3-4B",
}
