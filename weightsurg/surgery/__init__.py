"""Surgical weight delta synthesis engine (Algebraic & LLM-Guided)."""

from weightsurg.surgery.algebraic import (
    compute_low_rank_delta,
    compute_rank1_delta,
    compute_subspace_damping_delta,
    synthesize_algebraic_delta,
)
from weightsurg.surgery.llm_guided import (
    LLMWeightProgrammer,
    StabilityValidationResult,
)

__all__ = [
    "LLMWeightProgrammer",
    "StabilityValidationResult",
    "compute_low_rank_delta",
    "compute_rank1_delta",
    "compute_subspace_damping_delta",
    "synthesize_algebraic_delta",
]

