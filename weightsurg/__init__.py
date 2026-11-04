"""WeightSurg: Zero-Gradient Weight-Space Programming & Neural Circuit Surgery Toolkit."""

from weightsurg.core import (
    AttentionProjections,
    FFNProjections,
    ModelIngestor,
    SVDDecomposition,
    TensorAlgebraicProfile,
    compute_svd,
    create_synthetic_qwen2_model,
    profile_model,
    profile_tensor,
)
from weightsurg.diagnostics import (
    ComponentTraceScore,
    DiagnosticReport,
    FailureLocalizer,
    PatchRecipe,
)
from weightsurg.patcher import (
    HotPatcher,
    ValidationMetrics,
)
from weightsurg.surgery import (
    LLMWeightProgrammer,
    compute_low_rank_delta,
    compute_rank1_delta,
    compute_subspace_damping_delta,
    synthesize_algebraic_delta,
)

__version__ = "0.1.0"

__all__ = [
    "AttentionProjections",
    "ComponentTraceScore",
    "DiagnosticReport",
    "FFNProjections",
    "FailureLocalizer",
    "HotPatcher",
    "LLMWeightProgrammer",
    "ModelIngestor",
    "PatchRecipe",
    "SVDDecomposition",
    "TensorAlgebraicProfile",
    "ValidationMetrics",
    "compute_low_rank_delta",
    "compute_rank1_delta",
    "compute_subspace_damping_delta",
    "compute_svd",
    "create_synthetic_qwen2_model",
    "profile_model",
    "profile_tensor",
    "synthesize_algebraic_delta",
]

