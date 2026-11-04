"""Core model ingestion and algebraic profiling modules."""

from weightsurg.core.ingestion import (
    AttentionProjections,
    FFNProjections,
    ModelIngestor,
    create_synthetic_qwen2_model,
    detect_optimal_device,
    is_mlx_available,
    mlx_to_tensor,
    tensor_to_mlx,
)
from weightsurg.core.profiling import (
    OutlierReport,
    SVDDecomposition,
    TensorAlgebraicProfile,
    compute_sparsity_map,
    compute_spectral_metrics,
    compute_svd,
    detect_weight_outliers,
    profile_model,
    profile_tensor,
)

__all__ = [
    "AttentionProjections",
    "FFNProjections",
    "ModelIngestor",
    "OutlierReport",
    "SVDDecomposition",
    "TensorAlgebraicProfile",
    "compute_sparsity_map",
    "compute_spectral_metrics",
    "compute_svd",
    "create_synthetic_qwen2_model",
    "detect_optimal_device",
    "detect_weight_outliers",
    "is_mlx_available",
    "mlx_to_tensor",
    "profile_model",
    "profile_tensor",
    "tensor_to_mlx",
]

