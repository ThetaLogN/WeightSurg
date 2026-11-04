"""Zero-gradient hot-patcher, behavioral validator, and model exporter."""

from weightsurg.patcher.hotpatch import (
    DEFAULT_BASELINE_CORPUS,
    HotPatcher,
    ValidationMetrics,
    compute_token_repetition_rate,
)

__all__ = [
    "DEFAULT_BASELINE_CORPUS",
    "HotPatcher",
    "ValidationMetrics",
    "compute_token_repetition_rate",
]

