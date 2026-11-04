"""Fast Algebraic Profiling for Transformer Weights.

Provides Singular Value Decomposition (SVD), spectral norm computation, stable and
effective rank analysis, sparsity mapping, and outlier weight detection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
import torch

from weightsurg.core.ingestion import ModelIngestor


@dataclass
class SVDDecomposition:
    """Singular Value Decomposition (U, S, Vh) of a 2D weight matrix."""

    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor
    original_shape: tuple[int, int]
    original_dtype: torch.dtype
    original_device: torch.device

    def reconstruct(self, top_k: int | None = None) -> torch.Tensor:
        """Reconstruct the 2D matrix from (optionally truncated) SVD components."""
        if top_k is not None:
            k = min(top_k, self.s.numel())
            u_k = self.u[:, :k]
            s_k = self.s[:k]
            vh_k = self.vh[:k, :]
        else:
            u_k, s_k, vh_k = self.u, self.s, self.vh

        recon = (u_k * s_k.unsqueeze(0)) @ vh_k
        return recon.to(device=self.original_device, dtype=self.original_dtype)


@dataclass
class OutlierReport:
    """Summary of statistical weight outliers in a parameter matrix."""

    z_threshold: float
    outlier_count: int
    outlier_ratio: float
    max_abs_value: float
    mean_value: float
    std_value: float
    iqr_fence_upper: float
    iqr_fence_lower: float
    top_coordinates: list[tuple[int, int, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize outlier report to dictionary."""
        return asdict(self)


@dataclass
class TensorAlgebraicProfile:
    """Complete algebraic and statistical profile of a single 2D weight tensor."""

    tensor_path: str
    shape: tuple[int, ...]
    dtype: str
    spectral_norm: float
    frobenius_norm: float
    stable_rank: float
    effective_rank: float
    condition_number: float
    sparsity_ratio: float
    block_sparsity_grid: list[list[float]]
    top_singular_values: list[float]
    outliers: OutlierReport
    is_anomaly_hotspot: bool = False
    anomaly_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize tensor profile to dictionary."""
        return asdict(self)


def compute_svd(
    weight: torch.Tensor,
    top_k: int | None = None,
    use_scipy_truncated: bool = False,
) -> SVDDecomposition:
    """Compute full or truncated SVD of a 2D weight matrix.

    Args:
        weight: 2D tensor of shape `(out_features, in_features)`.
        top_k: Optional number of top singular components to retain.
        use_scipy_truncated: If True and `top_k < min(shape) - 1`, uses `scipy.sparse.linalg.svds`.
    """
    if weight.ndim != 2:
        raise ValueError(f"SVD requires a 2D matrix, got tensor of shape {tuple(weight.shape)}.")

    orig_shape = (int(weight.shape[0]), int(weight.shape[1]))
    orig_dtype = weight.dtype
    orig_device = weight.device

    with torch.no_grad():
        w_f32 = weight.detach().to(device="cpu", dtype=torch.float32)
        min_dim = min(orig_shape)

        if (
            use_scipy_truncated
            and top_k is not None
            and 0 < top_k < min_dim - 1
        ):
            w_np = w_f32.numpy()
            u_np, s_np, vh_np = spla.svds(w_np, k=top_k)
            # `svds` returns singular values in ascending order; sort descending
            order = np.argsort(s_np)[::-1].copy()
            u = torch.from_numpy(u_np[:, order].copy())
            s = torch.from_numpy(s_np[order].copy())
            vh = torch.from_numpy(vh_np[order, :].copy())
        else:
            u, s, vh = torch.linalg.svd(w_f32, full_matrices=False)
            if top_k is not None and top_k < s.numel():
                u = u[:, :top_k]
                s = s[:top_k]
                vh = vh[:top_k, :]

    return SVDDecomposition(
        u=u,
        s=s,
        vh=vh,
        original_shape=orig_shape,
        original_dtype=orig_dtype,
        original_device=orig_device,
    )


def compute_spectral_metrics(singular_values: torch.Tensor) -> dict[str, float]:
    """Compute spectral norm, Frobenius norm, stable rank, effective rank, and condition number."""
    with torch.no_grad():
        s = singular_values.detach().to(torch.float64)
        if s.numel() == 0 or float(s[0].item()) == 0.0:
            return {
                "spectral_norm": 0.0,
                "frobenius_norm": 0.0,
                "stable_rank": 0.0,
                "effective_rank": 0.0,
                "condition_number": 0.0,
            }

        spectral_norm = float(s[0].item())
        frob_sq = float(torch.sum(s * s).item())
        frobenius_norm = float(np.sqrt(frob_sq))
        stable_rank = frob_sq / (spectral_norm**2 + 1e-12)

        # Effective rank (Roy & Vetterli, 2007): exp(Shannon entropy of normalized singular spectrum)
        s_pos = s[s > 1e-12]
        p = s_pos / torch.sum(s_pos)
        entropy = -float(torch.sum(p * torch.log(p + 1e-15)).item())
        effective_rank = float(np.exp(entropy))

        min_sv = float(s[-1].item())
        condition_number = spectral_norm / max(min_sv, 1e-12)

        return {
            "spectral_norm": spectral_norm,
            "frobenius_norm": frobenius_norm,
            "stable_rank": stable_rank,
            "effective_rank": effective_rank,
            "condition_number": condition_number,
        }


def compute_sparsity_map(
    weight: torch.Tensor,
    eps: float = 1e-4,
    grid_size: tuple[int, int] = (4, 4),
) -> tuple[float, list[list[float]]]:
    """Compute overall near-zero sparsity ratio and a 2D block sparsity map."""
    with torch.no_grad():
        w = weight.detach().to(device="cpu", dtype=torch.float32)
        near_zero = (w.abs() < eps).to(torch.float32)
        overall_sparsity = float(near_zero.mean().item())

        if w.ndim != 2:
            return overall_sparsity, [[overall_sparsity]]

        rows, cols = w.shape
        g_r = min(grid_size[0], rows)
        g_c = min(grid_size[1], cols)
        row_splits = torch.chunk(near_zero, g_r, dim=0)
        grid: list[list[float]] = []
        for r_block in row_splits:
            col_splits = torch.chunk(r_block, g_c, dim=1)
            grid.append([round(float(c_block.mean().item()), 4) for c_block in col_splits])

        return overall_sparsity, grid


def detect_weight_outliers(
    weight: torch.Tensor,
    z_threshold: float = 4.0,
    top_n: int = 5,
) -> OutlierReport:
    """Identify extreme outlier weights using z-score thresholding and IQR fences."""
    with torch.no_grad():
        w = weight.detach().to(device="cpu", dtype=torch.float32)
        flat = w.flatten()
        mean_val = float(flat.mean().item())
        std_val = float(flat.std(unbiased=False).item())
        max_abs = float(flat.abs().max().item()) if flat.numel() > 0 else 0.0

        # Compute quartiles on a subsample if tensor is huge to stay fast
        sample = flat if flat.numel() <= 250_000 else flat[:: max(1, flat.numel() // 250_000)]
        q25, q75 = torch.quantile(sample, torch.tensor([0.25, 0.75])).tolist()
        iqr = q75 - q25
        fence_lower = q25 - 1.5 * iqr
        fence_upper = q75 + 1.5 * iqr

        if std_val < 1e-12:
            outlier_mask = torch.zeros_like(flat, dtype=torch.bool)
        else:
            z_scores = (flat - mean_val).abs() / std_val
            outlier_mask = z_scores > z_threshold

        outlier_count = int(outlier_mask.sum().item())
        outlier_ratio = outlier_count / max(1, flat.numel())

        top_coords: list[tuple[int, int, float]] = []
        if w.ndim == 2 and flat.numel() > 0:
            k = min(top_n, flat.numel())
            _, top_flat_idx = torch.topk(flat.abs(), k=k)
            num_cols = w.shape[1]
            for idx_tensor in top_flat_idx:
                idx = int(idx_tensor.item())
                r = idx // num_cols
                c = idx % num_cols
                val = float(w[r, c].item())
                top_coords.append((r, c, val))

        return OutlierReport(
            z_threshold=z_threshold,
            outlier_count=outlier_count,
            outlier_ratio=outlier_ratio,
            max_abs_value=max_abs,
            mean_value=mean_val,
            std_value=std_val,
            iqr_fence_upper=fence_upper,
            iqr_fence_lower=fence_lower,
            top_coordinates=top_coords,
        )


def profile_tensor(
    weight: torch.Tensor,
    tensor_path: str = "weight",
    top_k_sv: int = 8,
    z_threshold: float = 4.0,
    sparsity_eps: float = 1e-4,
) -> TensorAlgebraicProfile:
    """Compute a complete algebraic and anomaly profile for a 2D weight matrix."""
    svd_res = compute_svd(weight)
    metrics = compute_spectral_metrics(svd_res.s)
    sparsity_ratio, block_grid = compute_sparsity_map(weight, eps=sparsity_eps)
    outliers = detect_weight_outliers(weight, z_threshold=z_threshold)

    top_sv = [round(float(v), 6) for v in svd_res.s[:top_k_sv].tolist()]

    # Flag anomaly hotspots: low stable rank (spectral collapse) or high outlier concentration
    reasons: list[str] = []
    if len(top_sv) >= 2 and top_sv[0] > 3.5 * top_sv[1]:
        reasons.append(f"Rank-1 dominance (σ1/σ2 = {top_sv[0] / (top_sv[1] + 1e-9):.2f})")
    if outliers.outlier_ratio > 0.005:
        reasons.append(f"High outlier density ({outliers.outlier_ratio * 100:.2f}%)")
    if metrics["stable_rank"] < 1.8 and min(weight.shape) >= 4:
        reasons.append(f"Low stable rank ({metrics['stable_rank']:.2f})")

    return TensorAlgebraicProfile(
        tensor_path=tensor_path,
        shape=tuple(int(d) for d in weight.shape),
        dtype=str(weight.dtype),
        spectral_norm=metrics["spectral_norm"],
        frobenius_norm=metrics["frobenius_norm"],
        stable_rank=metrics["stable_rank"],
        effective_rank=metrics["effective_rank"],
        condition_number=metrics["condition_number"],
        sparsity_ratio=sparsity_ratio,
        block_sparsity_grid=block_grid,
        top_singular_values=top_sv,
        outliers=outliers,
        is_anomaly_hotspot=len(reasons) > 0,
        anomaly_reason="; ".join(reasons),
    )


def profile_model(
    ingestor: ModelIngestor,
    max_layers: int | None = None,
    top_k_sv: int = 6,
) -> list[TensorAlgebraicProfile]:
    """Profile key attention and FFN projection matrices across layers of a model."""
    profiles: list[TensorAlgebraicProfile] = []
    num_layers = ingestor.get_num_layers()
    if max_layers is not None:
        num_layers = min(num_layers, max_layers)

    if num_layers > 0:
        for layer_idx in range(num_layers):
            try:
                attn = ingestor.unpack_attention(layer_idx)
                for key, tensor in [
                    ("q", attn.w_q),
                    ("k", attn.w_k),
                    ("v", attn.w_v),
                    ("o", attn.w_o),
                ]:
                    profiles.append(
                        profile_tensor(tensor, tensor_path=attn.paths[key], top_k_sv=top_k_sv)
                    )
            except KeyError:
                pass

            try:
                ffn = ingestor.unpack_ffn(layer_idx)
                for key, tensor in [
                    ("gate", ffn.w_gate),
                    ("up", ffn.w_up),
                    ("down", ffn.w_down),
                ]:
                    profiles.append(
                        profile_tensor(tensor, tensor_path=ffn.paths[key], top_k_sv=top_k_sv)
                    )
            except KeyError:
                pass
    else:
        # Fallback for raw safetensors files without standard layer naming
        for path in ingestor.list_tensor_paths():
            tensor = ingestor.get_tensor(path)
            if tensor.ndim == 2:
                profiles.append(profile_tensor(tensor, tensor_path=path, top_k_sv=top_k_sv))

    return profiles

