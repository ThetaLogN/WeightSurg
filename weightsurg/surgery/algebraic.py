"""Algebraic / Closed-Form Weight Editing Engine.

Implements zero-gradient rank-one (ROME-style), low-rank (MEMIT-style), and
singular-subspace surgical updates in closed form without iterative backpropagation.
"""

from __future__ import annotations

import torch

from weightsurg.core.profiling import compute_svd
from weightsurg.diagnostics.schemas import PatchRecipe


def compute_rank1_delta(
    weight: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    covariance_inv: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute a closed-form rank-1 associative weight update (ROME-inspired).

    Solves `(W + ΔW) k = v` with minimal Frobenius norm:
        `ΔW = (v - W k) (k^T / ||k||^2)`

    When an optional inverse key covariance matrix `C^{-1}` is supplied, computes:
        `ΔW = (v - W k) ((C^{-1} k)^T / (k^T C^{-1} k))`

    Args:
        weight: 2D weight matrix `W` of shape `(d_out, d_in)`.
        key: 1D input key vector `k` of shape `(d_in,)`.
        value: 1D target output vector `v` of shape `(d_out,)`.
        covariance_inv: Optional 2D inverse covariance matrix of shape `(d_in, d_in)`.
        eps: Numerical stability floor for the denominator.

    Returns:
        Rank-1 weight delta `ΔW` of shape `(d_out, d_in)` matching `weight` dtype/device.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(weight.shape)}.")

    d_out, d_in = weight.shape
    k = key.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    v = value.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    w = weight.detach().to(device="cpu", dtype=torch.float64)

    if k.shape[0] != d_in:
        raise ValueError(f"Key dimension {k.shape[0]} does not match weight in_features {d_in}.")
    if v.shape[0] != d_out:
        raise ValueError(f"Value dimension {v.shape[0]} does not match weight out_features {d_out}.")

    with torch.no_grad():
        residual = v - (w @ k)  # shape: (d_out,)
        if covariance_inv is not None:
            c_inv = covariance_inv.detach().to(device="cpu", dtype=torch.float64)
            right_vec = c_inv @ k
            denom = float(torch.dot(k, right_vec).item())
        else:
            right_vec = k
            denom = float(torch.dot(k, k).item())

        if abs(denom) < eps:
            raise ValueError("Key vector has near-zero norm; cannot compute rank-1 update.")

        delta_w = torch.outer(residual, right_vec / denom)
        return delta_w.to(device=weight.device, dtype=weight.dtype)


def compute_low_rank_delta(
    weight: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    ridge_lambda: float = 1e-6,
) -> torch.Tensor:
    """Compute a closed-form low-rank multi-association update (MEMIT-inspired).

    Given key matrix `K` of shape `(d_in, r)` and target value matrix `V` of shape `(d_out, r)`,
    computes:
        `ΔW = (V - W K) K^T (K K^T + λ I)^{-1}`
    or via exact pseudoinverse when `ridge_lambda == 0.0`.
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {tuple(weight.shape)}.")

    d_out, d_in = weight.shape
    w = weight.detach().to(device="cpu", dtype=torch.float64)
    k_mat = keys.detach().to(device="cpu", dtype=torch.float64)
    v_mat = values.detach().to(device="cpu", dtype=torch.float64)

    if k_mat.ndim == 1:
        k_mat = k_mat.unsqueeze(1)
    if v_mat.ndim == 1:
        v_mat = v_mat.unsqueeze(1)

    if k_mat.shape[0] != d_in:
        raise ValueError(f"Keys shape {tuple(k_mat.shape)} incompatible with d_in={d_in}.")
    if v_mat.shape[0] != d_out or v_mat.shape[1] != k_mat.shape[1]:
        raise ValueError(
            f"Values shape {tuple(v_mat.shape)} incompatible with d_out={d_out} and rank={k_mat.shape[1]}."
        )

    with torch.no_grad():
        residuals = v_mat - (w @ k_mat)  # (d_out, r)
        if ridge_lambda <= 0.0:
            k_pinv = torch.linalg.pinv(k_mat)  # (r, d_in)
            delta_w = residuals @ k_pinv
        else:
            # Solve via Woodbury / Gram matrix of size (r, r) for efficiency when r << d_in
            r = k_mat.shape[1]
            gram = (k_mat.T @ k_mat) + ridge_lambda * torch.eye(r, dtype=torch.float64)
            coeff = torch.linalg.solve(gram, k_mat.T)  # (r, d_in)
            delta_w = residuals @ coeff

        return delta_w.to(device=weight.device, dtype=weight.dtype)


def compute_subspace_damping_delta(
    weight: torch.Tensor,
    singular_indices: list[int] | None = None,
    sub_rank: int = 1,
    damping_factor: float = 0.2,
    head_idx: int | None = None,
    num_heads: int | None = None,
    proj_type: str = "o",
) -> torch.Tensor:
    """Compute a closed-form singular-subspace damping delta using rank-1 associative edits.

    For each targeted singular mode `i` of `W` (or of the attention head slice `W^{(h)}`),
    uses the right singular vector `k_i = v_i` (`||k_i||^2 = 1`) and damped target left
    singular image `v_target = damping_factor * σ_i * u_i` in the closed-form equation:
        `ΔW_i = (v_target - W k_i) (k_i^T / ||k_i||^2) = -(1 - damping_factor) σ_i u_i v_i^T`

    Args:
        weight: Full 2D weight tensor (e.g., `self_attn.o_proj.weight`).
        singular_indices: Specific singular value indices to damp (defaults to `0..sub_rank-1`).
        sub_rank: Number of dominant singular components if `singular_indices` is None.
        damping_factor: Multiplicative retention factor in `[0.0, 1.0]` (`0.0` = full ablation
            of the singular mode, `1.0` = no change).
        head_idx: Optional attention head index to localize the edit to a single head slice.
        num_heads: Total number of heads in `weight` when `head_idx` is specified.
        proj_type: `'o'` (columns sliced per head) or `'q'`/`'k'`/`'v'` (rows sliced per head).
    """
    if not (0.0 <= damping_factor <= 2.0):
        raise ValueError(f"damping_factor must be in [0.0, 2.0], got {damping_factor}.")

    indices = singular_indices if singular_indices else list(range(max(1, sub_rank)))

    with torch.no_grad():
        full_delta = torch.zeros_like(weight, device="cpu", dtype=torch.float32)
        w_cpu = weight.detach().to(device="cpu", dtype=torch.float32)

        if head_idx is not None and num_heads is not None and num_heads > 1:
            if proj_type == "o":
                head_dim = w_cpu.shape[1] // num_heads
                c_start = head_idx * head_dim
                c_end = c_start + head_dim
                sub_w = w_cpu[:, c_start:c_end]
            else:
                head_dim = w_cpu.shape[0] // num_heads
                r_start = head_idx * head_dim
                r_end = r_start + head_dim
                sub_w = w_cpu[r_start:r_end, :]
        else:
            sub_w = w_cpu

        svd_res = compute_svd(sub_w)
        sub_delta = torch.zeros_like(sub_w)

        for idx in indices:
            if 0 <= idx < svd_res.s.numel():
                # Right singular vector k_i = Vh[idx, :], left output W k_i = σ_i u_i
                k_i = svd_res.vh[idx, :]
                sigma_i = svd_res.s[idx]
                u_i = svd_res.u[:, idx]
                v_target = damping_factor * sigma_i * u_i
                # Apply the exact rank-1 formula ΔW_i = (v_target - W k_i) (k_i^T / ||k_i||^2)
                mode_delta = compute_rank1_delta(sub_w, k_i, v_target)
                sub_delta = sub_delta + mode_delta

        if head_idx is not None and num_heads is not None and num_heads > 1:
            if proj_type == "o":
                head_dim = w_cpu.shape[1] // num_heads
                c_start = head_idx * head_dim
                c_end = c_start + head_dim
                full_delta[:, c_start:c_end] = sub_delta
            else:
                head_dim = w_cpu.shape[0] // num_heads
                r_start = head_idx * head_dim
                r_end = r_start + head_dim
                full_delta[r_start:r_end, :] = sub_delta
        else:
            full_delta = sub_delta

        return full_delta.to(device=weight.device, dtype=weight.dtype)


def synthesize_algebraic_delta(
    weight: torch.Tensor,
    recipe: PatchRecipe,
    inferred_num_heads: int | None = None,
) -> torch.Tensor:
    """Dispatch and compute the closed-form weight delta `ΔW` for a `PatchRecipe`."""
    num_heads = recipe.num_heads or inferred_num_heads
    proj_type = "o" if "o_proj" in recipe.target_tensor or "down_proj" in recipe.target_tensor else "q"

    if recipe.mode == "rank1_update":
        if recipe.key_vector is not None and recipe.value_vector is not None:
            key = torch.tensor(recipe.key_vector, dtype=torch.float32)
            value = torch.tensor(recipe.value_vector, dtype=torch.float32)
            return compute_rank1_delta(weight, key, value)
        # If no explicit k/v vectors are in the JSON recipe, perform rank-1 dominant mode damping
        return compute_subspace_damping_delta(
            weight=weight,
            singular_indices=[0],
            sub_rank=1,
            damping_factor=recipe.damping_factor,
            head_idx=recipe.head_idx,
            num_heads=num_heads,
            proj_type=proj_type,
        )

    if recipe.mode == "low_rank_update":
        if recipe.key_vector is not None and recipe.value_vector is not None:
            keys = torch.tensor(recipe.key_vector, dtype=torch.float32)
            values = torch.tensor(recipe.value_vector, dtype=torch.float32)
            return compute_low_rank_delta(weight, keys, values)
        return compute_subspace_damping_delta(
            weight=weight,
            singular_indices=recipe.singular_indices,
            sub_rank=recipe.sub_rank,
            damping_factor=recipe.damping_factor,
            head_idx=recipe.head_idx,
            num_heads=num_heads,
            proj_type=proj_type,
        )

    if recipe.mode in ("subspace_damping", "head_attenuation"):
        indices = (
            recipe.singular_indices
            if recipe.mode == "subspace_damping"
            else list(range(min(weight.shape)))
        )
        return compute_subspace_damping_delta(
            weight=weight,
            singular_indices=indices,
            sub_rank=recipe.sub_rank,
            damping_factor=recipe.damping_factor,
            head_idx=recipe.head_idx,
            num_heads=num_heads,
            proj_type=proj_type,
        )

    raise ValueError(f"Unsupported algebraic patch mode: '{recipe.mode}'.")

