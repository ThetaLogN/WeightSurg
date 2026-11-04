"""LLM-Guided Weight Programming (In-Context Surgical Synthesis).

Serializes low-dimensional sub-matrices or top-k SVD components into structured
tabular/JSON representations, formulates meta-prompts for frontier reasoning models
(e.g. GPT-4o, Claude 3.5 Sonnet), parses returned numerical modifications,
reconstructs ΔW, and validates spectral stability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from weightsurg.core.profiling import SVDDecomposition, compute_svd


@dataclass
class StabilityValidationResult:
    """Validation report for an LLM-proposed weight delta ΔW."""

    is_valid: bool
    has_nan_or_inf: bool
    relative_spectral_shift: float
    relative_frobenius_shift: float
    was_clamped: bool
    message: str


class LLMWeightProgrammer:
    """Serializes SVD subspaces for frontier LLMs and reconstructs validated weight deltas."""

    def __init__(
        self,
        llm_client: Callable[[str], str] | None = None,
        max_relative_spectral_shift: float = 0.5,
    ) -> None:
        """Initialize the LLM weight programmer.

        Args:
            llm_client: Optional callable taking a meta-prompt string and returning the
                frontier model's text response. If None, uses a deterministic heuristic
                surgical responder when invoked offline.
            max_relative_spectral_shift: Maximum allowed `||ΔW||_2 / ||W||_2` before clamping.
        """
        self.llm_client = llm_client
        self.max_relative_spectral_shift = max_relative_spectral_shift

    @staticmethod
    def serialize_svd_subspace(
        weight: torch.Tensor,
        top_k: int = 4,
        vector_preview_dim: int = 8,
    ) -> tuple[SVDDecomposition, dict[str, Any], str]:
        """Serialize the top-k SVD components into both a structured dict and tabular text."""
        svd_res = compute_svd(weight, top_k=top_k)
        k = svd_res.s.numel()
        d_prev = min(vector_preview_dim, svd_res.u.shape[0], svd_res.vh.shape[1])

        modes: list[dict[str, Any]] = []
        table_lines = [
            "| Mode idx | Singular Value (σ_i) | Energy Share | u_i[:preview] | v_i[:preview] |",
            "|---:|---:|---:|---|---|",
        ]
        total_energy = float(torch.sum(svd_res.s**2).item()) + 1e-12

        for i in range(k):
            sigma_i = round(float(svd_res.s[i].item()), 6)
            energy_share = round(float((svd_res.s[i].item() ** 2) / total_energy), 4)
            u_slice = [round(float(x), 5) for x in svd_res.u[:d_prev, i].tolist()]
            v_slice = [round(float(x), 5) for x in svd_res.vh[i, :d_prev].tolist()]
            modes.append(
                {
                    "index": i,
                    "sigma": sigma_i,
                    "energy_share": energy_share,
                    "u_preview": u_slice,
                    "v_preview": v_slice,
                }
            )
            table_lines.append(
                f"| {i} | {sigma_i:.6f} | {energy_share * 100:.2f}% | {u_slice} | {v_slice} |"
            )

        payload = {
            "matrix_shape": list(svd_res.original_shape),
            "top_k": k,
            "modes": modes,
        }
        tabular_str = "\n".join(table_lines)
        return svd_res, payload, tabular_str

    def build_meta_prompt(
        self,
        tensor_path: str,
        weight: torch.Tensor,
        surgical_goal: str,
        top_k: int = 4,
    ) -> tuple[SVDDecomposition, str]:
        """Formulate a structured meta-prompt for a frontier reasoning model."""
        svd_res, payload, tabular_str = self.serialize_svd_subspace(weight, top_k=top_k)
        prompt = f"""You are a Mechanistic Interpretability & Weight-Space Surgery Engine.
Target Tensor: `{tensor_path}`
Matrix Shape: {list(svd_res.original_shape)}
Surgical Objective: {surgical_goal}

### Top-{top_k} Singular Value Decomposition Summary
{tabular_str}

### Structured Subspace Payload
```json
{json.dumps(payload, indent=2)}
```

### Task
Propose closed-form modifications to the top-{top_k} singular modes to achieve the surgical objective while preserving orthogonal subspace capabilities.
Return ONLY a valid JSON object matching this schema:
```json
{{
  "rationale": "Brief mechanistic explanation of the singular value/vector edit",
  "modified_sigmas": [float, ...],
  "u_rotations": {{
    "0": [float, ...]
  }}
}}
```
"""
        return svd_res, prompt

    @staticmethod
    def parse_llm_response(response_text: str, current_sigmas: list[float]) -> dict[str, Any]:
        """Extract and validate numerical SVD edits from an LLM response string."""
        # Look for fenced ```json ... ``` blocks first, then any {...} object
        fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, flags=re.DOTALL)
        candidates = fenced if fenced else re.findall(r"(\{.*\})", response_text, flags=re.DOTALL)

        for cand in candidates:
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict) and "modified_sigmas" in parsed:
                    sigmas = [float(x) for x in parsed["modified_sigmas"]]
                    if len(sigmas) != len(current_sigmas):
                        # Pad or truncate to match top_k
                        sigmas = (sigmas + current_sigmas)[: len(current_sigmas)]
                    parsed["modified_sigmas"] = sigmas
                    return parsed
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

        raise ValueError("Could not parse valid 'modified_sigmas' JSON from LLM response.")

    def validate_and_stabilize_delta(
        self,
        original_weight: torch.Tensor,
        delta_w: torch.Tensor,
    ) -> tuple[torch.Tensor, StabilityValidationResult]:
        """Validate numerical stability of `ΔW` and clamp spectral norm if necessary."""
        with torch.no_grad():
            has_nan_or_inf = bool(torch.isnan(delta_w).any() or torch.isinf(delta_w).any())
            if has_nan_or_inf:
                zero_delta = torch.zeros_like(original_weight)
                return zero_delta, StabilityValidationResult(
                    is_valid=False,
                    has_nan_or_inf=True,
                    relative_spectral_shift=0.0,
                    relative_frobenius_shift=0.0,
                    was_clamped=True,
                    message="Rejected ΔW containing NaN or Inf entries.",
                )

            w_f32 = original_weight.detach().to(device="cpu", dtype=torch.float32)
            dw_f32 = delta_w.detach().to(device="cpu", dtype=torch.float32)

            w_spec = float(torch.linalg.matrix_norm(w_f32, ord=2).item()) + 1e-12
            dw_spec = float(torch.linalg.matrix_norm(dw_f32, ord=2).item())
            w_frob = float(torch.linalg.matrix_norm(w_f32, ord="fro").item()) + 1e-12
            dw_frob = float(torch.linalg.matrix_norm(dw_f32, ord="fro").item())

            rel_spec = dw_spec / w_spec
            rel_frob = dw_frob / w_frob
            was_clamped = False

            if rel_spec > self.max_relative_spectral_shift and dw_spec > 0:
                scale = self.max_relative_spectral_shift / rel_spec
                dw_f32 = dw_f32 * scale
                rel_spec = self.max_relative_spectral_shift
                rel_frob = rel_frob * scale
                was_clamped = True

            stabilized = dw_f32.to(device=original_weight.device, dtype=original_weight.dtype)
            return stabilized, StabilityValidationResult(
                is_valid=True,
                has_nan_or_inf=False,
                relative_spectral_shift=round(rel_spec, 6),
                relative_frobenius_shift=round(rel_frob, 6),
                was_clamped=was_clamped,
                message=(
                    "Clamped ΔW to spectral stability bound."
                    if was_clamped
                    else "ΔW passed spectral and numerical stability checks."
                ),
            )

    def synthesize_delta(
        self,
        tensor_path: str,
        weight: torch.Tensor,
        surgical_goal: str,
        top_k: int = 4,
    ) -> tuple[torch.Tensor, StabilityValidationResult, dict[str, Any]]:
        """Run end-to-end LLM-guided weight programming on a 2D matrix (or sub-matrix)."""
        svd_res, meta_prompt = self.build_meta_prompt(
            tensor_path=tensor_path,
            weight=weight,
            surgical_goal=surgical_goal,
            top_k=top_k,
        )
        orig_sigmas = [float(x) for x in svd_res.s.tolist()]

        if self.llm_client is not None:
            raw_response = self.llm_client(meta_prompt)
        else:
            # Deterministic fallback surgical synthesis: damp dominant singular component by 75%
            proposed = [
                round(orig_sigmas[0] * 0.25, 6),
                *[round(s, 6) for s in orig_sigmas[1:]],
            ]
            raw_response = json.dumps(
                {
                    "rationale": (
                        "Attenuate dominant singular vector σ_0 by 75% to suppress "
                        "degenerate identity copying on recurrent delimiter tokens."
                    ),
                    "modified_sigmas": proposed,
                }
            )

        parsed = self.parse_llm_response(raw_response, orig_sigmas)
        new_sigmas = torch.tensor(parsed["modified_sigmas"], dtype=svd_res.s.dtype)

        with torch.no_grad():
            u_mod = svd_res.u.clone()
            u_rotations = parsed.get("u_rotations", {})
            if isinstance(u_rotations, dict):
                for idx_str, vec_vals in u_rotations.items():
                    idx = int(idx_str)
                    if 0 <= idx < u_mod.shape[1] and isinstance(vec_vals, list) and vec_vals:
                        vec_t = torch.tensor(vec_vals, dtype=u_mod.dtype)
                        n = min(vec_t.numel(), u_mod.shape[0])
                        u_mod[:n, idx] = vec_t[:n]
                        norm = torch.linalg.norm(u_mod[:, idx])
                        if float(norm.item()) > 1e-9:
                            u_mod[:, idx] = u_mod[:, idx] / norm

            # Reconstruct ΔW = U_mod @ diag(S_new) @ Vh - U @ diag(S_old) @ Vh
            orig_subspace = (svd_res.u * svd_res.s.unsqueeze(0)) @ svd_res.vh
            new_subspace = (u_mod * new_sigmas.unsqueeze(0)) @ svd_res.vh
            raw_delta = (new_subspace - orig_subspace).to(
                device=weight.device, dtype=weight.dtype
            )

        stabilized_delta, validation = self.validate_and_stabilize_delta(weight, raw_delta)
        metadata = {
            "meta_prompt": meta_prompt,
            "parsed_edit": parsed,
            "original_sigmas": orig_sigmas,
            "validation": validation.__dict__,
        }
        return stabilized_delta, validation, metadata

