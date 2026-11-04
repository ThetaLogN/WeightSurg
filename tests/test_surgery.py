"""Unit tests for `weightsurg.surgery` mathematical guarantees."""

from __future__ import annotations

import unittest

import torch

from weightsurg.core.profiling import compute_svd
from weightsurg.surgery.algebraic import (
    compute_low_rank_delta,
    compute_rank1_delta,
    compute_subspace_damping_delta,
)
from weightsurg.surgery.llm_guided import LLMWeightProgrammer


class TestSurgicalSynthesis(unittest.TestCase):
    """Test mathematical invariants of closed-form and LLM-guided weight updates."""

    def test_rank1_associative_and_orthogonal_guarantees(self) -> None:
        torch.manual_seed(11)
        w = torch.randn(32, 16)
        k = torch.randn(16)
        v = torch.randn(32)

        delta_w = compute_rank1_delta(w, k, v)
        w_new = w + delta_w

        # Guarantee 1: Exact associative mapping (W + ΔW) k == v
        mapped = w_new @ k
        self.assertTrue(torch.allclose(mapped, v, atol=1e-5))

        # Guarantee 2: Rank(ΔW) == 1
        sv = torch.linalg.svdvals(delta_w)
        self.assertGreater(float(sv[0].item()), 0.0)
        self.assertAlmostEqual(float(sv[1].item()), 0.0, places=5)

        # Guarantee 3: Orthogonal vectors k_perp (k_perp^T k == 0) are completely unperturbed
        rand_vec = torch.randn(16)
        k_perp = rand_vec - (torch.dot(rand_vec, k) / torch.dot(k, k)) * k
        self.assertTrue(torch.allclose(w_new @ k_perp, w @ k_perp, atol=1e-5))

    def test_low_rank_multi_association_guarantee(self) -> None:
        torch.manual_seed(22)
        w = torch.randn(24, 16)
        # 3 linearly independent keys in R^16 and targets in R^24
        keys = torch.randn(16, 3)
        values = torch.randn(24, 3)

        delta_w = compute_low_rank_delta(w, keys, values, ridge_lambda=0.0)
        w_new = w + delta_w
        self.assertTrue(torch.allclose(w_new @ keys, values, atol=1e-4))

    def test_head_slice_subspace_damping_isolation(self) -> None:
        torch.manual_seed(33)
        # 4 heads of dimension 8 -> (32, 32) o_proj weight
        w_o = torch.randn(32, 32)
        orig_head2 = w_o[:, 16:24].clone()
        orig_sv = compute_svd(orig_head2).s

        delta_w = compute_subspace_damping_delta(
            weight=w_o,
            singular_indices=[0],
            sub_rank=1,
            damping_factor=0.25,
            head_idx=2,
            num_heads=4,
            proj_type="o",
        )

        # Non-target heads (0, 1, 3) must have strictly zero delta
        self.assertEqual(float(delta_w[:, 0:16].abs().max().item()), 0.0)
        self.assertEqual(float(delta_w[:, 24:32].abs().max().item()), 0.0)

        # Target head 2's dominant singular value σ_0 is scaled by 0.25, leaving others intact
        patched_head2 = orig_head2 + delta_w[:, 16:24]
        new_sv = compute_svd(patched_head2).s
        expected_sv = sorted([float(orig_sv[0].item() * 0.25), *[float(x) for x in orig_sv[1:]]], reverse=True)
        self.assertTrue(
            torch.allclose(new_sv, torch.tensor(expected_sv, dtype=new_sv.dtype), atol=1e-4)
        )

    def test_llm_guided_weight_programming_and_stability(self) -> None:
        torch.manual_seed(44)
        w = torch.randn(16, 16)
        programmer = LLMWeightProgrammer(max_relative_spectral_shift=0.3)

        delta_w, validation, meta = programmer.synthesize_delta(
            tensor_path="model.layers.0.self_attn.o_proj.weight",
            weight=w,
            surgical_goal="Suppress dominant copy mode",
            top_k=3,
        )
        self.assertTrue(validation.is_valid)
        self.assertFalse(validation.has_nan_or_inf)
        self.assertLessEqual(validation.relative_spectral_shift, 0.3 + 1e-5)
        self.assertIn("modified_sigmas", meta["parsed_edit"])

        # Verify NaN delta is rejected and zeroed safely
        nan_delta = torch.full_like(w, float("nan"))
        safe_delta, nan_val = programmer.validate_and_stabilize_delta(w, nan_delta)
        self.assertFalse(nan_val.is_valid)
        self.assertTrue(nan_val.has_nan_or_inf)
        self.assertEqual(float(safe_delta.abs().max().item()), 0.0)


if __name__ == "__main__":
    unittest.main()

