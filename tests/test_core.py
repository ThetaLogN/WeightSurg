"""Unit tests for `weightsurg.core` (Model ingestion, tensor slicing, SVD & profiling)."""

from __future__ import annotations

import unittest

import torch

from weightsurg.core.ingestion import ModelIngestor, create_synthetic_qwen2_model
from weightsurg.core.profiling import (
    compute_sparsity_map,
    compute_spectral_metrics,
    compute_svd,
    detect_weight_outliers,
    profile_model,
    profile_tensor,
)


class TestCoreIngestionAndProfiling(unittest.TestCase):
    """Test model unpacking, head-level tensor slicing, and algebraic profiling."""

    def setUp(self) -> None:
        self.model = create_synthetic_qwen2_model(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            seed=7,
        )
        self.ingestor = ModelIngestor.from_pretrained_objects(self.model, model_name="test-qwen2")

    def test_zero_gradient_enforcement(self) -> None:
        for param in self.ingestor.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_attention_and_ffn_unpacking_and_slicing(self) -> None:
        attn = self.ingestor.unpack_attention(0)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.num_kv_heads, 2)
        self.assertEqual(attn.head_dim, 8)

        # Q and O have 4 heads of dim 8 -> shape 32x32
        q_head0 = attn.get_head_slice("q", 0)
        o_head3 = attn.get_head_slice("o", 3)
        self.assertEqual(tuple(q_head0.shape), (8, 32))
        self.assertEqual(tuple(o_head3.shape), (32, 8))
        self.assertTrue(torch.allclose(q_head0, attn.w_q[0:8, :]))
        self.assertTrue(torch.allclose(o_head3, attn.w_o[:, 24:32]))

        # K and V have 2 KV heads of dim 8 -> shape 16x32
        k_head1 = attn.get_head_slice("k", 1)
        self.assertEqual(tuple(k_head1.shape), (8, 32))
        with self.assertRaises(ValueError):
            attn.get_head_slice("k", 2)

        ffn = self.ingestor.unpack_ffn(0)
        self.assertEqual(tuple(ffn.w_gate.shape), (64, 32))
        self.assertEqual(tuple(ffn.w_up.shape), (64, 32))
        self.assertEqual(tuple(ffn.w_down.shape), (32, 64))

    def test_svd_exact_and_truncated_reconstruction(self) -> None:
        torch.manual_seed(123)
        w = torch.randn(24, 16)
        svd_full = compute_svd(w)
        recon_full = svd_full.reconstruct()
        self.assertTrue(torch.allclose(w, recon_full, atol=1e-5))

        # Low-rank matrix of exact rank 3
        u_true = torch.randn(24, 3)
        v_true = torch.randn(3, 16)
        w_rank3 = u_true @ v_true
        svd_trunc = compute_svd(w_rank3, top_k=3, use_scipy_truncated=True)
        recon_rank3 = svd_trunc.reconstruct(top_k=3)
        self.assertTrue(torch.allclose(w_rank3, recon_rank3, atol=1e-4))

    def test_spectral_metrics_and_outliers(self) -> None:
        w = torch.zeros(16, 16)
        w[0, 0] = 10.0
        w[1, 1] = 2.0
        svd_res = compute_svd(w)
        metrics = compute_spectral_metrics(svd_res.s)
        self.assertAlmostEqual(metrics["spectral_norm"], 10.0, places=5)
        self.assertAlmostEqual(metrics["frobenius_norm"], (104.0) ** 0.5, places=5)
        self.assertAlmostEqual(metrics["stable_rank"], 104.0 / 100.0, places=5)

        sparsity, grid = compute_sparsity_map(w, eps=1e-4, grid_size=(2, 2))
        self.assertAlmostEqual(sparsity, (256 - 2) / 256, places=5)
        self.assertEqual(len(grid), 2)

        outliers = detect_weight_outliers(w, z_threshold=3.0)
        self.assertGreaterEqual(outliers.outlier_count, 1)
        self.assertEqual(outliers.top_coordinates[0][0], 0)
        self.assertEqual(outliers.top_coordinates[0][1], 0)
        self.assertAlmostEqual(outliers.top_coordinates[0][2], 10.0, places=5)

    def test_profile_tensor_and_model(self) -> None:
        attn = self.ingestor.unpack_attention(0)
        prof = profile_tensor(attn.w_o, tensor_path=attn.paths["o"])
        self.assertEqual(prof.shape, (32, 32))
        self.assertGreater(prof.spectral_norm, 0.0)

        all_profiles = profile_model(self.ingestor)
        # 2 layers * (4 attn + 3 ffn) = 14 matrices
        self.assertEqual(len(all_profiles), 14)


if __name__ == "__main__":
    unittest.main()

