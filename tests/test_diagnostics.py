"""Unit tests for `weightsurg.diagnostics` (Causal tracing & induction head detection)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.fix_repetition_loop import _induce_degenerate_copy_head
from weightsurg.core.ingestion import ModelIngestor, create_synthetic_qwen2_model
from weightsurg.diagnostics.localizer import FailureLocalizer
from weightsurg.diagnostics.schemas import DiagnosticReport, PatchRecipe


class TestFailureLocalizer(unittest.TestCase):
    """Test causal tracing, induction head detection, and diagnostic report serialization."""

    def setUp(self) -> None:
        self.model = create_synthetic_qwen2_model(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=4,
            seed=42,
        )
        self.ingestor = ModelIngestor.from_pretrained_objects(self.model, model_name="trace-test")

    def test_induction_head_detection_and_causal_localization(self) -> None:
        # Inject degenerate copy loop into Layer 1, Head 2 for token 82 ('R')
        _induce_degenerate_copy_head(
            self.ingestor,
            target_layer=1,
            target_head=2,
            loop_token_id=82,
        )
        localizer = FailureLocalizer(self.ingestor)
        report = localizer.trace_causal_components(prompt="RRRR", target_error="82")

        self.assertEqual(report.primary_layer_idx, 1)
        self.assertEqual(report.primary_head_idx, 2)
        self.assertEqual(report.primary_target_tensor, "model.layers.1.self_attn.o_proj.weight")

        with tempfile.TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / "diag.json"
            report.save(report_file)
            loaded = DiagnosticReport.load(report_file)
            self.assertEqual(loaded.primary_target_tensor, report.primary_target_tensor)
            self.assertEqual(loaded.primary_head_idx, 2)

            recipe = PatchRecipe.load(report_file)
            self.assertEqual(recipe.target_tensor, "model.layers.1.self_attn.o_proj.weight")
            self.assertEqual(recipe.head_idx, 2)


if __name__ == "__main__":
    unittest.main()

