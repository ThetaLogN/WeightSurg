"""Unit tests for `weightsurg.patcher`, `weightsurg.cli`, and the end-to-end demo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from examples.fix_repetition_loop import run_demo
from weightsurg.cli import main as cli_main
from weightsurg.core.ingestion import ModelIngestor, create_synthetic_qwen2_model
from weightsurg.patcher.hotpatch import HotPatcher


class TestPatcherCLIAndEndToEnd(unittest.TestCase):
    """Test hot-patching, rollback, Safetensors/GGUF export, CLI commands, and E2E demo."""

    def test_hotpatch_apply_and_revert(self) -> None:
        model = create_synthetic_qwen2_model(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            seed=99,
        )
        ingestor = ModelIngestor.from_pretrained_objects(model)
        patcher = HotPatcher(ingestor)

        path = "model.layers.0.self_attn.o_proj.weight"
        orig = ingestor.get_tensor(path).detach().clone()
        delta = torch.ones_like(orig) * 0.05

        patcher.apply_delta(path, delta)
        self.assertTrue(torch.allclose(ingestor.get_tensor(path), orig + delta, atol=1e-6))

        patcher.revert_tensor(path)
        self.assertTrue(torch.allclose(ingestor.get_tensor(path), orig, atol=1e-6))

    def test_end_to_end_demo_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_demo(output_dir=tmpdir, offline=True)
            metrics = res["metrics"]
            self.assertLess(metrics.post_patch_repetition_rate, metrics.pre_patch_repetition_rate)
            self.assertEqual(metrics.post_patch_repetition_rate, 0.0)
            self.assertFalse(metrics.catastrophic_forgetting_detected)
            self.assertTrue(Path(res["safetensors_path"]).exists())
            gguf_path = Path(res["gguf_path"])
            self.assertTrue(gguf_path.exists())
            with gguf_path.open("rb") as f:
                magic = f.read(4)
                self.assertEqual(magic, b"GGUF")

    def test_cli_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            inspect_json = tmp / "inspect.json"
            trace_json = tmp / "trace_report.json"
            patched_st = tmp / "patched.safetensors"

            rc_inspect = cli_main(
                ["inspect", "synthetic", "--max-layers", "2", "--json-out", str(inspect_json)]
            )
            self.assertEqual(rc_inspect, 0)
            self.assertTrue(inspect_json.exists())

            rc_trace = cli_main(
                [
                    "trace",
                    "--model",
                    "synthetic",
                    "--prompt",
                    "RRRR",
                    "--target-error",
                    "82",
                    "--output",
                    str(trace_json),
                ]
            )
            self.assertEqual(rc_trace, 0)
            self.assertTrue(trace_json.exists())

            rc_patch = cli_main(
                [
                    "patch",
                    "--model",
                    "synthetic",
                    "--spec",
                    str(trace_json),
                    "--output",
                    str(patched_st),
                ]
            )
            self.assertEqual(rc_patch, 0)
            self.assertTrue(patched_st.exists())


if __name__ == "__main__":
    unittest.main()
