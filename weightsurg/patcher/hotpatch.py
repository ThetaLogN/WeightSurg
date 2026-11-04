"""Zero-Gradient Hot-Patcher, Behavioral Evaluator, and Safetensors/GGUF Exporter."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as save_safetensors_file

from weightsurg.core.ingestion import ModelIngestor
from weightsurg.diagnostics.schemas import PatchRecipe
from weightsurg.surgery.algebraic import synthesize_algebraic_delta
from weightsurg.surgery.llm_guided import LLMWeightProgrammer

DEFAULT_BASELINE_CORPUS: list[str] = [
    "The fundamental theorem of calculus links the concept of differentiating a function with integrating a function.",
    "If a train travels 120 miles in 2 hours, its average speed is 60 miles per hour.",
    "Photosynthesis is the biochemical process by which plants convert light energy into chemical energy.",
    "In computer science, a binary search tree keeps its keys in sorted order for fast lookup.",
]


@dataclass
class ValidationMetrics:
    """Behavioral and perplexity validation report before and after hot-patching."""

    tensor_path: str
    delta_frobenius_norm: float
    delta_spectral_norm: float
    pre_patch_perplexity: float
    post_patch_perplexity: float
    perplexity_drift_ratio: float
    catastrophic_forgetting_detected: bool
    pre_patch_repetition_rate: float
    post_patch_repetition_rate: float
    failure_resolved: bool
    pre_patch_generation: str = ""
    post_patch_generation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation report to dictionary."""
        return asdict(self)


def compute_token_repetition_rate(token_ids: list[int], n: int = 2) -> float:
    """Measure the fraction of repeated consecutive tokens and n-grams in a sequence."""
    if len(token_ids) <= 1:
        return 0.0

    # Immediate adjacent token repetition rate
    adj_repeats = sum(1 for i in range(1, len(token_ids)) if token_ids[i] == token_ids[i - 1])
    adj_rate = adj_repeats / (len(token_ids) - 1)

    if len(token_ids) < n:
        return adj_rate

    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    unique_ngrams = len(set(ngrams))
    ngram_repeat_rate = 1.0 - (unique_ngrams / len(ngrams))
    return float(max(adj_rate, ngram_repeat_rate))


class HotPatcher:
    """Applies in-place zero-gradient weight deltas, validates behavior, and exports models."""

    def __init__(
        self,
        ingestor: ModelIngestor,
        max_perplexity_drift: float = 0.25,
    ) -> None:
        self.ingestor = ingestor
        self.max_perplexity_drift = max_perplexity_drift
        self._applied_deltas: dict[str, torch.Tensor] = {}

    def apply_delta(self, tensor_path: str, delta_w: torch.Tensor) -> torch.Tensor:
        """Apply `W_new = W_old + ΔW` in-place with zero gradient overhead."""
        target = self.ingestor.get_tensor(tensor_path)
        if target.shape != delta_w.shape:
            raise ValueError(
                f"Shape mismatch when patching '{tensor_path}': "
                f"target is {tuple(target.shape)}, delta_w is {tuple(delta_w.shape)}."
            )

        with torch.no_grad():
            delta_cast = delta_w.detach().to(device=target.device, dtype=target.dtype)
            target.add_(delta_cast)

            # Accumulate delta on CPU for exact rollback support
            delta_cpu = delta_cast.to(device="cpu", dtype=torch.float32)
            if tensor_path in self._applied_deltas:
                self._applied_deltas[tensor_path] = self._applied_deltas[tensor_path] + delta_cpu
            else:
                self._applied_deltas[tensor_path] = delta_cpu

        return target

    def revert_tensor(self, tensor_path: str) -> None:
        """Roll back all applied surgical deltas on `tensor_path` (`W_old = W_new - ΔW`)."""
        if tensor_path not in self._applied_deltas:
            raise KeyError(f"No active patch recorded for tensor '{tensor_path}'.")
        target = self.ingestor.get_tensor(tensor_path)
        delta_cpu = self._applied_deltas.pop(tensor_path)
        with torch.no_grad():
            target.sub_(delta_cpu.to(device=target.device, dtype=target.dtype))

    def revert_all(self) -> None:
        """Roll back all applied patches across the model."""
        for path in list(self._applied_deltas.keys()):
            self.revert_tensor(path)

    def compute_perplexity(self, texts: list[str] | None = None) -> float:
        """Compute zero-gradient perplexity on a baseline text dataset."""
        corpus = texts if texts else DEFAULT_BASELINE_CORPUS
        model = self.ingestor.model
        device = next(model.parameters()).device
        tokenizer = self.ingestor.tokenizer
        vocab_size = int(getattr(model.config, "vocab_size", 512))

        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for sample in corpus:
                if tokenizer is not None:
                    enc = tokenizer(sample, return_tensors="pt")
                    input_ids = enc["input_ids"].to(device)
                else:
                    byte_ids = [max(1, b % vocab_size) for b in sample.encode("utf-8")]
                    input_ids = torch.tensor([byte_ids], dtype=torch.long, device=device)

                if input_ids.shape[1] < 2:
                    continue

                outputs = model(input_ids=input_ids)
                shift_logits = outputs.logits[:, :-1, :].contiguous().to(torch.float32)
                shift_labels = input_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                total_nll += float(loss.item())
                total_tokens += int(shift_labels.numel())

        if total_tokens == 0:
            return 1.0
        avg_nll = min(20.0, total_nll / total_tokens)
        return float(math.exp(avg_nll))

    def generate_greedy(
        self,
        prompt: str | list[int],
        max_new_tokens: int = 16,
    ) -> tuple[list[int], str]:
        """Run deterministic greedy decoding under `torch.no_grad()`."""
        model = self.ingestor.model
        device = next(model.parameters()).device
        tokenizer = self.ingestor.tokenizer
        vocab_size = int(getattr(model.config, "vocab_size", 512))

        if isinstance(prompt, list):
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        elif tokenizer is not None:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        else:
            byte_ids = [max(1, b % vocab_size) for b in prompt.encode("utf-8")]
            input_ids = torch.tensor([byte_ids or [1]], dtype=torch.long, device=device)

        generated: list[int] = []
        cur_ids = input_ids
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = model(input_ids=cur_ids).logits[:, -1, :]
                next_tok = int(torch.argmax(logits, dim=-1).item())
                generated.append(next_tok)
                next_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                cur_ids = torch.cat([cur_ids, next_tensor], dim=1)

        if tokenizer is not None:
            decoded = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            decoded = " ".join(f"tok_{t}" for t in generated)
        return generated, decoded

    def execute_recipe(
        self,
        recipe: PatchRecipe,
        baseline_texts: list[str] | None = None,
        eval_max_new_tokens: int = 16,
        auto_rollback_on_regression: bool = False,
    ) -> ValidationMetrics:
        """Synthesize `ΔW` from `recipe`, apply it in-place, and validate behavior & perplexity."""
        target = self.ingestor.get_tensor(recipe.target_tensor)

        inferred_heads: int | None = None
        if hasattr(self.ingestor, "_model") and self.ingestor._model is not None:
            inferred_heads = int(getattr(self.ingestor._model.config, "num_attention_heads", 1))

        # Pre-patch evaluation
        pre_ppl = self.compute_perplexity(baseline_texts)
        eval_prompt: str | list[int] = recipe.prompt or "The quick brown fox jumps over the"
        pre_tokens, pre_gen = self.generate_greedy(eval_prompt, max_new_tokens=eval_max_new_tokens)
        pre_rep = compute_token_repetition_rate(pre_tokens)

        # Synthesize ΔW
        if recipe.mode == "llm_guided":
            programmer = LLMWeightProgrammer(max_relative_spectral_shift=recipe.max_spectral_shift)
            goal = recipe.llm_instruction or (
                f"Suppress repetition loop on target error '{recipe.target_error}'."
            )
            delta_w, _, _ = programmer.synthesize_delta(
                tensor_path=recipe.target_tensor,
                weight=target,
                surgical_goal=goal,
                top_k=max(1, recipe.sub_rank),
            )
        else:
            delta_w = synthesize_algebraic_delta(
                weight=target,
                recipe=recipe,
                inferred_num_heads=inferred_heads,
            )

        with torch.no_grad():
            dw_f32 = delta_w.detach().to(device="cpu", dtype=torch.float32)
            frob_norm = float(torch.linalg.matrix_norm(dw_f32, ord="fro").item())
            spec_norm = float(torch.linalg.matrix_norm(dw_f32, ord=2).item())

        # Apply hot-patch in memory
        self.apply_delta(recipe.target_tensor, delta_w)

        # Post-patch evaluation
        post_ppl = self.compute_perplexity(baseline_texts)
        post_tokens, post_gen = self.generate_greedy(eval_prompt, max_new_tokens=eval_max_new_tokens)
        post_rep = compute_token_repetition_rate(post_tokens)

        ppl_drift = (post_ppl - pre_ppl) / max(pre_ppl, 1e-8)
        forgetting = ppl_drift > self.max_perplexity_drift

        # Determine if the targeted failure was resolved
        failure_resolved = (post_rep < pre_rep) or (post_tokens != pre_tokens)
        if recipe.target_error and recipe.target_error in pre_gen and recipe.target_error not in post_gen:
            failure_resolved = True

        if forgetting and auto_rollback_on_regression:
            self.revert_tensor(recipe.target_tensor)

        return ValidationMetrics(
            tensor_path=recipe.target_tensor,
            delta_frobenius_norm=round(frob_norm, 6),
            delta_spectral_norm=round(spec_norm, 6),
            pre_patch_perplexity=round(pre_ppl, 4),
            post_patch_perplexity=round(post_ppl, 4),
            perplexity_drift_ratio=round(ppl_drift, 4),
            catastrophic_forgetting_detected=forgetting,
            pre_patch_repetition_rate=round(pre_rep, 4),
            post_patch_repetition_rate=round(post_rep, 4),
            failure_resolved=failure_resolved,
            pre_patch_generation=pre_gen,
            post_patch_generation=post_gen,
        )

    def export_safetensors(self, output_path: str | Path) -> Path:
        """Export the patched model to disk in standard `safetensors` format."""
        out = Path(output_path)
        if out.suffix == ".safetensors":
            out.parent.mkdir(parents=True, exist_ok=True)
            state_dict = {
                k: v.detach().to("cpu").contiguous()
                for k, v in self.ingestor.model.state_dict().items()
            }
            save_safetensors_file(state_dict, str(out))
            return out

        out.mkdir(parents=True, exist_ok=True)
        self.ingestor.model.save_pretrained(str(out), safe_serialization=True)
        if self.ingestor.tokenizer is not None:
            self.ingestor.tokenizer.save_pretrained(str(out))
        return out

    def export_gguf(self, output_path: str | Path, architecture: str = "qwen2") -> Path:
        """Export the patched model weights into a standard GGUF v3 binary file for llama.cpp/Ollama."""
        out = Path(output_path)
        if out.suffix != ".gguf":
            out = out / "model-patched.gguf" if out.is_dir() or not out.suffix else out.with_suffix(".gguf")
        out.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            k: v.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for k, v in self.ingestor.model.named_parameters()
        }

        metadata_kv: list[tuple[str, str]] = [
            ("general.architecture", architecture),
            ("general.name", "weightsurg-patched-model"),
            ("weightsurg.patched_tensors", json.dumps(list(self._applied_deltas.keys()))),
        ]

        def _pack_string(s: str) -> bytes:
            encoded = s.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        ALIGNMENT = 32

        with out.open("wb") as f:
            # 1. GGUF Header: magic ("GGUF"), version (3), tensor_count (uint64), kv_count (uint64)
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", len(state_dict)))
            f.write(struct.pack("<Q", len(metadata_kv)))

            # 2. Metadata KV pairs (string type = 8 in GGUF specification)
            for key, val in metadata_kv:
                f.write(_pack_string(key))
                f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
                f.write(_pack_string(val))

            # 3. Tensor Info Headers (F32 type = 0 in GGUF specification)
            offset = 0
            tensor_arrays: list[np.ndarray] = []
            for name, tensor in state_dict.items():
                arr = tensor.numpy()
                tensor_arrays.append(arr)
                f.write(_pack_string(name))
                n_dims = arr.ndim
                f.write(struct.pack("<I", n_dims))
                # GGUF dimensions are written in reverse (innermost first)
                for dim in reversed(arr.shape):
                    f.write(struct.pack("<Q", int(dim)))
                f.write(struct.pack("<I", 0))  # GGML_TYPE_F32
                f.write(struct.pack("<Q", offset))
                nbytes = arr.nbytes
                padded = ((nbytes + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
                offset += padded

            # Pad header section to 32-byte alignment
            pos = f.tell()
            header_pad = (ALIGNMENT - (pos % ALIGNMENT)) % ALIGNMENT
            if header_pad:
                f.write(b"\x00" * header_pad)

            # 4. Raw Tensor Payloads
            for arr in tensor_arrays:
                raw_bytes = arr.tobytes()
                f.write(raw_bytes)
                pad = (ALIGNMENT - (len(raw_bytes) % ALIGNMENT)) % ALIGNMENT
                if pad:
                    f.write(b"\x00" * pad)

        return out

