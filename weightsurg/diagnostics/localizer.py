"""Failure Localizer: Causal Tracing & Induction/Copy Head Detection.

Provides lightweight, zero-gradient mechanistic interpretability primitives to
pinpoint the exact transformer layer, attention head, or MLP projection responsible
for behavioral failures such as degenerative token repetition loops.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from weightsurg.core.ingestion import ModelIngestor
from weightsurg.core.profiling import compute_svd
from weightsurg.diagnostics.schemas import ComponentTraceScore, DiagnosticReport

logger = logging.getLogger(__name__)


class FailureLocalizer:
    """Mechanistic localizer combining activation patching and OV induction head detection."""

    def __init__(self, ingestor: ModelIngestor) -> None:
        self.ingestor = ingestor

    def _encode_prompt(self, text: str) -> torch.Tensor:
        """Encode text prompt into token IDs on the model's device."""
        model = self.ingestor.model
        device = next(model.parameters()).device
        tokenizer = self.ingestor.tokenizer
        if tokenizer is not None:
            encoded = tokenizer(text, return_tensors="pt")
            return encoded["input_ids"].to(device)

        # Fallback byte-level tokenization when running with a synthetic/tokenizer-free model
        vocab_size = int(getattr(model.config, "vocab_size", 512))
        byte_ids = [max(1, b % vocab_size) for b in text.encode("utf-8")]
        if not byte_ids:
            byte_ids = [1]
        return torch.tensor([byte_ids], dtype=torch.long, device=device)

    def _resolve_target_token_id(
        self,
        input_ids: torch.Tensor,
        target_error: str | int | None = None,
    ) -> int:
        """Determine the token ID corresponding to the target failure/error."""
        if isinstance(target_error, int):
            return target_error

        tokenizer = self.ingestor.tokenizer
        if isinstance(target_error, str) and target_error.strip():
            if target_error.strip().isdigit():
                vocab_size = int(getattr(self.ingestor.model.config, "vocab_size", 512))
                cand = int(target_error.strip())
                if 0 <= cand < vocab_size:
                    return cand
            if tokenizer is not None:
                toks = tokenizer.encode(target_error, add_special_tokens=False)
                if toks:
                    return int(toks[0])

        # Default to the model's top predicted token (or most frequent trailing token)
        with torch.no_grad():
            outputs = self.ingestor.model(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            return int(torch.argmax(next_token_logits).item())

    def detect_induction_and_copy_heads(
        self,
        input_ids: torch.Tensor | None = None,
    ) -> dict[tuple[int, int], float]:
        """Compute induction/copy head scores for every `(layer_idx, head_idx)` pair.

        Combines two zero-gradient mechanistic signals:
        1. **Static OV Circuit Copy Score**: Measures the positive trace / dominant singular
           alignment of `W_O^(h) @ W_V^(h_kv)`, which quantifies how strongly head `h`
           acts as an identity/copy operator on the residual stream.
        2. **Dynamic Attention Induction/Loop Score**: When `input_ids` is provided,
           inspects attention weights for induction diagonals (attending to the token
           immediately following a prior occurrence of the current token) and repetitive
           recent-token lock-in.
        """
        model = self.ingestor.model
        num_layers = self.ingestor.get_num_layers()
        scores: dict[tuple[int, int], float] = {}

        attn_weights_per_layer: tuple[torch.Tensor, ...] | None = None
        if input_ids is not None and input_ids.shape[1] >= 2:
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, output_attentions=True)
                    attn_weights_per_layer = out.attentions
                except Exception as exc:
                    logger.debug("Could not extract attention maps: %s", exc)

        for layer_idx in range(num_layers):
            attn_proj = self.ingestor.unpack_attention(layer_idx)
            num_heads = attn_proj.num_heads
            num_kv_heads = attn_proj.num_kv_heads
            kv_group_size = max(1, num_heads // num_kv_heads)

            for head_idx in range(num_heads):
                kv_head_idx = min(num_kv_heads - 1, head_idx // kv_group_size)
                w_v_h = attn_proj.get_head_slice("v", kv_head_idx).detach().to(torch.float32)
                w_o_h = attn_proj.get_head_slice("o", head_idx).detach().to(torch.float32)

                # Effective OV matrix on the residual stream: shape (hidden_size, hidden_size)
                with torch.no_grad():
                    w_ov = w_o_h @ w_v_h
                    diag = torch.diagonal(w_ov)
                    frob = float(torch.linalg.norm(w_ov, ord="fro").item()) + 1e-8
                    # Positive trace normalized by Frobenius norm & dimension measures identity copying
                    pos_trace_score = float(F.relu(diag.sum()).item()) / (
                        frob * (w_ov.shape[0] ** 0.5)
                    )
                    # Spectral concentration of the head's output slice
                    sv = torch.linalg.svdvals(w_o_h)
                    spectral_concentration = float(
                        (sv[0] / (sv.sum() + 1e-8)).item()
                    ) if sv.numel() > 0 else 0.0

                    static_score = 0.6 * pos_trace_score + 0.4 * spectral_concentration

                dynamic_score = 0.0
                if (
                    attn_weights_per_layer is not None
                    and layer_idx < len(attn_weights_per_layer)
                    and attn_weights_per_layer[layer_idx] is not None
                ):
                    # Shape: (batch, num_heads, seq_len, seq_len)
                    layer_attn = attn_weights_per_layer[layer_idx][0, head_idx]
                    seq_len = layer_attn.shape[0]
                    tokens = input_ids[0].tolist()

                    # Check induction pattern: for position i, if tokens[j-1] == tokens[i],
                    # how much attention is paid to position j?
                    induction_hits: list[float] = []
                    for i in range(1, seq_len):
                        for j in range(1, i):
                            if tokens[j - 1] == tokens[i]:
                                induction_hits.append(float(layer_attn[i, j].item()))

                    if induction_hits:
                        dynamic_score = float(sum(induction_hits) / len(induction_hits))
                    else:
                        # Measure sub-diagonal / local periodic attention concentration
                        last_row = layer_attn[-1]
                        dynamic_score = float(last_row.max().item())

                combined_score = float(0.5 * static_score + 0.5 * dynamic_score)
                scores[(layer_idx, head_idx)] = combined_score

        return scores

    def trace_causal_components(
        self,
        prompt: str,
        target_error: str | int | None = None,
        corrupted_prompt: str | None = None,
        top_k_singular: int = 4,
    ) -> DiagnosticReport:
        """Perform activation patching / causal mediation tracing across heads and MLPs.

        For each layer `l`:
        - Measures the causal mediation of each attention head `h` by intervening on the
          input slice `[h * head_dim : (h + 1) * head_dim]` to `self_attn.o_proj` (replacing
          it with the clean counterfactual activation if `corrupted_prompt` is given, or zero/mean
          ablation when localizing directly on a failing prompt) and measuring the drop in
          log-probability of the `target_error` token.
        - Measures the causal mediation of each layer's `mlp.down_proj` in the same manner.
        """
        model = self.ingestor.model
        num_layers = self.ingestor.get_num_layers()

        failing_text = corrupted_prompt if corrupted_prompt is not None else prompt
        failing_ids = self._encode_prompt(failing_text)
        target_token_id = self._resolve_target_token_id(failing_ids, target_error)

        # Baseline log-prob of the erroneous/repeating token on the failing input
        with torch.no_grad():
            base_logits = model(input_ids=failing_ids).logits[0, -1, :].to(torch.float32)
            base_log_probs = F.log_softmax(base_logits, dim=-1)
            base_target_lp = float(base_log_probs[target_token_id].item())

        # Optional clean counterfactual activations per layer o_proj input and down_proj input
        clean_o_inputs: dict[int, torch.Tensor] = {}
        clean_mlp_inputs: dict[int, torch.Tensor] = {}
        if corrupted_prompt is not None and prompt != corrupted_prompt:
            clean_ids = self._encode_prompt(prompt)
            hooks = []
            for layer_idx in range(num_layers):
                layer_mod = model.model.layers[layer_idx]

                def _make_o_capture(l_idx: int):
                    def _hook(_mod: Any, inputs: tuple[torch.Tensor, ...]) -> None:
                        clean_o_inputs[l_idx] = inputs[0].detach().clone()

                    return _hook

                def _make_mlp_capture(l_idx: int):
                    def _hook(_mod: Any, inputs: tuple[torch.Tensor, ...]) -> None:
                        clean_mlp_inputs[l_idx] = inputs[0].detach().clone()

                    return _hook

                hooks.append(layer_mod.self_attn.o_proj.register_forward_pre_hook(_make_o_capture(layer_idx)))
                hooks.append(layer_mod.mlp.down_proj.register_forward_pre_hook(_make_mlp_capture(layer_idx)))

            with torch.no_grad():
                model(input_ids=clean_ids)
            for h in hooks:
                h.remove()

        induction_scores = self.detect_induction_and_copy_heads(failing_ids)
        component_scores: list[ComponentTraceScore] = []

        for layer_idx in range(num_layers):
            attn_proj = self.ingestor.unpack_attention(layer_idx)
            ffn_proj = self.ingestor.unpack_ffn(layer_idx)
            layer_mod = model.model.layers[layer_idx]
            head_dim = attn_proj.head_dim

            # 1. Trace each individual attention head via o_proj input slice intervention
            for head_idx in range(attn_proj.num_heads):
                start = head_idx * head_dim
                end = start + head_dim

                def _head_patch_hook(
                    _mod: Any,
                    inputs: tuple[torch.Tensor, ...],
                    s: int = start,
                    e: int = end,
                    l_idx: int = layer_idx,
                ) -> tuple[torch.Tensor, ...]:
                    x = inputs[0].clone()
                    if l_idx in clean_o_inputs and clean_o_inputs[l_idx].shape == x.shape:
                        x[:, -1, s:e] = clean_o_inputs[l_idx][:, -1, s:e]
                    else:
                        x[:, -1, s:e] = 0.0
                    return (x,) + inputs[1:]

                handle = layer_mod.self_attn.o_proj.register_forward_pre_hook(_head_patch_hook)
                try:
                    with torch.no_grad():
                        patched_logits = model(input_ids=failing_ids).logits[0, -1, :].to(torch.float32)
                        patched_lp = float(F.log_softmax(patched_logits, dim=-1)[target_token_id].item())
                finally:
                    handle.remove()

                # Positive causal effect means this head actively drives the erroneous token
                causal_drop = max(0.0, base_target_lp - patched_lp)
                ind_score = induction_scores.get((layer_idx, head_idx), 0.0)

                head_slice = attn_proj.get_head_slice("o", head_idx)
                svd_res = compute_svd(head_slice, top_k=top_k_singular)
                top_sv = [round(float(v), 6) for v in svd_res.s.tolist()]

                component_scores.append(
                    ComponentTraceScore(
                        layer_idx=layer_idx,
                        component_type="attn_head",
                        tensor_path=attn_proj.paths["o"],
                        causal_effect=round(float(causal_drop + 0.35 * ind_score), 6),
                        head_idx=head_idx,
                        induction_score=round(float(ind_score), 6),
                        spectral_norm=top_sv[0] if top_sv else 0.0,
                        top_singular_values=top_sv,
                        sub_rank=1,
                    )
                )

            # 2. Trace the layer's MLP via down_proj input intervention
            def _mlp_patch_hook(
                _mod: Any,
                inputs: tuple[torch.Tensor, ...],
                l_idx: int = layer_idx,
            ) -> tuple[torch.Tensor, ...]:
                x = inputs[0].clone()
                if l_idx in clean_mlp_inputs and clean_mlp_inputs[l_idx].shape == x.shape:
                    x[:, -1, :] = clean_mlp_inputs[l_idx][:, -1, :]
                else:
                    x[:, -1, :] = 0.0
                return (x,) + inputs[1:]

            handle = layer_mod.mlp.down_proj.register_forward_pre_hook(_mlp_patch_hook)
            try:
                with torch.no_grad():
                    patched_logits = model(input_ids=failing_ids).logits[0, -1, :].to(torch.float32)
                    patched_lp = float(F.log_softmax(patched_logits, dim=-1)[target_token_id].item())
            finally:
                handle.remove()

            mlp_causal_drop = max(0.0, base_target_lp - patched_lp)
            mlp_svd = compute_svd(ffn_proj.w_down, top_k=top_k_singular)
            mlp_top_sv = [round(float(v), 6) for v in mlp_svd.s.tolist()]

            component_scores.append(
                ComponentTraceScore(
                    layer_idx=layer_idx,
                    component_type="mlp",
                    tensor_path=ffn_proj.paths["down"],
                    causal_effect=round(float(mlp_causal_drop * 0.5), 6),
                    head_idx=None,
                    induction_score=0.0,
                    spectral_norm=mlp_top_sv[0] if mlp_top_sv else 0.0,
                    top_singular_values=mlp_top_sv,
                    sub_rank=1,
                )
            )

        # Sort components descending by combined causal & induction score
        component_scores.sort(
            key=lambda c: (c.causal_effect, c.induction_score, c.spectral_norm or 0.0),
            reverse=True,
        )

        primary = component_scores[0]
        return DiagnosticReport(
            model_name_or_path=self.ingestor.model_name_or_path,
            clean_prompt=prompt,
            corrupted_prompt=corrupted_prompt,
            target_error=str(target_error) if target_error is not None else str(target_token_id),
            primary_target_tensor=primary.tensor_path,
            primary_layer_idx=primary.layer_idx,
            primary_head_idx=primary.head_idx,
            recommended_sub_rank=primary.sub_rank,
            dominant_singular_values=primary.top_singular_values,
            dominant_singular_indices=[0],
            ranked_components=component_scores[:15],
            recommended_patch_mode="subspace_damping",
            recommended_damping_factor=0.15,
            metadata={
                "target_token_id": target_token_id,
                "baseline_target_logprob": round(base_target_lp, 6),
            },
        )

