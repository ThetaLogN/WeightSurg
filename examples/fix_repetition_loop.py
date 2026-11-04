#!/usr/bin/env python3
"""Reproducible End-to-End Demo: Fixing a Degenerate Token Repetition Loop with Zero Gradients.

Workflow:
1. Loads `Qwen/Qwen2.5-0.5B-Instruct` (or a deterministic local `Qwen2ForCausalLM` fallback
   when running offline / in an air-gapped environment).
2. Reproduces a degenerative token repetition / reasoning loop triggered by a dominant
   recursive copy head in `self_attn.o_proj`.
3. Runs `FailureLocalizer` (Activation Patching + OV Induction Head Detection) to pinpoint
   the exact layer and attention head driving the loop.
4. Synthesizes a closed-form rank-1 surgical delta `ΔW = (v - W k)(k^T / ||k||^2)` on the
   offending head's dominant singular subspace.
5. Hot-patches the model in memory with zero gradient steps, verifies that the repetition
   loop is resolved without catastrophic forgetting, and exports the patched checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weightsurg.core.ingestion import ModelIngestor, create_synthetic_qwen2_model
from weightsurg.diagnostics.localizer import FailureLocalizer
from weightsurg.patcher.hotpatch import HotPatcher, compute_token_repetition_rate

console = Console()
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _induce_degenerate_copy_head(
    ingestor: ModelIngestor,
    target_layer: int = 2,
    target_head: int = 1,
    loop_token_id: int = 82,
) -> None:
    """Configure a clean sequential transition circuit in the synthetic model and inject a
    dominant rank-1 recursive copy spike into `(target_layer, target_head)` that traps
    generation in a deterministic repetition loop on `loop_token_id`.
    """
    model = ingestor.model
    hidden_size = int(model.config.hidden_size)

    with torch.no_grad():
        # 1. Give tokens 65..122 orthogonal/structured embeddings and a clean +1 step transition
        # in lm_head so the healthy model progresses cleanly (82 -> 83 -> 84 -> 85 ...)
        torch.manual_seed(42)
        ortho_basis = torch.linalg.qr(torch.randn(hidden_size, hidden_size))[0]
        for idx in range(hidden_size):
            tok_id = 65 + idx
            if tok_id < model.config.vocab_size:
                model.model.embed_tokens.weight[tok_id].copy_(ortho_basis[:, idx] * 0.30)
                prev_idx = (idx - 1) % hidden_size
                model.lm_head.weight[tok_id].copy_(ortho_basis[:, prev_idx] * 0.30)

        # 2. Inject a strong rank-1 recursive copy spike into (target_layer, target_head)
        # that maps loop_token_id's embedding directly onto loop_token_id's lm_head vector
        attn = ingestor.unpack_attention(target_layer)
        head_dim = attn.head_dim
        c_start = target_head * head_dim
        c_end = c_start + head_dim

        loop_embed = model.model.embed_tokens.weight[loop_token_id].detach().clone()
        loop_embed_unit = loop_embed / (torch.linalg.norm(loop_embed) + 1e-8)

        loop_unembed = model.lm_head.weight[loop_token_id].detach().clone()
        loop_unembed_unit = loop_unembed / (torch.linalg.norm(loop_unembed) + 1e-8)

        right_vec = torch.ones(head_dim, device=loop_embed.device, dtype=loop_embed.dtype)
        right_vec = right_vec / torch.linalg.norm(right_vec)

        # Rank-1 spike in head `target_head` of `o_proj`
        rank1_spike = 12.0 * torch.outer(loop_unembed_unit, right_vec)
        attn.w_o[:, c_start:c_end].add_(rank1_spike)

        # Align corresponding KV head's `v_proj` with `right_vec`
        kv_group = max(1, attn.num_heads // attn.num_kv_heads)
        kv_head = min(attn.num_kv_heads - 1, target_head // kv_group)
        r_start = kv_head * head_dim
        r_end = r_start + head_dim
        attn.w_v[r_start:r_end, :].add_(6.0 * torch.outer(right_vec, loop_embed_unit))


def load_target_or_fallback_model(
    model_id: str = DEFAULT_MODEL_ID,
    force_offline: bool = False,
) -> tuple[ModelIngestor, str, str]:
    """Load `Qwen/Qwen2.5-0.5B-Instruct` or fall back to a compact local Qwen2 architecture."""
    prompt = "Explain step by step: Why does a recursive function without a base case fail? Step 1:"
    target_error = "Step 1:"

    if not force_offline:
        try:
            ingestor = ModelIngestor(model_name_or_path=model_id)
            ingestor.load_model()
            return ingestor, prompt, target_error
        except Exception as exc:
            console.print(
                f"[yellow]Note:[/yellow] Could not download '{model_id}' ({exc.__class__.__name__}). "
                "Using deterministic local Qwen2.5 micro-architecture for offline reproducibility."
            )

    model = create_synthetic_qwen2_model(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        seed=42,
    )
    ingestor = ModelIngestor.from_pretrained_objects(
        model=model,
        model_name=f"{model_id} (offline-micro)",
    )
    loop_tok = ord("R")  # Token 82 in byte-encoded prompt
    _induce_degenerate_copy_head(ingestor, target_layer=2, target_head=1, loop_token_id=loop_tok)
    prompt = "RRRR"
    target_error = str(loop_tok)
    return ingestor, prompt, target_error


def run_demo(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str | Path = "outputs/qwen2.5-patched",
    offline: bool = False,
) -> dict[str, object]:
    """Run the complete zero-gradient localization, rank-1 surgery, and verification pipeline."""
    console.print(
        Panel(
            "[bold cyan]WeightSurg End-to-End Demo[/bold cyan]\n"
            "Zero-Gradient Weight-Space Programming & Neural Circuit Surgery",
            expand=False,
        )
    )

    ingestor, prompt, target_error = load_target_or_fallback_model(
        model_id=model_id,
        force_offline=offline,
    )
    patcher = HotPatcher(ingestor)

    # Step 1: Reproduce the token repetition loop prior to surgery
    pre_tokens, pre_text = patcher.generate_greedy(prompt, max_new_tokens=12)
    pre_rep_rate = compute_token_repetition_rate(pre_tokens)
    console.print(f"[bold red]1. Pre-Surgery Generation:[/bold red] {pre_text}")
    console.print(f"   Token IDs: {pre_tokens} (Repetition rate: {pre_rep_rate * 100:.1f}%)\n")

    # Step 2: Causal Localization & Induction Head Detection
    localizer = FailureLocalizer(ingestor)
    report = localizer.trace_causal_components(prompt=prompt, target_error=target_error)

    console.print(
        f"[bold magenta]2. Localized Offending Circuit:[/bold magenta] "
        f"[cyan]{report.primary_target_tensor}[/cyan] "
        f"(Layer {report.primary_layer_idx}, Head {report.primary_head_idx}, "
        f"σ_0 = {report.dominant_singular_values[0]:.4f})"
    )

    # Step 3: Formulate closed-form rank-1 surgical recipe & apply hot-patch
    recipe = report.to_patch_recipe()
    recipe.mode = "rank1_update"
    recipe.damping_factor = 0.0  # Surgically project out the degenerate rank-1 copy singular vector

    metrics = patcher.execute_recipe(recipe, eval_max_new_tokens=12)

    console.print(f"\n[bold green]3. Post-Surgery Generation:[/bold green] {metrics.post_patch_generation}")
    post_tokens, _ = patcher.generate_greedy(prompt, max_new_tokens=12)
    console.print(
        f"   Token IDs: {post_tokens} "
        f"(Repetition rate: {metrics.post_patch_repetition_rate * 100:.1f}%)\n"
    )

    # Step 4: Export patched model & diagnostic recipe
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = report.save(out_path / "diagnostic_report.json")
    recipe_path = recipe.save(out_path / "patch_recipe.json")
    st_path = patcher.export_safetensors(out_path / "model.safetensors")
    gguf_path = patcher.export_gguf(out_path / "model-patched.gguf")

    summary = Table(title="Zero-Gradient Surgery Verification Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Target Tensor", recipe.target_tensor)
    summary.add_row("Localized Layer / Head", f"Layer {recipe.layer_idx}, Head {recipe.head_idx}")
    summary.add_row("Closed-Form ΔW Norm (||ΔW||_F)", f"{metrics.delta_frobenius_norm:.4f}")
    summary.add_row("Pre-Patch Repetition Rate", f"{metrics.pre_patch_repetition_rate * 100:.1f}%")
    summary.add_row("Post-Patch Repetition Rate", f"{metrics.post_patch_repetition_rate * 100:.1f}%")
    summary.add_row("Pre-Patch Perplexity", f"{metrics.pre_patch_perplexity:.2f}")
    summary.add_row("Post-Patch Perplexity", f"{metrics.post_patch_perplexity:.2f}")
    summary.add_row("Gradient Steps / Optimizer States", "0 / 0 bytes")
    summary.add_row("Exported Safetensors", str(st_path))
    summary.add_row("Exported GGUF", str(gguf_path))
    console.print(summary)

    return {
        "report": report,
        "recipe": recipe,
        "metrics": metrics,
        "safetensors_path": st_path,
        "gguf_path": gguf_path,
        "report_path": report_path,
        "recipe_path": recipe_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WeightSurg End-to-End Repetition Loop Fix Demo")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or path.")
    parser.add_argument("--output-dir", default="outputs/qwen2.5-patched", help="Export directory.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic local Qwen2 micro-model without network access.",
    )
    args = parser.parse_args()
    run_demo(model_id=args.model, output_dir=args.output_dir, offline=args.offline)


if __name__ == "__main__":
    main()
