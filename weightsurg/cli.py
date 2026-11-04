"""Command-Line Interface (`weightsurg` / `weightsurg-cli`) for WeightSurg.

Provides three primary subcommands:
- `weightsurg inspect <model_path>`: Dumps layer statistics, SVD spectrum, and weight anomaly hotspots.
- `weightsurg trace --model <model> --prompt "..." --target-error <token_loop>`: Pinpoints the offending attention head or MLP.
- `weightsurg patch --model <model> --spec <patch_recipe.json> --output <patched_model_path>`: Applies surgical delta and benchmarks regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from weightsurg.core.ingestion import ModelIngestor, create_synthetic_qwen2_model
from weightsurg.core.profiling import profile_model
from weightsurg.diagnostics.localizer import FailureLocalizer
from weightsurg.diagnostics.schemas import PatchRecipe
from weightsurg.patcher.hotpatch import HotPatcher

console = Console()


def _resolve_ingestor(model_path: str, device: str | None = None) -> ModelIngestor:
    """Create a ModelIngestor for a Hugging Face ID, local path, or `'synthetic'`."""
    if model_path.strip().lower() == "synthetic":
        model = create_synthetic_qwen2_model()
        return ModelIngestor.from_pretrained_objects(model, model_name="synthetic-qwen2")
    ingestor = ModelIngestor(model_name_or_path=model_path, device=device)
    path_obj = Path(model_path)
    if path_obj.is_file() and path_obj.suffix == ".safetensors":
        ingestor.load_safetensors_state_dict()
    else:
        ingestor.load_model()
    return ingestor


def cmd_inspect(args: argparse.Namespace) -> int:
    """Execute `weightsurg inspect <model_path>`."""
    ingestor = _resolve_ingestor(args.model_path, device=args.device)
    profiles = profile_model(ingestor, max_layers=args.max_layers, top_k_sv=5)
    mem_mb = ingestor.estimate_parameter_memory_bytes() / (1024**2)

    console.print(
        Panel(
            f"[bold cyan]Model:[/bold cyan] {ingestor.model_name_or_path}\n"
            f"[bold cyan]Device:[/bold cyan] {ingestor.device} | "
            f"[bold cyan]Parameter Footprint:[/bold cyan] {mem_mb:.2f} MB | "
            f"[bold cyan]Profiled Matrices:[/bold cyan] {len(profiles)}",
            title="[bold green]WeightSurg Algebraic Inspector[/bold green]",
            expand=False,
        )
    )

    table = Table(title="Layer Projection Algebraic & Spectral Profile", show_lines=False)
    table.add_column("Tensor Path", style="cyan", no_wrap=True)
    table.add_column("Shape", style="white")
    table.add_column("Spectral Norm (σ1)", justify="right", style="magenta")
    table.add_column("Stable Rank", justify="right", style="green")
    table.add_column("Effective Rank", justify="right", style="green")
    table.add_column("Sparsity", justify="right", style="yellow")
    table.add_column("Outliers (>4σ)", justify="right", style="red")
    table.add_column("Top Singular Values", style="dim")
    table.add_column("Hotspot", style="bold red")

    hotspots = []
    for p in profiles:
        if p.is_anomaly_hotspot:
            hotspots.append(p)
        sv_str = ", ".join(f"{v:.2f}" for v in p.top_singular_values[:4])
        table.add_row(
            p.tensor_path,
            str(p.shape),
            f"{p.spectral_norm:.4f}",
            f"{p.stable_rank:.2f}",
            f"{p.effective_rank:.2f}",
            f"{p.sparsity_ratio * 100:.2f}%",
            str(p.outliers.outlier_count),
            f"[{sv_str}]",
            "HOTSPOT" if p.is_anomaly_hotspot else "OK",
        )

    console.print(table)

    if hotspots:
        console.print(f"\n[bold red]Detected {len(hotspots)} Weight Anomaly Hotspot(s):[/bold red]")
        for h in hotspots:
            console.print(f"  • [bold]{h.tensor_path}[/bold]: {h.anomaly_reason}")
    else:
        console.print("\n[bold green]No extreme spectral/outlier hotspots detected.[/bold green]")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([p.to_dict() for p in profiles], indent=2),
            encoding="utf-8",
        )
        console.print(f"[dim]Saved JSON inspection profile to {out_path}[/dim]")

    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    """Execute `weightsurg trace --model <model> --prompt "..." --target-error <token_loop>`."""
    ingestor = _resolve_ingestor(args.model, device=args.device)
    localizer = FailureLocalizer(ingestor)

    report = localizer.trace_causal_components(
        prompt=args.prompt,
        target_error=args.target_error,
        corrupted_prompt=args.corrupted_prompt,
    )

    console.print(
        Panel(
            f"[bold cyan]Model:[/bold cyan] {report.model_name_or_path}\n"
            f"[bold cyan]Prompt:[/bold cyan] {report.clean_prompt!r}\n"
            f"[bold cyan]Target Error:[/bold cyan] {report.target_error!r}\n"
            f"[bold yellow]Primary Localized Tensor:[/bold yellow] {report.primary_target_tensor} "
            f"(Layer {report.primary_layer_idx}, Head {report.primary_head_idx})",
            title="[bold magenta]WeightSurg Failure Localizer (Causal Trace)[/bold magenta]",
            expand=False,
        )
    )

    table = Table(title="Top Ranked Causal & Induction Components")
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Layer", justify="right", style="cyan")
    table.add_column("Component", style="white")
    table.add_column("Head", justify="right", style="green")
    table.add_column("Tensor Path", style="cyan")
    table.add_column("Causal Score", justify="right", style="magenta")
    table.add_column("Induction Score", justify="right", style="yellow")
    table.add_column("σ_max", justify="right", style="dim")

    for idx, comp in enumerate(report.ranked_components[:10], start=1):
        table.add_row(
            str(idx),
            str(comp.layer_idx),
            comp.component_type,
            str(comp.head_idx) if comp.head_idx is not None else "-",
            comp.tensor_path,
            f"{comp.causal_effect:.5f}",
            f"{comp.induction_score:.5f}",
            f"{comp.spectral_norm or 0.0:.4f}",
        )

    console.print(table)

    if args.output:
        saved = report.save(args.output)
        console.print(f"[bold green]Saved diagnostic report to:[/bold green] {saved}")
    else:
        console.print("\n[bold]Standardized JSON Diagnostic Report:[/bold]")
        console.print_json(report.to_json())

    return 0


def cmd_patch(args: argparse.Namespace) -> int:
    """Execute `weightsurg patch --model <model> --spec <patch_recipe.json> --output <patched_model_path>`."""
    ingestor = _resolve_ingestor(args.model, device=args.device)
    recipe = PatchRecipe.load(args.spec)
    patcher = HotPatcher(ingestor)

    metrics = patcher.execute_recipe(recipe)

    if args.format == "gguf" or str(args.output).endswith(".gguf"):
        exported_path = patcher.export_gguf(args.output)
    else:
        exported_path = patcher.export_safetensors(args.output)

    table = Table(title="Zero-Gradient Surgical Patch & Regression Benchmark")
    table.add_column("Metric", style="cyan")
    table.add_column("Before Patch", justify="right", style="yellow")
    table.add_column("After Patch", justify="right", style="green")

    table.add_row("Target Tensor", recipe.target_tensor, recipe.target_tensor)
    table.add_row("Surgical Mode", recipe.mode, f"||ΔW||_F = {metrics.delta_frobenius_norm:.5f}")
    table.add_row(
        "Baseline Perplexity",
        f"{metrics.pre_patch_perplexity:.4f}",
        f"{metrics.post_patch_perplexity:.4f} (drift {metrics.perplexity_drift_ratio * 100:+.2f}%)",
    )
    table.add_row(
        "Token Repetition Rate",
        f"{metrics.pre_patch_repetition_rate * 100:.1f}%",
        f"{metrics.post_patch_repetition_rate * 100:.1f}%",
    )
    table.add_row(
        "Catastrophic Forgetting",
        "N/A",
        "DETECTED" if metrics.catastrophic_forgetting_detected else "NONE (0 grad steps)",
    )

    console.print(table)
    console.print(f"[bold green]Exported patched model to:[/bold green] {exported_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for `weightsurg` / `weightsurg-cli`."""
    parser = argparse.ArgumentParser(
        prog="weightsurg",
        description="WeightSurg: Zero-Gradient Weight-Space Programming & Neural Circuit Surgery Toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. inspect
    p_inspect = subparsers.add_parser(
        "inspect",
        help="Dump layer statistics, SVD spectrum, and weight anomaly hotspots.",
    )
    p_inspect.add_argument("model_path", help="Model Hugging Face ID, local path, or 'synthetic'.")
    p_inspect.add_argument("--max-layers", type=int, default=None, help="Limit number of layers inspected.")
    p_inspect.add_argument("--device", type=str, default=None, help="Target device (cpu, mps, cuda).")
    p_inspect.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON profile.")
    p_inspect.set_defaults(func=cmd_inspect)

    # 2. trace
    p_trace = subparsers.add_parser(
        "trace",
        help="Pinpoint the specific attention head or MLP responsible for a behavioral failure.",
    )
    p_trace.add_argument("--model", required=True, help="Model Hugging Face ID, local path, or 'synthetic'.")
    p_trace.add_argument("--prompt", required=True, help="Input prompt triggering or testing the behavior.")
    p_trace.add_argument(
        "--target-error",
        required=True,
        help="Target repeating token, phrase, or token ID to localize.",
    )
    p_trace.add_argument(
        "--corrupted-prompt",
        default=None,
        help="Optional corrupted prompt for clean-vs-corrupted activation patching.",
    )
    p_trace.add_argument("--output", default=None, help="Optional file path to save the JSON diagnostic report.")
    p_trace.add_argument("--device", default=None, help="Target device (cpu, mps, cuda).")
    p_trace.set_defaults(func=cmd_trace)

    # 3. patch
    p_patch = subparsers.add_parser(
        "patch",
        help="Apply a surgical weight delta ΔW and benchmark behavioral & perplexity regression.",
    )
    p_patch.add_argument("--model", required=True, help="Model Hugging Face ID, local path, or 'synthetic'.")
    p_patch.add_argument("--spec", required=True, help="Path to patch recipe or diagnostic report JSON.")
    p_patch.add_argument("--output", required=True, help="Output directory or file path for patched model.")
    p_patch.add_argument(
        "--format",
        choices=["safetensors", "gguf"],
        default="safetensors",
        help="Export format (safetensors or gguf).",
    )
    p_patch.add_argument("--device", default=None, help="Target device (cpu, mps, cuda).")
    p_patch.set_defaults(func=cmd_patch)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `weightsurg` and `weightsurg-cli` executables."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

