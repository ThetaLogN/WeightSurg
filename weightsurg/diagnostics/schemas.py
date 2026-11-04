"""Data schemas for WeightSurg diagnostic reports and surgical patch recipes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

PatchMode = Literal[
    "rank1_update",
    "low_rank_update",
    "subspace_damping",
    "head_attenuation",
    "llm_guided",
]


@dataclass
class ComponentTraceScore:
    """Causal or mechanistic score for an individual model component (layer/head/MLP)."""

    layer_idx: int
    component_type: Literal["attn_head", "mlp", "layer_residual"]
    tensor_path: str
    causal_effect: float
    head_idx: int | None = None
    induction_score: float = 0.0
    spectral_norm: float | None = None
    top_singular_values: list[float] = field(default_factory=list)
    sub_rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize component score to dictionary."""
        return asdict(self)


@dataclass
class DiagnosticReport:
    """Standardized diagnostic report emitted by the Failure Localizer."""

    model_name_or_path: str
    clean_prompt: str
    corrupted_prompt: str | None
    target_error: str
    primary_target_tensor: str
    primary_layer_idx: int
    primary_head_idx: int | None
    recommended_sub_rank: int
    dominant_singular_values: list[float] = field(default_factory=list)
    dominant_singular_indices: list[int] = field(default_factory=list)
    ranked_components: list[ComponentTraceScore] = field(default_factory=list)
    recommended_patch_mode: PatchMode = "subspace_damping"
    recommended_damping_factor: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the diagnostic report to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the diagnostic report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the diagnostic report to disk as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticReport:
        """Reconstruct a DiagnosticReport from a parsed dictionary."""
        raw_components = data.get("ranked_components", [])
        components = [
            ComponentTraceScore(**comp) if isinstance(comp, dict) else comp
            for comp in raw_components
        ]
        copy_data = dict(data)
        copy_data["ranked_components"] = components
        return cls(**copy_data)

    @classmethod
    def load(cls, path: str | Path) -> DiagnosticReport:
        """Load a DiagnosticReport from a JSON file on disk."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def to_patch_recipe(self) -> PatchRecipe:
        """Convert this diagnostic report directly into an executable PatchRecipe."""
        return PatchRecipe(
            target_tensor=self.primary_target_tensor,
            mode=self.recommended_patch_mode,
            layer_idx=self.primary_layer_idx,
            head_idx=self.primary_head_idx,
            sub_rank=self.recommended_sub_rank,
            singular_indices=list(self.dominant_singular_indices or range(self.recommended_sub_rank)),
            damping_factor=self.recommended_damping_factor,
            prompt=self.clean_prompt,
            target_error=self.target_error,
            metadata={
                "source_model": self.model_name_or_path,
                "derived_from_diagnostic": True,
                **self.metadata,
            },
        )


@dataclass
class PatchRecipe:
    """Declarative specification for a zero-gradient surgical weight patch."""

    target_tensor: str
    mode: PatchMode = "subspace_damping"
    layer_idx: int | None = None
    head_idx: int | None = None
    num_heads: int | None = None
    sub_rank: int = 1
    singular_indices: list[int] = field(default_factory=lambda: [0])
    damping_factor: float = 0.25
    key_vector: list[float] | None = None
    value_vector: list[float] | None = None
    llm_instruction: str | None = None
    prompt: str | None = None
    target_error: str | None = None
    max_spectral_shift: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the patch recipe to a dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the patch recipe to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> Path:
        """Save the patch recipe to disk as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PatchRecipe:
        """Load a PatchRecipe from a dictionary (or convert from a DiagnosticReport dict)."""
        if "primary_target_tensor" in data and "target_tensor" not in data:
            return DiagnosticReport.from_dict(data).to_patch_recipe()
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str | Path) -> PatchRecipe:
        """Load a PatchRecipe (or DiagnosticReport) from a JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

