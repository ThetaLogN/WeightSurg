"""Model & Tensor Ingestion for WeightSurg.

Supports loading compact open-source Transformer models (e.g. Qwen 2.5 0.5B/1.5B/3B,
Llama 3.2 1B/3B) via Hugging Face `transformers` and `safetensors`, with optional
Apple Silicon `MLX` zero-copy array conversion hooks. Enforces zero-gradient execution
and strict memory footprints (< 6 GB peak RAM).
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2Config,
    Qwen2ForCausalLM,
)

logger = logging.getLogger(__name__)

MAX_RECOMMENDED_MEMORY_BYTES = 6 * 1024**3  # 6 GB ceiling for consumer hardware


@dataclass
class AttentionProjections:
    """Unpacked Multi-Head / Grouped-Query Attention projection matrices for a layer."""

    layer_idx: int
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    num_heads: int
    num_kv_heads: int
    head_dim: int
    paths: dict[str, str] = field(default_factory=dict)

    def get_head_slice(
        self,
        proj: Literal["q", "k", "v", "o"],
        head_idx: int,
    ) -> torch.Tensor:
        """Extract the sub-matrix corresponding to a specific attention head.

        For `q`, `k`, `v` projections of shape `(num_heads * head_dim, hidden_size)`,
        slices rows `[head_idx * head_dim : (head_idx + 1) * head_dim, :]`.
        For `o` projection of shape `(hidden_size, num_heads * head_dim)`,
        slices columns `[:, head_idx * head_dim : (head_idx + 1) * head_dim]`.
        """
        if proj in ("q", "o"):
            max_heads = self.num_heads
        else:
            max_heads = self.num_kv_heads

        if not (0 <= head_idx < max_heads):
            raise ValueError(
                f"head_idx {head_idx} out of bounds for projection '{proj}' "
                f"with {max_heads} heads."
            )

        start = head_idx * self.head_dim
        end = start + self.head_dim

        if proj == "q":
            return self.w_q[start:end, :]
        if proj == "k":
            return self.w_k[start:end, :]
        if proj == "v":
            return self.w_v[start:end, :]
        if proj == "o":
            return self.w_o[:, start:end]
        raise ValueError(f"Unsupported projection '{proj}'. Expected one of: q, k, v, o.")


@dataclass
class FFNProjections:
    """Unpacked SwiGLU / Feed-Forward Network projection matrices for a layer."""

    layer_idx: int
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    intermediate_size: int
    hidden_size: int
    paths: dict[str, str] = field(default_factory=dict)


def detect_optimal_device(preferred: str | None = None) -> torch.device:
    """Select the optimal compute device (MPS on Apple Silicon, CUDA, or CPU)."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_mlx_available() -> bool:
    """Return True if running on Apple Silicon with MLX installed."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def tensor_to_mlx(tensor: torch.Tensor) -> Any:
    """Zero-copy (or shared unified memory) conversion from PyTorch tensor to MLX array."""
    if not is_mlx_available():
        raise RuntimeError("MLX backend is not available in this environment.")
    import mlx.core as mx

    cpu_f32 = tensor.detach().cpu().to(torch.float32).numpy()
    return mx.array(cpu_f32)


def mlx_to_tensor(array: Any, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert an MLX array back into a PyTorch tensor."""
    np_arr = np.array(array, copy=False)
    return torch.from_numpy(np_arr).to(device=device, dtype=dtype)


class ModelIngestor:
    """Zero-gradient model and tensor loader for Qwen 2.5, Llama 3.2, and Safetensors checkpoints."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        use_mlx: bool = False,
        memory_limit_bytes: int = MAX_RECOMMENDED_MEMORY_BYTES,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = detect_optimal_device(str(device) if device is not None else None)
        self.dtype = dtype
        self.use_mlx = use_mlx and is_mlx_available()
        self.memory_limit_bytes = memory_limit_bytes

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._state_dict_cache: dict[str, torch.Tensor] | None = None

    @property
    def model(self) -> PreTrainedModel:
        """Return the loaded Hugging Face PreTrainedModel, loading it on demand."""
        if self._model is None:
            self.load_model()
        assert self._model is not None
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        """Return the loaded tokenizer if available."""
        return self._tokenizer

    @classmethod
    def from_pretrained_objects(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_name: str = "in-memory-model",
    ) -> ModelIngestor:
        """Wrap an already-instantiated PreTrainedModel with zero-grad enforcement."""
        ingestor = cls(
            model_name_or_path=model_name,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        ingestor._enforce_zero_grad(model)
        ingestor._model = model
        ingestor._tokenizer = tokenizer
        return ingestor

    @staticmethod
    def _enforce_zero_grad(model: nn.Module) -> None:
        """Disable gradient tracking on all parameters to guarantee zero optimizer overhead."""
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

    def estimate_parameter_memory_bytes(self) -> int:
        """Calculate total memory footprint in bytes of all model parameters and buffers."""
        if self._model is not None:
            param_bytes = sum(p.numel() * p.element_size() for p in self._model.parameters())
            buf_bytes = sum(b.numel() * b.element_size() for b in self._model.buffers())
            return param_bytes + buf_bytes
        if self._state_dict_cache is not None:
            return sum(t.numel() * t.element_size() for t in self._state_dict_cache.values())
        return 0

    def load_safetensors_state_dict(self) -> dict[str, torch.Tensor]:
        """Load tensors directly from a `.safetensors` file or checkpoint directory."""
        if self._state_dict_cache is not None:
            return self._state_dict_cache

        path = Path(self.model_name_or_path)
        tensors: dict[str, torch.Tensor] = {}

        if path.is_file() and path.suffix == ".safetensors":
            tensors = load_safetensors_file(str(path), device="cpu")
        elif path.is_dir():
            st_files = sorted(path.glob("*.safetensors"))
            if st_files:
                for st_file in st_files:
                    with safe_open(str(st_file), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensors[key] = f.get_tensor(key)

        if not tensors and self._model is not None:
            tensors = {name: param.detach() for name, param in self._model.named_parameters()}

        if not tensors:
            # Fall back to loading the full HF model and extracting its parameters
            self.load_model(load_tokenizer=False)
            assert self._model is not None
            tensors = {name: param.detach() for name, param in self._model.named_parameters()}

        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        if total_bytes > self.memory_limit_bytes:
            raise MemoryError(
                f"Checkpoint requires {total_bytes / 1024**3:.2f} GB, exceeding "
                f"configured limit of {self.memory_limit_bytes / 1024**3:.2f} GB."
            )

        self._state_dict_cache = tensors
        return tensors

    def load_model(self, load_tokenizer: bool = True) -> PreTrainedModel:
        """Load the causal LM and tokenizer with zero-gradient enforcement."""
        if self._model is not None:
            return self._model

        logger.info("Loading model '%s' on device '%s'...", self.model_name_or_path, self.device)
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            self._enforce_zero_grad(model)
            model.to(self.device)

        mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        if mem_bytes > self.memory_limit_bytes:
            raise MemoryError(
                f"Model parameter memory ({mem_bytes / 1024**3:.2f} GB) exceeds "
                f"6 GB hardware budget ({self.memory_limit_bytes / 1024**3:.2f} GB)."
            )

        self._model = model

        if load_tokenizer:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                self._tokenizer = tokenizer
            except Exception as exc:
                logger.warning("Could not load tokenizer for '%s': %s", self.model_name_or_path, exc)

        return self._model

    def list_tensor_paths(self) -> list[str]:
        """List all named parameter/tensor paths in the model or safetensors archive."""
        if self._model is not None:
            return [name for name, _ in self._model.named_parameters()]
        state = self.load_safetensors_state_dict()
        return list(state.keys())

    def get_tensor(self, tensor_path: str) -> torch.Tensor:
        """Retrieve a specific parameter tensor by its dotted path."""
        if self._model is not None:
            params = dict(self._model.named_parameters())
            if tensor_path in params:
                return params[tensor_path]
            # Handle optional 'model.' prefix differences
            alt_path = (
                tensor_path[len("model.") :]
                if tensor_path.startswith("model.")
                else f"model.{tensor_path}"
            )
            if alt_path in params:
                return params[alt_path]
            raise KeyError(f"Tensor path '{tensor_path}' not found in model parameters.")

        state = self.load_safetensors_state_dict()
        if tensor_path in state:
            return state[tensor_path]
        raise KeyError(f"Tensor path '{tensor_path}' not found in safetensors archive.")

    def get_num_layers(self) -> int:
        """Return the number of transformer decoder layers in the loaded model."""
        if self._model is not None and hasattr(self._model, "config"):
            return int(getattr(self._model.config, "num_hidden_layers", 0))
        paths = self.list_tensor_paths()
        layer_indices: set[int] = set()
        for p in paths:
            parts = p.split(".")
            for i, part in enumerate(parts):
                if part in ("layers", "h", "blocks") and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_indices.add(int(parts[i + 1]))
        return len(layer_indices)

    def unpack_attention(self, layer_idx: int) -> AttentionProjections:
        """Unpack W_q, W_k, W_v, and W_o for the specified transformer layer."""
        prefix_candidates = [
            f"model.layers.{layer_idx}.self_attn",
            f"layers.{layer_idx}.self_attn",
        ]
        avail = set(self.list_tensor_paths())
        chosen_prefix = None
        for cand in prefix_candidates:
            if f"{cand}.q_proj.weight" in avail:
                chosen_prefix = cand
                break
        if chosen_prefix is None:
            raise KeyError(f"Could not locate self_attn projections for layer {layer_idx}.")

        paths = {
            "q": f"{chosen_prefix}.q_proj.weight",
            "k": f"{chosen_prefix}.k_proj.weight",
            "v": f"{chosen_prefix}.v_proj.weight",
            "o": f"{chosen_prefix}.o_proj.weight",
        }
        w_q = self.get_tensor(paths["q"])
        w_k = self.get_tensor(paths["k"])
        w_v = self.get_tensor(paths["v"])
        w_o = self.get_tensor(paths["o"])

        if self._model is not None and hasattr(self._model, "config"):
            cfg = self._model.config
            num_heads = int(getattr(cfg, "num_attention_heads", 1))
            num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
            head_dim = int(getattr(cfg, "head_dim", w_q.shape[0] // num_heads))
        else:
            # Infer head geometry from Q and K matrix shapes
            q_out, hidden_dim = w_q.shape
            k_out, _ = w_k.shape
            head_dim = 64 if q_out >= 64 and q_out % 64 == 0 else max(1, min(q_out, k_out))
            num_heads = max(1, q_out // head_dim)
            num_kv_heads = max(1, k_out // head_dim)

        return AttentionProjections(
            layer_idx=layer_idx,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            w_o=w_o,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            paths=paths,
        )

    def unpack_ffn(self, layer_idx: int) -> FFNProjections:
        """Unpack SwiGLU / FFN projection matrices (W_gate, W_up, W_down) for a layer."""
        prefix_candidates = [
            f"model.layers.{layer_idx}.mlp",
            f"layers.{layer_idx}.mlp",
        ]
        avail = set(self.list_tensor_paths())
        chosen_prefix = None
        for cand in prefix_candidates:
            if f"{cand}.down_proj.weight" in avail:
                chosen_prefix = cand
                break
        if chosen_prefix is None:
            raise KeyError(f"Could not locate MLP/SwiGLU projections for layer {layer_idx}.")

        paths = {
            "gate": f"{chosen_prefix}.gate_proj.weight",
            "up": f"{chosen_prefix}.up_proj.weight",
            "down": f"{chosen_prefix}.down_proj.weight",
        }
        w_gate = self.get_tensor(paths["gate"])
        w_up = self.get_tensor(paths["up"])
        w_down = self.get_tensor(paths["down"])

        intermediate_size, hidden_size = w_gate.shape
        return FFNProjections(
            layer_idx=layer_idx,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            paths=paths,
        )


def create_synthetic_qwen2_model(
    vocab_size: int = 512,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 4,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 4,
    seed: int = 42,
) -> Qwen2ForCausalLM:
    """Create a deterministic compact Qwen2 model in memory for testing or offline demos."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    with torch.no_grad():
        model = Qwen2ForCausalLM(config)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model
