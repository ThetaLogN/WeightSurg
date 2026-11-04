# WeightSurg (`weightsurg`)

**Zero-Gradient Weight-Space Programming & Neural Circuit Surgery Toolkit**

WeightSurg is a lightweight diagnostic and surgical toolkit that shifts neural network post-training from iterative stochastic gradient descent (`loss -> backward -> AdamW`) toward deterministic **Weight-Space Programming** and closed-form parameter editing. Designed for resource-constrained hardware (including Apple Silicon 8GB/16GB Macs), WeightSurg operates with **zero optimizer state overhead** (`0` gradient tensors, `< 6 GB` peak RAM for 0.5B–3B models).

---

## Core Architecture

```text
weightsurg/
├── core/
│   ├── ingestion.py     # Zero-grad Safetensors & HF Transformers loader + MLX hooks + head slicing
│   └── profiling.py     # Fast SVD, spectral/Frobenius norms, stable/effective rank, sparsity & outliers
├── diagnostics/
│   ├── schemas.py       # Standardized DiagnosticReport & PatchRecipe JSON schemas
│   └── localizer.py     # Activation Patching / Causal Tracing + OV Induction/Copy Head Detector
├── surgery/
│   ├── algebraic.py     # Closed-form rank-1 (ROME), low-rank (MEMIT), and singular-subspace damping
│   └── llm_guided.py    # Frontier LLM SVD subspace serialization, meta-prompting & spectral validation
├── patcher/
│   └── hotpatch.py      # In-place W_new = W_old + ΔW hot-patching, rollback, perplexity & GGUF/Safetensors export
└── cli.py               # Rich CLI (`weightsurg` / `weightsurg-cli`) with `inspect`, `trace`, `patch`
```

### Mathematical Formulation

1. **Rank-One Associative Edit (ROME-style):**
   $$\Delta W = \frac{(v - W k) k^\top}{\|k\|^2}$$
   Guarantees $(W + \Delta W) k = v$ while leaving any orthogonal input $k_\perp \perp k$ completely unperturbed: $(W + \Delta W) k_\perp = W k_\perp$.

2. **Low-Rank Multi-Association Edit (MEMIT-style):**
   $$\Delta W = (V - W K) K^\top (K K^\top + \lambda I)^{-1}$$

3. **Head-Local Singular Subspace Damping:**
   Given offending head $h$ in $W_O[:, h d_h : (h+1) d_h]$ with dominant singular triple $(u_0, \sigma_0, v_0^\top)$, applies a rank-1 associative update with key $k = v_0$ and damped target $v = \alpha \sigma_0 u_0$ ($\alpha \in [0, 1)$):
   $$\Delta W_O^{(h)} = (\alpha \sigma_0 u_0 - W_O^{(h)} v_0) v_0^\top = -(1 - \alpha) \sigma_0 u_0 v_0^\top$$

---

## CLI Usage

```bash
# 1. Inspect layer statistics, SVD spectrum, and weight anomaly hotspots
weightsurg inspect Qwen/Qwen2.5-0.5B-Instruct --max-layers 4

# 2. Trace and localize the attention head or MLP responsible for a token repetition loop
weightsurg trace \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain step by step: Why does a recursive function without a base case fail? Step 1:" \
  --target-error "Step 1:" \
  --output artifacts/diagnostic_report.json

# 3. Apply a closed-form surgical patch and export to Safetensors or GGUF
weightsurg patch \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --spec artifacts/diagnostic_report.json \
  --output patched_models/qwen2.5-fixed.safetensors
```

---

## Reproducible End-to-End Demo

Run the self-contained repetition loop localization and surgical repair script:

```bash
python3 examples/fix_repetition_loop.py --offline
```

---

## Running the Test Suite

```bash
python3 -m unittest discover -s tests -v
```

