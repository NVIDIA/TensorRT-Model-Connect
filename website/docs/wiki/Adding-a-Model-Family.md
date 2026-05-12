# Adding a New Model Family

## Autopilot (Recommended)

The fastest way to add new model families is the autopilot system, which autonomously discovers unsupported models from HuggingFace and implements them end-to-end — Python plugin, C++ runtime plugin (if needed), validation, and E2E manifest.

```bash
# One command — discovers gaps, implements, validates, reports
python3 scripts/autopilot/autorun.py --auto

# Interactive mode (shows candidates, asks Y/n)
python3 scripts/autopilot/autorun.py

# Limit scope
python3 scripts/autopilot/autorun.py --auto --limit 4 --min-downloads 5000000
```

The autopilot dispatches parallel Claude Code agents across isolated workspaces. Each agent:
1. Scaffolds a plugin via `scripts/new_family.py`
2. Builds the TRT bundle
3. Validates correctness (TRT vs HuggingFace comparison — agent picks the right metric per modality)
4. Creates a C++ runtime plugin if no existing strategy handles the model
5. Iterates until `./build/trtmc run <bundle> --prompt "..."` produces correct output
6. Creates the E2E manifest (no skip)

**Prerequisites**: Agent workspaces bootstrapped (`./scripts/bootstrap_workspace.sh --id agent-N --detach`) and `claude` CLI in PATH.

See `scripts/autopilot/autorun.py` for full options and `CLAUDE.md` for detailed documentation.

---

## Manual Path

Adding support for a new HuggingFace model family manually is a Python task in `tensorrt_model_connect/` **when the model reuses an existing runtime strategy** already handled by a C++ model runtime folder in `src/runtime/models/`. C++ edits are needed only when introducing a new `runtime_strategy` that no existing model folder handles.

## Prerequisites

The standard decoder builder is parameterized and handles most decoder-only architectures:

**Norm types**: RMSNorm (LLaMA, Qwen, etc.) or LayerNorm (GPT-2, Falcon, StableLM, etc.)
**MLP types**: SwiGLU (LLaMA, Qwen, etc.) or GELU FC (GPT-2, Falcon, StarCoder2, etc.)
**Position types**: RoPE (most modern models) or learned absolute (GPT-2, OPT)
**Activations**: silu, gelu_new, gelu, relu

Pass these as keyword arguments to `build_standard_decoder_engine()`:
```python
build_standard_decoder_engine(config, weights, max_cache_length,
                              norm_type="layernorm", mlp_type="gelu_fc",
                              position_type="learned", activation="gelu_new")
```

If your model uses one of these combinations, you only need a plugin file with weight mapping.

If your model diverges further (MoE routing, SSM/Mamba, parallel attention), you will need a custom `build_engine()` — see [Advanced: Custom Build Engine](#advanced-custom-build-engine) below. For existing strategies such as `decoder_moe`, `ssm_recurrent`, and `vision_language`, this is still Python-only; add C++ only for new strategy/state semantics.

## Quick Path: Scaffolding Script

The fastest way to add a new family:

```bash
# 1. Generate a plugin from a HuggingFace model's config.json
python3 scripts/new_family.py \
  --model-type phi3 \
  --hf-repo microsoft/Phi-3-mini-4k-instruct \
  --family-name phi

# 2. Review the generated plugin (customize if needed)
$EDITOR tensorrt_model_connect/tensorrt_model_connect/families/phi/plugin.py

# 3. Validate end-to-end (build + diff_logits + diff_layers + runner parity)
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct
```

The scaffolding script:
- Downloads `config.json` from the HF repo
- Detects architecture features (GQA, tied embeddings, explicit head_dim, MoE, etc.)
- Generates a plugin `.py` with correct `matches()`, standard `load_weights()` and `build_engine()`
- Adds comments noting detected features that may need attention

## Manual Path: Step-by-Step

### Step 1: Create the plugin file

Create `tensorrt_model_connect/tensorrt_model_connect/families/<family>.py`. The file must:
- Define a class implementing the `FamilyPlugin` protocol (see `base.py`)
- Expose a module-level `plugin` attribute (instance of the class)

```python
"""Yi family plugin."""

from __future__ import annotations

from ..config import ModelConfig
from ..checkpoint_mapper import WeightDict, load_standard_weights
from ..standard_decoder_builder import build_standard_decoder_engine


class YiPlugin:
    name = "yi"

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("yi")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        return load_standard_weights(model_dir, config)

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
    ) -> bytes:
        return build_standard_decoder_engine(
            config, weights, max_cache_length, verbose=verbose)


plugin = YiPlugin()
```

That's it — the plugin is auto-discovered. `families/__init__.py` uses `pkgutil.iter_modules()` to find any `.py` file with a `plugin` attribute. No registration code needed.

### Step 2: Customize weight loading (if needed)

Most models use standard HF tensor naming (same as LLaMA):
```
model.embed_tokens.weight
model.layers.N.input_layernorm.weight
model.layers.N.self_attn.{q,k,v,o}_proj.weight
model.layers.N.post_attention_layernorm.weight
model.layers.N.mlp.{gate,up,down}_proj.weight
model.norm.weight
lm_head.weight
```

If your model uses standard naming, `load_standard_weights()` handles everything — including transposing, GQA expansion, tied embeddings, and optional q/k norms.

For non-standard models, customize `load_weights()`. Example from Gemma (adds +1.0 to RMSNorm gamma, scales embedding):

```python
def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
    weights = load_standard_weights(model_dir, config)

    # Gemma: (1 + gamma) * normalized
    for i in range(config.num_hidden_layers):
        weights[f"layer.{i}.input_norm"] += 1.0
        weights[f"layer.{i}.post_attn_norm"] += 1.0
    weights["final_norm"] += 1.0

    # Gemma: scale embedding by sqrt(hidden_size)
    weights["embedding"] *= math.sqrt(config.hidden_size)
    return weights
```

Some models use fused projections (e.g., Phi-3 ships a single `qkv_proj` instead of separate Q/K/V, and a single `gate_up_proj` instead of separate gate/up). In these cases, split the fused tensor during weight loading. See `tensorrt_model_connect/tensorrt_model_connect/families/phi/plugin.py` for an example.

### Step 3: Validate

Run the one-command validation gate:

```bash
./scripts/validate_family.sh <hf-repo-or-local-path>
```

This runs:
1. `trtmc-build build` — builds a `.trtfb` bundle
2. `diff_logits.py --battery` — E2E logit comparison (4 prompts)
3. `diff_layers.py` — per-layer hidden state comparison
4. `test_runner_parity.py` — Python-vs-C++ cross-validation

Or run each step individually:

```bash
# Build bundle
trtmc-build build <model> -o /tmp/test.trtfb --max-cache-length 256

# E2E logit comparison (per-step, all tokens)
python3 tools/diff_logits.py --model <model> --atol 1e-3 --battery

# Per-layer hidden state comparison
python3 tools/diff_layers.py --model <model> --atol 0.05

# Python-vs-C++ runner parity
python3 tools/test_runner_parity.py \
  --bundle /tmp/test.trtfb --binary ./build/trtmc \
  --hf-python .venv/bin/python --max-new-tokens 20
```

For models that require custom tokenizer code (e.g., Phi-3), add `--trust-remote-code` to diff_logits.py, diff_layers.py, and validate_family.sh.

**Memory note**: Large models (3B+ parameters) can require significant RAM during TRT engine compilation. Phi-3-mini (3.8B) peaks at ~44GB. On 64GB machines, 16GB swap is recommended.

## Checklist

| Step | Files | Lines |
|------|-------|-------|
| Plugin file | `families/<family>.py` | ~30 (standard decoder), ~60 (extended decoder), ~300+ (custom graph) |
| **Total** | **1 new file, 0 existing files edited** | **~30-300** |

## FamilyPlugin Protocol

From `tensorrt_model_connect/tensorrt_model_connect/families/base.py`:

```python
class FamilyPlugin(Protocol):
    name: str

    def matches(self, model_type: str) -> bool: ...
    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict: ...
    def build_engine(self, config: ModelConfig, weights: WeightDict,
                     max_cache_length: int, *, verbose: bool = False) -> bytes: ...
```

## Advanced: Custom Build Engine

If your model has an architecture not covered by the parameterized standard builder, override `build_engine()` to use custom graph construction. The shared TRT graph ops in `tensorrt_model_connect/tensorrt_model_connect/graph_ops.py` (RMSNorm, LayerNorm, RoPE, matmul, attention, SwiGLU, GELU, etc.) are reusable building blocks -- compose them differently for your architecture.

### Already implemented custom architectures

- **MoE (Phi-MoE)**: SparseMixer routing + per-expert SwiGLU MLPs. See `families/phi_moe.py`. Uses `runtime_strategy="decoder_moe"` (same KV-cache C++ backend).
- **Mamba/SSM**: Selective state space model with conv1d + selective scan. See `families/mamba.py`. Uses `runtime_strategy="ssm_recurrent"` and reuses the existing C++ `MambaBackend` (`src/runtime/pipelines/recurrent_pipeline.cpp`).
- **Vision-Language (Qwen-VL)**: Vision encoder (ViT + 3D RoPE + spatial merge) + text decoder with embed_input. See `families/qwen_vl.py`. Uses `runtime_strategy="vision_language"`. Requires `build_vision_engine()` and `get_vl_config()` methods.

### Adding a Vision-Language Family

VL plugins require two additional methods beyond the standard `FamilyPlugin` protocol:

1. **`build_vision_engine()`**: Build a TRT engine for the vision encoder. Return serialized engine bytes or `None`.
2. **`get_vl_config()`**: Return a dict with VL configuration to inject into the bundle's config.json:
   - `preprocessor_type`: Image preprocessing strategy (`"qwen_merge_group"`, `"simple_chw"`, `"center_crop_chw"`, or `"aspect_preserve_chw"`)
   - `interpolation`: Resize interpolation mode (`"bicubic"` (default), `"bilinear"`, or `"nearest"`)
   - `image_token_id`, `num_image_pad_tokens`, `vision_output_dim`
   - `vl_prompt_template`, `image_token_str`

The C++ runtime handles 4 image preprocessing strategies out of the box. No C++ changes needed for standard VL models.

### Architectures needing custom `build_engine()`

- **MLA (Multi-head Latent Attention)**: Compressed KV cache (DeepSeek-V2/V3). Needs Python graph builder + C++ cache shape changes.
- **Parallel attention**: Attention and MLP computed in parallel (GPT-J style).
- **Hybrid SSM+Attention (Jamba)**: Mix of attention and Mamba layers.
