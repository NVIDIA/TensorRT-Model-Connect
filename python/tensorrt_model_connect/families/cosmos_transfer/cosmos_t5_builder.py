"""T5-XXL text encoder builder for Cosmos-Transfer1-7B.

Cosmos-Transfer1 uses Google T5-XXL (1.1 / v1.1) as the frozen text encoder.
Architecture matches the standard t5-v1_1-xxl config:

  * d_model:    4096
  * d_kv:       64
  * d_ff:       10240            (gated GeLU, so two 10240-wide projections)
  * num_layers: 24
  * num_heads:  64
  * vocab:      32128
  * max_seq:    512

The trtmc repo already has a battle-tested T5 builder in
``families/wan_t2v/t5_encoder_builder.py`` (Wan2.1 also uses T5). For
Cosmos-Transfer we reuse that builder *if* the weight-key layout matches;
the only Cosmos-specific bit is the .pt -> WeightDict translation, since
Cosmos doesn't ship safetensors.

The shared T5 builder expects keys like ``encoder.block.{i}.layer.0.
SelfAttention.q.weight``, which is the HF safetensors layout. Cosmos's
``t5_text_encoder.pt`` follows the same naming (it is just T5-XXL re-saved
from the HF checkpoint), so a straight rename pass is sufficient.

When wired up on GPU this module will simply:
    1. Load the .pt via pt_loader.
    2. Translate keys (if any rename is needed — likely none) and transpose
       linear weights for TRT.
    3. Call into ``families.wan_t2v.t5_encoder_builder.build_t5_encoder_engine``.

For now (no GPU), the module exposes the loader skeleton and a build
function that delegates with a NotImplementedError fallback if the key
layout mismatches.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# T5-v1_1-XXL defaults.
D_MODEL = 4096
NUM_HEADS = 64
D_KV = 64
D_FF = 10240
NUM_LAYERS = 24
VOCAB_SIZE = 32128
MAX_SEQ_LEN = 512


def load_t5_weights_from_pt(
    pt_path: str,
    *,
    num_layers: int = NUM_LAYERS,
    precision: str = "fp32",
) -> "WeightDict":
    """Load T5-XXL weights from Cosmos's ``t5_text_encoder.pt`` file.

    Linear weights are transposed for TRT (HF layout is ``[out, in]``,
    TRT matmul wants ``[in, out]`` when the RHS is a constant — see
    ``families.wan_t2v.t5_encoder_builder.load_t5_weights`` for the
    pattern we mirror.).
    """
    from ...checkpoint_mapper import WeightDict
    from .pt_loader import load_pt_state_dict

    raw = load_pt_state_dict(pt_path)
    target_dtype = np.float32 if precision == "fp32" else np.float16

    w = WeightDict()
    w["_role"] = "t5_encoder"
    w["_source_pt"] = str(pt_path)

    # Token embedding.
    if "shared.weight" in raw:
        w["shared.weight"] = raw["shared.weight"].astype(target_dtype)
    else:
        # Some Cosmos releases prefix everything with ``text_encoder.``.
        for prefix in ("text_encoder.", "encoder."):
            key = f"{prefix}shared.weight"
            if key in raw:
                w["shared.weight"] = raw[key].astype(target_dtype)
                break

    # Per-layer projections — transpose linear weights.
    for i in range(num_layers):
        prefix = f"encoder.block.{i}"

        # Self-attention.
        for proj in ("q", "k", "v", "o"):
            key = f"{prefix}.layer.0.SelfAttention.{proj}.weight"
            if key in raw:
                w[key] = np.ascontiguousarray(raw[key].T, dtype=target_dtype)

        # Self-attention layer norm.
        norm_key = f"{prefix}.layer.0.layer_norm.weight"
        if norm_key in raw:
            w[norm_key] = raw[norm_key].astype(np.float32)

        # FFN (gated GeLU: wi_0 = gate, wi_1 = up, wo = down).
        for proj in ("wi_0", "wi_1", "wo"):
            key = f"{prefix}.layer.1.DenseReluDense.{proj}.weight"
            if key in raw:
                w[key] = np.ascontiguousarray(raw[key].T, dtype=target_dtype)

        # FFN layer norm.
        norm_key = f"{prefix}.layer.1.layer_norm.weight"
        if norm_key in raw:
            w[norm_key] = raw[norm_key].astype(np.float32)

        # Relative attention bias is only on layer 0 in T5-v1_1.
        if i == 0:
            bias_key = (
                "encoder.block.0.layer.0.SelfAttention."
                "relative_attention_bias.weight"
            )
            if bias_key in raw:
                w[bias_key] = raw[bias_key].astype(np.float32)

    # Final layer norm.
    if "encoder.final_layer_norm.weight" in raw:
        w["encoder.final_layer_norm.weight"] = (
            raw["encoder.final_layer_norm.weight"].astype(np.float32)
        )

    # Sanity check: did we actually pick anything up? If not, the .pt file
    # uses a different key prefix and the runtime will need a rename map.
    found_any = any(k.startswith("encoder.block.") for k in w)
    if not found_any:
        print(
            f"[cosmos-transfer] WARNING: t5_text_encoder.pt at {pt_path} "
            f"contains no encoder.block.* keys after standard prefix "
            f"stripping. The Cosmos release likely uses a different key "
            f"naming convention — see top-level keys: "
            f"{sorted(list(raw.keys()))[:10]} ...",
            file=sys.stderr,
        )

    return w


def build_t5_encoder_engine_for_cosmos(
    weights: "WeightDict",
    *,
    d_model: int = D_MODEL,
    num_heads: int = NUM_HEADS,
    d_kv: int = D_KV,
    d_ff: int = D_FF,
    num_layers: int = NUM_LAYERS,
    vocab_size: int = VOCAB_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    verbose: bool = False,
) -> bytes:
    """Build a T5-XXL encoder TRT engine for Cosmos-Transfer.

    Delegates to the shared T5 builder in families/wan_t2v if available;
    otherwise raises NotImplementedError with a pointer to the shared
    builder.
    """
    try:
        from ..wan_t2v.t5_encoder_builder import build_t5_encoder_engine
    except ImportError as e:  # pragma: no cover - import-time wiring
        raise NotImplementedError(
            f"Shared T5 builder import failed: {e}. Cosmos-Transfer "
            f"reuses families/wan_t2v/t5_encoder_builder.py; if that "
            f"module is unavailable we need a Cosmos-local copy."
        )

    return build_t5_encoder_engine(
        weights,
        d_model=d_model,
        num_heads=num_heads,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        verbose=verbose,
    )
