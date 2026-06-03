"""Base DiT denoiser builder for Cosmos-Transfer1-7B / Cosmos-Predict1-7B.

The base model is a 7B DiT (Diffusion Transformer) with:
  * Hidden dim:   4096            (32 heads * 128 head_dim — head count
                                   confirmed via NVIDIA HF model card)
  * Num blocks:   28              (Predict1-7B; TODO confirm on weights)
  * FFN:          SwiGLU, ~16384  (4 * dim with gated MLP)
  * Text cond.:   T5-XXL, 4096-d, max_seq_len 512
  * Time cond.:   adaLN-Zero per block (6 modulation params: shift/scale/gate
                  for self-attn and FFN)
  * RoPE:         3-axis (T, H, W) — see Cosmos-Predict1 reference impl,
                  cosmos_predict1/diffusion/module/position_embedding.py
  * Patch:        (1, 2, 2)        (no temporal patching, 2x2 spatial)
  * In channels:  16               (z_dim of Cosmos-Tokenizer CV8x8x8)

ControlNet injection
--------------------
Cosmos-Transfer adds an *additional* input to the base DiT: a list of
control-feature maps from the active ControlNet branches. For each base
block ``i < num_control_blocks`` the hidden state is updated as:

    hidden += sum_m alpha_m * control_features_m[i]

where ``m`` iterates over the active modalities (edge / depth / ...) and
``alpha_m`` is a per-modality scalar weight (typically 1.0 / num_active).

The injection happens *before* the block's adaLN normalization, matching
the reference impl in github.com/nvidia-cosmos/cosmos-transfer1
cosmos_transfer1/diffusion/module/blocks.py:ControlNetTransformerBlock.

This builder is currently a *signature-only* scaffold. The body raises
NotImplementedError with a clear list of open questions because the
exact Cosmos DiT block layout cannot be implemented without a GPU
checkpoint inspection (the public reference impl exists, but the weight
tensor shapes need to be confirmed against the actual ``base_model.pt``).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# Default architecture for Cosmos-Transfer1-7B / Cosmos-Predict1-7B.
# Values come from the HF model card + arxiv 2503.14492. The ones marked
# TODO are best-effort guesses and must be confirmed against base_model.pt
# during the first GPU validation run.
DIM = 4096
NUM_HEADS = 32           # confirmed: HF model card "Attention heads: 32"
NUM_LAYERS = 28          # TODO confirm
HEAD_DIM = DIM // NUM_HEADS  # 128
FFN_DIM = 16384          # TODO confirm; 4 * dim is the standard SwiGLU width
TEXT_SEQ_LEN = 512       # T5-XXL default
TEXT_CONTEXT_DIM = 4096  # T5-XXL d_model
IN_CHANNELS = 16         # Cosmos-Tokenizer CV8x8x8 z_dim
PATCH_SIZE = (1, 2, 2)
NUM_CONTROL_BLOCKS = 7   # TODO confirm; arxiv §3.1 mentions "first N blocks"


def load_base_dit_weights(pt_path: str) -> "WeightDict":
    """Load base DiT weights from ``base_model.pt``.

    Thin wrapper around pt_loader.load_pt_state_dict that returns a typed
    WeightDict and stamps provenance fields used by the plugin's debug
    output.
    """
    from ...checkpoint_mapper import WeightDict
    from .pt_loader import load_pt_state_dict

    raw = load_pt_state_dict(pt_path)
    w = WeightDict()
    w["_role"] = "base_dit"
    w["_source_pt"] = str(pt_path)
    for k, v in raw.items():
        w[k] = v
    return w


def build_cosmos_dit_engine(
    weights: "WeightDict",
    *,
    dim: int = DIM,
    num_heads: int = NUM_HEADS,
    num_layers: int = NUM_LAYERS,
    ffn_dim: int = FFN_DIM,
    context_dim: int = TEXT_CONTEXT_DIM,
    num_patches: int,
    text_seq_len: int = TEXT_SEQ_LEN,
    num_control_inputs: int = 0,
    num_control_blocks: int = NUM_CONTROL_BLOCKS,
    verbose: bool = False,
) -> bytes:
    """Build the Cosmos base-DiT denoiser TRT engine plan.

    Engine I/O signature
    --------------------
    Inputs:
        hidden_states       [1, num_patches, dim] float32
        timestep_embedding  [1, 6 * dim]          float32   (external MLP)
        time_embed          [1, dim]              float32   (for final out
                                                              modulation)
        encoder_hidden      [1, text_seq_len, context_dim] float32
        rotary_cos_t        [1, num_patches, 1, head_dim//3] float32
        rotary_sin_t        [1, num_patches, 1, head_dim//3] float32
        rotary_cos_h        [1, num_patches, 1, head_dim//3] float32
        rotary_sin_h        [1, num_patches, 1, head_dim//3] float32
        rotary_cos_w        [1, num_patches, 1, head_dim//3] float32
        rotary_sin_w        [1, num_patches, 1, head_dim//3] float32
            (3-axis RoPE; head_dim is split across T/H/W as in Predict1)

        # ControlNet injection (only when num_control_inputs > 0):
        control_features    [num_control_inputs, num_control_blocks,
                             num_patches, dim] float32
        control_weights     [num_control_inputs] float32     (alpha_m
                                                              per modality)
    Outputs:
        denoised            [1, num_patches, dim] float32

    Args:
        weights: Output of ``load_base_dit_weights``.
        num_control_inputs: How many ControlNet branches are wired in this
            build. 0 disables the injection inputs entirely (so the engine
            can also serve the plain Cosmos-Predict1 use case).
        num_control_blocks: Number of leading blocks that consume control
            features (must match what the ControlNet branches produce).
    """
    head_dim = dim // num_heads
    if head_dim % 3 != 0:
        # 3-axis RoPE requires head_dim divisible by 3. The 7B model uses
        # head_dim=128, which is NOT cleanly divisible by 3 — Cosmos handles
        # this with a 44/44/40 axis split. This will need to be confirmed
        # against the actual reference impl before the builder is written.
        print(
            f"[cosmos-transfer] NOTE: head_dim={head_dim} is not divisible "
            f"by 3 — Cosmos splits the head dim as (44, 44, 40) for "
            f"(T, H, W) RoPE in the 7B model. Confirm before implementing.",
            file=sys.stderr,
        )

    raise NotImplementedError(
        "cosmos_dit_builder is a structural stub; the TRT graph "
        "construction is pending GPU-side weight-key confirmation. "
        "Open questions (must answer before implementing): "
        "(1) Exact weight-key prefix in base_model.pt — 'blocks.{i}.' "
        "(predict1 style) or 'transformer.h.{i}.' (HF style)? "
        "(2) SwiGLU activation key layout — separate w_gate/w_up or "
        "fused w_gate_up like Llama? "
        "(3) adaLN-Zero: is the scale_shift_table per-block (Wan-style, "
        "[1,6,dim]) or a single global table indexed by block id? "
        "(4) 3-axis RoPE head-dim split — 44/44/40 for d=128? Or "
        "different fractions? "
        "(5) Text conditioning: cross-attention K/V projected inside the "
        "engine (Wan/FLUX-style) or projected externally and passed in "
        "pre-projected? "
        "(6) Are the ControlNet injection points before or after the "
        "block's adaLN-Zero normalization? (impacts where we wire the "
        "additive elementwise op)."
    )
