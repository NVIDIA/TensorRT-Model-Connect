"""ControlNet branch builder for Cosmos-Transfer1-7B.

A Cosmos-Transfer ControlNet branch is structurally a *copy* of the first
``N_ctrl`` transformer blocks of the base DiT (typically 7 of 28 in the
7B model, per the arxiv paper §3.1 "Control Branch"). It takes:

  1. A control video latent ``c_lat`` of the same shape as the base
     latent ``z_t`` (after the Cosmos VAE has encoded the control video).
  2. The same timestep / text-conditioning the base DiT sees.

and produces a list of per-block feature maps ``[f_0, f_1, ..., f_{N_ctrl-1}]``
that are *additively injected* into the corresponding base DiT blocks before
their self-attention LayerNorm, i.e.

    hidden = hidden + alpha_i * f_i        (additive feature injection)

where ``alpha_i`` is a per-block scalar (often called the ControlNet weight
in NVIDIA's reference impl, defaults to 1.0). The conditioning input goes
through a learned zero-initialized linear projection — that's what makes
ControlNet training stable.

NOTE: This builder is currently a *structural scaffold* — it lays out the
expected weight keys, input/output shapes, and conditioning math, but the
core block body delegates to a NotImplementedError until the exact Cosmos
DiT block variant is wired in (see cosmos_dit_builder.py). At that point,
this file should be refactored to share block construction code with the
base DiT, since the ControlNet branch is literally the first N blocks of
the base model.

References:
  * arxiv 2503.14492 §3, Figure 3 (control branch diagram).
  * github.com/nvidia-cosmos/cosmos-transfer1 cosmos_transfer1/diffusion/
    module/blocks.py — exact PyTorch reference.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# Default number of control blocks in the 7B variant (28 base blocks total).
DEFAULT_NUM_CONTROL_BLOCKS = 7


def build_cosmos_controlnet_engine(
    weights: "WeightDict",
    *,
    modality: str,
    dim: int,
    num_heads: int,
    num_control_blocks: int = DEFAULT_NUM_CONTROL_BLOCKS,
    ffn_dim: int,
    context_dim: int,
    num_patches: int,
    text_seq_len: int = 512,
    in_channels: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    verbose: bool = False,
) -> bytes:
    """Build a TRT engine for one Cosmos-Transfer ControlNet branch.

    Args:
        weights: Flat WeightDict produced by ``pt_loader.load_pt_state_dict``
            on the corresponding ``<modality>_control.pt`` file. Expected keys
            (mirroring the base DiT, prefix ``blocks.{i}.``):
                - patch_embedding.weight (Conv3D, [out, in*kt*kh*kw])
                  for the *control* input (separate from base patch embed).
                - blocks.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
                - blocks.{i}.cross_attn.{q,k,v,o}_proj.{weight,bias}
                - blocks.{i}.mlp.{fc1,fc2}.{weight,bias}
                - blocks.{i}.adaLN.{weight,bias}        (time conditioning)
                - blocks.{i}.norm1, norm2, norm3        (no weight, affine=False)
                - zero_proj.{i}.weight                  (zero-init output proj
                                                         from this block into
                                                         the base DiT feature
                                                         map; shape [dim, dim])
            and optionally:
                - cond_embed.{0,2}.weight/bias  (optional MLP between control
                                                 latent and DiT hidden dim)
        modality: One of "edge" / "depth" / "seg" / "vis" / "keypoint". Used
            only for engine-name tagging / debug prints; the architecture is
            identical across modalities (Cosmos-Transfer ships separate
            *weights* per modality, not separate *shapes*).
        dim: DiT hidden dim (e.g. 4096 for the 7B model).
        num_heads: Attention heads (32 in the 7B model — confirmed via
            HF model card, "Attention heads: 32").
        num_control_blocks: How many DiT blocks the control branch replicates.
            Defaults to 7 for the 7B variant (TODO: confirm on weights).
        ffn_dim: MLP inner dim (e.g. 4*dim with SwiGLU = 16384).
        context_dim: Text-encoder output dim (T5-XXL -> 4096).
        num_patches: T_lat/pt * H_lat/ph * W_lat/pw, must match base DiT.
        text_seq_len: Max text tokens (T5-XXL: 512 in Cosmos default).
        in_channels: Channels in the control-latent input. Cosmos uses the
            same VAE for control + base, so this equals the base latent
            channel count (z_dim = 16 for Cosmos-Tokenizer CV8x8x8).
        patch_size: 3D patch (pt, ph, pw); Cosmos uses (1, 2, 2) per Predict1.
        verbose: TRT builder verbose logging.

    Returns:
        Serialized TRT engine plan bytes. The engine signature is:
            Inputs:
              control_latent      [1, num_patches, dim] float32
              timestep_embedding  [1, 6 * dim]          float32
              encoder_hidden      [1, text_seq_len, context_dim] float32
              rotary_cos          [1, num_patches, 1, head_dim]  float32
              rotary_sin          [1, num_patches, 1, head_dim]  float32
            Outputs:
              control_features    [num_control_blocks, num_patches, dim]
                                  float32   (one feature map per block,
                                             already zero-projected; the
                                             base DiT just adds them in.)
    """
    # ------------------------------------------------------------------
    # Pre-flight: weight shape sanity. This is cheap to check before we
    # call into TensorRT, and gives a clear error if the .pt file is the
    # wrong modality / a different release.
    # ------------------------------------------------------------------
    head_dim = dim // num_heads
    if dim % num_heads != 0:
        raise ValueError(
            f"dim={dim} not divisible by num_heads={num_heads}; "
            f"refusing to build {modality} ControlNet engine.")

    expected_keys = [
        "patch_embedding.weight",
    ]
    for i in range(num_control_blocks):
        expected_keys.extend([
            f"blocks.{i}.self_attn.q_proj.weight",
            f"blocks.{i}.self_attn.k_proj.weight",
            f"blocks.{i}.self_attn.v_proj.weight",
            f"blocks.{i}.self_attn.o_proj.weight",
            f"zero_proj.{i}.weight",
        ])

    missing = [k for k in expected_keys if k not in weights]
    if missing:
        # Don't hard-fail — the exact Cosmos weight names are not yet
        # locked down (different releases use slightly different keys,
        # see pt_loader.py docstring). Log a clear warning so a GPU
        # validation run surfaces the renames immediately.
        print(
            f"[cosmos-transfer] WARNING: {modality} ControlNet checkpoint is "
            f"missing {len(missing)} expected keys (first: "
            f"{missing[0]!r}); the runtime will need a key rename. "
            f"See pt_loader.py for known aliases.",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Stub: actual TRT graph construction is not implemented yet because
    # the exact Cosmos block layout (RoPE variant, attention mask flavor,
    # adaLN-Zero vs adaLN, SwiGLU vs GELU) cannot be confirmed without a
    # GPU repro. See module docstring for the missing pieces.
    # ------------------------------------------------------------------
    raise NotImplementedError(
        f"cosmos-transfer {modality} ControlNet TRT graph construction "
        f"is pending GPU-side validation. Expected I/O signature is "
        f"documented in this function's docstring; once the base DiT "
        f"builder lands (cosmos_dit_builder.py) this should share its "
        f"block body. Open questions: "
        f"(1) RoPE axis layout — 3D RoPE (Cosmos-Predict1 style) vs 1D "
        f"flattened? "
        f"(2) Attention mask — full bidirectional or causal in temporal "
        f"dim? "
        f"(3) adaLN scale_shift_table layout: per-block [1, 6, dim] (Wan) "
        f"or global [num_blocks, 6, dim]? "
        f"(4) ControlNet output: zero-init linear per block, or zero-init "
        f"applied only to the first/last block?"
    )


def load_controlnet_weights(
    pt_path: str,
    *,
    modality: str,
) -> "WeightDict":
    """Load a Cosmos-Transfer ControlNet ``.pt`` file into a WeightDict.

    Thin wrapper around ``pt_loader.load_pt_state_dict`` that:
      * Returns a typed WeightDict (so downstream type hints line up).
      * Tags ``_modality`` into the dict so debug / error messages can
        identify which branch a weight set belongs to.
    """
    from ...checkpoint_mapper import WeightDict
    from .pt_loader import load_pt_state_dict

    raw = load_pt_state_dict(pt_path)
    w = WeightDict()
    w["_modality"] = modality
    w["_source_pt"] = str(pt_path)
    for k, v in raw.items():
        w[k] = v
    return w


def count_expected_control_keys(
    num_control_blocks: int = DEFAULT_NUM_CONTROL_BLOCKS,
) -> int:
    """Return the rough number of tensor keys we expect per branch.

    Useful for the plugin to assert the .pt file isn't truncated /
    corrupted before we ever launch a build.
    """
    # Per-block: 4x attn (qkvo) + 2x mlp + 2x norm bias + 1x zero_proj  ~= 12
    # Plus global: patch_embed (1), cond_embed (2), final_norm (1)       ~= 4
    return 12 * num_control_blocks + 4
