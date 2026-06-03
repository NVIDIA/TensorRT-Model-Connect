"""byT5 text encoder builder for HunyuanImage-2.1 (scaffold).

HunyuanImage-2.1 uses two text encoders:
  1. byT5 (google/byt5-small or similar byte-level T5)
  2. Qwen2.5-VL (LLM-style sequence encoder)

byT5 is a byte-level T5 variant that operates on UTF-8 bytes rather than
SentencePiece subwords. Architecturally it shares the standard T5 encoder
stack (RMSNorm + relative-attention-bias self-attention + GEGLU MLP), so
we can reuse the shared ``flux.t5_encoder_builder`` infrastructure as
long as the actual byT5 config (vocab=256+offset, d_model/d_ff/num_layers
that differ from FLUX's T5-XXL) is plumbed through.

byT5 reference (Xue et al., 2022, "ByT5: Towards a Token-Free Future"):
  - vocab_size  = 384  (256 UTF-8 bytes + 128 special/extra ids)
  - byt5-small  : d_model=1472, d_ff=3584, num_layers=12, num_heads=6, d_kv=64
  - byt5-base   : d_model=1536, d_ff=3968, num_layers=18, num_heads=12, d_kv=64

The HunyuanImage repo packages a customized byT5; the exact variant
(small / base / fine-tuned head) is read from ``text_encoder/config.json``
in :class:`HunyuanImagePlugin.load_weights` and forwarded to this builder.

----------------------------------------------------------------------------
GAPS — fill in when ``tencent/HunyuanImage-2.1`` is fetched on a GPU host:

  1. Confirm whether HunyuanImage uses the byt5-small or byt5-base head
     (or a custom finetune). Read ``text_encoder/config.json``::
       - architectures / model_type
       - d_model, d_ff, num_layers, num_heads, d_kv
       - vocab_size  (byT5 typically 384)
       - max_seq_len contract used by the HunyuanImage pipeline
         (Tencent's reference code clamps to 128 tokens for byT5).

  2. Decide whether to reuse ``flux.t5_encoder_builder.build_t5_encoder_engine``
     directly or fork it: byT5 uses the same T5 ops, but the embedding
     table is much smaller and the relative-attention bucket count may
     differ. The FLUX builder hardcodes some FLUX-T5-XXL behaviour
     (e.g. ``relative_attention_num_buckets=32``), which is also the
     byT5 default — so reuse should be safe.

  3. byT5 weight prefix in safetensors. T5 family uses
     ``shared.weight`` + ``encoder.block.{i}.layer.0|1.*``; verify that
     the HunyuanImage checkpoint preserves this prefix or whether it
     wraps it under ``text_encoder.*``.

Until those are confirmed on hardware, this module exposes a
``build_byt5_encoder_engine`` shim that delegates to the shared FLUX T5
builder. The shim is GPU-validated when the surrounding plugin lands.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# byT5 architectural defaults (byt5-small). Override from config.json
# in the plugin once we read the real checkpoint.
DEFAULT_BYT5_D_MODEL = 1472
DEFAULT_BYT5_NUM_HEADS = 6
DEFAULT_BYT5_D_KV = 64
DEFAULT_BYT5_D_FF = 3584
DEFAULT_BYT5_NUM_LAYERS = 12
DEFAULT_BYT5_VOCAB_SIZE = 384
DEFAULT_BYT5_MAX_SEQ_LEN = 128


def load_byt5_weights(
    text_encoder_dir: str,
    *,
    d_model: int = DEFAULT_BYT5_D_MODEL,
    num_heads: int = DEFAULT_BYT5_NUM_HEADS,
    d_kv: int = DEFAULT_BYT5_D_KV,
    d_ff: int = DEFAULT_BYT5_D_FF,
    num_layers: int = DEFAULT_BYT5_NUM_LAYERS,
    vocab_size: int = DEFAULT_BYT5_VOCAB_SIZE,
    precision: str = "fp32",
):
    """Load byT5 encoder weights from a diffusers-format text_encoder dir.

    Delegates to ``flux.t5_encoder_builder.load_t5_weights`` because byT5
    shares the T5 weight layout. Override defaults using values read from
    ``text_encoder/config.json``.

    NOTE: This is a scaffold. Validate the safetensors key layout when
    the real HunyuanImage byT5 checkpoint is downloaded -- some Tencent
    checkpoints wrap encoder weights under a ``text_encoder.`` prefix.
    """
    from ..flux.t5_encoder_builder import load_t5_weights

    return load_t5_weights(
        text_encoder_dir,
        d_model=d_model,
        num_heads=num_heads,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        vocab_size=vocab_size,
        precision=precision,
    )


def build_byt5_encoder_engine(
    weights: "WeightDict",
    *,
    d_model: int = DEFAULT_BYT5_D_MODEL,
    num_heads: int = DEFAULT_BYT5_NUM_HEADS,
    d_kv: int = DEFAULT_BYT5_D_KV,
    d_ff: int = DEFAULT_BYT5_D_FF,
    num_layers: int = DEFAULT_BYT5_NUM_LAYERS,
    vocab_size: int = DEFAULT_BYT5_VOCAB_SIZE,
    max_seq_len: int = DEFAULT_BYT5_MAX_SEQ_LEN,
    verbose: bool = False,
) -> bytes:
    """Build the byT5 encoder TRT engine for HunyuanImage-2.1.

    Engine I/O matches the shared FLUX T5 builder:
        Input  : input_ids       [1, max_seq_len] int32
                 attention_mask  [1, max_seq_len] float32 (0 valid / -1e9 pad)
        Output : text_embeddings [1, max_seq_len, d_model] float32

    GAP: confirm against ``text_encoder/config.json`` -- in particular,
    ``relative_attention_num_buckets`` and ``relative_attention_max_distance``
    are passed through with T5 defaults (32 / 128); byT5 inherits these
    but a custom head could override them.
    """
    from ..flux.t5_encoder_builder import build_t5_encoder_engine

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
