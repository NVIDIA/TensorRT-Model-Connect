"""Cosmos3 autoregressive (AR) reasoner TRT engine builder.

The Cosmos 3 AR reasoner is architecturally identical to Qwen3-VL 32B Instruct
(per ARCH.md; the config explicitly names ``Qwen/Qwen3-VL-32B-Instruct`` as
the reasoner backbone). It is the text-side subsequence in the Cosmos 3
Mixture-of-Transformers — when ``joint_attn_implementation`` is in single-lane
text-only mode (no DM tokens present), it behaves exactly like Qwen3-VL.

This builder therefore composes the existing Qwen-VL TP decoder primitives
rather than re-implementing them, with two responsibilities:

  1. Inject Cosmos 3 architectural constants (verified against
     ``nvidia/Cosmos3-Super/config.json``) when the caller does not supply a
     fully-populated ``ModelConfig``.
  2. Default ``deepstack_num_levels=3`` (Cosmos 3 uses ViT layers 8/16/24 as
     visual feature levels, matching ``deepstack_visual_indexes`` in the
     vision config).

For the joint AR+DM mode required by image/video generation, see
``cosmos3_dm_generator_builder.py`` (Phase 4) and the joint-attention runtime
(Phase 6) — those layers interleave AR and DM tokens through a shared
attention pool, which the single-lane reasoner engine emitted here does not
handle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..qwen_vl.decoder_tp_builder import build_qwen_vl_tp_decoder_engine
from ..qwen_vl.standard_decoder_builder import build_standard_decoder_engine
from ...parallel_config import normalize_parallel_config

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig


# Cosmos3-Super reasoner constants (locked from config.json — see ARCH.md).
COSMOS3_SUPER_REASONER_HIDDEN_SIZE = 5120
COSMOS3_SUPER_REASONER_NUM_LAYERS = 64
COSMOS3_SUPER_REASONER_NUM_HEADS = 64
COSMOS3_SUPER_REASONER_NUM_KV_HEADS = 8
COSMOS3_SUPER_REASONER_HEAD_DIM = 128
COSMOS3_SUPER_REASONER_INTERMEDIATE_SIZE = 25600
COSMOS3_SUPER_REASONER_VOCAB_SIZE = 151936
COSMOS3_SUPER_REASONER_ROPE_THETA = 5_000_000
COSMOS3_SUPER_REASONER_RMS_NORM_EPS = 1e-6
COSMOS3_SUPER_REASONER_MAX_POSITION_EMBEDDINGS = 262_144
COSMOS3_SUPER_REASONER_MROPE_SECTION = (24, 20, 20)

# Cosmos3-Super ViT exports intermediate features at these layers (deepstack).
COSMOS3_SUPER_DEEPSTACK_INDEXES = (8, 16, 24)


def build_cosmos3_ar_reasoner_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "bf16",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
    deepstack_num_levels: int = len(COSMOS3_SUPER_DEEPSTACK_INDEXES),
) -> bytes:
    """Build a TRT engine for the Cosmos 3 AR reasoner (single-lane / text-only mode).

    Args:
      config: ``ModelConfig`` describing the reasoner. Must already carry the
        Qwen3-VL text fields (hidden_size, num_hidden_layers, etc.); load via
        the standard checkpoint_mapper pathway. Cosmos 3 Super values are
        listed as module-level constants for reference.
      weights: weight dict produced by ``Cosmos3Plugin.load_weights``; must
        contain the reasoner backbone keys (``model.language_model.layers.{i}.*``).
      max_cache_length: KV cache length the engine will be built for.
      precision: ``bf16`` recommended (Cosmos 3 is bf16-only per HF model card).
      quant_ctx: optional quantization context (None for the L0 lane).
      verbose: verbose engine build logging.
      debug_layer_outputs: mark intermediate tensors as graph outputs.
      parallel_config: optional ``ParallelConfig``; when ``enabled`` the TP
        decoder builder is used, else the single-device standard builder.
      deepstack_num_levels: number of ViT feature levels (Cosmos 3 has 3).

    Returns:
      Serialized TRT engine bytes ready for ``trtmc run``.

    Notes:
      Joint attention between AR and DM lanes is **not** handled here. This
      engine is correct for ``parallel.mode in {single, tensor_parallel}`` and
      for the reasoner-only text→text capability. Anything that requires the
      DM lane (text→image, text→video, image→video, action gen) must build
      both this engine and the DM generator engine and dispatch through the
      Phase 6 C++ runtime.
    """
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        return build_qwen_vl_tp_decoder_engine(
            config, weights, max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            embed_input=True,
            deepstack_num_levels=deepstack_num_levels,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            parallel_config=parallel)

    return build_standard_decoder_engine(
        config, weights, max_cache_length,
        precision=precision,
        quant_ctx=quant_ctx,
        verbose=verbose)
