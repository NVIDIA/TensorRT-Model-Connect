# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The language model as an engine the runtime can actually drive.

:mod:`.language_model_builder` builds the decoder *stack* -- hidden states in,
hidden states out. That graph is numerically correct and was checked against
the reference, but it cannot serve a generation: it has no embedding lookup, no
output projection and no key/value cache, so the runtime has nothing to feed it
token ids with and no way to avoid recomputing the whole prefix at every step.

This module builds the engine the runtime contract asks for instead. The
inputs and outputs are the repository's standard decoder names, which
``IoMap`` (``include/trtmc/runtime/pipeline_plugin.h``) already defaults to::

    inputs   token_id, position_id, attention_mask, cache_k_{i}, cache_v_{i},
             input_embed, use_input_embed
    outputs  logits, present_k_{i}, present_v_{i}, hidden_state

``hidden_state`` is requested explicitly. It is not decoration: the depth
decoder conditions on the frame's hidden state, so a logits-only engine would
force a second forward pass to recover what the first one already computed.

The decoder machinery is this family's own copy, per the architecture's
duplication rule. It came from ``qwen3_omni``, whose stack is the same Qwen3
shape -- RMS norm, SwiGLU, grouped attention and the per-head ``q_norm`` /
``k_norm`` before the rotation.
"""

from __future__ import annotations

from .config import ModelConfig
from .language_model import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_HIDDEN_LAYERS,
    NUM_KEY_VALUE_HEADS,
    RMS_NORM_EPS,
    ROPE_THETA,
    VOCAB_SIZE,
)
from .pipeline_spec import MAX_AUDIO_FRAMES, MAX_PROMPT_TOKENS

#: Engine tensor names, matching the runtime's IoMap defaults.
TOKEN_INPUT = "token_id"
EMBED_INPUT = "input_embed"
USE_EMBED_INPUT = "use_input_embed"
POSITION_INPUT = "position_id"
ATTENTION_MASK_INPUT = "attention_mask"
LOGITS_OUTPUT = "logits"
HIDDEN_STATES_OUTPUT = "hidden_state"
CACHE_K_PATTERN = "cache_k_{i}"
CACHE_V_PATTERN = "cache_v_{i}"
PRESENT_K_PATTERN = "present_k_{i}"
PRESENT_V_PATTERN = "present_v_{i}"

#: One generation holds the prompt and then every audio frame it emits, so the
#: cache has to span both. Neither bound is arbitrary: the prompt cap is
#: prompt_format.MAX_PROMPT_TOKENS and the frame cap is
#: pipeline_spec.MAX_AUDIO_FRAMES, both taken from the reference.
DEFAULT_MAX_CACHE_LENGTH = MAX_PROMPT_TOKENS + MAX_AUDIO_FRAMES


def model_config() -> ModelConfig:
    """Return the decoder's architecture in the builder's own vocabulary."""

    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
    )


def build_engine(tensors: dict, *, max_cache_length: int = DEFAULT_MAX_CACHE_LENGTH,
                 precision: str = "fp32", num_layers: int | None = None,
                 verbose: bool = False) -> bytes:
    """Build the cached decoder engine from the checkpoint's tensors."""

    from .default_decoder import build_standard_decoder_engine
    from .language_model_weights import build_weight_dict

    config = model_config()
    if num_layers is not None:
        config.num_hidden_layers = num_layers
    weights = build_weight_dict(tensors, num_layers=config.num_hidden_layers)
    return build_standard_decoder_engine(
        config,
        weights,
        max_cache_length,
        precision=precision,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        position_type="rope",
        activation="silu",
        # Qwen3 rotates the whole head, unlike the diffusion transformer's
        # leading half.
        partial_rotary_factor=1.0,
        interleaved_rope=False,
        hidden_state_output=True,
        # Generation does not feed this model token ids. After the prompt, each
        # step's input is the whole frame's embedding -- the semantic code's
        # plus the seven residual codes', summed -- so the engine needs the
        # input_embed path as well as the lookup.
        embed_input=True,
        verbose=verbose,
    )


def expected_io_shapes(max_cache_length: int = DEFAULT_MAX_CACHE_LENGTH,
                       num_layers: int = NUM_HIDDEN_LAYERS) -> dict:
    """Return the engine's decode-step input and output shapes.

    The engine is a single-token decode step: one id in, one row of logits out,
    with the prefix living in the cache rather than in the input.
    """

    head_dim = HIDDEN_SIZE // NUM_ATTENTION_HEADS
    kv_width = NUM_KEY_VALUE_HEADS * head_dim
    shapes: dict = {
        TOKEN_INPUT: (1,),
        POSITION_INPUT: (1,),
        ATTENTION_MASK_INPUT: (1, max_cache_length),
        LOGITS_OUTPUT: (1, VOCAB_SIZE),
        HIDDEN_STATES_OUTPUT: (1, HIDDEN_SIZE),
    }
    for layer in range(num_layers):
        cache = (1, max_cache_length, kv_width)
        shapes[CACHE_K_PATTERN.format(i=layer)] = cache
        shapes[CACHE_V_PATTERN.format(i=layer)] = cache
        shapes[PRESENT_K_PATTERN.format(i=layer)] = cache
        shapes[PRESENT_V_PATTERN.format(i=layer)] = cache
    return shapes
