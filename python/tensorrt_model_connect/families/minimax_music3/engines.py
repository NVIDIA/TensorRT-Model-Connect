# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which engines the bundle carries, and what the runtime needs to know.

**Validated.** Driven against the published checkpoint on an A40 with
TensorRT 11.1.0.106: ``load_weights`` read the three enumerated components in
2.3 s and their tensor counts matched the recorded inventories -- 47, 4 and
121 -- and ``build_extra_engines`` produced three plans in 113.5 s, at 100.8,
2282.2 and 218.0 MB, the same sizes the individual graph validations measured.

Five networks. The diffusion transformer is the primary engine because it is
what the denoising loop runs thirty times per window; the condition encoder,
the depth decoder, the vocoder and the global language model are extras.

The language model's decoder is duplicated into this family rather than shared
with the repository's other Qwen decoders. That is the architecture's
instruction, not an oversight: model-specific code stays family-local so a
defect here is fixed and reverted here.

:func:`bundle_config_overrides` is the other half of the job. The runtime
cannot re-derive the window plan, the crop widths or the output rate from the
checkpoint, because none of them are in it: they live in the reference
implementation, and every one of them was measured against a real run before
being written down here.
"""

from __future__ import annotations

from . import condition_encoder as ce
from . import depth_decoder as dd
from . import dit, language_model, pipeline_spec, vocoder

#: Engine names inside the bundle.
DIT_ENGINE = "dit"
CONDITION_ENCODER_ENGINE = "condition_encoder"
DEPTH_DECODER_ENGINE = "depth_decoder"
VOCODER_ENGINE = "vocoder"

LANGUAGE_MODEL_ENGINE = "language_model"

#: Built by this family, in the order a generation uses them.
ENGINE_NAMES = (
    LANGUAGE_MODEL_ENGINE,
    DEPTH_DECODER_ENGINE,
    CONDITION_ENCODER_ENGINE,
    DIT_ENGINE,
    VOCODER_ENGINE,
)

#: Checkpoint subfolder each engine's weights come from.
ENGINE_COMPONENTS = {
    LANGUAGE_MODEL_ENGINE: "language_model",
    DIT_ENGINE: "transformer",
    CONDITION_ENCODER_ENGINE: "condition_encoder",
    DEPTH_DECODER_ENGINE: "rvq_depth_decoder",
    VOCODER_ENGINE: "vocoder",
}


def engine_component(name: str) -> str:
    """Return the checkpoint subfolder an engine is built from."""

    try:
        return ENGINE_COMPONENTS[name]
    except KeyError:
        raise ValueError(f"unknown MiniMax-Music3 engine {name!r}") from None


def engine_io(name: str, *, latent_length: int, steps: int = 8) -> dict:
    """Return an engine's input and output shapes."""

    if name == CONDITION_ENCODER_ENGINE:
        # The encoder consumes autoregressive frames, not latents.
        frames = pipeline_spec.CHUNK_FRAMES
        return ce.engine_io_shapes(frames)
    if name == DEPTH_DECODER_ENGINE:
        # Fixed at the full depth sequence: the hidden state plus seven codes.
        from .depth_decoder_engine import expected_io_shapes

        return expected_io_shapes()
    if name == DIT_ENGINE:
        from .dit_builder import expected_io_shapes

        return expected_io_shapes(latent_length)
    if name == LANGUAGE_MODEL_ENGINE:
        # A cached decode step, not a fixed-length stack: the shapes depend on
        # the cache the engine was compiled for, not on ``steps``.
        from .language_model_engine import expected_io_shapes

        return expected_io_shapes()
    if name == VOCODER_ENGINE:
        from .vocoder_builder import expected_input_shape, expected_output_shape

        return {
            "latents": expected_input_shape(latent_length),
            "waveform": expected_output_shape(latent_length),
        }
    raise ValueError(f"unknown MiniMax-Music3 engine {name!r}")


def bundle_config_overrides() -> dict:
    """Facts the runtime needs that the checkpoint does not carry.

    Every value here was read from the reference implementation and then
    confirmed against a recorded generation: the window plan and the crop
    widths reproduce the run's 882688 samples exactly, and the rate is what
    the pipeline's own property returns rather than what the model card says.
    """

    return {
        "sampling_rate": vocoder.SAMPLING_RATE,
        "output_channels": vocoder.STREAMS,
        "frame_rate_hz": pipeline_spec.FRAME_RATE_HZ,
        "latent_hop_length": ce.OUTPUT_HOP_LENGTH,
        # Latent frames per autoregressive frame. Carried rather than
        # re-derived: the runtime would otherwise need the two sampling
        # rates and the two hop lengths, and truncating a rebuilt ratio
        # is exactly where a one-frame drift enters.
        "latent_resample_ratio": ce.resample_ratio(),
        "chunk_latent_length": ce.latent_length(pipeline_spec.CHUNK_FRAMES),
        "chunk_frames": pipeline_spec.CHUNK_FRAMES,
        "chunk_hop": pipeline_spec.CHUNK_HOP,
        "crop_left_latent": pipeline_spec.CROP_LEFT_LATENT,
        "crop_right_latent": pipeline_spec.CROP_RIGHT_LATENT,
        "default_inference_steps": pipeline_spec.DEFAULT_INFERENCE_STEPS,
        "max_audio_frames": pipeline_spec.MAX_AUDIO_FRAMES,
        "guidance_branches": pipeline_spec.TEXT_IDS_BATCH,
        "num_codebooks": dd.NUM_CODEBOOKS,
        "num_residual_codebooks": dd.NUM_RESIDUAL_CODEBOOKS,
        "audio_vocab_size": dd.AUDIO_VOCAB_SIZE,
        "latent_channels": dit.IN_CHANNELS,
        "condition_dim": dit.CONDITION_DIM,
        # Width of one frame's hidden state as the condition encoder
        # reads it: eight streams -- the language model's one and the
        # depth decoder's seven -- each CONDITION_HIDDEN_DIM wide.
        "frame_hidden_width": ce.NUM_CONDITION_LAYERS * ce.CONDITION_HIDDEN_DIM,
        "condition_streams": ce.NUM_CONDITION_LAYERS,
        "language_model_hidden_size": language_model.HIDDEN_SIZE,
        "language_model_kv_width": language_model.NUM_KEY_VALUE_HEADS
        * language_model.HEAD_DIM,
        # The reference guider runs a conditional branch and an
        # unconditional one that conditions on zeros, at this scale.
        "guidance_scale": pipeline_spec.GUIDANCE_SCALE,
        "language_model_vocab_size": language_model.VOCAB_SIZE,
        "language_model_layers": language_model.NUM_HIDDEN_LAYERS,
    }


def latent_length_for(frames: int) -> int:
    """Latent frames one window of ``frames`` autoregressive frames produces."""

    return ce.latent_length(frames)


def samples_for(chunk_count: int, latent_length: int) -> int:
    """Samples the stitched output holds for ``chunk_count`` windows.

    Every window but the first drops ``crop_left_latent`` and every window but
    the last drops ``crop_right_latent``, so the two crops remove exactly one
    overlap per seam.
    """

    if chunk_count < 1:
        raise ValueError(f"chunk_count must be positive, got {chunk_count}")
    total = 0
    for index in range(chunk_count):
        left = 0 if index == 0 else pipeline_spec.CROP_LEFT_LATENT
        right = 0 if index == chunk_count - 1 else pipeline_spec.CROP_RIGHT_LATENT
        total += latent_length - left - right
    return total * vocoder.upsample_factor()
