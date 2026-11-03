# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental OmniRef PDD schedule and build-time output-head composition.

These primitives do not make the student a supported build/runtime profile.
The released BF16 student uses eight blocks of a fixed 32-interval base grid.
Its native H3 heads emit x0-eps; the ordinary H3 scheduler retains that sign.
"""

from dataclasses import dataclass

import numpy as np


MODEL_ID = "FastVideo/FastVideo-FastH3-Omni-8-Step-V1"
MODEL_REVISION = "53e0d26a1691aa473b6262d8196a8feb2c297df0"
FINE_STEPS = 32
STEP_INDICES = tuple(range(0, 33, 4))
GRID_MAX = 0.999
SHIFTS = {"video": 12.0, "audio": 3.0}
HEAD_FEATURES = {"video": 96, "audio": 32}
CONTRACT = {
    "schema": "fasth3-inference-contract-v1",
    "schema_version": "fasth3-inference-contract-v1",
    "model_type": "ref2va",
    "transformer_component": "transformer_ref",
    "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
    "conditioning": "fixed_ordered_references_target_only_flow",
    "pdd_steps": FINE_STEPS,
    "pdd_step_indices": list(STEP_INDICES),
    "num_inference_steps": 8,
    "transformer_forwards": 8,
    "grid_max_t": GRID_MAX,
    "video_scheduler_shift": SHIFTS["video"],
    "audio_scheduler_shift": SHIFTS["audio"],
    "guidance_scale": 1.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
    "vsa_ref_keep_rate": 0.1,
    "vsa_ref_policy": "p2_multi_region",
}
TRANSFORMER_CONFIG = {
    "_class_name": "MiniMaxH3Transformer3DModel",
    "attention_head_dim": 128,
    "audio_in_channels": 32,
    "ffn_dim": 14336,
    "final_norm_eps": 1e-5,
    "freq_dim": 256,
    "hidden_size": 5376,
    "in_channels": 24,
    "norm_eps": 1e-5,
    "num_attention_heads": 56,
    "num_layers": 50,
    "num_refiner_layers": 2,
    "patch_size": [1, 2, 2],
    "pdd_steps": FINE_STEPS,
    "qk_norm_eps": 1e-5,
    "rope_freq_dim": 16,
    "rope_theta": 10000.0,
    "text_dim": 5120,
    "time_embed_dim": 2688,
    "time_embed_hidden_dim": 5376,
}


def _same(actual, expected):
    if isinstance(expected, dict):
        return (
            type(actual) is dict
            and set(actual) == set(expected)
            and all(_same(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, float):
        return type(actual) in (int, float) and np.isfinite(actual) and actual == expected
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(_same(a, b) for a, b in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def validate_contract(value):
    """Accept the one released BF16 sampling contract, without inferred defaults."""
    if not isinstance(value, dict) or set(value) != set(CONTRACT):
        raise ValueError("OmniRef requires its complete explicit PDD contract")
    if any(not _same(value[key], expected) for key, expected in CONTRACT.items()):
        raise ValueError("Unsupported OmniRef PDD sampling or attention contract")


def validate_transformer_config(value):
    if (
        not isinstance(value, dict)
        or set(value) - set(TRANSFORMER_CONFIG) - {"_diffusers_version"}
        or any(not _same(value.get(key), expected) for key, expected in TRANSFORMER_CONFIG.items())
    ):
        raise ValueError("Unsupported OmniRef PDD transformer topology")


def checkpoint_partition():
    """Partition the student without reusing the teacher's learned AdaLN values."""
    from . import adaln_builder, dit_builder

    modulation = adaln_builder.checkpoint_keys()
    denoiser = (
        *dit_builder.checkpoint_keys(),
        *(f"transformer_blocks.{index}.attn.to_gate_compress.weight" for index in range(50)),
    )
    if (
        set(modulation) & set(denoiser)
        or len(set(modulation)) != len(modulation)
        or len(set(denoiser)) != len(denoiser)
        or len(modulation) + len(denoiser) != 688
    ):
        raise ValueError("OmniRef student parameter partition changed")
    return {"adaln": modulation, "denoiser": denoiser}


def validate_checkpoint_index(value):
    """Admit every released student tensor and only its fourteen owned shards."""
    import re

    if not isinstance(value, dict) or set(value) != {"metadata", "weight_map"}:
        raise ValueError("OmniRef requires its complete safetensors index")
    if not _same(value["metadata"], {"total_size": 70142174464}):
        raise ValueError("OmniRef student tensor payload size differs from the release")
    mapping = value["weight_map"]
    partition = checkpoint_partition()
    expected = {key for keys in partition.values() for key in keys}
    shards = {f"diffusion_pytorch_model-{index:05}-of-00014.safetensors" for index in range(1, 15)}
    if (
        not isinstance(mapping, dict)
        or set(mapping) != expected
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"diffusion_pytorch_model-\d{5}-of-00014\.safetensors", name) is None
            for name in mapping.values()
        )
        or set(mapping.values()) != shards
    ):
        raise ValueError("OmniRef index has missing, unexpected or redirected student parameters")


def _readonly(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ModalityPlan:
    """FP64 head coefficients and the separate FP32 scheduler/score clocks."""

    fine_grid: np.ndarray
    integration_weights: np.ndarray
    sigmas: np.ndarray
    timesteps: np.ndarray


def sampling_plan(modality):
    if modality not in SHIFTS:
        raise ValueError("OmniRef PDD supports video and audio modalities only")
    shift = SHIFTS[modality]
    fine = np.linspace(GRID_MAX, 0.0, FINE_STEPS + 1, dtype=np.float64)
    start, end = fine[:-1], fine[1:]
    # Keep the cancellation-resistant rational difference used by the release.
    weights = (
        shift
        * GRID_MAX**2
        * (end - start)
        / ((GRID_MAX + (shift - 1.0) * start) * (GRID_MAX + (shift - 1.0) * end))
    )
    nodes = fine[list(STEP_INDICES)]
    sigmas = (shift * nodes * GRID_MAX / (nodes * (shift - 1.0) + GRID_MAX)).astype(np.float32)
    times = np.float32(1.0) - sigmas[:-1]
    return ModalityPlan(*(_readonly(a) for a in (fine, weights, sigmas, times)))


def _block(step):
    if type(step) is not int or not 0 <= step < len(STEP_INDICES) - 1:
        raise ValueError("OmniRef PDD step must select one of the eight trained blocks")
    return STEP_INDICES[step], STEP_INDICES[step + 1]


def _fp32(value, name):
    array = np.asarray(value)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite FP32 values")
    return array


def _round_bf16(value):
    # Return decoded BF16 values in an FP32 container, with nearest-even ties.
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    result = rounded.view(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("OmniRef fused parameters overflow BF16")
    return result


def fuse_output_projection(weight, bias, modality, step):
    """Compose one released channel-major head block with FP32 accumulation.

    Inputs contain decoded BF16 checkpoint values. Normalize coefficients in
    FP64, accumulate in FP32, then round back to BF16 before the projection.
    Video rows are (grid,96); audio rows are (grid,32), not patch-major heads.
    """
    plan = sampling_plan(modality)
    start, end = _block(step)
    width = HEAD_FEATURES[modality]
    weight = _fp32(weight, "PDD weight")
    if weight.ndim != 2 or weight.shape[0] != FINE_STEPS * width or weight.shape[1] == 0:
        raise ValueError("OmniRef output weight differs from its widened head layout")
    bias = None if bias is None else _fp32(bias, "PDD bias")
    if bias is not None and bias.shape != (FINE_STEPS * width,):
        raise ValueError("OmniRef output bias differs from its widened head layout")
    coefficients = plan.integration_weights[start:end]
    normalized = (coefficients / coefficients.sum()).astype(np.float32)

    def compose(parameter):
        heads = parameter.reshape(FINE_STEPS, width, *parameter.shape[1:])[start:end]
        return _round_bf16(np.einsum("n,n...->...", normalized, heads, optimize=False))

    return compose(weight), None if bias is None else compose(bias)


def initial_target_noise(noise, modality):
    """Scale only target rows; callers must retain fixed conditioning rows."""
    return _fp32(noise, "PDD target noise") * sampling_plan(modality).sigmas[0]


def scheduler_step(sample, model_output, modality, step):
    """Apply the H3 x0-eps convention with separate FP32 score and grid paths."""
    plan = sampling_plan(modality)
    _block(step)
    sample, model_output = _fp32(sample, "PDD state"), _fp32(model_output, "PDD output")
    if sample.shape != model_output.shape:
        raise ValueError("OmniRef PDD state and output shapes differ")
    sigma_from_time = np.float32(1.0) - plan.timesteps[step]
    denoised = sample + sigma_from_time * model_output
    ratio = plan.sigmas[step + 1] / plan.sigmas[step]
    return ratio * sample + (np.float32(1.0) - ratio) * denoised
