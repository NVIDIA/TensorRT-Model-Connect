# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the prepared Omni model and all native component graph construction."""

from .config import OmniConfig
from .native_builder import build_native
from .prepared_quantization import checkpoint_policy
from .support import SUPPORTED_TASKS


def build(request, writer):
    """Build the full checkpoint, never a text-only substitute for Omni."""
    if request.task not in SUPPORTED_TASKS:
        raise ValueError("Nemotron Omni requires a supported text or multimodal reasoning task")
    if request.backend != "trt" or request.precision not in {"fp16", "fp32"}:
        raise ValueError("Omni requires TensorRT with fp16 or fp32 computation")
    if request.dynamic_kv_cache:
        raise NotImplementedError("nemotron_omni does not support dynamic_kv_cache")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.fp32_layers
        or request.graph_transform is not None
        or request.mtp_seq_len is not None
    ):
        raise ValueError("Omni implements one non-speculative request on one TensorRT device")
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError("Omni media shapes are dynamic, not fixed generic dimensions")
    if request.quantization not in {None, "none", "nvfp4"}:
        raise ValueError("Omni preserves the documented prepared NVFP4/FP8 weights")
    policy = checkpoint_policy(request.model_dir)
    if policy is None or policy.request_format != "nvfp4":
        raise ValueError("Omni requires its documented prepared NVFP4 checkpoint")
    config = OmniConfig.from_dir(request.model_dir)
    # Dynamic images consume four raw patches per decoder visual token. The
    # reference processor also caps its grid by the available context budget.
    capacity = 2048 if request.max_sequence_length is None else request.max_sequence_length
    if type(capacity) is not int or not 1 <= capacity <= config.decoder.max_position_embeddings:
        raise ValueError("Omni decoder capacity must be within the checkpoint context limit")
    maximum = config.vision["args"]["max_num_patches"]
    patches = min(maximum, max(1024, capacity * 4))

    def native(request, writer):
        writer.set_header(family="nemotron_omni", task=request.task, backend=request.backend)
        build_native(request, writer, max_visual_patches=patches, verbose=request.verbose)

    from .edge_llm import dispatch

    dispatch.build(request, writer, native, capacity)
