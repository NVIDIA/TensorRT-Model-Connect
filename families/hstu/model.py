# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build public HSTU using the unchanged staged serving contract."""

import hashlib
import json
from pathlib import Path
from .checkpoint import load_weights
from .config import Config, REVISION, UPSTREAM


def build(request, writer):
    if request.family != "hstu" or request.task != "structured_ranking" or request.backend != "trt":
        raise ValueError("HSTU requires family=hstu task=structured_ranking backend=trt")
    if request.precision not in ("fp32", "bf16"):
        raise ValueError("HSTU supports fp32 and bf16")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("HSTU supports one device per instance")
    if (
        request.dynamic_kv_cache
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.mtp_seq_len
    ):
        raise ValueError("unsupported HSTU build option")
    if (
        request.image_height
        or request.image_width
        or request.video_num_frames
        or request.max_sequence_length not in (None, 128)
    ):
        raise ValueError("HSTU requires fixed history length 128")
    c = Config.load(request.model_dir)
    weights = load_weights(request.model_dir, c)
    import tensorrt as trt
    from .native import Network

    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    writer.set_header(family="hstu", task=request.task, backend=request.backend)
    runtime = c.runtime(request.max_batch_size, request.precision)
    writer.add_json("runtime.json", runtime)
    for stage in runtime["stages"]:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        owner = Network(network, c, weights, request.precision).define(stage["name"])
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            minimum = runtime["min_batch_size"]
            low = tuple(minimum if v == -1 else v for v in tensor.shape)
            high = tuple(request.max_batch_size if v == -1 else v for v in tensor.shape)
            profile.set_shape(tensor.name, low, high, high)
        options = builder.create_builder_config()
        options.builder_optimization_level = 1
        options.clear_flag(trt.BuilderFlag.TF32)
        options.max_aux_streams = 0
        options.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        options.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, options)
        if plan is None:
            raise RuntimeError("HSTU TensorRT build failed: " + stage["name"])
        writer.add_bytes(stage["name"] + ".plan", bytes(plan))
        del owner
    path = Path(request.model_dir)
    writer.add_json(
        "hstu_source.json",
        dict(
            architecture_source=UPSTREAM,
            architecture_revision=REVISION,
            weights_sha256=hashlib.sha256((path / "hstu_weights.npz").read_bytes()).hexdigest(),
            adapter="public target-aware greedy slate ranking, one block",
            checkpoint=json.loads((path / "checkpoint_provenance.json").read_text())
            if (path / "checkpoint_provenance.json").exists()
            else {"kind": "user-supplied"},
        ),
    )
