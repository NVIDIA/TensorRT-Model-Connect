# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the complete official DLRMv3 dense model and immutable history cache."""

import json
from pathlib import Path
from .checkpoint import load_weights
import uuid
from .config import Config


def build(request, writer):
    if request.family != "hstu" or request.task != "structured_ranking" or request.backend != "trt":
        raise ValueError("DLRMv3 requires family=hstu task=structured_ranking backend=trt")
    if request.precision not in ("fp32", "bf16"):
        raise ValueError("supported precisions are fp32 and bf16")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("DLRMv3 supports one device per instance")
    optional = (
        request.fp32_layers,
        request.mtp_seq_len,
        request.image_height,
        request.image_width,
        request.video_num_frames,
    )
    if any(optional) or request.quantization not in (None, "none"):
        raise ValueError("unsupported DLRMv3 build option")

    if request.dynamic_kv_cache:
        raise NotImplementedError("hstu does not support dynamic_kv_cache")
    config = Config.load(request.model_dir)
    if request.max_sequence_length not in (None, config.serving.history_length):
        raise ValueError("sequence length must match the serving profile")
    weights = load_weights(request.model_dir, config)
    import tensorrt as trt
    from .native import Network

    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    runtime = config.runtime(request.max_batch_size, request.precision)
    runtime["bundle_id"] = str(uuid.uuid4())
    writer.set_header(family="hstu", task=request.task, backend=request.backend)
    writer.add_json("runtime.json", runtime)
    for stage in runtime["stages"]:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        owner = Network(network, config, weights, request.precision).define(stage["name"])
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            shapes = [
                tuple(runtime[key] if v == -1 else v for v in tensor.shape)
                for key in ("min_batch_size", "opt_batch_size", "max_batch_size")
            ]
            profile.set_shape(tensor.name, *shapes)
        options = builder.create_builder_config()
        options.builder_optimization_level = 1
        options.clear_flag(trt.BuilderFlag.TF32)
        options.max_aux_streams = 0
        options.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        options.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        options.add_optimization_profile(profile)
        plan = builder.build_serialized_network(network, options)
        if plan is None:
            raise RuntimeError("DLRMv3 build failed: " + stage["name"])
        writer.add_bytes(stage["name"] + ".plan", bytes(plan))
        del owner
    root = Path(request.model_dir)
    writer.add_json(
        "checkpoint_provenance.json", json.loads((root / "checkpoint_provenance.json").read_text())
    )
    owner = Path(__file__).parent
    writer.add_bytes("LICENSE.txt", (owner / "UPSTREAM_LICENSE.txt").read_bytes())
    writer.add_bytes("NOTICE", (owner / "NOTICE").read_bytes())
