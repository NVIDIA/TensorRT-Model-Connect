# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build PP-DocLayoutV3's full layout network without an intermediate export."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import backbone_builder, decoder_builder, encoder_builder
from . import graph as g


def resolve(model_dir):
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    expected = {
        "model_type": "pp_doclayout_v3",
        "d_model": 256,
        "encoder_hidden_dim": 256,
        "encoder_layers": 1,
        "encoder_attention_heads": 8,
        "encode_proj_layers": [2],
        "encoder_ffn_dim": 1024,
        "decoder_layers": 6,
        "decoder_attention_heads": 8,
        "decoder_ffn_dim": 1024,
        "decoder_n_points": 4,
        "num_queries": 300,
        "num_feature_levels": 3,
        "mask_enhanced": True,
        "num_prototypes": 32,
        "global_pointer_head_size": 64,
        "x4_feat_dim": 128,
        "mask_feature_channels": [64, 64],
        "learn_initial_query": False,
        "activation_function": "silu",
        "encoder_activation_function": "gelu",
        "decoder_activation_function": "relu",
        "normalize_before": False,
        "hidden_expansion": 1.0,
        "layer_norm_eps": 1e-5,
        "batch_norm_eps": 1e-5,
        "positional_encoding_temperature": 10000,
        "encoder_in_channels": [512, 1024, 2048],
        "decoder_in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "feature_strides": [8, 16, 32],
        "anchor_image_size": None,
        "eval_size": None,
        "use_timm_backbone": False,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if raw.get(key, value) != value:
            raise NotImplementedError(f"pp_doclayout_v3 requires {key}={value!r}")
    backbone = raw.get("backbone_config", {})
    if backbone.get("model_type") != "hgnet_v2" or backbone.get("arch", "L") != "L":
        raise NotImplementedError("pp_doclayout_v3 requires the HGNetV2 L backbone")
    backbone_defaults = {
        "stem_channels": [3, 32, 48],
        "stage_in_channels": [48, 128, 512, 1024],
        "stage_mid_channels": [48, 96, 192, 384],
        "stage_out_channels": [128, 512, 1024, 2048],
        "stage_num_blocks": [1, 1, 3, 1],
        "stage_downsample": [False, True, True, True],
        "stage_light_block": [False, False, True, True],
        "stage_kernel_size": [3, 3, 5, 5],
        "stage_numb_of_layers": [6, 6, 6, 6],
        "use_learnable_affine_block": False,
        "hidden_act": "relu",
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
    }
    for key, value in backbone_defaults.items():
        if backbone.get(key, value) != value:
            raise NotImplementedError(f"pp_doclayout_v3 requires backbone {key}={value!r}")
    if len(raw.get("id2label", {})) != 25:
        raise NotImplementedError("pp_doclayout_v3 requires the 25-class layout checkpoint")
    processor = json.loads((Path(model_dir) / "preprocessor_config.json").read_text())
    if (
        processor.get("size") != {"height": 800, "width": 800}
        or processor.get("resample") != 3
        or processor.get("image_mean") != [0, 0, 0]
        or processor.get("image_std") != [1, 1, 1]
        or processor.get("rescale_factor") != 1 / 255
        or not all(processor.get(k) is True for k in ("do_resize", "do_rescale", "do_normalize"))
        or processor.get("do_center_crop", False) not in (False, None)
        or processor.get("do_pad", False) not in (False, None)
    ):
        raise NotImplementedError(
            "pp_doclayout_v3 requires the published 800px bicubic preprocessor"
        )
    return {
        **expected,
        "image_size": 800,
        "shapes": [(100, 100), (50, 50), (25, 25)],
        "bottlenecks": 3,
        "offset_scale": 0.5,
    }


def build_layout_engine(weights, cfg, *, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, 800, 800))
    levels = backbone_builder.build_backbone(network, pixels, weights)
    fused = encoder_builder.build_encoder(network, levels[1:], weights, cfg)
    mask_features = encoder_builder.build_mask_features(network, fused, levels[0], weights)
    projected = encoder_builder.project_for_decoder(network, fused, weights)
    memory = g.concat(network, [g.spatial_to_tokens(network, x) for x in projected], 1)
    outputs = decoder_builder.build_decoder(
        network, memory, cfg["shapes"], mask_features, weights, cfg
    )
    for tensor, name in zip(outputs, ("logits", "boxes", "order_logits", "masks")):
        tensor.name = name
        network.mark_output(tensor)
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("pp_doclayout_v3 engine build failed")
    return bytes(plan)


def build(request, writer):
    if request.task != "document_layout" or request.backend != "trt":
        raise ValueError("pp_doclayout_v3 supports document_layout on backend=trt")
    if request.precision != "fp32":
        raise ValueError("pp_doclayout_v3 supports precision=fp32")
    if request.dynamic_kv_cache:
        raise NotImplementedError("pp_doclayout_v3 does not support dynamic_kv_cache")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.max_sequence_length is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("pp_doclayout_v3 supports a single unquantized image per request")
    if request.image_height not in (None, 0, 800) or request.image_width not in (None, 0, 800):
        raise NotImplementedError("pp_doclayout_v3 requires an 800x800 network input")
    cfg = resolve(request.model_dir)
    with safe_open(str(Path(request.model_dir) / "model.safetensors"), framework="numpy") as reader:
        weights = {key: reader.get_tensor(key) for key in reader.keys()}
    if any(value.dtype != np.float32 for value in weights.values()):
        raise ValueError("pp_doclayout_v3 requires float32 safetensors weights")
    writer.set_header(family="pp_doclayout_v3", task=request.task, backend=request.backend)
    writer.add_bytes("layout.plan", build_layout_engine(weights, cfg, verbose=request.verbose))
    writer.add_json(
        "runtime.json",
        {
            "input_size": 800,
            "num_queries": 300,
            "num_labels": 25,
            "mask_size": 200,
            "score_threshold": 0.5,
        },
    )
