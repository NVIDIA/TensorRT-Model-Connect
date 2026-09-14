# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DETR object-detection family plugin.

Supports Hugging Face DETR checkpoints whose backbone is a ResNet-50 and whose
configuration uses sine spatial position embeddings.  The initial target is:

    facebook/detr-resnet-50

The builder constructs the full DETR graph (ResNet-50 backbone, 1x1 input
projection, sine position embeddings, 6 encoder layers, 6 decoder layers, class
and bbox heads) with TensorRT Network API calls rather than routing through ONNX.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .graph import model as graph_ops
from .weights import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_torch_checkpoint,
    _target_np_dtype,
)
from .config import ModelConfig


_BN_EPS = 1e-5
_LAYERNORM_EPS = 1e-5

_RESNET50_STAGES = (
    (1, 64, 256, 1),
    (2, 128, 512, 2),
    (3, 256, 1024, 2),
    (4, 512, 2048, 2),
)

_BACKBONE_PREFIX = "model.backbone.conv_encoder.model"


def _pretrained_cfg(raw: dict) -> dict:
    return raw


def _resolve_config(raw: dict) -> dict:
    fb = raw.get("_family_build_options") or {}
    image_h = int(fb.get("input_image_h", raw.get("_detr_image_h", 800)))
    image_w = int(fb.get("input_image_w", raw.get("_detr_image_w", 800)))
    return {
        "image_size_h": image_h,
        "image_size_w": image_w,
        "num_queries": int(raw.get("num_queries", 100)),
        "num_labels": int(raw.get("num_labels", len(raw.get("id2label", {})) or 91)),
        "d_model": int(raw.get("d_model", 256)),
        "encoder_layers": int(raw.get("encoder_layers", 6)),
        "decoder_layers": int(raw.get("decoder_layers", 6)),
        "encoder_attention_heads": int(raw.get("encoder_attention_heads", 8)),
        "decoder_attention_heads": int(raw.get("decoder_attention_heads", 8)),
        "encoder_ffn_dim": int(raw.get("encoder_ffn_dim", 2048)),
        "decoder_ffn_dim": int(raw.get("decoder_ffn_dim", 2048)),
        "position_embedding_type": str(raw.get("position_embedding_type", "sine")),
    }


def _discover_backbone_blocks(readers) -> dict:
    tensor_map = getattr(readers, "tensor_map", None)
    if tensor_map is not None:
        names = set(tensor_map)
    else:
        names = set()
        for reader in readers:
            names.update(reader.keys())

    blocks: dict[int, int] = {}
    for layer in (1, 2, 3, 4):
        indices = set()
        pattern = re.compile(rf"^{_BACKBONE_PREFIX}\.layer{layer}\.(\d+)\.conv1\.weight$")
        for name in names:
            match = pattern.match(name)
            if match:
                indices.add(int(match.group(1)))
        if not indices:
            raise ValueError(f"DETR backbone has no layer{layer} blocks")
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"DETR backbone layer{layer} block indices are not contiguous")
        blocks[layer] = len(indices)
    return blocks


class _DetrModel:

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        readers = _open_torch_checkpoint(Path(model_dir))
        raw = config.raw
        cfg = _resolve_config(raw)
        cfg["backbone_blocks"] = _discover_backbone_blocks(readers)
        raw["_detr_config"] = cfg
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        tensor_map = getattr(readers, "tensor_map", None)
        if tensor_map is not None:
            names = list(tensor_map)
        else:
            names = []
            for reader in readers:
                names.extend(reader.keys())

        for name in names:
            if name.endswith("num_batches_tracked"):
                continue
            if name.endswith(("running_mean", "running_var")):
                weights[name] = _load_tensor(readers, name).astype(np.float32)
            elif name.endswith(("weight", "bias")):
                weights[name] = _load_tensor(readers, name).astype(target_dtype)

        # Ensure the tensors our builder expects are present.
        required = (
            f"{_BACKBONE_PREFIX}.conv1.weight",
            f"{_BACKBONE_PREFIX}.bn1.weight",
            "model.input_projection.weight",
            "model.query_position_embeddings.weight",
            "class_labels_classifier.weight",
            "class_labels_classifier.bias",
            "bbox_predictor.layers.0.weight",
            "bbox_predictor.layers.2.bias",
        )
        for key in required:
            if not _has_tensor(readers, key):
                raise KeyError(f"Tensor not found: {key}")
        return weights

    def _bn(self, network, x, weights, prefix, dtype):
        return graph_ops.add_bn_folded(
            network, x,
            weights[f"{prefix}.weight"], weights[f"{prefix}.bias"],
            weights[f"{prefix}.running_mean"], weights[f"{prefix}.running_var"],
            _BN_EPS, dtype=dtype)

    def _layernorm(self, network, x, weights, prefix, hidden, dtype):
        return graph_ops.add_layer_norm_v2(
            network, x, hidden,
            weights[f"{prefix}.weight"], weights[f"{prefix}.bias"],
            _LAYERNORM_EPS, dtype=dtype)

    def _linear(self, network, x, weights, key, out_features, dtype):
        bias = weights.get(f"{key}.bias")
        return graph_ops.add_linear(
            network, x, weights[f"{key}.weight"], bias, out_features, dtype=dtype)

    def _resnet_bottleneck(self, network, x, weights, prefix, in_ch, width, out_ch,
                           stride, dtype):
        identity = x

        w = weights[f"{prefix}.conv1.weight"]
        x = graph_ops.add_conv2d(network, x, w, None, int(w.shape[0]), (1, 1), dtype=dtype)
        x = self._bn(network, x, weights, f"{prefix}.bn1", dtype)
        x = graph_ops.add_relu(network, x)

        w = weights[f"{prefix}.conv2.weight"]
        x = graph_ops.add_conv2d(
            network, x, w, None, int(w.shape[0]), (3, 3),
            stride=(stride, stride), padding=(1, 1), dtype=dtype)
        x = self._bn(network, x, weights, f"{prefix}.bn2", dtype)
        x = graph_ops.add_relu(network, x)

        w = weights[f"{prefix}.conv3.weight"]
        x = graph_ops.add_conv2d(network, x, w, None, int(w.shape[0]), (1, 1), dtype=dtype)
        x = self._bn(network, x, weights, f"{prefix}.bn3", dtype)

        if stride != 1 or in_ch != out_ch:
            dw = weights[f"{prefix}.downsample.0.weight"]
            identity = graph_ops.add_conv2d(
                network, identity, dw, None, int(dw.shape[0]), (1, 1),
                stride=(stride, stride), dtype=dtype)
            identity = self._bn(network, identity, weights, f"{prefix}.downsample.1", dtype)

        x = graph_ops.add_sum(network, x, identity)
        return graph_ops.add_relu(network, x)

    def _build_backbone(self, network, pixel_values, weights, cfg, dtype):
        p = _BACKBONE_PREFIX
        x = graph_ops.add_conv2d(
            network, pixel_values, weights[f"{p}.conv1.weight"], None,
            int(weights[f"{p}.conv1.weight"].shape[0]), (7, 7),
            stride=(2, 2), padding=(3, 3), dtype=dtype)
        x = self._bn(network, x, weights, f"{p}.bn1", dtype)
        x = graph_ops.add_relu(network, x)
        x = graph_ops.add_max_pool2d(network, x, 3, 2, 1)

        in_ch = 64
        for layer, width, out_ch, stride in _RESNET50_STAGES:
            num_blocks = cfg["backbone_blocks"][layer]
            for idx in range(num_blocks):
                block_prefix = f"{p}.layer{layer}.{idx}"
                block_stride = stride if idx == 0 else 1
                x = self._resnet_bottleneck(
                    network, x, weights, block_prefix, in_ch, width, out_ch,
                    block_stride, dtype)
                in_ch = out_ch
        return x

    def _attention(self, network, x, weights, prefix, hidden_size, num_heads,
                   dtype, pos_embedding=None, kv_input=None, kv_pos_embedding=None):
        head_dim = hidden_size // num_heads
        seq_len = int(x.shape[-2])

        q_input = x if pos_embedding is None else graph_ops.add_sum(network, x, pos_embedding)
        q = self._linear(network, q_input, weights, f"{prefix}.q_proj", hidden_size, dtype)
        if kv_input is None:
            k = self._linear(network, q_input, weights, f"{prefix}.k_proj", hidden_size, dtype)
            v = self._linear(network, x, weights, f"{prefix}.v_proj", hidden_size, dtype)
            kv_seq = seq_len
        else:
            k_input = kv_input if kv_pos_embedding is None else graph_ops.add_sum(network, kv_input, kv_pos_embedding)
            k = self._linear(network, k_input, weights, f"{prefix}.k_proj", hidden_size, dtype)
            v = self._linear(network, kv_input, weights, f"{prefix}.v_proj", hidden_size, dtype)
            kv_seq = int(kv_input.shape[-2])

        q_4d = graph_ops.reshape_rows_to_heads_4d(network, q, num_heads, head_dim, seq_len)
        k_4d = graph_ops.reshape_rows_to_heads_4d(network, k, num_heads, head_dim, kv_seq)
        v_4d = graph_ops.reshape_rows_to_heads_4d(network, v, num_heads, head_dim, kv_seq)
        attn = graph_ops.add_attention_core(network, q_4d, k_4d, v_4d)
        out = graph_ops.reshape_heads_4d_to_rows(network, attn, hidden_size, seq_len)
        return self._linear(network, out, weights, f"{prefix}.out_proj", hidden_size, dtype)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str,
        verbose: bool = False,
    ) -> bytes:
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"Unsupported detr precision: {precision}")

        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported detr precision: {precision}")

        cfg = config.raw.get("_detr_config")
        if cfg is None:
            raise RuntimeError("load_weights must run before build_engine to resolve the layout")
        image_h = cfg["image_size_h"]
        image_w = cfg["image_size_w"]
        num_queries = cfg["num_queries"]
        num_labels = cfg["num_labels"]
        d_model = cfg["d_model"]
        enc_layers = cfg["encoder_layers"]
        dec_layers = cfg["decoder_layers"]
        enc_heads = cfg["encoder_attention_heads"]
        dec_heads = cfg["decoder_attention_heads"]
        enc_ffn = cfg["encoder_ffn_dim"]
        dec_ffn = cfg["decoder_ffn_dim"]
        if cfg["position_embedding_type"] != "sine":
            raise NotImplementedError("detr only supports sine position embeddings")

        feature_h = (image_h + 31) // 32
        feature_w = (image_w + 31) // 32
        seq_len = feature_h * feature_w

        if verbose:
            print(
                f"[trtmc build] detr: image={image_h}x{image_w}, "
                f"feature={feature_h}x{feature_w}, queries={num_queries}, "
                f"classes={num_labels + 1}, precision={precision}",
                file=sys.stderr)

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.builder_optimization_level = 1
        trt_config.avg_timing_iterations = 8
        trt_config.max_aux_streams = 0
        trt_config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

        pixel_values = network.add_input(
            "pixel_values", trt.float32, (1, 3, image_h, image_w))
        hidden_4d = pixel_values
        if hidden_4d.dtype != work_trt_dtype:
            hidden_4d = network.add_cast(hidden_4d, work_trt_dtype).get_output(0)

        # Backbone + input projection.
        hidden_4d = self._build_backbone(network, hidden_4d, weights, cfg, work_np_dtype)
        proj = graph_ops.add_conv2d(
            network, hidden_4d, weights["model.input_projection.weight"],
            weights.get("model.input_projection.bias"),
            int(weights["model.input_projection.weight"].shape[0]), (1, 1),
            dtype=work_np_dtype)
        proj_flat = network.add_shuffle(proj)
        proj_flat.first_transpose = trt.Permutation([0, 2, 3, 1])
        proj_flat.reshape_dims = (seq_len, d_model)
        hidden = proj_flat.get_output(0)

        spatial_pos_np = graph_ops.make_sine_position_embedding(
            feature_h, feature_w, d_model)
        spatial_pos = graph_ops.add_constant(
            network, (seq_len, d_model), spatial_pos_np.reshape(seq_len, d_model), dtype=np.float32)
        if spatial_pos.dtype != hidden.dtype:
            spatial_pos = network.add_cast(spatial_pos, hidden.dtype).get_output(0)

        # Encoder.
        for layer_idx in range(enc_layers):
            prefix = f"model.encoder.layers.{layer_idx}"
            residual = hidden
            hidden = self._attention(
                network, hidden, weights, f"{prefix}.self_attn", d_model,
                enc_heads, work_np_dtype, pos_embedding=spatial_pos)
            hidden = graph_ops.add_sum(network, hidden, residual)
            hidden = self._layernorm(
                network, hidden, weights, f"{prefix}.self_attn_layer_norm",
                d_model, work_np_dtype)

            residual = hidden
            hidden = self._linear(network, hidden, weights, f"{prefix}.fc1", enc_ffn, work_np_dtype)
            hidden = graph_ops.add_relu(network, hidden)
            hidden = self._linear(network, hidden, weights, f"{prefix}.fc2", d_model, work_np_dtype)
            hidden = graph_ops.add_sum(network, hidden, residual)
            hidden = self._layernorm(
                network, hidden, weights, f"{prefix}.final_layer_norm",
                d_model, work_np_dtype)

        # Decoder.
        object_queries_pos_np = weights["model.query_position_embeddings.weight"].astype(np.float32)
        object_queries_pos = graph_ops.add_constant(
            network, (num_queries, d_model), object_queries_pos_np, dtype=np.float32)
        if object_queries_pos.dtype != hidden.dtype:
            object_queries_pos = network.add_cast(object_queries_pos, hidden.dtype).get_output(0)

        queries = graph_ops.add_constant(
            network, (num_queries, d_model),
            np.zeros((num_queries, d_model), dtype=np.float32), dtype=np.float32)
        if queries.dtype != hidden.dtype:
            queries = network.add_cast(queries, hidden.dtype).get_output(0)

        for layer_idx in range(dec_layers):
            prefix = f"model.decoder.layers.{layer_idx}"

            residual = queries
            queries = self._attention(
                network, queries, weights, f"{prefix}.self_attn", d_model,
                dec_heads, work_np_dtype, pos_embedding=object_queries_pos)
            queries = graph_ops.add_sum(network, queries, residual)
            queries = self._layernorm(
                network, queries, weights, f"{prefix}.self_attn_layer_norm",
                d_model, work_np_dtype)

            residual = queries
            queries = self._attention(
                network, queries, weights, f"{prefix}.encoder_attn", d_model,
                dec_heads, work_np_dtype,
                pos_embedding=object_queries_pos,
                kv_input=hidden, kv_pos_embedding=spatial_pos)
            queries = graph_ops.add_sum(network, queries, residual)
            queries = self._layernorm(
                network, queries, weights, f"{prefix}.encoder_attn_layer_norm",
                d_model, work_np_dtype)

            residual = queries
            queries = self._linear(network, queries, weights, f"{prefix}.fc1", dec_ffn, work_np_dtype)
            queries = graph_ops.add_relu(network, queries)
            queries = self._linear(network, queries, weights, f"{prefix}.fc2", d_model, work_np_dtype)
            queries = graph_ops.add_sum(network, queries, residual)
            queries = self._layernorm(
                network, queries, weights, f"{prefix}.final_layer_norm",
                d_model, work_np_dtype)

        queries = self._layernorm(
            network, queries, weights, "model.decoder.layernorm", d_model, work_np_dtype)

        # Heads operate on [1, num_queries, d_model].
        head_input = network.add_shuffle(queries)
        head_input.reshape_dims = (1, num_queries, d_model)
        head_input = head_input.get_output(0)

        logits = self._linear(
            network, head_input, weights, "class_labels_classifier",
            num_labels + 1, work_np_dtype)

        box = self._linear(
            network, head_input, weights, "bbox_predictor.layers.0", d_model, work_np_dtype)
        box = graph_ops.add_relu(network, box)
        box = self._linear(
            network, box, weights, "bbox_predictor.layers.1", d_model, work_np_dtype)
        box = graph_ops.add_relu(network, box)
        box = self._linear(
            network, box, weights, "bbox_predictor.layers.2", 4, work_np_dtype)
        box = network.add_activation(box, trt.ActivationType.SIGMOID).get_output(0)

        if logits.dtype != trt.float32:
            logits = network.add_cast(logits, trt.float32).get_output(0)
        if box.dtype != trt.float32:
            box = network.add_cast(box, trt.float32).get_output(0)
        logits.name = "logits"
        box.name = "pred_boxes"
        network.mark_output(logits)
        network.mark_output(box)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT detr engine build failed")
        return bytes(plan)

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict:
        cfg = config.raw.get("_detr_config") or _resolve_config(config.raw)
        return {
            "model_type": config.model_type,
            "input_image_h": cfg["image_size_h"],
            "input_image_w": cfg["image_size_w"],
            "shortest_edge": 800,
            "longest_edge": 1333,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "num_queries": cfg["num_queries"],
            "num_labels": cfg["num_labels"],
        }


def build(request, writer) -> None:
    """Build one DETR object-detection bundle."""
    if request.task != "object_detection":
        raise ValueError("detr supports only task=object_detection")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("detr supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("detr does not support dynamic_kv_cache")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("detr supports only max_sequence_length=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("detr does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("detr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("detr does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("detr does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("detr does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("detr does not support mixed-precision layers")

    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if config.model_type.lower() != "detr":
        raise ValueError(f"detr does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported detr precision: {precision}")

    if request.image_height is not None or request.image_width is not None:
        build_options = config.raw.setdefault("_family_build_options", {})
        if request.image_height is not None:
            build_options["input_image_h"] = int(request.image_height)
        if request.image_width is not None:
            build_options["input_image_w"] = int(request.image_width)

    model = _DetrModel()
    weights = model.load_weights(str(model_dir), config, precision=precision)
    plan = model.build_engine(
        config,
        weights,
        precision=precision,
        verbose=bool(request.verbose),
    )

    runtime = model.get_bundle_config_overrides(config)
    writer.set_header(family="detr", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            key: runtime[key]
            for key in (
                "input_image_h",
                "input_image_w",
                "shortest_edge",
                "longest_edge",
                "image_mean",
                "image_std",
                "num_queries",
                "num_labels",
            )
        },
    )
