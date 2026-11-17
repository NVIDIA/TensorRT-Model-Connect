# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph for the source BEVFusion fusion-bev256d2-lss BEV path."""

from pathlib import Path

import numpy as np


CLASSES = ("drivable_area", "ped_crossing", "walkway", "stop_line", "carpark_area", "divider")
INPUT_SHAPES = {"camera_bev": (1, 80, 128, 128), "lidar_bev": (1, 256, 128, 128)}
OUTPUT_SHAPE = (1, 6, 200, 200)


def grid_coordinates() -> np.ndarray:
    """Source BEVGridTransform grid, with x in the last dimension's first slot."""
    centers = np.arange(-49.75, 50.0, 0.5, dtype=np.float32)
    normalized = (centers + np.float32(51.2)) / np.float32(102.4) * 2 - 1
    y, x = np.meshgrid(normalized, normalized, indexing="ij")
    return np.ascontiguousarray(np.stack((x, y), axis=-1)[None])


class Weights:
    """Read only the selected checkpoint's BEV stages and validate every operand."""

    def __init__(self, checkpoint: Path):
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        prefixes = ("fuser.", "decoder.", "heads.map.")
        self.values = {name: value for name, value in state.items() if name.startswith(prefixes)}
        self.used = set()

    def get(self, name, shape):
        value = self.values.get(name)
        if value is None or tuple(value.shape) != tuple(shape):
            raise ValueError(f"BEVFusion checkpoint requires {name} with shape {tuple(shape)}")
        result = value.detach().float().numpy()
        if not np.isfinite(result).all():
            raise ValueError(f"BEVFusion checkpoint has non-finite values in {name}")
        self.used.add(name)
        return np.ascontiguousarray(result)

    def finish(self):
        unused = set(self.values) - self.used
        unused = {name for name in unused if not name.endswith(".num_batches_tracked")}
        if unused:
            raise ValueError(f"BEVFusion checkpoint has unexpected BEV weights: {sorted(unused)}")


class Graph:
    def __init__(self, network, weights, precision, fp8_scales=None):
        import tensorrt as trt

        self.trt = trt
        self.network = network
        self.weights = weights
        self.fp8_scales = fp8_scales or {}
        self.dtype = trt.float16 if precision == "fp16" else trt.float32
        self.numpy_dtype = np.float16 if precision == "fp16" else np.float32
        self.storage = []

    def keep(self, value, dtype=None):
        array = np.ascontiguousarray(value, dtype=dtype)
        self.storage.append(array)
        return array

    def constant(self, value):
        array = self.keep(value, np.float32)
        return self.network.add_constant(array.shape, array).get_output(0)

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.network.add_cast(value, dtype).get_output(0)

    def conv(self, value, name, channels, kernel=3, stride=1, transpose=False, bias=False):
        inputs = int(value.shape[1])
        shape = (
            (inputs, channels, kernel, kernel) if transpose else (channels, inputs, kernel, kernel)
        )
        source_weight = self.weights.get(name + ".weight", shape)
        offset = (
            self.keep(self.weights.get(name + ".bias", (channels,)), self.numpy_dtype)
            if bias
            else self.trt.Weights(self.dtype)
        )
        if self.fp8_scales.get(name) is not None:
            from .quantization import add_convolution

            if transpose or bias:
                raise ValueError("BEVFusion FP8 recipe requires ordinary, bias-free convolutions")
            layer = add_convolution(self, value, source_weight, self.fp8_scales[name])
        else:
            weight = self.keep(source_weight, self.numpy_dtype)
            add = (
                self.network.add_deconvolution_nd if transpose else self.network.add_convolution_nd
            )
            layer = add(value, channels, (kernel, kernel), weight, offset)
        layer.name = name
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (kernel // 2, kernel // 2) if not transpose else (0, 0)
        return layer.get_output(0)

    def norm_relu(self, value, name, epsilon):
        channels = int(value.shape[1])
        gamma, beta, mean, variance = (
            self.weights.get(name + suffix, (channels,))
            for suffix in (".weight", ".bias", ".running_mean", ".running_var")
        )
        if (variance < 0).any():
            raise ValueError(f"BEVFusion checkpoint has negative variance in {name}")
        # PyTorch BatchNorm keeps its statistics in FP32 for half activations.
        scale = gamma / np.sqrt(variance + np.float32(epsilon))
        shift = beta - mean * scale
        value = self.cast(value, self.trt.float32)
        value = self.network.add_elementwise(
            value,
            self.constant(scale.reshape(1, channels, 1, 1)),
            self.trt.ElementWiseOperation.PROD,
        ).get_output(0)
        value = self.network.add_elementwise(
            value,
            self.constant(shift.reshape(1, channels, 1, 1)),
            self.trt.ElementWiseOperation.SUM,
        ).get_output(0)
        value = self.network.add_activation(value, self.trt.ActivationType.RELU).get_output(0)
        return self.cast(value, self.dtype)

    def concat(self, values):
        layer = self.network.add_concatenation(values)
        layer.axis = 1
        return layer.get_output(0)

    def create(self):
        inputs = [
            self.cast(self.network.add_input(name, self.trt.float32, shape), self.dtype)
            for name, shape in INPUT_SHAPES.items()
        ]
        value = self.conv(self.concat(inputs), "fuser.0", 256)
        value = self.norm_relu(value, "fuser.1", 1e-5)
        stages = []
        for stage, (channels, stride) in enumerate(((128, 1), (256, 2))):
            prefix = f"decoder.backbone.blocks.{stage}"
            for index in range(6):
                value = self.conv(
                    value, f"{prefix}.{3 * index}", channels, stride=stride if index == 0 else 1
                )
                value = self.norm_relu(value, f"{prefix}.{3 * index + 1}", 1e-3)
            stages.append(value)
        branches = []
        for stage, value in enumerate(stages):
            prefix = f"decoder.neck.deblocks.{stage}"
            value = self.conv(
                value, prefix + ".0", 256, kernel=stage + 1, stride=stage + 1, transpose=stage == 1
            )
            branches.append(self.norm_relu(value, prefix + ".1", 1e-3))
        value = self.concat(branches)
        # The source resamples BEV features before the classifier, not its output.
        value = self.cast(value, self.trt.float32)
        sample = self.network.add_grid_sample(value, self.constant(grid_coordinates()))
        sample.align_corners = False
        sample.interpolation_mode = self.trt.InterpolationMode.LINEAR
        sample.sample_mode = self.trt.SampleMode.FILL
        value = self.cast(sample.get_output(0), self.dtype)
        for index in (0, 3):
            value = self.conv(value, f"heads.map.classifier.{index}", 512)
            value = self.norm_relu(value, f"heads.map.classifier.{index + 1}", 1e-5)
        value = self.conv(value, "heads.map.classifier.6", 6, kernel=1, bias=True)
        value = self.cast(value, self.trt.float32)
        result = self.network.add_activation(value, self.trt.ActivationType.SIGMOID).get_output(0)
        result.name = "map_probabilities"
        assert tuple(result.shape) == OUTPUT_SHAPE
        self.network.mark_output(result)
        self.weights.finish()


def build_engine(
    checkpoint: Path, precision="fp32", verbose=False, quantization=None, *, metadata=None
) -> bytes:
    """Build one fixed-shape BEV forward engine; inputs and output use FP32."""
    import tensorrt as trt

    if precision not in {"fp32", "fp16"}:
        raise ValueError("BEVFusion supports only fp32 and fp16")
    if quantization not in (None, "none", "fp8"):
        raise ValueError("BEVFusion supports only fp8 quantization or none")
    if quantization == "fp8" and precision != "fp16":
        raise ValueError("BEVFusion FP8 requires precision fp16")
    weights = Weights(Path(checkpoint))
    fp8_scales, calibration = {}, None
    if quantization == "fp8":
        from .quantization import load_scales

        fp8_scales, calibration = load_scales(checkpoint, weights.values)
    if metadata is not None:
        metadata.update(
            quantization=quantization or "none",
            fp8_layers=[name for name, scale in fp8_scales.items() if scale is not None],
            calibration=calibration,
        )
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Graph(network, weights, precision, fp8_scales)
    graph.create()
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if fp8_scales:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the BEVFusion BEV engine")
    return bytes(plan)


def build(request, writer):
    """Build the fixed-shape BEV feature-to-map bundle through the public builder."""
    if request.task != "spatial_features_to_class_scores" or request.backend != "trt":
        raise ValueError("BEVFusion requires spatial_features_to_class_scores with the trt backend")
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
    ):
        raise ValueError("BEVFusion supports batch size one without parallel sharding")
    if any(
        value is not None
        for value in (
            request.max_sequence_length,
            request.image_height,
            request.image_width,
            request.video_num_frames,
            request.mtp_seq_len,
        )
    ):
        raise ValueError("BEVFusion uses fixed BEV feature shapes without text/image/video options")
    if request.dynamic_kv_cache:
        raise NotImplementedError("bevfusion does not support dynamic_kv_cache")
    if request.fp32_layers:
        raise ValueError("BEVFusion does not support KV caches or selected FP32 layers")
    checkpoint = Path(request.model_dir) / "bevfusion-seg.pth"
    metadata = {}
    plan = build_engine(
        checkpoint, request.precision, request.verbose, request.quantization, metadata=metadata
    )
    writer.set_header(family="bevfusion", task=request.task, backend=request.backend)
    writer.add_json(
        "runtime.json",
        {
            **metadata,
            "precision": request.precision,
            "input_shapes": INPUT_SHAPES,
            "output_shape": OUTPUT_SHAPE,
            "classes": CLASSES,
            "checkpoint_file": checkpoint.name,
            "architecture_source": "https://github.com/mit-han-lab/bevfusion",
            "architecture_revision": "326653dc06e0938edf1aae7d01efcd158ba83de5",
            "architecture_config": "configs/nuscenes/seg/fusion-bev256d2-lss.yaml",
        },
    )
    writer.add_bytes("engine.plan", plan)
