# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned LocateAnything vision-stage oracle."""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np

_BUNDLE_MAGIC = b"BUNDLE\x01\x00"
VISION_FEATURE_COSINE = 0.5


def _bundle_section(bundle: Path, name: str) -> bytes:
    with bundle.open("rb") as stream:
        assert stream.read(8) == _BUNDLE_MAGIC
        encoded_length = stream.read(8)
        assert len(encoded_length) == 8
        header_length = struct.unpack("<Q", encoded_length)[0]
        header = json.loads(stream.read(header_length))
        section = header["sections"][name]
        stream.seek(16 + header_length + int(section["offset"]))
        data = stream.read(int(section["length"]))
    assert data
    return data


def _patchify_pixels(image_path: Path, config: dict) -> dict[str, np.ndarray]:
    from PIL import Image

    size = int(config["fixed_image_size"])
    patch = int(config["patch_size"])
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(config["image_mean"], dtype=np.float32)
    std = np.asarray(config["image_std"], dtype=np.float32)
    chw = ((pixels - mean) / std).transpose(2, 0, 1)
    channels = chw.shape[0]
    grid = size // patch
    patches = chw.reshape(channels, grid, patch, grid, patch)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(grid * grid, channels, patch, patch)
    return {
        "pixel_values": np.ascontiguousarray(patches, dtype=np.float32),
        "image_grid_hws": np.array([[grid, grid]], dtype=np.int32),
    }


def _torch_dtype(dtype):
    import tensorrt as trt
    import torch

    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }[dtype]


def _execute_vision_plan(plan: bytes, inputs: dict[str, np.ndarray]) -> np.ndarray:
    import tensorrt as trt
    import torch

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    tensors = {}
    output_names = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(int(value) for value in engine.get_tensor_shape(name))
        assert shape and all(value > 0 for value in shape)
        dtype = _torch_dtype(engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            assert name in inputs
            value = np.asarray(inputs[name])
            assert value.shape == shape
            tensor = torch.as_tensor(np.ascontiguousarray(value), device="cuda", dtype=dtype)
        else:
            tensor = torch.empty(shape, device="cuda", dtype=dtype)
            output_names.append(name)
        tensors[name] = tensor
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert output_names == ["image_features"]
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    result = tensors["image_features"].float().cpu().numpy()
    del tensors, context, engine
    torch.cuda.empty_cache()
    return result


def native_vision_features(bundle: Path, image_path: Path) -> np.ndarray:
    config = json.loads(_bundle_section(bundle, "runtime.json"))
    assert config["preprocessor_type"] == "patchify_chw"
    inputs = _patchify_pixels(image_path, config)
    return _execute_vision_plan(_bundle_section(bundle, "vision.plan"), inputs)


def official_vision_features(model_dir: Path, image_path: Path) -> np.ndarray:
    import torch
    from PIL import Image

    from families.locateanything.config import ModelConfig
    from families.locateanything.vision_builder import (
        _build_moonvit,
        _build_projector,
        _load_modeling_vit,
        _load_vision_and_projector_weights,
    )

    config = ModelConfig.from_dir(model_dir)
    vision_model = _build_moonvit(_load_modeling_vit(model_dir), config)
    projector = _build_projector(config)
    _load_vision_and_projector_weights(model_dir, vision_model, projector)
    vision_model = vision_model.to(device="cuda", dtype=torch.float32).eval()
    projector = projector.to(device="cuda", dtype=torch.float32).eval()

    processor_path = model_dir / "image_processing_locateanything.py"
    spec = importlib.util.spec_from_file_location(
        "trtmc_locateanything_vision_oracle", processor_path
    )
    assert spec is not None and spec.loader is not None
    processor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor_module)
    vision_config = config.raw["vision_config"]
    image_processor = processor_module.LocateAnythingImageProcessor(
        patch_size=int(vision_config["patch_size"]),
        merge_kernel_size=vision_config["merge_kernel_size"],
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    image = Image.open(image_path).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
    encoded = image_processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        vit_features = vision_model(
            encoded["pixel_values"].to(device="cuda", dtype=torch.float32),
            encoded["image_grid_hws"].to(device="cuda"),
        )
        features = projector(torch.cat(vit_features, dim=0))
    result = features.float().cpu().numpy()
    del vision_model, projector
    torch.cuda.empty_cache()
    return result


def assert_vision_parity(native, official) -> float:
    left = np.asarray(native, dtype=np.float32)
    right = np.asarray(official, dtype=np.float32)
    assert left.size > 0 and right.size > 0
    assert np.isfinite(left).all() and np.isfinite(right).all()
    assert np.any(left != 0.0) and np.any(right != 0.0)
    assert left.ndim == 2 and right.ndim == 2
    rows = min(left.shape[0], right.shape[0])
    columns = min(left.shape[1], right.shape[1])
    left = left[:rows, :columns].reshape(-1)
    right = right[:rows, :columns].reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    assert denominator > 0.0
    cosine = float(np.dot(left, right) / denominator)
    assert cosine >= VISION_FEATURE_COSINE
    return cosine
