# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

_BUNDLE_MAGIC = b"BUNDLE\x01\x00"
VISION_FEATURE_COSINE = 0.5


def _bundle_section(bundle: Path, name: str, *, output: Path | None = None) -> bytes:
    """Read metadata or stream a weight section without allocating the whole checkpoint."""
    with bundle.open("rb") as stream:
        assert stream.read(8) == _BUNDLE_MAGIC
        encoded_length = stream.read(8)
        assert len(encoded_length) == 8
        header_length = struct.unpack("<Q", encoded_length)[0]
        assert 0 < header_length <= min(100 * 1024 * 1024, bundle.stat().st_size - 16)
        header = json.loads(stream.read(header_length))
        section = header["sections"][name]
        offset, length = section["offset"], section["length"]
        assert type(offset) is int and type(length) is int
        assert offset >= 0 and length > 0
        assert 16 + header_length + offset + length <= bundle.stat().st_size
        stream.seek(16 + header_length + offset)
        if output is None:
            data = stream.read(length)
            assert len(data) == length
            return data
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as destination:
            while length:
                chunk = stream.read(min(length, 64 * 1024))
                assert chunk
                destination.write(chunk)
                length -= len(chunk)
    return b""


def _edge_vision_features(bundle: Path, image_path: Path, marker: dict) -> np.ndarray:
    """Keep the existing health test on actual features, not generated text."""
    from PIL import Image

    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    with tempfile.TemporaryDirectory(prefix="internvl-vision-") as temporary:
        root = Path(temporary)
        for name in marker["artifacts"]:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != name
                or "\\" in name
                or "\0" in name
                or not name.startswith(("edge_llm/engine/", "edge_llm/checkpoint/"))
            ):
                raise ValueError(f"Unsafe Edge artifact: {name}")
            if name.startswith(("edge_llm/engine/visual/", "edge_llm/checkpoint/")):
                _bundle_section(bundle, name, output=root / name)
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        rgb = root / "image.rgb"
        rgb.write_bytes(image.tobytes())
        features = root / "features.fp16"
        subprocess.run(
            [
                str(runtime / "families/internvl/internvl_edge_vision_features"),
                str(root / "edge_llm/engine"),
                str(root / "edge_llm/checkpoint"),
                str(runtime / "libNvInfer_edgellm_plugin.so"),
                str(rgb),
                str(image.shape[0]),
                str(image.shape[1]),
                str(marker["max_sequence_length"]),
                str(features),
            ],
            check=True,
            timeout=1800,
        )
        return np.fromfile(features, dtype=np.float16).astype(np.float32)


def _native_pixels(image_path: Path, config: dict) -> np.ndarray:
    from PIL import Image

    size = int(config["fixed_image_size"])
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(config["image_mean"], dtype=np.float32)
    std = np.asarray(config["image_std"], dtype=np.float32)
    return np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1))


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

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
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
    del tensors, context, engine, runtime, logger
    torch.cuda.empty_cache()
    return result


def native_vision_features(bundle: Path, image_path: Path) -> np.ndarray:
    try:
        marker = json.loads(_bundle_section(bundle, "edge_llm.json"))
    except KeyError:
        marker = None
    if marker is not None:
        return _edge_vision_features(bundle, image_path, marker)
    config = json.loads(_bundle_section(bundle, "runtime.json"))
    pixels = _native_pixels(image_path, config)
    return _execute_vision_plan(_bundle_section(bundle, "vision.plan"), {"pixel_values": pixels})


def official_vision_features(model, processor, image) -> np.ndarray:
    import torch

    encoded = processor.image_processor(images=image, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device="cuda", dtype=model.dtype)
    with torch.no_grad():
        output = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=model.config.vision_feature_layer,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        )
    return output.pooler_output.float().cpu().numpy().squeeze(0)


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
