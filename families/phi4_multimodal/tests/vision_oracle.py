# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

import numpy as np

_BUNDLE_MAGIC = b"BUNDLE\x01\x00"
_IMAGE_TOKEN_ID = 200010
VISION_FEATURE_COSINE = 0.5
# Dynamic-HD geometry of runtime/phi4_dynamic_hd.h.
_HD_CROP = 448
_HD_PATCH = 14
_HD_PATCH_GRID = _HD_CROP // _HD_PATCH
_HD_POOLED_GRID = _HD_PATCH_GRID // 2
_HD_MAX_CROPS = 36
# Rows the vision engine prepends to its pooled crop features: sub_GN, then glb_GN.
_HD_SUB_SEPARATOR_ROW = 0
_HD_GLOBAL_SEPARATOR_ROW = 1
_HD_FIRST_FEATURE_ROW = 2


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


def _edge_vision_features(bundle: Path, image, marker: dict) -> np.ndarray:
    """Keep the existing health test on actual features, not generated text."""
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    native_build = os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    helper_root = Path(native_build) if native_build else runtime
    helper_dir = helper_root / "families/phi4_multimodal"
    helper = helper_dir / "phi4_multimodal_edge_vision_features"
    plugin = (
        helper_dir / "libNvInfer_edgellm_plugin.so"
        if native_build
        else runtime / "libNvInfer_edgellm_plugin.so"
    )
    with tempfile.TemporaryDirectory(prefix="phi4_multimodal-vision-") as temporary:
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
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        rgb = root / "image.rgb"
        rgb.write_bytes(image.tobytes())
        features = root / "features.fp16"
        subprocess.run(
            [
                str(helper),
                str(root / "edge_llm/engine"),
                str(root / "edge_llm/checkpoint"),
                str(plugin),
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


def _hd_crop_grid(width: int, height: int) -> tuple[int, int]:
    """dynamic_preprocess's (cols, rows) crop grid, as runtime/phi4_dynamic_hd.h ports it."""
    cols = math.ceil(width / _HD_CROP)
    rows = math.ceil(height / _HD_CROP)
    if cols * rows <= _HD_MAX_CROPS:
        return cols, rows
    aspect = width / height
    best, best_difference = (1, 1), math.inf
    for count in range(1, _HD_MAX_CROPS + 1):
        for ratio in ((i, count // i) for i in range(1, count + 1) if count % i == 0):
            difference = abs(aspect - ratio[0] / ratio[1])
            if difference < best_difference:
                best, best_difference = ratio, difference
            elif difference == best_difference and (
                width * height > 0.5 * _HD_CROP * _HD_CROP * ratio[0] * ratio[1]
            ):
                best = ratio
    return best


def _hd_crop_valid_and_token_rows(
    cols: int, rows: int, padding_width: int, padding_height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Valid patch rows/cols per vision batch entry and the sub_glb image-token rows."""
    valid_cols = cols * _HD_PATCH_GRID - padding_width // _HD_PATCH
    valid_rows = rows * _HD_PATCH_GRID - padding_height // _HD_PATCH
    crop_valid = [(_HD_PATCH_GRID, _HD_PATCH_GRID)]
    for row in range(rows):
        for col in range(cols):
            crop_cols = min(max(valid_cols - col * _HD_PATCH_GRID, 0), _HD_PATCH_GRID)
            crop_rows = min(max(valid_rows - row * _HD_PATCH_GRID, 0), _HD_PATCH_GRID)
            # SiglipVisionEmbeddings counts rows on the first column and columns on the first row.
            crop_valid.append((crop_rows if crop_cols else 0, crop_cols if crop_rows else 0))

    def feature_row(entry: int, y: int, x: int) -> int:
        return _HD_FIRST_FEATURE_ROW + (entry * _HD_POOLED_GRID + y) * _HD_POOLED_GRID + x

    token_rows = []
    for y in range((valid_rows + 1) // 2):
        for x in range((valid_cols + 1) // 2):
            crop = 1 + (y // _HD_POOLED_GRID) * cols + x // _HD_POOLED_GRID
            token_rows.append(feature_row(crop, y % _HD_POOLED_GRID, x % _HD_POOLED_GRID))
        token_rows.append(_HD_SUB_SEPARATOR_ROW)
    token_rows.append(_HD_GLOBAL_SEPARATOR_ROW)
    for y in range(_HD_POOLED_GRID):
        token_rows.extend(feature_row(0, y, x) for x in range(_HD_POOLED_GRID))
        token_rows.append(_HD_SUB_SEPARATOR_ROW)
    return np.asarray(crop_valid, dtype=np.int32), np.asarray(token_rows, dtype=np.int32)


def _native_dynamic_hd_inputs(image, config: dict) -> dict[str, np.ndarray]:
    """The vision engine inputs of the native runtime's Dynamic-HD preprocessing."""
    import torch
    from PIL import Image
    from torch.nn import functional

    assert int(config["fixed_image_size"]) == _HD_CROP
    assert config["preprocessor_type"] == "phi4_hd_chw"
    assert config["interpolation"] == "bilinear"
    image = image.convert("RGB")
    cols, rows = _hd_crop_grid(image.width, image.height)
    target_width, target_height = _HD_CROP * cols, _HD_CROP * rows
    ratio_width = target_width / image.width
    ratio_height = target_height / image.height
    if ratio_width < ratio_height:
        new_width, new_height = target_width, int(image.height * ratio_width)
    else:
        new_width, new_height = int(image.width * ratio_height), target_height
    assert min(new_width, new_height) >= 10
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    canvas.paste(resized, (0, 0))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.asarray(config["image_mean"], dtype=np.float32)
    std = np.asarray(config["image_std"], dtype=np.float32)
    chw = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1))
    global_view = functional.interpolate(
        torch.from_numpy(chw).unsqueeze(0),
        size=(_HD_CROP, _HD_CROP),
        mode="bicubic",
        align_corners=False,
    ).numpy()
    crops = (
        chw.reshape(3, rows, _HD_CROP, cols, _HD_CROP)
        .transpose(1, 3, 0, 2, 4)
        .reshape(rows * cols, 3, _HD_CROP, _HD_CROP)
    )
    crop_valid, token_rows = _hd_crop_valid_and_token_rows(
        cols, rows, target_width - new_width, target_height - new_height
    )
    return {
        "pixel_values": np.ascontiguousarray(np.concatenate((global_view, crops))),
        "crop_valid": crop_valid,
        "image_token_rows": token_rows,
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

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    input_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    output_names = [name for name in names if name not in input_names]
    assert sorted(input_names) == sorted(inputs)
    assert output_names == ["image_features"]
    tensors = {}
    for name in input_names:
        value = np.ascontiguousarray(inputs[name])
        declared = tuple(int(size) for size in engine.get_tensor_shape(name))
        assert len(declared) == value.ndim
        assert all(size in (-1, actual) for size, actual in zip(declared, value.shape))
        assert context.set_input_shape(name, value.shape)
        dtype = _torch_dtype(engine.get_tensor_dtype(name))
        tensors[name] = torch.as_tensor(value, device="cuda", dtype=dtype)
    for name in output_names:
        shape = tuple(int(size) for size in context.get_tensor_shape(name))
        assert shape and all(size > 0 for size in shape)
        dtype = _torch_dtype(engine.get_tensor_dtype(name))
        tensors[name] = torch.empty(shape, device="cuda", dtype=dtype)
    for name, tensor in tensors.items():
        assert context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    result = tensors["image_features"].float().cpu().numpy()
    del tensors, context, engine, runtime, logger
    torch.cuda.empty_cache()
    return result


def native_vision_features(bundle: Path, image) -> np.ndarray:
    try:
        marker = json.loads(_bundle_section(bundle, "edge_llm.json"))
    except KeyError:
        marker = None
    if marker is not None:
        return _edge_vision_features(bundle, image, marker)
    config = json.loads(_bundle_section(bundle, "runtime.json"))
    inputs = _native_dynamic_hd_inputs(image, config)
    features = _execute_vision_plan(_bundle_section(bundle, "vision.plan"), inputs)
    assert features.shape[0] == inputs["image_token_rows"].shape[0]
    return features


def official_vision_features(model, processor, image) -> np.ndarray:
    import torch

    encoded = processor.image_processor(images=[image], return_tensors="pt")
    input_image_embeds = encoded["input_image_embeds"].to("cuda")
    image_sizes = encoded["image_sizes"].to("cuda")
    image_attention_mask = encoded["image_attention_mask"].to("cuda")
    token_count = int(encoded["num_img_tokens"][0])
    input_ids = torch.full((1, token_count), _IMAGE_TOKEN_ID, dtype=torch.long, device="cuda")
    with torch.no_grad():
        features = model.model.embed_tokens_extend(
            input_ids=input_ids,
            input_embeds=None,
            input_image_embeds=input_image_embeds,
            image_sizes=image_sizes,
            image_attention_mask=image_attention_mask,
            wte=model.model.embed_tokens,
        )
    return features[0].float().cpu().numpy()


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
