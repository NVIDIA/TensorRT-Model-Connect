# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Qwen-VL vision-stage oracle."""

from __future__ import annotations

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


def _merge_group_pixels(image_path: Path, config: dict) -> np.ndarray:
    from PIL import Image

    height = int(config.get("fixed_image_height") or config["fixed_image_size"])
    width = int(config.get("fixed_image_width") or config["fixed_image_size"])
    patch = int(config["patch_size"])
    merge = int(config["merge_size"])
    temporal = int(config["temporal_patch_size"])
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(config["image_mean"], dtype=np.float32)
    std = np.asarray(config["image_std"], dtype=np.float32)
    chw = ((pixels - mean) / std).transpose(2, 0, 1)

    channels = chw.shape[0]
    grid_h, grid_w = height // patch, width // patch
    patches = chw.reshape(channels, grid_h, patch, grid_w, patch).transpose(1, 3, 0, 2, 4)
    grouped = patches.reshape(
        grid_h // merge, merge, grid_w // merge, merge, channels, patch, patch
    ).transpose(0, 2, 1, 3, 4, 5, 6)
    pseudo = grouped.reshape(grid_h, grid_w, channels, patch, patch)
    pseudo = pseudo.transpose(2, 0, 3, 1, 4).reshape(channels, height, width)
    return np.ascontiguousarray(np.repeat(pseudo, temporal, axis=0), dtype=np.float32)


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
    for name, value in inputs.items():
        assert context.set_input_shape(name, np.asarray(value).shape)
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(int(value) for value in context.get_tensor_shape(name))
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
    assert "image_features" in output_names
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    result = tensors["image_features"].float().cpu().numpy()
    del tensors, context, engine
    torch.cuda.empty_cache()
    return result


def native_vision_features(bundle: Path, image_path: Path) -> np.ndarray:
    config = json.loads(_bundle_section(bundle, "runtime.json"))
    if config["preprocessor_type"] in {"qwen3_smart_resize_patchify", "qwen_smart_resize_patchify"}:
        inputs = _smart_resize_inputs(image_path, config)
    else:
        assert config["preprocessor_type"] == "merge_group_chw"
        inputs = {"pixel_values": _merge_group_pixels(image_path, config)}
    return _execute_vision_plan(_bundle_section(bundle, "vision.plan"), inputs)


def _smart_resize_inputs(image_path: Path, config: dict) -> dict[str, np.ndarray]:
    """Independent upstream preprocessing and PyTorch geometry for dynamic plans."""
    import torch
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    processor = Qwen2VLImageProcessorFast(
        patch_size=config["patch_size"], merge_size=config["merge_size"],
        temporal_patch_size=config["temporal_patch_size"],
        image_mean=config["image_mean"], image_std=config["image_std"],
        size={"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]},
    )
    with Image.open(image_path) as image:
        encoded = processor(images=[image.convert("RGB")], return_tensors="pt")
    _, height, width = encoded["image_grid_thw"][0].tolist()
    merge = config["merge_size"]
    rows, cols = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    def group(tensor):
        return tensor.reshape(height // merge, merge, width // merge, merge).permute(
            0, 2, 1, 3).reshape(-1)

    rows, cols = group(rows), group(cols)
    half = config["vision_embed_dim"] // config["vision_num_heads"] // 2
    inv = 1.0 / config["vision_rope_theta"] ** (torch.arange(0, half, 2).float() / half)
    angles = torch.cat([rows[:, None] * inv, cols[:, None] * inv], dim=1)
    inputs = {"pixel_values": encoded["pixel_values"].numpy()}
    if config["preprocessor_type"] == "qwen_smart_resize_patchify":
        from types import SimpleNamespace
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
        )

        geometry = SimpleNamespace(
            window_size=config["vision_window_size"], patch_size=config["patch_size"],
            spatial_merge_size=merge, spatial_merge_unit=merge**2)
        order, cumulative = Qwen2_5_VisionTransformerPretrainedModel.get_window_index(
            geometry, encoded["image_grid_thw"])
        angles = angles.reshape(-1, merge**2, half)[order].reshape(-1, half)
        inputs.update({"vision_cos_half": angles.cos().numpy(),
                       "vision_sin_half": angles.sin().numpy(),
                       "vision_window_indices": order.int().numpy(),
                       "vision_reverse_indices": order.argsort().int().numpy()})
        lengths = torch.unique_consecutive(torch.tensor(cumulative)).diff().tolist()
        per_window = (geometry.window_size // merge // geometry.patch_size)**2 * merge**2
        padded, compact, masks = [], [], []
        offset = 0
        for count in lengths:
            compact.extend(range(len(padded), len(padded) + count))
            padded.extend(list(range(offset, offset + count)) + [offset] * (per_window - count))
            masks.extend([0.0] * count + [-1e9] * (per_window - count))
            offset += count
        inputs["vision_padded_window_indices"] = np.array(padded, dtype=np.int32)
        inputs["vision_compact_window_indices"] = np.array(compact, dtype=np.int32)
        inputs["vision_window_mask"] = np.array(masks, dtype=np.float32).reshape(-1, 1, 1, per_window)
        return inputs

    side = config["vision_position_grid_size"]
    h = torch.linspace(0, side - 1, height)[rows]
    w = torch.linspace(0, side - 1, width)[cols]
    h0, w0 = h.long(), w.long()
    h1, w1 = (h0 + 1).clamp(max=side - 1), (w0 + 1).clamp(max=side - 1)
    dh, dw = h - h0, w - w0
    indices = [h0 * side + w0, h0 * side + w1, h1 * side + w0, h1 * side + w1]
    blends = [(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw]
    inputs.update({"vision_cos_half": angles.cos().numpy(), "vision_sin_half": angles.sin().numpy()})
    for corner in range(4):
        inputs[f"vision_position_indices_{corner}"] = indices[corner].int().numpy()
        inputs[f"vision_position_weights_{corner}"] = blends[corner][:, None].numpy()
    return inputs


def official_vision_features(
    model_dir: Path, processor, image, fixed_image_size: int
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    image = image.resize((fixed_image_size, fixed_image_size), Image.Resampling.BICUBIC)
    encoded = processor.image_processor(images=[image], return_tensors="pt")
    model = (
        AutoModelForImageTextToText.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.float32
        )
        .to("cuda")
        .eval()
    )
    inner = model.model
    visual = inner.visual
    device = visual.patch_embed.proj.weight.device
    with torch.no_grad():
        result = inner.get_image_features(
            encoded["pixel_values"].to(device=device, dtype=visual.dtype),
            image_grid_thw=encoded["image_grid_thw"].to(device=device),
            return_dict=True,
        )
    features = result.pooler_output
    if isinstance(features, (list, tuple)):
        features = torch.cat(features, dim=0)
    features = features.float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return features


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
