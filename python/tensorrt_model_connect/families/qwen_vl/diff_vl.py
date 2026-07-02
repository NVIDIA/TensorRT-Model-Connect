# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VL diff hooks owned by the Qwen VL family."""

from __future__ import annotations

import sys

import numpy as np


def _free_gpu() -> None:
    """Force-free all GPU memory used by temporary reference models."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        from cuda import cudart  # type: ignore[import-untyped]

        cudart.cudaDeviceSynchronize()
    except ImportError:
        pass


def _detect_model_type(model_id: str) -> str:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(cfg, "model_type", "unknown")


def handles_model_type(model_type: str) -> bool:
    mt = model_type.lower()
    return "qwen" in mt and "vl" in mt


def get_hf_vision_features(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> tuple[np.ndarray, np.ndarray]:
    """Qwen-specific HF vision feature extraction."""
    import torch
    from PIL import Image

    model_type = _detect_model_type(model_id)
    is_qwen3 = "qwen3" in model_type.lower()

    if is_qwen3:
        print(f"[diff_vl] Loading HF Qwen3 VL model {model_id} ...", file=sys.stderr)
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    else:
        print(f"[diff_vl] Loading HF Qwen2.5 VL model {model_id} ...", file=sys.stderr)
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_cls.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device)
    model.eval()

    image = Image.open(image_path).convert("RGB")

    if is_qwen3:
        from transformers import AutoImageProcessor

        image_processor = AutoImageProcessor.from_pretrained(
            model_id, trust_remote_code=True)
    else:
        from transformers import Qwen2VLImageProcessor

        image_processor = Qwen2VLImageProcessor.from_pretrained(model_id)

    image_fixed = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)
    image_inputs = image_processor(images=[image_fixed], return_tensors="pt")
    pixel_values_hf = image_inputs["pixel_values"]
    image_grid_thw = image_inputs["image_grid_thw"]

    print(f"[diff_vl] HF pixel_values: {pixel_values_hf.shape}, "
          f"grid_thw: {image_grid_thw.tolist()}", file=sys.stderr)

    inner = model.model
    with torch.no_grad():
        vis_out = inner.get_image_features(
            pixel_values_hf.to(device=inner.visual.patch_embed.proj.weight.device,
                               dtype=inner.visual.dtype),
            image_grid_thw=image_grid_thw.to(
                device=inner.visual.patch_embed.proj.weight.device),
            return_dict=True)
        pooler = vis_out.pooler_output
        if isinstance(pooler, (list, tuple)):
            hf_features = torch.cat(pooler, dim=0)
        else:
            hf_features = pooler

    hf_features_np = hf_features.float().cpu().numpy()
    print(f"[diff_vl] HF features: shape={hf_features_np.shape}, "
          f"mean={hf_features_np.mean():.6f}, std={hf_features_np.std():.6f}",
          file=sys.stderr)

    image_resized = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)
    img_np = np.array(image_resized, dtype=np.float32) / 255.0
    mean = np.array(image_processor.image_mean, dtype=np.float32)
    std = np.array(image_processor.image_std, dtype=np.float32)
    img_np = (img_np - mean) / std
    img_chw = img_np.transpose(2, 0, 1)
    temporal = getattr(image_processor, "temporal_patch_size", 1)
    trt_pixel_values = np.tile(img_chw, (temporal, 1, 1)).astype(np.float32)

    del model
    _free_gpu()
    return hf_features_np, trt_pixel_values


def debug_layers(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> bool:
    """Per-layer HF vision encoder analysis for Qwen VL."""
    import torch
    from PIL import Image
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLImageProcessor,
    )

    print(f"[diff_vl] Debug layers: loading HF model {model_id} ...",
          file=sys.stderr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device)
    model.eval()

    visual = model.model.visual

    image_processor = Qwen2VLImageProcessor.from_pretrained(model_id)
    image = Image.open(image_path).convert("RGB")
    image_fixed = image.resize(
        (fixed_image_size, fixed_image_size), Image.BICUBIC)
    image_inputs = image_processor(images=[image_fixed], return_tensors="pt")
    pixel_values_hf = image_inputs["pixel_values"].to(
        device=visual.patch_embed.proj.weight.device, dtype=visual.dtype)
    grid_thw = image_inputs["image_grid_thw"].to(
        device=visual.patch_embed.proj.weight.device)

    print(f"[diff_vl] pixel_values: {pixel_values_hf.shape}, "
          f"grid_thw: {grid_thw.tolist()}", file=sys.stderr)

    layer_outputs: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook_fn(_module, _input, output):
            if isinstance(output, torch.Tensor):
                layer_outputs[name] = output.detach().float().cpu()
            elif isinstance(output, tuple):
                layer_outputs[name] = output[0].detach().float().cpu()

        return hook_fn

    hooks = []
    hooks.append(visual.patch_embed.register_forward_hook(
        make_hook("patch_embed")))
    for i, blk in enumerate(visual.blocks):
        hooks.append(blk.register_forward_hook(make_hook(f"block.{i}")))
    hooks.append(visual.merger.register_forward_hook(make_hook("merger")))

    with torch.no_grad():
        hf_out = visual(pixel_values_hf, grid_thw=grid_thw)

    for hook in hooks:
        hook.remove()

    print(f"\n{'Layer':<20} {'Shape':<20} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)

    for name in sorted(layer_outputs.keys(),
                       key=lambda x: (0 if x == "patch_embed" else
                                      1 if x.startswith("block") else 2,
                                      int(x.split(".")[-1])
                                      if "." in x else 0)):
        t = layer_outputs[name].numpy()
        print(f"{name:<20} {str(t.shape):<20} {t.mean():>10.4f} "
              f"{t.std():>10.4f} {t.min():>10.4f} {t.max():>10.4f}",
              file=sys.stderr)

    block_keys = [k for k in sorted(layer_outputs.keys())
                  if k.startswith("block.")]
    if len(block_keys) >= 2:
        print(f"\n{'Layer Pair':<30} {'Cosine Sim':>12} {'Norm Ratio':>12}",
              file=sys.stderr)
        print("-" * 54, file=sys.stderr)
        prev_key = block_keys[0]
        for key in block_keys[1:]:
            a = layer_outputs[prev_key].numpy().flatten()
            b = layer_outputs[key].numpy().flatten()
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            ratio = np.linalg.norm(b) / (np.linalg.norm(a) + 1e-8)
            print(f"{prev_key} -> {key:<15} {cos:>12.6f} {ratio:>12.4f}",
                  file=sys.stderr)
            prev_key = key

    if "merger" in layer_outputs:
        merger_np = layer_outputs["merger"].numpy()
        print(f"\n[diff_vl] Merger output: shape={merger_np.shape}, "
              f"mean={merger_np.mean():.4f}, std={merger_np.std():.4f}",
              file=sys.stderr)

    if hasattr(hf_out, "pooler_output") and hf_out.pooler_output is not None:
        pooler = hf_out.pooler_output
        if isinstance(pooler, (list, tuple)):
            pooler = torch.cat(pooler, dim=0)
        pooler_np = pooler.float().cpu().numpy()
        merger_np = layer_outputs.get("merger", torch.zeros(1)).numpy()
        if pooler_np.shape == merger_np.shape:
            cos = (np.dot(pooler_np.flatten(), merger_np.flatten()) /
                   (np.linalg.norm(pooler_np) * np.linalg.norm(merger_np)
                    + 1e-8))
            print(f"[diff_vl] merger vs pooler_output cosine: {cos:.6f} "
                  "(should be < 1.0 due to reverse_indices reorder)",
                  file=sys.stderr)

    del model
    _free_gpu()

    print("[diff_vl] Debug layers: DONE", file=sys.stderr)
    return True
