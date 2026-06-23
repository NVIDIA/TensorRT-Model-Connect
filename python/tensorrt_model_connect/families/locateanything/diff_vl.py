"""VL diff hooks owned by the LocateAnything family."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def handles_model_type(model_type: str) -> bool:
    return model_type.lower() == "locateanything"


def get_hf_vision_features(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> tuple[np.ndarray, np.ndarray]:
    """LocateAnything HF vision features: MoonViT output after mlp1."""
    import torch
    from PIL import Image

    from tensorrt_model_connect.families.locateanything.config import ModelConfig
    from tensorrt_model_connect.families.locateanything.vision_builder import (
        _build_moonvit,
        _build_projector,
        _load_modeling_vit,
        _load_vision_and_projector_weights,
    )

    model_dir = Path(model_id)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(
            repo_id=model_id,
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "chat_template.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "*.py",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        ))

    print(f"[diff_vl] Loading HF LocateAnything vision path from {model_dir} ...",
          file=sys.stderr)

    cfg = ModelConfig.from_dir(model_dir)
    modeling_vit = _load_modeling_vit(model_dir)
    vision_model = _build_moonvit(modeling_vit, cfg)
    projector = _build_projector(cfg)
    _load_vision_and_projector_weights(model_dir, vision_model, projector)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision_model.to(device=device, dtype=torch.float32).eval()
    projector.to(device=device, dtype=torch.float32).eval()

    processor_path = model_dir / "image_processing_locateanything.py"
    spec = importlib.util.spec_from_file_location(
        "trtmc_locateanything_image_processing_ref", processor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import LocateAnything image processor: {processor_path}")
    processor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(processor_module)
    vision_config = cfg.raw.get("vision_config", {})
    image_processor = processor_module.LocateAnythingImageProcessor(
        patch_size=int(vision_config.get("patch_size", 14)),
        merge_kernel_size=vision_config.get("merge_kernel_size", [2, 2]),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    image = Image.open(image_path).convert("RGB")
    image_fixed = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)
    image_inputs = image_processor(images=[image_fixed], return_tensors="pt")
    pixel_values_hf = image_inputs["pixel_values"]
    image_grid_hws = image_inputs["image_grid_hws"]

    print(f"[diff_vl] HF pixel_values: {pixel_values_hf.shape}, "
          f"image_grid_hws: {image_grid_hws.tolist()}", file=sys.stderr)

    with torch.no_grad():
        vit_features = vision_model(
            pixel_values_hf.to(device=device, dtype=torch.float32),
            image_grid_hws.to(device=device),
        )
        hf_features = projector(torch.cat(vit_features, dim=0))

    hf_features_np = hf_features.float().cpu().numpy()
    print(f"[diff_vl] HF features: shape={hf_features_np.shape}, "
          f"mean={hf_features_np.mean():.6f}, std={hf_features_np.std():.6f}",
          file=sys.stderr)

    pixel_values_np = pixel_values_hf.float().cpu().numpy()
    del vision_model, projector
    _free_gpu()
    return hf_features_np, pixel_values_np
