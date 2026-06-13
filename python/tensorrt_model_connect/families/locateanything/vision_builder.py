"""LocateAnything MoonViT vision engine builder.

The HF MoonViT implementation uses Python shape loops and complex-valued RoPE.
For TRT MC's initial LocateAnything contract we export a fixed single-image
path: 448x448 input, 14x14 patches, 32x32 patch grid, 2x2 merge, producing
256 projected image features.
"""

from __future__ import annotations

import importlib.util
import io
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from ...checkpoint_mapper import _has_tensor, _load_tensor, _open_safetensors
from ...config import ModelConfig
from ..qwen_vl.onnx_vision_builder import build_engine_from_onnx


def build_locateanything_vision_engine(
    model_dir: str,
    config: ModelConfig,
    *,
    fixed_image_size: int = 448,
    verbose: bool = False,
) -> bytes:
    """Build MoonViT + mlp1 as a TRT vision engine.

    Inputs:
        pixel_values: [1024, 3, 14, 14] float32 for the fixed 448x448 path.
        The export is traced with image_grid_hws=[[32, 32]]; TensorRT may
        constant-fold that fixed grid and omit it from the final engine inputs.

    Output:
        image_features: [256, hidden_size] float32
    """
    import torch

    model_dir_path = Path(model_dir)
    modeling_vit = _load_modeling_vit(model_dir_path)
    _patch_moonvit_for_onnx(modeling_vit)

    vision_model = _build_moonvit(modeling_vit, config)
    projector = _build_projector(config)
    _load_vision_and_projector_weights(model_dir_path, vision_model, projector)

    wrapper = _LocateAnythingVisionWrapper(vision_model, projector).eval()

    vision_config = config.raw.get("vision_config", {})
    patch_size = int(vision_config.get("patch_size", 14))
    grid_h = fixed_image_size // patch_size
    grid_w = fixed_image_size // patch_size
    num_patches = grid_h * grid_w

    dummy_pixel = torch.zeros(num_patches, 3, patch_size, patch_size, dtype=torch.float32)
    dummy_grid = torch.tensor([[grid_h, grid_w]], dtype=torch.int32)

    if verbose:
        print(
            "[trtmc build] Exporting LocateAnything MoonViT vision path "
            f"to ONNX (grid={grid_h}x{grid_w}, patches={num_patches}) ...",
            file=sys.stderr,
        )

    onnx_buffer = io.BytesIO()
    with torch.no_grad():
        export_kwargs = dict(
            opset_version=17,
            input_names=["pixel_values", "image_grid_hws"],
            output_names=["image_features"],
        )
        try:
            torch.onnx.export(
                wrapper,
                (dummy_pixel, dummy_grid),
                onnx_buffer,
                dynamo=False,
                **export_kwargs,
            )
        except TypeError:
            torch.onnx.export(
                wrapper,
                (dummy_pixel, dummy_grid),
                onnx_buffer,
                **export_kwargs,
            )

    onnx_bytes = onnx_buffer.getvalue()
    if verbose:
        print(
            f"[trtmc build] LocateAnything vision ONNX export done "
            f"({len(onnx_bytes) / (1024 * 1024):.1f} MB)",
            file=sys.stderr,
        )
    return build_engine_from_onnx(onnx_bytes, verbose=verbose)


class _LocateAnythingVisionWrapper:
    def __new__(cls, vision_model, projector):
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, vision_model, projector):
                super().__init__()
                self.vision_model = vision_model
                self.projector = projector

            def forward(self, pixel_values, image_grid_hws):
                features = self.vision_model(pixel_values, image_grid_hws)
                return self.projector(torch.cat(features, dim=0))

        return Wrapper(vision_model, projector)


def _load_modeling_vit(model_dir: Path) -> ModuleType:
    path = model_dir / "modeling_vit.py"
    if not path.exists():
        raise RuntimeError(
            "LocateAnything vision build requires modeling_vit.py from the "
            "HF repository. Re-download the model with trust_remote_code files."
        )
    module_name = f"trtmc_locateanything_modeling_vit_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import LocateAnything MoonViT module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_moonvit_for_onnx(modeling_vit: ModuleType) -> None:
    """Replace export-hostile HF helpers with equivalent real-valued ops."""

    def real_apply_rope(xq, xk, freqs_cis):
        import torch

        freqs_real = torch.view_as_real(freqs_cis)
        cos = freqs_real.select(-1, 0).unsqueeze(-2)
        sin = freqs_real.select(-1, 1).unsqueeze(-2)

        def rotate(x):
            x_pair = x.float().reshape(*x.shape[:-1], -1, 2)
            x0 = x_pair[..., 0]
            x1 = x_pair[..., 1]
            y = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
            return y.flatten(-2).to(dtype=x.dtype)

        return rotate(xq), rotate(xk)

    def full_sequence_attention(q, k, v, q_cu_seqlens=None, k_cu_seqlens=None):
        import torch

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        out = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype) @ v
        return out.transpose(0, 1).reshape(q.shape[1], -1)

    modeling_vit.apply_rope = real_apply_rope
    modeling_vit.VL_VISION_ATTENTION_FUNCTIONS["eager"] = full_sequence_attention


def _build_moonvit(modeling_vit: ModuleType, config: ModelConfig):
    vision_config = dict(config.raw.get("vision_config", {}))
    vision_config.pop("auto_map", None)
    vision_config.pop("model_type", None)
    vision_config.pop("torch_dtype", None)
    vision_config["_attn_implementation"] = "eager"
    cfg = modeling_vit.MoonViTConfig(**vision_config)
    cfg._attn_implementation = "eager"
    return modeling_vit.MoonVitPretrainedModel(cfg).eval()


def _build_projector(config: ModelConfig):
    import torch

    vit_hidden = int(config.raw.get("vision_config", {}).get("hidden_size", 1152))
    llm_hidden = int(config.hidden_size)
    return torch.nn.Sequential(
        torch.nn.LayerNorm(vit_hidden * 4),
        torch.nn.Linear(vit_hidden * 4, llm_hidden),
        torch.nn.GELU(),
        torch.nn.Linear(llm_hidden, llm_hidden),
    ).eval()


def _load_vision_and_projector_weights(model_dir: Path, vision_model, projector) -> None:
    import torch

    readers = _open_safetensors(model_dir)
    _load_module_state(readers, vision_model, ("vision_model", "model.vision_model"))
    _load_module_state(readers, projector, ("mlp1", "model.mlp1"))

    # Force float32 export even when the source checkpoint stores bf16/fp16.
    vision_model.to(dtype=torch.float32)
    projector.to(dtype=torch.float32)


def _load_module_state(readers, module, prefixes: tuple[str, ...]) -> None:
    import torch

    state = {}
    missing = []
    for key in module.state_dict().keys():
        tensor_key = _find_prefixed_tensor(readers, prefixes, key)
        if tensor_key is None:
            missing.append(key)
            continue
        state[key] = torch.from_numpy(_load_tensor(readers, tensor_key).astype(np.float32))
    if missing:
        preview = ", ".join(missing[:8])
        raise RuntimeError(
            f"LocateAnything vision checkpoint is missing {len(missing)} tensors: {preview}"
        )
    module.load_state_dict(state, strict=True)


def _find_prefixed_tensor(readers, prefixes: tuple[str, ...], key: str) -> str | None:
    for prefix in prefixes:
        candidate = f"{prefix}.{key}"
        if _has_tensor(readers, candidate):
            return candidate
    return None
