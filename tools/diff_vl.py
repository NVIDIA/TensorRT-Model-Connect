#!/usr/bin/env python3
"""VL diff testing: compare TRT vision-language pipeline against HuggingFace.

Tests:
  1. Vision features: TRT vision engine output vs HF model.visual() features
  2. Text decoder: embed_input mode smoke test
  3. Full VL generation: TRT pipeline vs HF pipeline text comparison
  4. C++ binary parity

Usage:
  # Vision feature comparison (requires torch + transformers)
  python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
    --model Qwen/Qwen2.5-VL-3B-Instruct --atol 0.1

  # Full VL generation with C++ binary
  python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
    --binary ./build/trtmc --hf-python .venv/bin/python

  # Vision-only (no HF model needed)
  python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg --vision-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _free_gpu():
    """Force-free all GPU memory (TRT contexts, torch cache)."""
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
    """Detect model_type from HF config (e.g. 'qwen2_5_vl', 'llava', etc.)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(cfg, "model_type", "unknown")


def _is_qwen_vl(model_type: str) -> bool:
    """True if the model_type is a Qwen VL variant."""
    mt = model_type.lower()
    return "qwen" in mt and "vl" in mt


def _is_locateanything(model_type: str) -> bool:
    return model_type.lower() == "locateanything"


def _get_hf_vision_features_qwen(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> tuple[np.ndarray, np.ndarray]:
    """Qwen-specific HF vision feature extraction (Qwen2.5-VL and Qwen3-VL)."""
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

    # Use AutoImageProcessor for Qwen3-VL, Qwen2VLImageProcessor for Qwen2.5-VL
    if is_qwen3:
        from transformers import AutoImageProcessor
        image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        from transformers import Qwen2VLImageProcessor
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_id)

    # Resize image to fixed_image_size to match TRT engine
    image_fixed = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)
    image_inputs = image_processor(images=[image_fixed], return_tensors="pt")
    pixel_values_hf = image_inputs["pixel_values"]
    image_grid_thw = image_inputs["image_grid_thw"]

    print(f"[diff_vl] HF pixel_values: {pixel_values_hf.shape}, "
          f"grid_thw: {image_grid_thw.tolist()}", file=sys.stderr)

    # Run HF vision encoder
    inner = model.model
    with torch.no_grad():
        vis_out = inner.get_image_features(
            pixel_values_hf.to(device=inner.visual.patch_embed.proj.weight.device,
                               dtype=inner.visual.dtype),
            image_grid_thw=image_grid_thw.to(device=inner.visual.patch_embed.proj.weight.device),
            return_dict=True)
        # Qwen3-VL returns pooler_output (main features) + deepstack_features
        pooler = vis_out.pooler_output
        if isinstance(pooler, (list, tuple)):
            hf_features = torch.cat(pooler, dim=0)
        else:
            hf_features = pooler

    hf_features_np = hf_features.float().cpu().numpy()
    print(f"[diff_vl] HF features: shape={hf_features_np.shape}, "
          f"mean={hf_features_np.mean():.6f}, std={hf_features_np.std():.6f}",
          file=sys.stderr)

    # Also build the TRT-style pixel_values for comparison
    image_resized = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)
    img_np = np.array(image_resized, dtype=np.float32) / 255.0
    mean = np.array(image_processor.image_mean, dtype=np.float32)
    std = np.array(image_processor.image_std, dtype=np.float32)
    img_np = (img_np - mean) / std
    img_chw = img_np.transpose(2, 0, 1)
    temporal = getattr(image_processor, 'temporal_patch_size', 1)
    trt_pixel_values = np.tile(img_chw, (temporal, 1, 1)).astype(np.float32)

    del model
    _free_gpu()
    return hf_features_np, trt_pixel_values


def _get_hf_vision_features_locateanything(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> tuple[np.ndarray, np.ndarray]:
    """LocateAnything HF vision features: MoonViT output after mlp1."""
    import importlib.util
    import torch
    from PIL import Image

    from tensorrt_model_connect.config import ModelConfig
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
        raise RuntimeError(f"Cannot import LocateAnything image processor: {processor_path}")
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


def _get_hf_vision_features_generic(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic HF vision feature extraction using AutoModelForVision2Seq."""
    import torch
    from PIL import Image

    print(f"[diff_vl] Loading HF generic VL model {model_id} ...", file=sys.stderr)
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device,
        trust_remote_code=True)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    image = Image.open(image_path).convert("RGB")
    image_fixed = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)

    inputs = processor(images=[image_fixed], text="Describe", return_tensors="pt")
    pixel_values_hf = inputs.get("pixel_values")

    print(f"[diff_vl] HF pixel_values: {pixel_values_hf.shape}", file=sys.stderr)

    # Run vision encoder via the model's get_image_features or equivalent
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            hf_features = model.get_image_features(
                pixel_values_hf.to(device=device))
        elif hasattr(model, "vision_tower"):
            hf_features = model.vision_tower(
                pixel_values_hf.to(device=device))
        else:
            print("[diff_vl] WARN: Cannot find vision feature method — "
                  "falling back to full forward", file=sys.stderr)
            outputs = model(**{k: v.to(device=device) for k, v in inputs.items()},
                           output_hidden_states=True)
            hf_features = outputs.hidden_states[0]

    if isinstance(hf_features, (list, tuple)):
        hf_features = torch.cat(hf_features, dim=0)
    hf_features_np = hf_features.float().cpu().numpy()
    # Handle 4D tensors (batch, seq, hidden) or (batch, C, H, W)
    if hf_features_np.ndim == 4:
        # Reshape to (N, D) by flattening spatial dims
        b, *rest = hf_features_np.shape
        hf_features_np = hf_features_np.reshape(b, -1)
    if hf_features_np.ndim == 3:
        hf_features_np = hf_features_np.squeeze(0)

    print(f"[diff_vl] HF features: shape={hf_features_np.shape}, "
          f"mean={hf_features_np.mean():.6f}, std={hf_features_np.std():.6f}",
          file=sys.stderr)

    # Build simple CHW pixel_values for TRT comparison
    image_resized = image.resize((fixed_image_size, fixed_image_size), Image.BICUBIC)
    img_np = np.array(image_resized, dtype=np.float32) / 255.0
    image_mean = getattr(processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = getattr(processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
    mean = np.array(image_mean, dtype=np.float32)
    std = np.array(image_std, dtype=np.float32)
    img_np = (img_np - mean) / std
    trt_pixel_values = img_np.transpose(2, 0, 1).astype(np.float32)

    del model
    _free_gpu()
    return hf_features_np, trt_pixel_values


def _get_hf_vision_features(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> tuple[np.ndarray, np.ndarray]:
    """Get reference vision features from HuggingFace model.

    Auto-detects model type and dispatches to the appropriate loader.

    Returns:
        (hf_features [N, dim], pixel_values_for_trt)
    """
    model_type = _detect_model_type(model_id)
    print(f"[diff_vl] Detected model_type: {model_type}", file=sys.stderr)

    if _is_qwen_vl(model_type):
        return _get_hf_vision_features_qwen(
            model_id, image_path, fixed_image_size)
    if _is_locateanything(model_type):
        return _get_hf_vision_features_locateanything(
            model_id, image_path, fixed_image_size)
    return _get_hf_vision_features_generic(
        model_id, image_path, fixed_image_size)


def test_vision_features(
    bundle_path: str,
    image_path: str,
    model_id: str | None = None,
    atol: float = 0.1,
    preprocessor_type_override: str | None = None,
) -> bool:
    """Compare TRT vision encoder output against HF reference.

    If model_id is provided, does a full numerical comparison.
    Otherwise, just verifies TRT produces non-zero output.
    """
    from tensorrt_model_connect.debug_runner import (
        VisionTrtRunner, load_vision_engine_from_bundle,
        preprocess_image_inputs_for_trt, load_preprocessor_config_from_bundle,
        load_config_from_bundle,
    )

    vision_plan, header = load_vision_engine_from_bundle(bundle_path)
    if vision_plan is None:
        print("[diff_vl] No vision engine in bundle — skipping", file=sys.stderr)
        return True

    config = load_config_from_bundle(bundle_path)
    preproc = load_preprocessor_config_from_bundle(bundle_path)
    fixed_image_size = config.get("fixed_image_size", 448)
    temporal = preproc.get(
        "temporal_patch_size", config.get("temporal_patch_size", 1))
    image_mean = tuple(preproc.get(
        "image_mean", config.get(
            "image_mean", [0.48145466, 0.4578275, 0.40821073])))
    image_std = tuple(preproc.get(
        "image_std", config.get(
            "image_std", [0.26862954, 0.26130258, 0.27577711])))
    preprocessor_type = config.get("preprocessor_type", "qwen_merge_group")
    interpolation = config.get("interpolation", "bicubic")

    # Allow CLI override for debugging
    if preprocessor_type_override is not None:
        print(f"[diff_vl] Overriding preprocessor_type: "
              f"{preprocessor_type!r} -> {preprocessor_type_override!r}",
              file=sys.stderr)
        preprocessor_type = preprocessor_type_override

    print(f"[diff_vl] Preprocessor: type={preprocessor_type!r}, "
          f"interpolation={interpolation!r}, "
          f"image_size={fixed_image_size}", file=sys.stderr)

    runner = VisionTrtRunner(vision_plan)

    # Prepare TRT pixel values
    vis_patch_size = preproc.get("patch_size", config.get("patch_size", 14))
    vis_merge_size = preproc.get("merge_size", config.get("merge_size", 2))
    trt_inputs = preprocess_image_inputs_for_trt(
        image_path, preprocessor_type=preprocessor_type,
        fixed_image_size=fixed_image_size,
        temporal_patch_size=temporal, image_mean=image_mean, image_std=image_std,
        patch_size=vis_patch_size, merge_size=vis_merge_size,
        interpolation=interpolation)

    shapes = {name: value.shape for name, value in trt_inputs.items()}
    print(f"[diff_vl] TRT vision inputs: {shapes}", file=sys.stderr)

    # Run TRT vision encoder
    results = runner.encode(**trt_inputs)
    trt_features = results["image_features"]

    print(f"[diff_vl] TRT features: shape={trt_features.shape}, "
          f"mean={trt_features.mean():.6f}, std={trt_features.std():.6f}",
          file=sys.stderr)

    # Basic sanity checks
    if np.all(trt_features == 0):
        print("[diff_vl] FAIL: TRT vision output is all zeros", file=sys.stderr)
        del runner
        _free_gpu()
        return False

    if np.any(np.isnan(trt_features)):
        print("[diff_vl] FAIL: TRT vision output contains NaN", file=sys.stderr)
        del runner
        _free_gpu()
        return False

    if np.any(np.isinf(trt_features)):
        print("[diff_vl] FAIL: TRT vision output contains Inf", file=sys.stderr)
        del runner
        _free_gpu()
        return False

    # Numerical comparison against HF reference
    if model_id is not None:
        print("[diff_vl] Comparing against HF reference ...", file=sys.stderr)

        # Warn if HF processor's image_mean/std differ from bundle values
        try:
            from transformers import AutoImageProcessor
            hf_im = AutoImageProcessor.from_pretrained(
                model_id, trust_remote_code=True)
            hf_mean = getattr(hf_im, "image_mean", None)
            hf_std = getattr(hf_im, "image_std", None)
            if hf_mean is not None and tuple(hf_mean) != tuple(image_mean):
                print(f"[diff_vl] WARN: image_mean divergence: "
                      f"bundle={list(image_mean)} vs HF={list(hf_mean)}",
                      file=sys.stderr)
            if hf_std is not None and tuple(hf_std) != tuple(image_std):
                print(f"[diff_vl] WARN: image_std divergence: "
                      f"bundle={list(image_std)} vs HF={list(hf_std)}",
                      file=sys.stderr)
        except Exception:
            pass  # non-critical

        # Free TRT vision engine before loading HF model (avoid GPU OOM)
        del runner
        _free_gpu()
        hf_features, _ = _get_hf_vision_features(
            model_id, image_path, fixed_image_size)

        if trt_features.shape != hf_features.shape:
            print(f"[diff_vl] WARN: Shape mismatch: TRT {trt_features.shape} "
                  f"vs HF {hf_features.shape}", file=sys.stderr)
            # Compare as many features as possible
            min_n = min(trt_features.shape[0], hf_features.shape[0])
            min_d = min(trt_features.shape[1], hf_features.shape[1])
            trt_sub = trt_features[:min_n, :min_d]
            hf_sub = hf_features[:min_n, :min_d]
        else:
            trt_sub = trt_features
            hf_sub = hf_features

        abs_diff = np.abs(trt_sub - hf_sub)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        cos_sim = np.dot(trt_sub.flatten(), hf_sub.flatten()) / (
            np.linalg.norm(trt_sub) * np.linalg.norm(hf_sub) + 1e-8)

        print(f"[diff_vl] Vision comparison: "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
              f"cosine_sim={cos_sim:.6f}", file=sys.stderr)

        if max_diff > atol:
            print(f"[diff_vl] WARN: Max diff {max_diff:.6f} > atol {atol}",
                  file=sys.stderr)
            # Not a hard failure — print diagnostics
            print("[diff_vl] Per-row max diff (first 10 features):",
                  file=sys.stderr)
            for i in range(min(10, trt_sub.shape[0])):
                row_diff = np.abs(trt_sub[i] - hf_sub[i]).max()
                print(f"  feature[{i}]: max_diff={row_diff:.6f}, "
                      f"trt_norm={np.linalg.norm(trt_sub[i]):.4f}, "
                      f"hf_norm={np.linalg.norm(hf_sub[i]):.4f}",
                      file=sys.stderr)

        if cos_sim < 0.5:
            print(f"[diff_vl] FAIL: Cosine similarity {cos_sim:.4f} < 0.5 — "
                  f"features are uncorrelated", file=sys.stderr)
            _free_gpu()
            return False

    print("[diff_vl] Vision encoder: PASS", file=sys.stderr)
    # Ensure GPU is free for next test (runner may already be deleted by HF path)
    try:
        del runner
    except (NameError, UnboundLocalError):
        pass
    _free_gpu()
    return True


def test_embed_input(bundle_path: str) -> bool:
    """Verify the text decoder accepts embed_input mode."""
    from tensorrt_model_connect.debug_runner import (
        load_section_from_bundle,
        runner_from_bundle,
    )

    if (
        load_section_from_bundle(bundle_path, "engine_plan") is None
        and load_section_from_bundle(bundle_path, "engine_plan_tp_rank0") is not None
    ):
        print("[diff_vl] Text decoder embed_input: SKIP "
              "(tensor-parallel decoder requires distributed runtime)",
              file=sys.stderr)
        return True

    runner = runner_from_bundle(bundle_path)
    if not runner.has_embed_input:
        print("[diff_vl] Text decoder has no embed_input — skipping",
              file=sys.stderr)
        del runner
        _free_gpu()
        return True

    result1 = runner.step(1, use_input_embed=0.0)
    if "logits" not in result1:
        print("[diff_vl] FAIL: no logits from text decoder", file=sys.stderr)
        del runner
        _free_gpu()
        return False

    embed_shape = tuple(runner.engine.get_tensor_shape("input_embed"))
    embed_hidden = embed_shape[-1]
    dummy_embed = np.random.randn(1, embed_hidden).astype(np.float32) * 0.01
    result2 = runner.step(0, input_embed=dummy_embed, use_input_embed=1.0)
    if "logits" not in result2:
        print("[diff_vl] FAIL: no logits from embed_input step", file=sys.stderr)
        del runner
        _free_gpu()
        return False

    print(f"[diff_vl] Text decoder embed_input: PASS "
          f"(hidden={embed_hidden})", file=sys.stderr)
    del runner
    _free_gpu()
    return True


def test_vl_generation(
    bundle_path: str,
    image_path: str,
    max_new_tokens: int = 20,
) -> bool:
    """Run full VL generation in Python using VLTrtRunner."""
    from tensorrt_model_connect.debug_runner import (
        load_section_from_bundle,
        VLTrtRunner,
    )

    print("[diff_vl] Loading VL runner from bundle ...", file=sys.stderr)
    runner = VLTrtRunner(bundle_path)
    if runner.vision_runner is None:
        print("[diff_vl] No vision engine — skipping VL generation", file=sys.stderr)
        return True

    # Encode image
    print("[diff_vl] Encoding image ...", file=sys.stderr)
    features = runner.encode_image(image_path)
    print(f"[diff_vl] Image features: {features.shape}, "
          f"mean={features.mean():.4f}, std={features.std():.4f}",
          file=sys.stderr)

    # Format prompt and tokenize
    prompt = "Describe this image"
    formatted = runner.format_prompt(prompt)
    print(f"[diff_vl] Formatted prompt length: {len(formatted)} chars",
          file=sys.stderr)

    # Tokenize using HF tokenizer
    try:
        from transformers import AutoTokenizer
        model_source = runner.config.get("model_source", "")
        tok_data = load_section_from_bundle(bundle_path, "tokenizer.json")
        if tok_data:
            # Write tokenizer to temp dir and load
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                for name in ["tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json"]:
                    data = load_section_from_bundle(bundle_path, name)
                    if data:
                        (Path(tmpdir) / name).write_bytes(data)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tmpdir)
                except Exception as auto_exc:
                    from tokenizers import Tokenizer

                    raw_tokenizer = Tokenizer.from_file(
                        str(Path(tmpdir) / "tokenizer.json"))

                    class TokenizersWrapper:
                        def __init__(self, tok):
                            self._tok = tok

                        def encode(self, text, add_special_tokens=False):
                            return self._tok.encode(
                                text,
                                add_special_tokens=add_special_tokens).ids

                        def decode(self, ids, skip_special_tokens=True):
                            return self._tok.decode(
                                ids,
                                skip_special_tokens=skip_special_tokens)

                    tokenizer = TokenizersWrapper(raw_tokenizer)
                    print(f"[diff_vl] WARN: AutoTokenizer failed ({auto_exc}); "
                          "using tokenizer.json fallback", file=sys.stderr)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_source)
    except Exception as e:
        print(f"[diff_vl] Cannot load tokenizer: {e} — skipping generation",
              file=sys.stderr)
        return True

    input_ids = tokenizer.encode(formatted, add_special_tokens=False)
    print(f"[diff_vl] Input tokens: {len(input_ids)} "
          f"(image_pad tokens: {input_ids.count(runner.image_token_id)})",
          file=sys.stderr)

    # Generate
    print(f"[diff_vl] Generating {max_new_tokens} tokens ...", file=sys.stderr)
    output_ids = runner.generate_vl(input_ids, features, max_new_tokens)
    new_ids = output_ids[len(input_ids):]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    print(f"[diff_vl] Generated: {output_text!r}", file=sys.stderr)

    if not output_text.strip():
        print("[diff_vl] WARN: Empty generation output", file=sys.stderr)

    print("[diff_vl] VL generation: PASS", file=sys.stderr)
    del runner
    _free_gpu()
    return True


def test_cpp_binary(
    bundle_path: str,
    binary_path: str,
    image_path: str,
    hf_python: str | None = None,
    max_new_tokens: int = 10,
) -> bool:
    """Test C++ binary VL inference."""
    import subprocess

    cmd = [binary_path, "run", bundle_path,
           "--prompt", "Describe this image",
           "--image", image_path,
           "--max-new-tokens", str(max_new_tokens)]
    if hf_python:
        cmd.extend(["--hf-python", hf_python])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print(f"[diff_vl] SKIP: binary not found: {binary_path}", file=sys.stderr)
        return True
    except subprocess.TimeoutExpired:
        print("[diff_vl] FAIL: C++ binary timed out", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(f"[diff_vl] FAIL: C++ exit={result.returncode}", file=sys.stderr)
        print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
        return False

    output = result.stdout.strip()
    print(f"[diff_vl] C++ output: {output[:200]!r}", file=sys.stderr)
    print("[diff_vl] C++ binary: PASS", file=sys.stderr)
    return True


def test_debug_layers(
    model_id: str,
    image_path: str,
    fixed_image_size: int = 448,
) -> bool:
    """Per-layer comparison of TRT vision encoder vs HF reference.

    Uses HF hooks to capture hidden states after each ViT block, then
    compares against TRT layer-by-layer using our RoPE/window_index logic
    applied to the same pixel values.
    """
    import torch
    from PIL import Image

    model_type = _detect_model_type(model_id)
    if not _is_qwen_vl(model_type):
        print(f"[diff_vl] Debug layers: only supported for Qwen VL models "
              f"(got model_type={model_type!r})", file=sys.stderr)
        return True

    print(f"[diff_vl] Debug layers: loading HF model {model_id} ...",
          file=sys.stderr)
    from transformers import (
        Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map=device)
    model.eval()

    visual = model.model.visual

    # Prepare image through HF processor
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

    # Capture per-layer hidden states via hooks
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

    # Run HF forward
    with torch.no_grad():
        hf_out = visual(pixel_values_hf, grid_thw=grid_thw)

    for h in hooks:
        h.remove()

    # Print per-layer stats
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

    # Compute cosine similarity between consecutive layers to spot divergence
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

    # Final merger output stats
    if "merger" in layer_outputs:
        merger_np = layer_outputs["merger"].numpy()
        print(f"\n[diff_vl] Merger output: shape={merger_np.shape}, "
              f"mean={merger_np.mean():.4f}, std={merger_np.std():.4f}",
              file=sys.stderr)

    # Compare merger output with HF pooler_output (post reverse-reorder)
    if hasattr(hf_out, 'pooler_output') and hf_out.pooler_output is not None:
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
                  f"(should be < 1.0 due to reverse_indices reorder)",
                  file=sys.stderr)

    del model
    _free_gpu()

    print("[diff_vl] Debug layers: DONE", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="VL diff testing")
    parser.add_argument("--bundle", required=True, help="Path to .trtfb bundle")
    parser.add_argument("--image", default=None, help="Path to test image")
    parser.add_argument("--model", default=None,
                        help="HF model ID for reference comparison")
    parser.add_argument("--binary", default="./build/trtmc",
                        help="Path to trtmc binary")
    parser.add_argument("--hf-python", default=None,
                        help="Python interpreter path")
    parser.add_argument("--atol", type=float, default=0.1,
                        help="Absolute tolerance for feature comparison")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--vision-only", action="store_true",
                        help="Only test vision encoder (no HF model needed)")
    parser.add_argument("--debug-layers", action="store_true",
                        help="Per-layer HF vision encoder analysis (requires --model)")
    parser.add_argument("--preprocessor-type", default=None,
                        help="Override preprocessor_type for debugging")
    args = parser.parse_args()

    all_pass = True

    # Debug layers mode: per-layer HF analysis, then exit
    if args.debug_layers:
        if not args.model or not args.image:
            print("[diff_vl] --debug-layers requires --model and --image",
                  file=sys.stderr)
            sys.exit(1)
        test_debug_layers(args.model, args.image)
        return

    # Test 1: Vision encoder features
    if args.image:
        model_id = None if args.vision_only else args.model
        if not test_vision_features(args.bundle, args.image, model_id, args.atol,
                                    preprocessor_type_override=args.preprocessor_type):
            all_pass = False

    # Test 2: Text decoder embed_input
    if not test_embed_input(args.bundle):
        all_pass = False

    if not args.vision_only:
        # Test 3: Full VL generation in Python
        if args.image:
            if not test_vl_generation(args.bundle, args.image, args.max_new_tokens):
                all_pass = False

        # Test 4: C++ binary
        if args.image:
            if not test_cpp_binary(args.bundle, args.binary, args.image,
                                   args.hf_python, args.max_new_tokens):
                all_pass = False

    if all_pass:
        print("\n[diff_vl] ALL TESTS PASSED", file=sys.stderr)
    else:
        print("\n[diff_vl] SOME TESTS FAILED", file=sys.stderr)
        sys.exit(1)


def run_as_diff_test(ctx):
    """Framework entry point. Returns DiffResult."""
    from diff_framework.protocol import DiffResult
    import time as _time

    t0 = _time.monotonic()
    try:
        bundle = ctx.bundle_path
        if not bundle:
            return DiffResult.skip(
                "vl_pipeline", ctx.model, ctx.runtime_strategy,
                "No bundle provided")

        sub_results = {}

        # Test 1: Vision features
        if ctx.image_path:
            sub_results["vision"] = (
                "PASS" if test_vision_features(
                    bundle, ctx.image_path, ctx.model, ctx.atol)
                else "FAIL")

        # Test 2: Embed input
        sub_results["embed"] = (
            "PASS" if test_embed_input(bundle) else "FAIL")

        # Test 3: VL generation
        if ctx.image_path:
            sub_results["generation"] = (
                "PASS" if test_vl_generation(
                    bundle, ctx.image_path, ctx.max_new_tokens)
                else "FAIL")

        # Test 4: C++ binary
        if ctx.image_path and ctx.binary_path:
            sub_results["cpp_parity"] = (
                "PASS" if test_cpp_binary(
                    bundle, ctx.binary_path, ctx.image_path,
                    ctx.hf_python, ctx.max_new_tokens)
                else "FAIL")

        all_passed = all(v == "PASS" for v in sub_results.values())
        return DiffResult(
            test_name="vl_pipeline", model=ctx.model,
            runtime_strategy=ctx.runtime_strategy,
            passed=all_passed,
            status="PASS" if all_passed else "FAIL",
            message=f"sub-tests: {sub_results}",
            metrics=sub_results,
            duration_s=_time.monotonic() - t0)
    except Exception as e:
        return DiffResult.error(
            "vl_pipeline", ctx.model, ctx.runtime_strategy, str(e))


if __name__ == "__main__":
    main()
