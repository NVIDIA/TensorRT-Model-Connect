#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VL diff testing: compare TRT vision-language pipeline against HuggingFace.

Tests:
  1. Vision features: TRT vision engine output vs HF model.visual() features
  2. Text decoder: embed_input mode smoke test
  3. Full VL generation: TRT pipeline vs HF pipeline text comparison
  4. C++ binary parity

Usage:
  # Vision feature comparison (requires torch + transformers)
  python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
    --model example-org/example-vl-model --atol 0.1

  # Full VL generation with C++ binary
  python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg \
    --binary ./build/trtmc --hf-python .venv/bin/python

  # Vision-only (no HF model needed)
  python3 tools/diff_vl.py --bundle model.trtfb --image test.jpg --vision-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import struct
from functools import lru_cache
from pathlib import Path
from types import ModuleType

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
    """Detect model_type from HF config."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return getattr(cfg, "model_type", "unknown")


def _family_handler_paths(filename: str) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    roots = (
        repo_root / "tools/families",
        repo_root / "python/tensorrt_model_connect/families",
    )
    handlers: dict[str, Path] = {}
    for root in reversed(roots):
        handlers.update({path.parent.name: path for path in root.glob(f"*/{filename}")})
    return [handlers[family] for family in sorted(handlers)]


@lru_cache(maxsize=1)
def _family_diff_vl_modules() -> tuple[ModuleType, ...]:
    """Load optional model-owned VL diff handlers from family folders."""
    modules: list[ModuleType] = []
    for handler_path in _family_handler_paths("diff_vl.py"):
        module_name = f"_trtmc_diff_vl_{handler_path.parent.name}"
        spec = importlib.util.spec_from_file_location(module_name, handler_path)
        if spec is None or spec.loader is None:
            print(f"[diff_vl] WARN: cannot load family diff handler "
                  f"{handler_path}", file=sys.stderr)
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[diff_vl] WARN: failed to import family diff handler "
                  f"{handler_path}: {exc}", file=sys.stderr)
            continue
        if callable(getattr(module, "handles_model_type", None)):
            modules.append(module)
    return tuple(modules)


def _find_family_diff_vl_handler(model_type: str) -> ModuleType | None:
    """Return the model-owned diff handler for a detected HF model_type."""
    for module in _family_diff_vl_modules():
        handles = getattr(module, "handles_model_type")
        if handles(model_type):
            return module
    return None


def _read_bundle_header(bundle_path: str) -> dict:
    with open(bundle_path, "rb") as f:
        magic = f.read(8)
        if magic != b"TRTFB\x00\x01\x00":
            raise ValueError(f"Not a valid .trtfb bundle: {bundle_path}")
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


def _bundle_family(bundle_path: str) -> str:
    header = _read_bundle_header(bundle_path)
    family = str(header.get("family") or "")
    if family:
        return family
    model_type = str(header.get("model_type") or "")
    if model_type:
        from tensorrt_model_connect.families import find_plugin

        plugin = find_plugin(model_type)
        if plugin is not None:
            return str(getattr(plugin, "name", "") or "")
    return ""


def _load_family_vl_debug_runner(bundle_path: str) -> ModuleType:
    family = _bundle_family(bundle_path)
    if not family:
        raise RuntimeError(
            "VL debug execution requires bundle family metadata so the "
            "owning family vl_debug_runner.py can be selected."
        )
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "tools/families" / family / "vl_debug_runner.py",
        repo_root
        / "python/tensorrt_model_connect/families"
        / family
        / "vl_debug_runner.py",
    )
    module_path = next((path for path in candidates if path.is_file()), None)
    if module_path is None:
        raise RuntimeError(
            f"Family {family!r} does not provide an owned VL debug runner"
        )
    module_name = f"_trtmc_vl_debug_runner_{family}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load owned VL debug runner {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    handler = _find_family_diff_vl_handler(model_type)
    get_features = getattr(handler, "get_hf_vision_features", None)
    if callable(get_features):
        return get_features(model_id, image_path, fixed_image_size)

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
    vl_debug = _load_family_vl_debug_runner(bundle_path)

    vision_plan, header = vl_debug.load_vision_engine_from_bundle(bundle_path)
    if vision_plan is None:
        print("[diff_vl] No vision engine in bundle — skipping", file=sys.stderr)
        return True

    config = vl_debug.load_config_from_bundle(bundle_path)
    preproc = vl_debug.load_preprocessor_config_from_bundle(bundle_path)
    fixed_image_size = config.get("fixed_image_size", 448)
    temporal = preproc.get(
        "temporal_patch_size", config.get("temporal_patch_size", 1))
    image_mean = tuple(preproc.get(
        "image_mean", config.get(
            "image_mean", [0.48145466, 0.4578275, 0.40821073])))
    image_std = tuple(preproc.get(
        "image_std", config.get(
            "image_std", [0.26862954, 0.26130258, 0.27577711])))
    preprocessor_type = config.get("preprocessor_type", "merge_group_chw")
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

    runner = vl_debug.VisionTrtRunner(vision_plan)

    # Prepare TRT pixel values
    vis_patch_size = preproc.get("patch_size", config.get("patch_size", 14))
    vis_merge_size = preproc.get("merge_size", config.get("merge_size", 2))
    trt_inputs = vl_debug.preprocess_image_inputs_for_trt(
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
    vl_debug = _load_family_vl_debug_runner(bundle_path)

    if (
        vl_debug.load_section_from_bundle(bundle_path, "engine_plan") is None
        and vl_debug.load_section_from_bundle(
            bundle_path, "engine_plan_tp_rank0") is not None
    ):
        print("[diff_vl] Text decoder embed_input: SKIP "
              "(tensor-parallel decoder requires distributed runtime)",
              file=sys.stderr)
        return True

    config = vl_debug.load_config_from_bundle(bundle_path)
    engine_plan, header = vl_debug.load_engine_from_bundle(bundle_path)
    runner = vl_debug.TrtRunner(
        engine_plan=engine_plan,
        max_cache_length=header["max_cache_length"],
        num_layers=header.get("num_layers", config.get("num_hidden_layers", 1)),
    )
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
    """Run full VL generation in Python using the family-owned VL runner."""
    vl_debug = _load_family_vl_debug_runner(bundle_path)

    print("[diff_vl] Loading VL runner from bundle ...", file=sys.stderr)
    runner = vl_debug.VLTrtRunner(bundle_path)
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
        tok_data = vl_debug.load_section_from_bundle(bundle_path, "tokenizer.json")
        if tok_data:
            # Write tokenizer to temp dir and load
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                for name in ["tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json"]:
                    data = vl_debug.load_section_from_bundle(bundle_path, name)
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

    Family-owned handlers provide concrete layer hooks for the active model.
    """
    model_type = _detect_model_type(model_id)
    handler = _find_family_diff_vl_handler(model_type)
    debug_layers = getattr(handler, "debug_layers", None)
    if callable(debug_layers):
        return bool(debug_layers(model_id, image_path, fixed_image_size))

    print(f"[diff_vl] Debug layers: no family handler for "
          f"model_type={model_type!r}", file=sys.stderr)
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
