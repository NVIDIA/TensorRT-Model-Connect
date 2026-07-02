# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ONNX-based vision encoder builder.

Strategy A for vision encoders: trace a HuggingFace vision model to ONNX,
then convert to a TensorRT engine via trt.OnnxParser.

Works for simple ViTs (CLIP, SigLIP, DINOv2, etc.) that can be cleanly
exported to ONNX without custom ops.

Usage from a family plugin:
    def build_vision_engine(self, model_dir, config, weights, *, verbose=False):
        return trace_hf_vision_encoder(model_dir, config, verbose=verbose)
"""

from __future__ import annotations

import io
import sys

from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()

def build_engine_from_onnx(
    onnx_bytes: bytes,
    *,
    verbose: bool = False,
) -> bytes:
    """Convert ONNX model bytes to a TRT engine plan via trt.OnnxParser.

    Args:
        onnx_bytes: Serialized ONNX model.
        verbose: Print TRT builder logs.

    Returns:
        Serialized TRT engine plan bytes.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        trt_compat.network_creation_flags(
            explicit_batch=True,
            strongly_typed=True,
        )
    )
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_bytes):
        errors = []
        for i in range(parser.num_errors):
            errors.append(str(parser.get_error(i)))
        raise RuntimeError(
            "ONNX parsing failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if verbose:
        print(f"[trtmc build] Building vision TRT engine from ONNX "
              f"({network.num_layers} layers) ...", file=sys.stderr)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT vision engine build failed")

    return bytes(plan)


def build_vision_engine_from_onnx(
    onnx_bytes: bytes,
    *,
    verbose: bool = False,
) -> bytes:
    """Backward-compatible alias for the generic ONNX -> TRT builder."""
    return build_engine_from_onnx(onnx_bytes, verbose=verbose)


def trace_hf_vision_encoder(
    model_dir: str,
    config: "ModelConfig",  # noqa: F821
    *,
    image_size: int | None = None,
    num_channels: int = 3,
    verbose: bool = False,
) -> bytes:
    """Export a HuggingFace vision encoder to ONNX and build a TRT engine.

    Loads the vision model from the HF model directory, creates a dummy input,
    exports to ONNX via torch.onnx.export, then converts to TRT.

    Args:
        model_dir: Path to HF model directory.
        config: ModelConfig (must have raw["vision_config"]).
        image_size: Override image size (default: from vision_config).
        num_channels: Number of input channels (default: 3).
        verbose: Print detailed logs.

    Returns:
        Serialized TRT engine plan bytes.
    """
    import torch

    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers is required for ONNX vision encoder tracing. "
            "Install it with: pip install transformers")

    vision_config = config.raw.get("vision_config", {})
    if image_size is None:
        image_size = vision_config.get("image_size", 224)

    if verbose:
        print(f"[trtmc build] Tracing vision encoder to ONNX "
              f"(image_size={image_size}) ...", file=sys.stderr)

    # Load the full model and extract the vision encoder
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=False)
    vision_model = None
    for attr in ("vision_model", "visual", "vision_tower", "image_encoder"):
        if hasattr(model, attr):
            vision_model = getattr(model, attr)
            break

    if vision_model is None:
        raise RuntimeError(
            "Could not find vision encoder in model. "
            "Tried: vision_model, visual, vision_tower, image_encoder")

    vision_model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, num_channels, image_size, image_size)

    # Export to ONNX
    onnx_buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            vision_model,
            dummy_input,
            onnx_buffer,
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_features": {0: "batch"},
            },
        )

    onnx_bytes = onnx_buffer.getvalue()
    if verbose:
        print(f"[trtmc build] ONNX export done ({len(onnx_bytes) / 1024:.0f} KB)",
              file=sys.stderr)

    return build_vision_engine_from_onnx(onnx_bytes, verbose=verbose)
