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
        raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if verbose:
        print(
            f"[trtmc build] Building vision TRT engine from ONNX ({network.num_layers} layers) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT vision engine build failed")

    return bytes(plan)
