# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned export of the complete official Qwen3-Omni Code2Wav graph."""

from __future__ import annotations

import gc
import io
import sys

import numpy as np
import tensorrt as trt

from .checkpoint_mapper import WeightDict


def build_code2wav_engine(
    weights: WeightDict,
    code2wav_cfg: dict,
    verbose: bool = False,
) -> bytes | None:
    """Build the official Code2Wav graph: RVQ codes -> speech waveform.

    The released checkpoint uses an eight-layer sliding-attention
    pre-transformer, two ConvNeXt upsamplers, and four causal HiFi-GAN-style
    decoder blocks. Exporting the upstream model-owned module to ONNX
    preserves all 230 checkpoint tensors for the complete static decoder.
    """
    if not code2wav_cfg.get("available"):
        return None

    import torch
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeCode2WavConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeCode2Wav,
    )

    model_config = Qwen3OmniMoeCode2WavConfig(**code2wav_cfg["config"])
    model = Qwen3OmniMoeCode2Wav(model_config)

    state = {}
    for name in model.state_dict():
        key = f"code2wav.{name}"
        value = weights.get(key)
        if value is None:
            raise RuntimeError(f"Qwen3-Omni Code2Wav checkpoint tensor is missing: {key}")
        state[name] = torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
    model.load_state_dict(state, strict=True)
    model.eval()

    class _StaticCode2Wav(torch.nn.Module):
        """Static, export-safe equivalent of the official forward method."""

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, codes):
            codes = codes.to(torch.int64)
            hidden = self.module.code_embedding(codes + self.module.code_offset).mean(1)

            # Supplying the official 72-frame sliding causal mask explicitly
            # avoids the torch.diff-based dynamic mask helper, which is not
            # representable in ONNX opset 17. The engine shape is static.
            length = hidden.shape[1]
            indices = torch.arange(length, device=hidden.device)
            row = indices[:, None]
            col = indices[None, :]
            allowed = (col <= row) & (col > row - self.module.config.sliding_window)
            mask = torch.where(
                allowed,
                torch.zeros((), dtype=hidden.dtype, device=hidden.device),
                torch.full(
                    (),
                    torch.finfo(hidden.dtype).min,
                    dtype=hidden.dtype,
                    device=hidden.device,
                ),
            )[None, None]
            hidden = self.module.pre_transformer(
                inputs_embeds=hidden,
                attention_mask={
                    "sliding_attention": mask,
                    "full_attention": mask,
                },
            ).last_hidden_state
            hidden = hidden.permute(0, 2, 1)
            for blocks in self.module.upsample:
                for block in blocks:
                    hidden = block(hidden)
            for block in self.module.decoder:
                hidden = block(hidden)
            return hidden.clamp(min=-1, max=1)

    max_frames = int(code2wav_cfg["max_frames"])
    num_quantizers = int(model_config.num_quantizers)
    export_module = _StaticCode2Wav(model).eval()
    dummy_codes = torch.zeros((1, num_quantizers, max_frames), dtype=torch.int32)
    onnx_buffer = io.BytesIO()
    if verbose:
        print(
            "[trtmc build]   Exporting official Qwen3-Omni Code2Wav "
            f"({num_quantizers} codebooks x {max_frames} frames) ...",
            file=sys.stderr,
        )
    with torch.inference_mode():
        torch.onnx.export(
            export_module,
            dummy_codes,
            onnx_buffer,
            opset_version=17,
            input_names=["codec_tokens"],
            output_names=["waveform"],
            dynamo=False,
        )

    del export_module, model, state, dummy_codes
    gc.collect()

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_buffer.getvalue()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Qwen3-Omni Code2Wav ONNX parsing failed:\n" + "\n".join(errors))

    expected_samples = max_frames * int(code2wav_cfg["upsample_factor"]) - int(
        code2wav_cfg["output_delay"]
    )
    if tuple(network.get_input(0).shape) != (1, num_quantizers, max_frames) or tuple(
        network.get_output(0).shape
    ) != (1, 1, expected_samples):
        raise RuntimeError(
            "Qwen3-Omni Code2Wav ONNX shape contract mismatch: "
            f"input={tuple(network.get_input(0).shape)}, "
            f"output={tuple(network.get_output(0).shape)}"
        )

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if verbose:
        print(
            "[trtmc build]   Building complete Qwen3-Omni Code2Wav "
            f"TensorRT engine ({expected_samples} samples) ...",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen3-Omni Code2Wav build failed")
    return bytes(plan)


__all__ = ["build_code2wav_engine"]
