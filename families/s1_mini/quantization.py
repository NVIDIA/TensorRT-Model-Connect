# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned FP8 calibration and TensorRT Q/DQ graph context."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorrt as trt


CALIBRATION_SAMPLE_COUNT = 64
FP8_EXCLUDE_PATTERNS = (
    "embedding",
    "final_norm",
    "w_out",
    "lm_head",
    "*.input_norm",
    "*.post_attn_norm",
    "*_norm*",
    "layer.*.w_q",
    "layer.*.w_k",
    "layer.*.w_v",
    "layer.*.w_o",
    "layer.*.w_gate",
    "layer.*.w_down",
)
_PROMPTS = (
    "What is the capital of France? Answer in one sentence.",
    "Summarize why photosynthesis is important for life on Earth.",
    "Translate 'Good morning, how are you?' into Chinese.",
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain the difference between RAM and storage in simple terms.",
    "What causes the seasons to change on Earth?",
    "Give three bullet points about the benefits of exercise.",
    "Write a short email asking to reschedule a meeting.",
    "What is the derivative of x^2 + 3x + 1?",
    "If a train travels 60 miles in 1.5 hours, what is its average speed?",
    "Describe the plot of Romeo and Juliet in three sentences.",
    "What is the purpose of unit testing in software engineering?",
    "List five countries in South America.",
    "Explain what a GPU does in machine learning.",
    "Write a haiku about the ocean.",
    "What is the boiling point of water at sea level?",
    "Compare democracy and monarchy in two sentences.",
    "Write SQL to select users created in the last seven days.",
    "What is Newton's second law?",
    "Describe how to make a peanut butter sandwich.",
    "Why do programmers use version control?",
    "Name three applications of linear algebra.",
    "What is the tallest mountain in the world?",
    "Explain recursion to a beginner.",
    "What is the difference between a Python list and tuple?",
    "Write a short product description for wireless headphones.",
    "How does a solar panel generate electricity?",
    "What are the main themes of 1984 by George Orwell?",
    "Summarize the water cycle in one paragraph.",
    "Write a polite response declining an invitation.",
    "What is the role of mitochondria in a cell?",
    "Convert the fraction 3/4 into a percentage.",
)


@dataclass(frozen=True)
class _LayerScales:
    input_scale: float
    weight_scale: float


class _FP8Format:
    name = "fp8"

    @staticmethod
    def wrap_matmul(
        network,
        activation,
        weight_array: np.ndarray,
        scales: _LayerScales,
        *,
        lhs_width: int,
        rhs_width: int,
        dtype: np.dtype,
        graph_ops,
    ):
        output_dtype = activation.dtype
        accumulation_dtype = trt.float32 if output_dtype == trt.float16 else output_dtype
        weight = graph_ops.add_constant(network, (lhs_width, rhs_width), weight_array, dtype=dtype)
        if weight.dtype != output_dtype:
            weight = network.add_cast(weight, output_dtype).get_output(0)

        def scale(value: float, target_dtype):
            tensor = graph_ops.add_constant(
                network, (1,), np.asarray([value], dtype=np.float32), dtype=np.float32
            )
            return (
                tensor
                if tensor.dtype == target_dtype
                else network.add_cast(tensor, target_dtype).get_output(0)
            )

        weight_quant = network.add_quantize(
            weight, scale(scales.weight_scale, output_dtype), trt.fp8
        )
        weight_dequant = network.add_dequantize(
            weight_quant.get_output(0),
            scale(scales.weight_scale, accumulation_dtype),
            accumulation_dtype,
        )
        activation_quant = network.add_quantize(
            activation, scale(scales.input_scale, output_dtype), trt.fp8
        )
        activation_dequant = network.add_dequantize(
            activation_quant.get_output(0),
            scale(scales.input_scale, accumulation_dtype),
            accumulation_dtype,
        )
        output = network.add_matrix_multiply(
            activation_dequant.get_output(0),
            trt.MatrixOperation.NONE,
            weight_dequant.get_output(0),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        return (
            output
            if output.dtype == output_dtype
            else network.add_cast(output, output_dtype).get_output(0)
        )


@dataclass(frozen=True)
class _Profile:
    format: _FP8Format
    scales: dict[str, _LayerScales]
    exclude_patterns: tuple[str, ...]

    def should_quantize(self, name: str) -> bool:
        return name in self.scales and not any(
            fnmatch.fnmatch(name, pattern) for pattern in self.exclude_patterns
        )


@dataclass(frozen=True)
class QwenFP8Context:
    profile: _Profile
    graph_ops: Any

    def maybe_quantized_matmul(
        self,
        network,
        lhs,
        lhs_width: int,
        rhs_width: int,
        rhs_weights: np.ndarray,
        weight_name: str,
        dtype: np.dtype = np.float32,
    ):
        if not self.profile.should_quantize(weight_name):
            return self.graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_width, rhs_width, rhs_weights, dtype=dtype
            )
        return self.profile.format.wrap_matmul(
            network,
            lhs,
            rhs_weights,
            self.profile.scales[weight_name],
            lhs_width=lhs_width,
            rhs_width=rhs_width,
            dtype=dtype,
            graph_ops=self.graph_ops,
        )


def _scalar(value) -> float:
    array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else value
    flattened = np.asarray(array, dtype=np.float32).reshape(-1)
    if flattened.size != 1 or not np.isfinite(flattened[0]) or flattened[0] <= 0:
        raise ValueError("Qwen FP8 calibration produced an invalid amax")
    return float(flattened[0]) / 448.0


def _extract_up_projection_scales(state: dict, num_layers: int) -> dict[str, _LayerScales]:
    captured: dict[int, dict[str, float]] = {}
    pattern = re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj\.(input|weight)_quantizer\._amax$")
    for name, value in state.items():
        match = pattern.match(name)
        if match is None:
            continue
        layer = int(match.group(1))
        captured.setdefault(layer, {})[f"{match.group(2)}_scale"] = _scalar(value)
    scales: dict[str, _LayerScales] = {}
    for layer in range(num_layers):
        values = captured.get(layer)
        if values is None or set(values) != {"input_scale", "weight_scale"}:
            raise RuntimeError(f"Qwen FP8 calibration did not produce layer {layer} up-proj scales")
        scales[f"layer.{layer}.w_up"] = _LayerScales(**values)
    return scales


def calibrate_qwen_fp8(model_dir: Path, config, graph_ops) -> QwenFP8Context:
    """Calibrate exactly 64 text samples and return the family Q/DQ context."""
    try:
        import modelopt.torch.quantization as mtq
    except ImportError as error:
        raise RuntimeError(
            "Qwen FP8 calibration requires families/qwen/requirements.txt"
        ) from error
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen FP8 calibration requires CUDA")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=False,
        )
        .eval()
        .to("cuda")
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)

    def forward_loop(calibration_model) -> None:
        for index in range(CALIBRATION_SAMPLE_COUNT):
            batch = tokenizer(
                _PROMPTS[index % len(_PROMPTS)],
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids.to(calibration_model.device)
            with torch.no_grad():
                calibration_model(batch)

    quantized = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    scales = _extract_up_projection_scales(quantized.state_dict(), int(config.num_hidden_layers))
    del quantized, model
    torch.cuda.empty_cache()
    return QwenFP8Context(
        profile=_Profile(_FP8Format(), scales, FP8_EXCLUDE_PATTERNS),
        graph_ops=graph_ops,
    )
