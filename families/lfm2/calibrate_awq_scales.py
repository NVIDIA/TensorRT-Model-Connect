# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Developer tool: calibrate INT4-AWQ smoothing scales for one LFM2 checkpoint.

Not used when building an engine. Calibrate each supported checkpoint and
commit the resulting ``awq_scales/<name>.npz``::

    python -m families.lfm2.calibrate_fp8_scales fetch --out prompts.jsonl
    python -m families.lfm2.calibrate_awq_scales \
        --model-dir <original BF16 LFM2 checkpoint> --prompts prompts.jsonl \
        --hf-id <org/name> --revision <commit> --name <file stem>

The tool loads the model in PyTorch, runs ModelOpt ``INT4_AWQ_CFG``
(``awq_lite``, no weight-clipping search) over the prompts with the depthwise
short convolution left unquantized, and stores the activation
``pre_quant_scale`` of every block projection (``ALL_PROJECTIONS``, including
the attention ``out_proj``) with float64 sums that identify the
checkpoint. Block scales are not stored: the build recomputes them as ``amax / 7`` of
the smoothed weight, and this tool checks that the recomputed amax matches the
one ModelOpt calibrated. It needs a GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .calibrate_fp8_scales import render_prompt
from .checkpoint_mapper import _open_safetensors
from .quantization import ALL_PROJECTIONS, AWQ_SCALES_DIR, INT4_AWQ_BLOCK_SIZE, awq_identity

_PROJECTION_NAME = re.compile(
    r"layers\.(\d+)\.(" + "|".join(re.escape(stem) for stem, _ in ALL_PROJECTIONS) + r")$"
)
_WEIGHT_STEM = dict(ALL_PROJECTIONS)
# Relative amax difference allowed for ModelOpt's 16-bit smoothed weight copy.
_AMAX_TOLERANCE = 2e-2


def _check_block_amax(name: str, weight: np.ndarray, smooth: np.ndarray, amax) -> None:
    """Fail when the build's recomputed block amax drifts from ModelOpt's calibrated amax."""
    out_features, in_features = weight.shape
    blocks = (weight / smooth.astype(np.float32)).reshape(
        out_features, in_features // INT4_AWQ_BLOCK_SIZE, INT4_AWQ_BLOCK_SIZE
    )
    expected = np.abs(blocks).max(axis=2).reshape(-1)
    calibrated = amax.detach().float().cpu().numpy().reshape(-1)
    if calibrated.shape != expected.shape:
        raise RuntimeError(f"{name}: amax shape {calibrated.shape}, expected {expected.shape}")
    error = np.abs(calibrated - expected) / np.maximum(expected, 1e-8)
    if float(error.max()) > _AMAX_TOLERANCE:
        raise RuntimeError(f"{name}: recomputed block amax differs by {float(error.max()):.3g}")


def awq_config(base: dict | None = None) -> dict:
    """``base`` (ModelOpt ``INT4_AWQ_CFG`` by default) with the depthwise short convolution left unquantized."""
    if base is None:
        import modelopt.torch.quantization as mtq

        base = mtq.INT4_AWQ_CFG
    config = copy.deepcopy(base)
    pattern = "*conv.conv.*"
    if isinstance(config["quant_cfg"], list):  # ModelOpt 0.48 and later
        config["quant_cfg"].append({"quantizer_name": pattern, "enable": False})
    else:
        config["quant_cfg"][pattern] = {"enable": False}
    return config


def calibrate(
    model_dir: Path, prompts: Path, max_tokens: int, hf_id: str, revision: str, name: str
) -> Path:
    """Run ModelOpt INT4-AWQ calibration and store the smoothing scales for this checkpoint."""
    import modelopt.torch.quantization as mtq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .config import validate_dense_lfm2_config

    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = validate_dense_lfm2_config(SimpleNamespace(raw=raw))
    readers = _open_safetensors(model_dir)
    identity = awq_identity(readers)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rows = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines() if line]
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).cuda().eval()

    def forward_loop(calibration_model) -> None:
        """Feed every calibration prompt through the model once."""
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(
                    render_prompt(tokenizer, row["messages"]),
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_tokens,
                    add_special_tokens=False,
                ).input_ids.cuda()
                calibration_model(ids)

    model = mtq.quantize(model, awq_config(), forward_loop)

    scales: dict[str, np.ndarray] = {}
    for module_name, module in model.named_modules():
        match = _PROJECTION_NAME.search(module_name)
        if match is None or not hasattr(module, "input_quantizer"):
            continue
        smooth = module.input_quantizer.pre_quant_scale
        if smooth is None:
            raise RuntimeError(f"{module_name} has no calibrated pre_quant_scale")
        smooth = smooth.detach().float().cpu().numpy().astype(np.float16)
        layer, stem = match.group(1), match.group(2)
        key = f"model.layers.{layer}.{stem}.weight"
        weight = np.asarray(readers.tensor_map[key].get_tensor(key), dtype=np.float32)
        _check_block_amax(module_name, weight, smooth, module.weight_quantizer.amax)
        scales[f"layer.{layer}.{_WEIGHT_STEM[stem]}"] = smooth
    # Conv layers: conv in/out + SwiGLU (5); attention layers: q/k/v/out + SwiGLU (7).
    expected = 5 * config.num_conv_layers + 7 * config.num_attention_layers
    if len(scales) != expected:
        raise RuntimeError(f"calibrated {len(scales)} projections, expected {expected}")

    AWQ_SCALES_DIR.mkdir(exist_ok=True)
    path = AWQ_SCALES_DIR / f"{name}.npz"
    np.savez_compressed(
        path,
        hf_id=np.array(hf_id),
        hf_revision=np.array(revision),
        algorithm=np.array("modelopt INT4_AWQ_CFG awq_lite, no clipping search"),
        **identity,
        **dict(sorted(scales.items())),
    )
    print(f"wrote {len(scales)} smoothing scales for {hf_id}@{revision} to {path}")
    return path


def main() -> None:
    """Parse the command line and run the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--hf-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    calibrate(args.model_dir, args.prompts, args.max_tokens, args.hf_id, args.revision, args.name)


if __name__ == "__main__":
    main()
