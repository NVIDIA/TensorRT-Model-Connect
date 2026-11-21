# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the native reference of the family's INT4-AWQ profiles: each checkpoint with every decoder projection
replaced by the INT4-AWQ weight the bundle runs, under ``<environment>/trtmc-reference/<manifest name>``.

The build quantizes with the committed AWQ scales (``quantization.quantize_awq_weight``) and the engine computes
``(x * pre_quant_scale) @ (codes * block_scale)^T``. The reference stores ``codes * block_scale * pre_quant_scale``
(per input channel) as float16, the same product, so PyTorch runs the quantized model the bundle runs and the
comparison measures the TensorRT implementation, not INT4 quantization.

usage (``reference.prepare``, run once per reference environment): python prepare_int4_awq.py <environment>
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANIFESTS = ROOT / "families/llama/tests/manifests"
MARKER = "trtmc-int4-awq-reference.json"


def effective_weight(weight, pre_quant_scale):
    """The ``[out, in]`` float32 weight an INT4-AWQ engine multiplies the unscaled activation by."""
    import numpy as np

    from families.llama import quantization

    packed = quantization.quantize_awq_weight(weight, pre_quant_scale)
    rows, columns = packed.out_features, packed.in_features
    codes = np.empty((rows, columns), np.int8)
    codes[:, 0::2] = (packed.packed & 15).astype(np.int8)
    codes[:, 1::2] = (packed.packed >> 4).astype(np.int8)
    codes = np.where(codes > 7, codes - 16, codes).astype(np.float32)
    block = quantization.INT4_AWQ_BLOCK_SIZE
    dequantized = (
        codes.reshape(rows, columns // block, block)
        * packed.block_scale.astype(np.float32)[..., None]
    )
    dequantized = (
        dequantized.reshape(rows, columns).astype(np.float16).astype(np.float32)
    )  # the engine's fp16 dequantize
    return dequantized * packed.pre_quant_scale.astype(np.float32)[None, :]


def write_reference(manifest: dict, destination: Path) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    from families.llama import quantization

    source = Path(
        snapshot_download(manifest["hf_id"], revision=manifest.get("hf_revision") or None)
    )
    tensors = quantization._open_raw_readers(source)
    table = quantization._load_awq_scales(
        quantization.AWQ_SCALES_DIR, quantization.awq_identity(tensors)
    )
    stems = dict(quantization._PROJECTIONS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        for path in source.iterdir():
            if path.suffix != ".safetensors":
                (shutil.copytree if path.is_dir() else shutil.copy)(
                    path.resolve(), work / path.name
                )
        replaced = 0
        for shard in sorted(source.glob("*.safetensors")):
            state = load_file(str(shard.resolve()))
            for key in list(state):
                parts = key.split(".")
                stem = ".".join(parts[3:-1])
                if key.startswith("model.layers.") and key.endswith(".weight") and stem in stems:
                    scale = table[f"layer.{parts[2]}.{stems[stem]}"]
                    state[key] = torch.from_numpy(
                        effective_weight(state[key].float().numpy(), scale)
                    ).to(torch.float16)
                    replaced += 1
            save_file(state, str(work / shard.name), metadata={"format": "pt"})
        (work / MARKER).write_text(
            json.dumps(
                {
                    "hf_id": manifest["hf_id"],
                    "hf_revision": manifest.get("hf_revision"),
                    "projections": replaced,
                }
            )
        )
        shutil.rmtree(destination, ignore_errors=True)
        work.rename(destination)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise


def main(environment: Path) -> int:
    sys.path.insert(0, str(ROOT))
    for path in sorted(MANIFESTS.glob("*.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("quantization") != "int4_awq":
            continue
        destination = environment / "trtmc-reference" / manifest["name"]
        if not (destination / MARKER).is_file():
            write_reference(manifest, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sys.prefix)))
