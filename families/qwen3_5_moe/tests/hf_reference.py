# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit official packed-source oracle; no custom dequantization or relaxed metric."""

from __future__ import annotations
import argparse
import contextlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from families.qwen3_5_moe.tests.quality import (
    normalized_edit_distance,
    assert_quality,
    combined_evidence,
)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
REVISION = "3af5ca2972faf6de1fd6f4efc4d8d319ca751e8b"


def _reference_device(name: str):
    """Honor explicit oracle placement without hiding CUDA or choosing by quality."""
    import torch

    if name not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported reference device: {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected reference requires an available CUDA device")
    return torch.device("cuda:0" if name == "cuda" else "cpu")


def _reference_precision(checkpoint: Path) -> str:
    """Honor the packed checkpoint's declared arithmetic; never guess FP16."""
    config = json.loads((checkpoint / "config.json").read_text())
    dtype = config.get("text_config", {}).get("dtype")
    precisions = {"float16": "fp16", "bfloat16": "bf16"}
    if not isinstance(dtype, str) or dtype not in precisions:
        raise ValueError(f"Unsupported packed-reference checkpoint dtype: {dtype!r}")
    return precisions[dtype]



def _reference_format(checkpoint: Path, index: dict[str, str]) -> str:
    """Infer only an unambiguous symmetric source format; never rewrite packed tensors."""
    import numpy as np
    from safetensors import safe_open

    quant = json.loads((checkpoint / "config.json").read_text())["quantization_config"]
    if quant.get("quant_method") != "gptq" or quant.get("bits") != 4 or quant.get("sym") is not True:
        raise ValueError("Packed reference requires symmetric four-bit GPTQ")
    keys = [key for key in index if key.endswith(".qzeros") and not key.startswith("mtp.")]
    if not keys:
        raise ValueError("Packed reference has no source zero points")
    formats = set()
    with contextlib.ExitStack() as stack:
        shards = {
            name: stack.enter_context(safe_open(checkpoint / name, framework="np"))
            for name in {index[key] for key in keys}
        }
        for key in keys:
            zeros = shards[index[key]].get_tensor(key)
            if zeros.dtype != np.int32 or zeros.ndim != 2 or not zeros.size:
                raise ValueError(f"Invalid packed source zero points: {key}")
            words = zeros.view(np.uint32)
            if np.all(words == np.uint32(0x77777777)):
                formats.add("gptq")
            elif np.all(words == np.uint32(0x88888888)):
                formats.add("gptq_v2")
            else:
                raise ValueError(f"Ambiguous symmetric source zero points: {key}")
    if len(formats) != 1:
        raise ValueError("Mixed packed source zero-point formats")
    source_format = formats.pop()
    for field in ("format", "checkpoint_format"):
        if quant.get(field) is not None and quant[field] != source_format:
            raise ValueError("Declared GPTQ format disagrees with source zero points")
    return source_format


def run(checkpoint: Path, output: Path, image_path: Path, *, device: str = "cuda") -> None:
    """Load the official route, verify every source tensor, then compare frozen requests.

    Args:
        checkpoint: Complete immutable original GPTQ source checkpoint.
        output: Directory containing the fresh public/direct proof outputs.
        image_path: Exact RGB image used by the public proof.
        device: CUDA for the GPU E2E; CPU only when explicitly requested for diagnostics.

    Raises:
        Exception: Missing dependencies, source mismatch, loader error or strict quality failure.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
    from safetensors import safe_open
    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    assert os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    reference_device = _reference_device(device)
    assert importlib.metadata.version("gptqmodel") == "7.4.0"
    assert importlib.metadata.version("transformers") == "5.14.1"
    started = time.time()
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    precision = _reference_precision(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    source_format = _reference_format(checkpoint, index)
    reference_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    model = GPTQModel.load(
        str(checkpoint),
        device=str(reference_device),
        backend=BACKEND.TORCH,
        dtype=reference_dtype,
        format=source_format,
        trust_remote_code=False,
        local_files_only=True,
        attn_implementation="eager",
    )
    print("MODEL LOADED", type(model), time.time() - started, flush=True)
    assert type(model).__name__ == "Qwen3_5_MoeQModel"
    quantized = {
        name: module
        for name, module in model.model.named_modules()
        if isinstance(module, BaseQuantLinear)
    }
    assert len(quantized) == 40 * 256 * 3, len(quantized)
    assert all(".mlp.experts." in name for name in quantized)
    qcfg = model.quantize_config
    runtime_format = getattr(qcfg.runtime_format, "value", str(qcfg.runtime_format))
    assert runtime_format in ("gptq", "gptq_v2"), runtime_format
    requires_v2 = any(module.REQUIRES_FORMAT_V2 for module in quantized.values())
    assert (runtime_format == "gptq_v2") == requires_v2
    state = model.model.state_dict()
    expected = {name for name in index if not name.startswith("mtp.")}
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    print("STATE KEYS", {"missing": missing, "unexpected": unexpected}, flush=True)
    (output / "reference-source-keys.json").write_text(
        json.dumps(
            {
                "missing": missing,
                "unexpected": unexpected,
                "quantized_module_count": len(quantized),
                "runtime_format": runtime_format,
            },
            indent=2,
        )
        + "\n"
    )
    assert not missing and not unexpected
    counts = {"qweight": 0, "qzeros": 0, "g_idx": 0, "scales": 0, "original": 0}
    with contextlib.ExitStack() as stack:
        shards = {
            name: stack.enter_context(safe_open(checkpoint / name, framework="pt", device="cpu"))
            for name in set(index.values())
        }
        for name, tensor in state.items():
            assert not tensor.is_meta and tensor.device == reference_device, name
            source = shards[index[name]].get_tensor(name)
            field = name.rsplit(".", 1)[-1]
            if field == "qzeros" and source_format == "gptq" and runtime_format == "gptq_v2":
                assert (
                    tensor.dtype == torch.int32
                    and quantized[name.rsplit(".", 1)[0]].REQUIRES_FORMAT_V2
                )
                source = source + 0x11111111
            assert torch.equal(tensor.cpu(), source.to(dtype=tensor.dtype)), name
            counts[field if field in counts else "original"] += 1
    assert all(
        counts[field] == 40 * 256 * 3 for field in ("qweight", "qzeros", "g_idx", "scales")
    ), counts
    for layer in range(40):
        mlp = model.model.model.language_model.layers[layer].mlp
        assert isinstance(mlp.gate, Qwen3_5MoeTopKRouter) and isinstance(
            mlp.shared_expert_gate, torch.nn.Linear
        )
        assert isinstance(mlp.shared_expert.gate_proj, torch.nn.Linear)
    print("ALL SOURCE TENSORS VERIFIED", counts, "runtime_format", runtime_format, flush=True)
    model.model.eval()
    processor = AutoProcessor.from_pretrained(
        checkpoint, trust_remote_code=False, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=False, local_files_only=True
    )
    image = Image.open(image_path).convert("RGB")
    actual = {
        item["id"]: item for item in json.loads((output / "public.json").read_text())["results"]
    }
    generation = json.loads((checkpoint / "generation_config.json").read_text())
    results = []
    for kind in ("raw", "image"):
        if kind == "raw":
            prompt = "The capital of France is"
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {
            k: v.to(device=reference_device,
                    dtype=reference_dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        input_ids = inputs["input_ids"][0].tolist()
        print("GENERATE", kind, "input_count", len(input_ids), flush=True)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=generation["eos_token_id"],
                pad_token_id=generation["pad_token_id"],
                temperature=None,
                top_k=None,
                top_p=None,
            )
        ids = generated[0, len(input_ids) :].tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        ned = normalized_edit_distance(actual[kind]["text"], text)
        record = {
            "id": kind,
            "prompt": prompt,
            "input_token_ids": input_ids,
            "input_count": len(input_ids),
            "token_ids": ids,
            "text": text,
            "actual_text": actual[kind]["text"],
            "normalized_edit_distance": ned,
            "passed": bool(text.strip()) and ned <= 0.15,
        }
        results.append(record)
        report = {
            "checkpoint": MODEL_ID,
            "revision": REVISION,
            "source_verified": counts,
            "source_format": source_format,
            "runtime_format": runtime_format,
            "precision": precision,
            "backend": "official GPTQModel7.4.0 Qwen3_5_MoeQModel / Torch",
            "device": str(reference_device),
            "threshold": 0.15,
            "results": results,
            "passed": len(results) == 2 and all(row["passed"] for row in results),
            "elapsed_seconds": time.time() - started,
        }
        (output / "hf.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    report["dependencies"] = {
        name: importlib.metadata.version(name)
        for name in (
            "gptqmodel",
            "transformers",
            "torch",
            "safetensors",
            "accelerate",
            "numpy",
            "torchao",
        )
    }
    (output / "hf.json").write_text(json.dumps(report, indent=2) + "\n")
    evidence = json.loads((output / "gpu.evidence.json").read_text())
    evidence = combined_evidence(evidence, report)
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for row in results:
        assert_quality(row["actual_text"], row["text"])


def main() -> None:
    """Run only with the reviewed offline reference environment and explicit placement."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "output", "image"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="Use the leased GPU; CPU is an explicit diagnostic mode")
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.image, device=args.device)


if __name__ == "__main__":
    main()
