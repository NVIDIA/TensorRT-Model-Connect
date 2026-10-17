# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit official packed-source CPU oracle; no custom dequantization or relaxed metric."""

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


def run(checkpoint: Path, output: Path, image_path: Path) -> None:
    """Load the official CPU route, verify every source tensor, then compare frozen requests.

    Args:
        checkpoint: Complete immutable original GPTQ source checkpoint.
        output: Directory containing the fresh public/direct proof outputs.
        image_path: Exact RGB image used by the public proof.

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
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
    assert importlib.metadata.version("gptqmodel") == "7.4.0"
    assert importlib.metadata.version("transformers") == "5.14.1"
    started = time.time()
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    model = GPTQModel.load(
        str(checkpoint),
        device="cpu",
        backend=BACKEND.TORCH,
        dtype=torch.float16,
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
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
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
            assert not tensor.is_meta and tensor.device.type == "cpu", name
            source = shards[index[name]].get_tensor(name)
            field = name.rsplit(".", 1)[-1]
            if field == "qzeros" and runtime_format == "gptq_v2":
                assert (
                    tensor.dtype == torch.int32
                    and quantized[name.rsplit(".", 1)[0]].REQUIRES_FORMAT_V2
                )
                source = source + 0x11111111
            assert torch.equal(tensor, source.to(dtype=tensor.dtype)), name
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
            k: (v.to(dtype=torch.float16) if v.is_floating_point() else v)
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
            "runtime_format": runtime_format,
            "precision": "fp16",
            "backend": "official GPTQModel7.4.0 Qwen3_5_MoeQModel / Torch CPU",
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
    """Run only when explicitly provisioned with the reviewed offline CPU reference environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "output", "image"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.image)


if __name__ == "__main__":
    main()
