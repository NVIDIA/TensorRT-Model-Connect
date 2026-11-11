# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the independent, pinned Transformers checkpoint implementation.

The reference consumes exactly the diagnostic request accepted by the native
family test executable. Fixed canvases isolate graph correctness from the two
implementations' different random number generators. This module deliberately
does not import the family's builder, graph helpers, or runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRANSFORMERS_REVISION = "07cc8e8c1b19160cfba6893ceb3099dc8f09d90c"


def save_tensor(directory: Path, tensors: dict, name: str, tensor) -> None:
    import torch

    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    array.tofile(directory / f"{name}.bin")
    tensors[name] = {
        "file": f"{name}.bin",
        "shape": list(array.shape),
        "dtype": "float32",
    }


def _assert_official_source() -> dict:
    """Require the recorded source revision instead of an ambient installation."""
    from importlib.metadata import distribution

    document = distribution("transformers").read_text("direct_url.json")
    source = json.loads(document) if document else {}
    expected_url = (
        "https://github.com/huggingface/transformers/archive/"
        f"{TRANSFORMERS_REVISION}.tar.gz"
    )
    if source.get("url") != expected_url and (
        source.get("vcs_info", {}).get("commit_id") != TRANSFORMERS_REVISION
    ):
        raise RuntimeError(
            "The reference requires Transformers installed from the pinned official "
            f"commit {TRANSFORMERS_REVISION}; use tests/requirements.txt."
        )
    return source


def run(checkpoint: Path, request_path: Path, output: Path) -> None:
    import torch
    from transformers import DiffusionGemmaForBlockDiffusion

    source = _assert_official_source()
    if not torch.cuda.is_available():
        raise RuntimeError("The exact DiffusionGemma reference requires a CUDA device")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "transformers_revision": TRANSFORMERS_REVISION,
        "transformers_source": source,
        "request": request,
        "checkpoint_config": json.loads((checkpoint / "config.json").read_text()),
        "checkpoint_index": json.loads((checkpoint / "model.safetensors.index.json").read_text()),
        "torch_version": torch.__version__,
        "hardware": {
            "name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "tensors": {},
        "cache_readonly": True,
    }
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    ids = torch.tensor([request["input_ids"]], dtype=torch.long, device="cuda")
    canvases = request["decoder_input_ids"]
    if not canvases or isinstance(canvases[0], int):
        canvases = [canvases]
    temperatures = request.get("temperatures", [0.8] * len(canvases))
    if len(temperatures) != len(canvases):
        raise ValueError("One temperature is required for every fixed canvas")
    tensors = metadata["tensors"]
    with torch.inference_mode():
        encoded = model.model.encoder(input_ids=ids)
        cache = encoded.past_key_values
        saved_cache = []
        for index, layer in enumerate(cache.layers):
            key, value = layer.keys, layer.values
            save_tensor(output, tensors, f"encoder_key_{index}", key.squeeze(0))
            save_tensor(output, tensors, f"encoder_value_{index}", value.squeeze(0))
            saved_cache.append((key.clone(), value.clone()))
        previous_logits = None
        for step, (canvas, temperature) in enumerate(zip(canvases, temperatures, strict=True)):
            canvas_ids = torch.tensor([canvas], dtype=torch.long, device="cuda")
            decoded = model.model.decoder(
                decoder_input_ids=canvas_ids,
                past_key_values=cache,
                self_conditioning_logits=previous_logits,
            )
            # Use the upstream LM head and its exact FP32 softcapping order.
            logits = model.lm_head(decoded.last_hidden_state).float()
            cap = model.final_logit_softcapping
            logits = (logits / cap).tanh() * cap
            save_tensor(output, tensors, f"step_{step}_raw_logits", logits.squeeze(0))
            tensors["raw_logits"] = tensors[f"step_{step}_raw_logits"]
            # Generation casts processed logits to the embedding dtype before
            # handing them to the next denoising step.
            previous_logits = (logits / float(temperature)).to(torch.bfloat16)
            embedding = model.model.decoder.embed_tokens
            signal = (
                previous_logits.softmax(dim=-1, dtype=torch.float32).to(torch.bfloat16)
                @ embedding.weight
            ) * embedding.embed_scale.to(torch.bfloat16)
            save_tensor(output, tensors, f"step_{step}_self_conditioning", signal.squeeze(0))
            for layer, (key, value) in zip(cache.layers, saved_cache, strict=True):
                if not torch.equal(layer.keys, key) or not torch.equal(layer.values, value):
                    raise AssertionError("The upstream decoder mutated its encoder KV cache")
    metadata["decoder_steps"] = len(canvases)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.request, args.output)


if __name__ == "__main__":
    main()
