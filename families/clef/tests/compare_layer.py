# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layer-local native TensorRT versus unmodified Transformers comparison."""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from safetensors import safe_open

from families.clef.tests.compare_head import dump_inputs


def read_layer(checkpoint, index):
    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{index}."
    weights = {}
    for shard in sorted({shard for name, shard in mapping.items() if name.startswith(prefix)}):
        with safe_open(checkpoint / shard, framework="pt") as reader:
            for name in reader.keys():
                if name.startswith(prefix):
                    weights[name] = reader.get_tensor(name).bfloat16()
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from families.clef.model import build_decoder_layer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
    )

    raw = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    config = Qwen3_5TextConfig(**raw)
    config._attn_implementation = "sdpa"
    weights = read_layer(args.checkpoint, args.index)
    plan = args.output / f"layer-{args.index}.plan"
    plan.write_bytes(build_decoder_layer(weights, raw, args.index, max_sequence_length=1024))
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, args.index)
    prefix = f"model.language_model.layers.{args.index}."
    layer.load_state_dict(
        {name.removeprefix(prefix): tensor for name, tensor in weights.items()}, assign=True
    )
    layer = layer.cuda().eval()
    rotary = Qwen3_5TextRotaryEmbedding(config).cuda()
    receipts = []
    torch.manual_seed(71)
    with torch.inference_mode():
        for sequence in (64, 127, 256, 273, 512):
            root = args.output / str(sequence)
            x = torch.randn(1, sequence, raw["hidden_size"], dtype=torch.bfloat16, device="cuda")
            cos, sin = rotary(x, torch.arange(sequence, device="cuda")[None])
            mask = torch.full(
                (1, 1, sequence, sequence), -torch.inf, device="cuda", dtype=torch.bfloat16
            ).triu(1)
            expected = layer(
                x,
                position_embeddings=(cos, sin),
                attention_mask=mask if raw["layer_types"][args.index] == "full_attention" else None,
            ).cpu()
            inputs = {"hidden_states": x[0]}
            if raw["layer_types"][args.index] == "full_attention":
                inputs.update(rope_cos=cos[0, :, None], rope_sin=sin[0, :, None])
            dump_inputs(root, inputs)
            expected.view(torch.int16).numpy().tofile(root / "output.reference.bin")
            subprocess.run([str(args.probe), str(plan), str(root)], check=True)
            actual = (
                torch.from_numpy(np.fromfile(root / "output.out.bin", dtype=np.int16))
                .view(torch.bfloat16)
                .reshape(expected.shape)
            )
            difference = (actual.float() - expected.float()).abs()
            receipt = {
                "sequence": sequence,
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "cosine": torch.nn.functional.cosine_similarity(
                    actual.double().flatten(), expected.double().flatten(), dim=0
                ).item(),
            }
            print(json.dumps(receipt), flush=True)
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.0625, rtol=0.02)
            receipts.append(receipt)
    (args.output / "comparison.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
