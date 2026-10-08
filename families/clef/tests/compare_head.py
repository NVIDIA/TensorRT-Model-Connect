# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the original joint head and its native TensorRT graph on the same tensors."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def dump_inputs(root, inputs):
    root.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for name, tensor in inputs.items():
        tensor = tensor.detach().cpu().contiguous()
        dtype = {torch.bfloat16: "bf16", torch.float32: "float32", torch.int32: "int32"}[
            tensor.dtype
        ]
        data = (
            tensor.view(torch.int16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
        )
        data.tofile(root / (name + ".bin"))
        metadata[name] = {"dtype": dtype, "shape": list(tensor.shape)}
    (root / "inputs.json").write_text(json.dumps(metadata))


def head_inputs(hidden, lexical_weight, record):
    sequence, width = hidden.shape
    question_count = len(record.questions)
    option_count = sum(len(q.option_ids) for q in record.questions)
    q_pool = torch.zeros(question_count, sequence)
    o_pool = torch.zeros(option_count, sequence)
    group_mask = torch.full((question_count, option_count), -torch.inf, dtype=torch.bfloat16)
    fields, types, lexical = [], [], []
    for qi, q in enumerate(record.questions):
        start, end = q.question_span
        q_pool[qi, start:end] = 1 / (end - start)
        types.append(q.question_type)
        for start, end in q.option_spans:
            oi = len(fields)
            o_pool[oi, start:end] = 1 / (end - start)
            group_mask[qi, oi] = 0
            fields.append(qi)
            lexical.append(lexical_weight[list(record.input_ids[start:end])].mean(0))
    return {
        "hidden_states": hidden,
        "lexical_options": torch.stack(lexical),
        "question_pool": q_pool,
        "option_pool": o_pool,
        "option_fields": torch.tensor(fields, dtype=torch.int32),
        "type_ids": torch.tensor(types, dtype=torch.int32),
        "group_mask": group_mask,
        "last_index": torch.tensor([sequence - 1], dtype=torch.int32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkpoint))
    from joint_schema_model import JointSchemaHead, encode_record
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    torch.manual_seed(1042)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = json.loads((args.checkpoint / "joint_head_config.json").read_text())
    head = JointSchemaHead(**config)
    head.load_state_dict(load_file(args.checkpoint / "joint_head.safetensors"))
    head = head.cuda().bfloat16().eval()
    receipts = []
    with torch.inference_mode():
        for fixture in sorted((Path(__file__).parent / "fixtures").glob("*.json")):
            record = encode_record(tokenizer, json.loads(fixture.read_text()))
            seq = len(record.input_ids)
            hidden = torch.randn(seq, config["hidden_size"], device="cuda", dtype=torch.bfloat16)
            # Only token IDs present in the record need embeddings in this test.
            ids = torch.tensor(record.input_ids, device="cuda")[None]
            lexical_weight = torch.randn(
                int(ids.max()) + 1, config["hidden_size"], device="cuda", dtype=torch.bfloat16
            )
            expected = (
                torch.cat(
                    head(hidden[None], ids, torch.ones_like(ids), [record], lexical_weight)[0]
                )
                .float()
                .cpu()
                .numpy()
            )
            root = args.output / fixture.stem
            dump_inputs(root, head_inputs(hidden, lexical_weight, record))
            expected.tofile(root / "logits.reference.bin")
            subprocess.run([str(args.probe), str(args.plan), str(root)], check=True)
            actual = np.fromfile(root / "logits.out.bin", dtype=np.float32)
            print(
                fixture.stem, "expected", expected.tolist(), "actual", actual.tolist(), flush=True
            )
            # One BF16 ULP is ~0.8% around unit scale; do not substitute a
            # classification-only check for per-option numerical agreement.
            np.testing.assert_allclose(actual, expected, atol=0.025, rtol=0.01)
            receipts.append(
                {"fixture": fixture.stem, "max_abs_error": float(np.max(np.abs(actual - expected)))}
            )
    (args.output / "head-comparison.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
