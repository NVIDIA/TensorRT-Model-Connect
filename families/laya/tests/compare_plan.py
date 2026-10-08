# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the released Laya model and the C++ TensorRT probe on identical tensors."""

import argparse
import json
from pathlib import Path
import subprocess

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--record", type=Path, default=Path(__file__).parent / "fixtures/email.json"
    )
    args = parser.parse_args()
    from laya import Agent
    from laya.common import collate_items

    agent = Agent(str(args.checkpoint), device="cuda", fast=False, compile=False)
    assert agent.device.type == "cuda"
    record = json.loads(args.record.read_text())
    question_ids = list(record["questions"])
    internal = {name: agent._to_internal(record["questions"][name]) for name in question_ids}
    items = agent._encode_state(record["state"], question_ids, internal)
    batch = collate_items([items], agent.tok.pad_token_id)
    observed = {}

    def capture_rope(module, inputs, outputs):
        observed["rope_input"] = str(inputs[0].dtype)
        observed["rope_output"] = str(outputs[0].dtype)

    handle = agent.model.encoder.rotary_emb.register_forward_hook(capture_rope)
    with torch.inference_mode():
        logits, act = agent._infer(batch)
        expected = {"logits": logits.float().cpu().numpy(), "act_logits": act.float().cpu().numpy()}
    handle.remove()
    print("reference dtypes", observed, flush=True)
    inputs = {
        name: batch[name].to(torch.int32)
        for name in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    }
    inputs["position_ids"] = torch.arange(batch["input_ids"].shape[1], dtype=torch.int32)
    dump_inputs(args.output, inputs)
    (args.output / "reference.json").write_text(
        json.dumps({name: x.tolist() for name, x in expected.items()}, indent=2)
    )
    print(
        "reference response",
        json.dumps(agent.predict(record["state"], record["questions"])),
        flush=True,
    )
    del agent
    torch.cuda.empty_cache()
    subprocess.run([str(args.probe), str(args.plan), str(args.output)], check=True)
    for name, reference in expected.items():
        actual = np.fromfile(args.output / (name + ".out.bin"), np.float32).reshape(reference.shape)
        print(
            name,
            "max_abs",
            float(np.max(np.abs(actual - reference))),
            "actual",
            actual.tolist(),
            "reference",
            reference.tolist(),
            flush=True,
        )
        np.testing.assert_allclose(actual, reference, atol=0.125, rtol=0.015)
        p, q = torch.tensor(actual).softmax(-1).numpy(), torch.tensor(reference).softmax(-1).numpy()
        np.testing.assert_allclose(p, q, atol=0.002, rtol=0.01)


if __name__ == "__main__":
    main()
