#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TimesFM-owned ETTh1 Accuracy reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    import transformers

    if request.get("precision") != "fp32":
        raise ValueError("TimesFM Accuracy requires an fp32 reference")
    if not torch.cuda.is_available():
        raise RuntimeError("TimesFM Accuracy reference requires CUDA")
    options = {
        "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
    }
    if request.get("revision"):
        options["revision"] = request["revision"]
    model = (
        transformers.TimesFmModelForPrediction.from_pretrained(
            str(request["model"]), torch_dtype=torch.float32, **options
        )
        .eval()
        .to("cuda")
    )
    context = int(model.config.context_length)
    samples = []
    with torch.inference_mode():
        for sample in request["samples"]:
            raw = [float(value) for value in sample["past_values"]]
            count = min(len(raw), context)
            values = [0.0] * context
            values[-count:] = raw[-count:]
            padding = [1] * context
            padding[-count:] = [0] * count
            decoded = model.decoder(
                past_values=torch.tensor(
                    values, dtype=torch.float32, device="cuda"
                ).reshape(1, context),
                past_values_padding=torch.tensor(
                    padding, dtype=torch.int32, device="cuda"
                ).reshape(1, context),
                freq=torch.tensor(
                    [[int(sample.get("frequency", 0))]],
                    dtype=torch.long,
                    device="cuda",
                ),
                output_attentions=False,
                output_hidden_states=False,
            )
            output = model._postprocess_output(
                decoded.last_hidden_state, (decoded.loc, decoded.scale)
            )[:, -1, : model.config.horizon_length, 0].detach().float().cpu()
            samples.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "shape": [int(size) for size in output.shape],
                    "values": output.reshape(-1).tolist(),
                }
            )
    arguments.output.write_text(
        json.dumps(
            {
                "schema_version": "trtmc.accuracy-reference/v1",
                "model": request["model"],
                "revision": request.get("revision"),
                "precision": "fp32",
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
