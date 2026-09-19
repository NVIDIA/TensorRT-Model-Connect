#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PatchTSMixer-owned ETTh1 Accuracy reference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def prediction_model_class(transformers):
    model_class = transformers.PatchTSMixerForPrediction
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    return model_class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text(encoding="utf-8"))

    import torch
    import transformers

    if request.get("precision") != "fp32":
        raise ValueError("PatchTSMixer Accuracy requires an fp32 reference")
    if not torch.cuda.is_available():
        raise RuntimeError("PatchTSMixer Accuracy reference requires CUDA")
    options = {
        "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1"
    }
    if request.get("revision"):
        options["revision"] = request["revision"]
    config = transformers.AutoConfig.from_pretrained(str(request["model"]), **options)
    model_class = prediction_model_class(transformers)
    model = (
        model_class.from_pretrained(
            str(request["model"]), torch_dtype=torch.float32, **options
        )
        .eval()
        .to("cuda")
    )
    context = int(config.context_length)
    channels = int(config.num_input_channels)
    samples = []
    with torch.inference_mode():
        for sample in request["samples"]:
            raw = [float(value) for value in sample["past_values"]]
            count = min(len(raw), context * channels)
            values = [0.0] * (context * channels)
            values[-count:] = raw[-count:]
            observed = [0.0] * (context * channels)
            observed[-count:] = [1.0] * count
            output = model(
                past_values=torch.tensor(
                    values, dtype=torch.float32, device="cuda"
                ).reshape(1, context, channels),
                observed_mask=torch.tensor(
                    observed, dtype=torch.float32, device="cuda"
                ).reshape(1, context, channels),
                return_loss=False,
                return_dict=True,
            ).prediction_outputs.detach().float().cpu()
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
