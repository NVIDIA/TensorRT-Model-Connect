#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-Bolt official internal Accuracy reference."""

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
    from chronos import ChronosBoltPipeline

    if request.get("precision") != "fp32":
        raise ValueError("Chronos-Bolt Accuracy requires an fp32 reference")
    if not torch.cuda.is_available():
        raise RuntimeError("Chronos-Bolt Accuracy reference requires CUDA")
    options = {
        "device_map": "cuda",
        "dtype": torch.float32,
        "local_files_only": os.environ.get("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY") == "1",
    }
    if request.get("revision"):
        options["revision"] = request["revision"]
    pipeline = ChronosBoltPipeline.from_pretrained(str(request["model"]), **options)
    samples = []
    with torch.inference_mode():
        for sample in request["samples"]:
            context = torch.tensor(sample["past_values"], dtype=torch.float32, device="cuda")
            value = pipeline.predict(
                context,
                prediction_length=pipeline.model_prediction_length,
                limit_prediction_length=True,
            ).detach().float().cpu()
            samples.append(
                {
                    "sample_id": str(sample["sample_id"]),
                    "shape": [int(size) for size in value.shape],
                    "values": value.reshape(-1).tolist(),
                }
            )
    result = {
        "schema_version": "trtmc.accuracy-reference/v1",
        "model": request["model"],
        "revision": request.get("revision"),
        "precision": "fp32",
        "samples": samples,
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
