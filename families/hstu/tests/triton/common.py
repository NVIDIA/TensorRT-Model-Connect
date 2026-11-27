# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json
import re
import time
import urllib.request
import numpy as np
from tritonclient.utils import np_to_triton_dtype

import os

root = Path(os.environ.get("HSTU_TEST_ROOT", "/work"))
runtime_root = Path(os.environ.get("TRTMC_RUNTIME_ROOT", str(root / "deploy")))
repository = root / "triton-models"
FIELDS = (
    "supported",
    "budget_bytes",
    "reserved_bytes",
    "used_bytes",
    "peak_bytes",
    "active_sequences",
    "allocation_failures",
)


def scrape(model):
    text = urllib.request.urlopen("http://localhost:8002/metrics", timeout=10).read().decode()
    samples = []
    for line in text.splitlines():
        if line.startswith("#") or f'model="{model}"' not in line:
            continue
        name = line.split("{", 1)[0]
        if "kv_cache" in name:
            labels = dict(re.findall(r'(\w+)="([^"\\]*)"', line))
            samples.append(dict(name=name, labels=labels, value=float(line.rsplit(" ", 1)[1])))
    fields = {}
    for key in FIELDS:
        found = [
            row
            for row in samples
            if row["name"] in ("hstu_kv_cache_" + key, "hstu_kv_cache_" + key + "_total")
        ]
        if found:
            fields[key] = sum(row["value"] for row in found)
    return text, samples, fields


def wait_metrics(model, predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while True:
        text, samples, fields = scrape(model)
        if predicate(fields):
            return text, samples, fields
        if time.monotonic() > deadline:
            raise AssertionError(f"Metrics did not converge for {model}: {fields}")
        time.sleep(0.02)


def make_model(name, variant, parameters=None, instances=1):
    source = repository / ("native_" + variant.replace("-", "_")) / "config.pbtxt"
    directory = repository / name
    (directory / "1").mkdir(parents=True, exist_ok=True)
    text = source.read_text().replace("native_" + variant.replace("-", "_"), name)
    text = text.replace("count: 1", f"count: {instances}")
    for key, value in (parameters or {}).items():
        text += (
            "parameters { key: "
            + json.dumps(key)
            + " value { string_value: "
            + json.dumps(str(value))
            + " } }\n"
        )
    (directory / "config.pbtxt").write_text(text)
    link = directory / "1/model.bundle"
    target = root / "output" / variant / "model.bundle"
    if link.is_symlink():
        assert link.readlink() == target
    else:
        link.symlink_to(target)
    return directory


def infer(client, module, model, features, outputs):
    tensors = []
    for name, value in features.items():
        value = np.ascontiguousarray(value)
        tensor = module.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
        tensor.set_data_from_numpy(value)
        tensors.append(tensor)
    return client.infer(
        model, tensors, outputs=[module.InferRequestedOutput(name) for name in outputs]
    )
