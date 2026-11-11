# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full checkpoint -> bundle -> native Task -> independent logit oracle."""

import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest


def test_flint_bf16_logits(request, tmp_path):
    directory = Path(__file__).parent
    manifest = json.loads((directory / "manifests/flint-bf16.json").read_text())
    case = manifest["testcases"][0]
    values = (request.config.getoption("--e2e-model", default=[]) or []) + (
        request.config.getoption("--e2e-testcase", default=[]) or []
    )
    selected = {item.strip() for value in values for item in value.split(",") if item.strip()}
    models_file = request.config.getoption("--e2e-models-file", default=None)
    if models_file:
        selected.update(
            line.strip()
            for line in Path(models_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if selected and not selected.intersection({"flint", manifest["name"], case["name"]}):
        pytest.skip("Flint was not selected")
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("requires explicit Flint E2E selection")
    from tokenizers import Tokenizer
    import torch
    from tensorrt_model_connect import BuildRequest, build
    from tensorrt_model_connect.model_support import load_model_metadata, resolve_family
    from families.flint.tests.reference import Reference

    assert manifest["trust_remote_code"] is False
    assert case["reference_precision"] == manifest["precision"]
    thresholds = json.loads((directory / "thresholds" / (case["name"] + ".json")).read_text())[
        "threshold_overrides"
    ]
    checkpoint_value = os.environ.get("TRTMC_FLINT_CHECKPOINT")
    if checkpoint_value:
        checkpoint = Path(checkpoint_value)
    else:
        from huggingface_hub import snapshot_download

        checkpoint = Path(
            snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"])
        )
    # These local-dir download receipts identify every fetched shard's commit;
    # a manually supplied checkpoint must provide the same pinned receipts.
    if checkpoint.parent.name == "snapshots":
        assert checkpoint.name == manifest["hf_revision"]
    else:
        for filename in [
            "config.json",
            "model.safetensors.index.json",
            *sorted(path.name for path in checkpoint.glob("*.safetensors")),
        ]:
            receipt = checkpoint / ".cache/huggingface/download" / (filename + ".metadata")
            assert receipt.read_text().splitlines()[0] == manifest["hf_revision"]
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    binary = Path(os.environ["TRTMC_BINARY"])
    assert binary.is_file()
    assert (runtime / "libtrtmc_model_flint.so").is_file()
    family, support = resolve_family(load_model_metadata(checkpoint))
    assert family == "flint" and support.default_task == manifest["task"]
    # An explicitly supplied prebuilt artifact is validated below through its
    # provenance. Otherwise this direct E2E owns the complete public build call.
    supplied = os.environ.get("TRTMC_FLINT_BUNDLE")
    bundle = Path(supplied) if supplied else tmp_path / manifest["bundle"]
    if not supplied:
        build(
            BuildRequest(
                model_dir=checkpoint,
                output_path=bundle,
                family=family,
                task=manifest["task"],
                precision=manifest["precision"],
                max_sequence_length=manifest["max_sequence_length"],
            )
        )
    assert bundle.is_file() and bundle.stat().st_size > 70_000_000_000
    inspected = subprocess.run(
        [str(binary), "inspect", str(bundle)], check=True, capture_output=True, text=True
    )
    info = json.loads(inspected.stdout)
    assert info["family"] == "flint" and info["task"] == "text_generation"
    assert info["provenance"]["build"]["precision"] == manifest["precision"]
    assert info["provenance"]["build"]["max_sequence_length"] == manifest["max_sequence_length"]
    assert info["provenance"]["build"]["tensor_parallel_size"] == manifest["tensor_parallel_size"]
    trace_binary = runtime / "flint_logits_trace"
    assert trace_binary.is_file()
    prefix = tmp_path / "native"
    subprocess.run(
        [
            str(trace_binary),
            str(bundle),
            str(runtime),
            str(prefix),
            case["prompt"],
            str(case["max_new_tokens"]),
        ],
        check=True,
    )
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    assert metadata["request_replay"]
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    ids = tokenizer.encode(case["prompt"], add_special_tokens=False).ids
    assert ids == metadata["prompt_ids"], "native tokenization must match upstream exactly"
    actual = np.fromfile(prefix.with_suffix(".f32"), dtype=np.float32).reshape(
        metadata["rows"], metadata["columns"]
    )
    assert np.isfinite(actual).all()
    torch.set_num_threads(8)
    reference = Reference(checkpoint)
    # Teacher forcing uses exactly the native prefix at every compared step.
    tokens = ids + metadata["generated_ids"][:-1]
    expected = np.stack(
        [row for i, row in enumerate(reference.logits_for_tokens(tokens)) if i >= len(ids) - 1]
    )
    assert actual.shape == expected.shape
    relative = np.linalg.norm(actual - expected, axis=1) / np.maximum(
        np.linalg.norm(expected, axis=1), 1e-12
    )
    cosine = np.sum(actual * expected, axis=1) / np.maximum(
        np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1), 1e-12
    )
    top1 = float(np.mean(actual.argmax(-1) == expected.argmax(-1)))
    evidence = {
        "checkpoint": manifest["hf_id"],
        "revision": manifest["hf_revision"],
        "relative_l2": relative.tolist(),
        "cosine_similarity": cosine.tolist(),
        "top1_agreement": top1,
        "native": metadata,
    }
    (tmp_path / "comparison.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)
    assert float(relative.max()) < thresholds["max_relative_l2"]
    assert float(cosine.min()) > thresholds["min_cosine_similarity"]
    assert top1 >= thresholds["min_top1_agreement"]
