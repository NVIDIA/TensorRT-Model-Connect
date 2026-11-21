# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full checkpoint -> bundle -> native Task -> independent logit oracle."""

import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest

from families.qwen3_next.tests.checkpoint import assert_bundle_checkpoint, assert_pinned_checkpoint


def test_qwen3_next_bf16_logits(request, tmp_path):
    directory = Path(__file__).parent
    manifest = json.loads((directory / "manifests/qwen3-next-bf16.json").read_text())
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
    if selected and not selected.intersection({"qwen3_next", manifest["name"], case["name"]}):
        pytest.skip("Qwen3Next was not selected")
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("requires explicit Qwen3Next E2E selection")
    from tokenizers import Tokenizer
    import torch
    from tensorrt_model_connect import BuildRequest, build
    from tensorrt_model_connect.model_support import load_model_metadata, resolve_family
    from families.qwen3_next.tests.reference import load_full_model

    assert manifest["trust_remote_code"] is False
    assert case["reference_precision"] == manifest["precision"]
    thresholds = json.loads((directory / "thresholds" / (case["name"] + ".json")).read_text())[
        "threshold_overrides"
    ]
    checkpoint_value = os.environ.get("TRTMC_QWEN3_NEXT_CHECKPOINT")
    if checkpoint_value:
        checkpoint = Path(checkpoint_value)
    else:
        from huggingface_hub import snapshot_download

        checkpoint = Path(
            snapshot_download(repo_id=manifest["hf_id"], revision=manifest["hf_revision"])
        )
    assert_pinned_checkpoint(checkpoint, manifest)
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    binary = Path(os.environ["TRTMC_BINARY"])
    assert binary.is_file()
    assert (runtime / "libtrtmc_model_qwen3_next.so").is_file()
    family, support = resolve_family(load_model_metadata(checkpoint))
    assert family == "qwen3_next" and support.default_task == manifest["task"]
    # An explicitly supplied prebuilt artifact is validated below through its
    # provenance. Otherwise this direct E2E owns the complete public build call.
    supplied = os.environ.get("TRTMC_QWEN3_NEXT_BUNDLE")
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
    assert bundle.is_file() and bundle.stat().st_size > 150_000_000_000
    inspected = subprocess.run(
        [str(binary), "inspect", str(bundle)], check=True, capture_output=True, text=True
    )
    info = json.loads(inspected.stdout)
    assert info["family"] == "qwen3_next" and info["task"] == "text_generation"
    assert info["provenance"]["build"]["precision"] == manifest["precision"]
    assert info["provenance"]["build"]["max_sequence_length"] == manifest["max_sequence_length"]
    assert info["provenance"]["build"]["tensor_parallel_size"] == manifest["tensor_parallel_size"]
    if supplied:
        assert_bundle_checkpoint(info, manifest)
    trace_binary = runtime / "qwen3_next_logits_trace"
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
    reference = load_full_model(checkpoint)
    rows = []
    prefix_ids = torch.tensor([ids], dtype=torch.long)
    cache = None
    with torch.inference_mode():
        for i in range(len(metadata["generated_ids"])):
            output = reference(input_ids=prefix_ids, past_key_values=cache, use_cache=True)
            rows.append(output.logits[0, -1].float().numpy().copy())
            cache = output.past_key_values
            prefix_ids = torch.tensor([[metadata["generated_ids"][i]]], dtype=torch.long)
    expected = np.stack(rows)
    assert tokenizer.decode(metadata["generated_ids"], skip_special_tokens=True) == metadata["text"]
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
        "reference_top1_ids": expected.argmax(-1).tolist(),
        "prompt": case["prompt"],
        "native": metadata,
    }
    (tmp_path / "comparison.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)
    assert float(relative.max()) < thresholds["max_relative_l2"]
    assert float(cosine.min()) > thresholds["min_cosine_similarity"]
    assert top1 >= thresholds["min_top1_agreement"]
