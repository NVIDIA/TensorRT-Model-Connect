# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clef checkpoint -> bundle -> C++ Task -> original per-option probabilities."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parent
MANIFEST = json.loads((ROOT / "manifests/clef.json").read_text())


@pytest.fixture(scope="module")
def clef_bundle(tmp_path_factory):
    """All cases exercise one freshly built checkpoint/bundle unless explicitly supplied."""
    # Clef's graph and C++ pipeline execute the complete model on one device.
    assert MANIFEST["tensor_parallel_size"] == 1
    checkpoint = os.environ.get("TRTMC_CLEF_CHECKPOINT")
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download(MANIFEST["hf_id"], revision=MANIFEST["hf_revision"])
    checkpoint = Path(checkpoint)
    bundle_value = os.environ.get("TRTMC_CLEF_BUNDLE")
    bundle = (
        Path(bundle_value)
        if bundle_value
        else tmp_path_factory.mktemp("clef-bundle") / MANIFEST["bundle"]
    )
    if bundle_value is None:
        from families.clef.cli import build

        build(
            model=str(checkpoint),
            output=bundle,
            max_sequence_length=MANIFEST["max_sequence_length"],
        )
    assert bundle.is_file()
    return checkpoint, bundle


@pytest.mark.parametrize("case", MANIFEST["testcases"], ids=lambda case: case["name"])
def test_e2e(case, request, tmp_path):
    assert MANIFEST["precision"] == "bf16"
    selected = set(request.config.getoption("--e2e-model", default=[]) or [])
    selected.update(request.config.getoption("--e2e-testcase", default=[]) or [])
    if not selected and os.environ.get("TRTMC_E2E") != "1":
        pytest.skip("real Clef E2E requires explicit selection")
    if selected and not selected.intersection({"clef", case["name"]}):
        pytest.skip("Clef case was not selected")
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    build_root = Path(os.environ.get("TRTMC_NATIVE_BUILD_DIR", runtime))
    binary = build_root / "clef_task_probe"
    assert binary.is_file()
    checkpoint, bundle = request.getfixturevalue("clef_bundle")
    sys.path.insert(0, str(checkpoint))
    from joint_schema_model import (
        collate_records,
        encode_record,
        load_release_model,
        systemone_answer,
    )
    from families.clef.tests.media_fixtures import reference_record
    import torch

    fixture = ROOT / case["inputs"]["document_path"]
    inputs = case["inputs"]
    if inputs.get("image_paths") or inputs.get("video_frame_paths"):
        document = json.loads(fixture.read_text())
        if inputs.get("image_paths"):
            document["images"] = [str(ROOT / path) for path in inputs["image_paths"]]
        if inputs.get("video_frame_paths"):
            document["videos"] = [
                [str(ROOT / path) for path in frames] for frames in inputs["video_frame_paths"]
            ]
        fixture = tmp_path / "record.json"
        fixture.write_text(json.dumps(document))
    record = reference_record(fixture)
    # Loading on CPU then moving the unchanged BF16 module avoids Transformers'
    # single 51 GiB allocator-warmup reservation. The complete reference forward
    # still runs on CUDA with the release's original operators and weights.
    model, processor = load_release_model(checkpoint, device="cpu")
    model = model.to("cuda")
    encoded = encode_record(processor.tokenizer, record, processor=processor)
    batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
    with torch.inference_mode():
        logits = [v.float().cpu().numpy() for v in model(batch)[0]]
    del model, batch
    torch.cuda.empty_cache()
    command = [str(binary), str(bundle), str(runtime), str(fixture), "2"]
    native = json.loads(subprocess.check_output(command, text=True))
    assert not any(
        "python" in name.lower() or "torch" in name.lower() or "c10" in name.lower()
        for name in native["loaded_libraries"]
    )
    first = native["results"][0]
    assert first["scores"] == native["results"][1]["scores"]
    assert first["response"]["usage"] == {
        "input_tokens": len(encoded.input_ids),
        "output_tokens": 0,
    }
    for question, reference, actual in zip(encoded.questions, logits, first["scores"], strict=True):
        assert actual["question_id"] == question.question_id
        assert actual["option_ids"] == list(question.option_ids)
        probabilities = torch.tensor(reference).softmax(-1).numpy()
        np.testing.assert_allclose(actual["logits"], reference, atol=0.125, rtol=0.015)
        np.testing.assert_allclose(actual["probabilities"], probabilities, atol=0.002, rtol=0.01)
        assert int(np.argmax(actual["probabilities"])) == int(np.argmax(probabilities))
        assert first["response"]["answers"][question.question_id] == systemone_answer(
            record["questions"][question.question_id],
            dict(zip(question.option_ids, actual["probabilities"], strict=True)),
        )
    (tmp_path / "native.json").write_text(json.dumps(native, indent=2))
