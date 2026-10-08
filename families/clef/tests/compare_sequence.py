# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check changing requests through one loaded native Task and original model."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from families.clef.tests.media_fixtures import reference_record


def main():
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "bundle", "runtime-root", "probe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", choices=("clef", "clef-flash"), required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    manifest = json.loads((root / "manifests" / (args.model + ".json")).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.checkpoint))
    from joint_schema_model import (
        collate_records,
        encode_record,
        load_release_model,
        systemone_answer,
    )

    model, processor = load_release_model(args.checkpoint, device="cpu")
    model = model.cuda()
    expected, documents, paths = [], [], []
    with torch.inference_mode():
        for case in manifest["testcases"]:
            inputs = case["inputs"]
            document = json.loads((root / inputs["document_path"]).read_text())
            if inputs.get("image_paths"):
                document["images"] = [str(root / name) for name in inputs["image_paths"]]
            if inputs.get("video_frame_paths"):
                document["videos"] = [
                    [str(root / name) for name in frames] for frames in inputs["video_frame_paths"]
                ]
            path = args.output / (case["name"] + ".json")
            path.write_text(json.dumps(document))
            record = reference_record(path)
            encoded = encode_record(processor.tokenizer, record, processor=processor)
            batch = collate_records(
                [encoded], processor.tokenizer.pad_token_id, torch.device("cuda")
            )
            logits = model(batch)[0]
            expected.append(
                {
                    "name": case["name"],
                    "tokens": len(encoded.input_ids),
                    "scores": [
                        {
                            "id": question.question_id,
                            "options": list(question.option_ids),
                            "logits": values.float().cpu().numpy(),
                        }
                        for question, values in zip(encoded.questions, logits, strict=True)
                    ],
                }
            )
            documents.append(document)
            paths.append(str(path))
    del model, batch, logits
    torch.cuda.empty_cache()
    (args.output / "reference.json").write_text(
        json.dumps(expected, default=lambda value: value.tolist(), indent=2)
    )
    native = json.loads(
        subprocess.check_output(
            [
                str(args.probe),
                str(args.bundle),
                str(args.runtime_root),
                "--sequence",
                *paths,
                *paths,
            ],
            text=True,
        )
    )
    (args.output / "native.json").write_text(json.dumps(native, indent=2))
    assert not any(
        any(token in name.lower() for token in ("python", "torch", "c10"))
        for name in native["loaded_libraries"]
    )
    count = len(expected)
    assert len(native["results"]) == 2 * count
    receipts = []
    for index, (reference, document) in enumerate(zip(expected, documents, strict=True)):
        actual = native["results"][index]
        repeated = native["results"][index + count]
        assert actual["scores"] == repeated["scores"]
        assert actual["response"] == repeated["response"]
        assert actual["response"]["usage"] == {
            "input_tokens": reference["tokens"],
            "output_tokens": 0,
        }
        errors = []
        for wanted, score in zip(reference["scores"], actual["scores"], strict=True):
            assert score["question_id"] == wanted["id"]
            assert score["option_ids"] == wanted["options"]
            probabilities = torch.tensor(wanted["logits"]).softmax(-1).numpy()
            label = reference["name"] + ":" + wanted["id"]
            np.testing.assert_allclose(
                score["logits"], wanted["logits"], atol=0.125, rtol=0.015, err_msg=label
            )
            np.testing.assert_allclose(
                score["probabilities"], probabilities, atol=0.002, rtol=0.01, err_msg=label
            )
            assert np.argmax(score["probabilities"]) == np.argmax(probabilities)
            assert actual["response"]["answers"][wanted["id"]] == systemone_answer(
                document["questions"][wanted["id"]],
                dict(zip(wanted["options"], score["probabilities"], strict=True)),
            )
            errors.append(float(np.max(np.abs(np.asarray(score["probabilities"]) - probabilities))))
        receipt = {
            "case": reference["name"],
            "max_probability_error": max(errors),
            "repeated_exactly": True,
        }
        print(json.dumps(receipt), flush=True)
        receipts.append(receipt)
    (args.output / "comparison.json").write_text(
        json.dumps(
            {
                "checkpoint": manifest["hf_id"],
                "revision": manifest["hf_revision"],
                "results": receipts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
