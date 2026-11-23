# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the document oracle in its original isolated Transformers environment."""

import json
import os
from pathlib import Path
import subprocess
import sys


def run_reference(model_dir, pixels, words, boxes, length, work, *, reference_environment):
    import numpy as np

    from tools.ci.environment import prepared_python
    from tools.e2e_evidence import record_evidence

    if reference_environment != "isolated":
        raise ValueError("This reference requires its declared isolated environment")
    source = Path(__file__).resolve().parents[4]
    python = prepared_python(source, "layoutlmv3", dict(os.environ))
    work.mkdir()
    np.save(work / "pixels.npy", pixels, allow_pickle=False)
    job = work / "request.json"
    job.write_text(
        json.dumps({"model_dir": str(model_dir), "words": words, "boxes": boxes, "length": length})
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    completed = subprocess.run(
        [str(python), str(Path(__file__).resolve()), str(job)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    record_evidence(
        "reference_process",
        {
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        },
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    metadata = json.loads((work / "environment.json").read_text())
    assert metadata["transformers"] == "5.20.0"
    record_evidence("reference_environment", {"requested": reference_environment, **metadata})
    with np.load(work / "result.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    import numpy as np
    import torch
    import transformers
    from transformers import LayoutLMv3ImageProcessor, LayoutLMv3Model, LayoutLMv3Tokenizer

    assert transformers.__version__ == "5.20.0"
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    work = job_path.parent
    tokenizer = LayoutLMv3Tokenizer.from_pretrained(job["model_dir"])
    processor = LayoutLMv3ImageProcessor.from_pretrained(job["model_dir"], apply_ocr=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    reference = (
        LayoutLMv3Model.from_pretrained(job["model_dir"], dtype=torch.float32).eval().to("cuda")
    )
    text = tokenizer(
        job["words"],
        boxes=job["boxes"],
        padding="max_length",
        max_length=job["length"],
        return_tensors="pt",
    )
    pixels = np.load(work / "pixels.npy", allow_pickle=False)
    visual = processor(
        images=pixels, input_data_format="channels_last", return_tensors="pt"
    ).pixel_values
    words = [
        word if word is not None else (-1 if text.attention_mask[0, i] else -2)
        for i, word in enumerate(text.word_ids())
    ]
    result = {
        "input_ids": text.input_ids.numpy(),
        "attention_mask": text.attention_mask.numpy(),
        "bbox": text.bbox.numpy(),
        "word_indices": np.asarray(words),
        "pixels": visual[0].numpy(),
    }
    all_boxes = torch.cat([text.bbox[0], reference.visual_bbox.cpu()], dim=0)
    for name, axis in (("relative_x", 0), ("relative_y", 3)):
        coordinates = all_boxes[:, axis].cuda()
        result[name] = (
            reference.encoder.relative_position_bucket(
                coordinates[None, :] - coordinates[:, None],
                num_buckets=reference.config.rel_2d_pos_bins,
                max_distance=reference.config.max_rel_2d_pos,
            )
            .cpu()
            .numpy()
        )
    with torch.no_grad():
        result["features"] = (
            reference(
                **text.to("cuda"),
                pixel_values=visual.to("cuda"),
            )
            .last_hidden_state[0, : job["length"]]
            .cpu()
            .numpy()
        )
    np.savez_compressed(work / "result.npz", **result)
    (work / "environment.json").write_text(
        json.dumps({"python": sys.executable, "transformers": transformers.__version__})
    )


if __name__ == "__main__":
    main()
