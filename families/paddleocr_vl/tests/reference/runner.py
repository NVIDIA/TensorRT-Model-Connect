# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged PaddleOCR processor and greedy oracle in isolation."""

import json
import os
from pathlib import Path
import subprocess
import sys


def run_reference(model_dir, work, *, reference_environment, mode, **inputs):
    import numpy as np

    from tools.ci.environment import prepared_python
    from tools.e2e_evidence import record_evidence

    if reference_environment != "isolated":
        raise ValueError("PaddleOCR reference requires its declared isolated environment")
    python = prepared_python(Path(__file__).resolve().parents[4], "paddleocr_vl", dict(os.environ))
    work.mkdir()
    request = work / "request.json"
    request.write_text(json.dumps({"model_dir": str(model_dir), "mode": mode, **inputs}))
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    completed = subprocess.run(
        [str(python), str(Path(__file__).resolve()), str(request)],
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
    result = json.loads((work / "result.json").read_text())
    assert result["environment"]["transformers"] == "5.20.0"
    record_evidence(
        "reference_environment",
        {
            "requested": reference_environment,
            **result["environment"],
        },
    )
    arrays = {}
    if (work / "arrays.npz").is_file():
        with np.load(work / "arrays.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    return arrays, result["values"]


def main():
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    assert transformers.__version__ == "5.20.0"
    request = Path(sys.argv[1])
    job = json.loads(request.read_text())
    work = request.parent
    config = AutoConfig.from_pretrained(job["model_dir"])
    processor = AutoProcessor.from_pretrained(job["model_dir"], use_fast=False)
    values = {}
    arrays = {}
    if job["mode"] == "metadata":
        marker = "<|IMAGE_PLACEHOLDER|>"
        sentinel = "TRTMC_QUESTION_PLACEHOLDER"
        framed = processor.apply_chat_template(
            [dict(role="user", content=[dict(type="image"), dict(type="text", text=sentinel)])],
            add_generation_prompt=True,
        )
        prefix, rest = framed.split(marker)
        between, suffix = rest.split(sentinel)
        values = dict(
            prefix=prefix,
            between=between,
            suffix=suffix,
            tokenizer_json=processor.tokenizer.backend_tokenizer.to_str(),
            image_id=config.image_token_id,
        )
    else:
        page = Image.open(job["image"]).convert("RGB")
        text = processor.apply_chat_template(
            [
                dict(
                    role="user",
                    content=[dict(type="image"), dict(type="text", text=job["question"])],
                )
            ],
            add_generation_prompt=True,
        )
        inputs = processor(text=text, images=page, return_tensors="pt")
        arrays["input_ids"] = inputs["input_ids"].numpy()
        if job["mode"] == "preprocess":
            with torch.device("meta"):
                position_oracle = AutoModelForImageTextToText.from_config(
                    config,
                    attn_implementation="eager",
                )
            positions, _ = position_oracle.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
            )
            arrays["positions"] = positions[:, 0].numpy()
            _, gh, gw = inputs["image_grid_thw"][0].tolist()
            arrays["pixels"] = (
                inputs["pixel_values"]
                .numpy()
                .reshape(
                    gh,
                    gw,
                    3,
                    14,
                    14,
                )
                .transpose(2, 0, 3, 1, 4)
                .reshape(3, gh * 14, gw * 14)
            )
        elif job["mode"] == "generate":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
            reference = (
                AutoModelForImageTextToText.from_pretrained(
                    job["model_dir"],
                    dtype=torch.float32,
                    attn_implementation="eager",
                )
                .eval()
                .to("cuda")
            )
            inputs = inputs.to("cuda")
            with torch.no_grad():
                generated = reference.generate(
                    **inputs, do_sample=False, max_new_tokens=job["max_new_tokens"]
                )
            token_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
            values = {
                "token_ids": token_ids,
                "text": processor.batch_decode([token_ids], skip_special_tokens=True)[0],
            }
        else:
            raise ValueError("Unsupported PaddleOCR reference operation")
    if arrays:
        np.savez_compressed(work / "arrays.npz", **arrays)
    (work / "result.json").write_text(
        json.dumps(
            {
                "values": values,
                "environment": {
                    "python": sys.executable,
                    "transformers": transformers.__version__,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
