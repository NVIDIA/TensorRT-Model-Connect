# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional independent vLLM replay; never updates the checked-in oracle itself."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .test_e2e import REFERENCE, make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve the previous independent result")
    from PIL import Image
    import vllm
    from vllm import LLM, SamplingParams

    assert vllm.__version__ == "0.20.0", "replay the pinned independent reference version"
    llm = LLM(model=str(args.model_dir), **REFERENCE["reference_arguments"])
    sampling = SamplingParams(**REFERENCE["sampling"])
    output = {
        "checkpoint": {
            "path": str(args.model_dir.resolve()),
            "identity_status": "unverified_local_directory",
        },
        "expected_checkpoint": {
            "model": REFERENCE["model"],
            "revision": REFERENCE["revision"],
        },
        "reference_backend": f"vllm=={vllm.__version__}",
        "reference_arguments": REFERENCE["reference_arguments"],
        "sampling": REFERENCE["sampling"],
        "chat_template": REFERENCE["chat_template"],
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="omni-reference-", dir=args.output.parent) as directory:
        requests, _ = make_inputs([row["id"] for row in REFERENCE["cases"]], Path(directory))
        for request in requests:
            media, content = {}, []
            for kind in ("image", "video", "audio"):
                if kind not in request:
                    continue
                content.append({"type": kind + "_url", kind + "_url": {"url": "unused"}})
                spec = request[kind]
                if kind == "image":
                    media[kind] = Image.fromarray(
                        np.fromfile(spec["path"], np.uint8).reshape(
                            spec["height"], spec["width"], 3
                        )
                    )
                elif kind == "audio":
                    media[kind] = (np.fromfile(spec["path"], "<f4"), spec["sample_rate"])
                else:
                    frames = np.stack(
                        [
                            np.fromfile(f["path"], np.uint8).reshape(f["height"], f["width"], 3)
                            for f in spec["frames"]
                        ]
                    )
                    original = REFERENCE["video"]
                    media[kind] = (
                        frames,
                        {
                            "fps": original["original_fps"],
                            "frames_indices": original["frame_indices"],
                            "total_num_frames": original["original_frame_count"],
                            "duration": original["original_frame_count"] / original["original_fps"],
                            "do_sample_frames": False,
                            "video_backend": "opencv",
                        },
                    )
            content.append({"type": "text", "text": request["prompt"]})
            prompt = llm.get_tokenizer().apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, **REFERENCE["chat_template"]
            )
            payload = {"prompt": prompt}
            if media:
                payload["multi_modal_data"] = media
            result = llm.generate(payload, sampling_params=sampling, use_tqdm=False)[0].outputs[0]
            output["cases"].append(
                {
                    "id": request["id"],
                    "text": result.text,
                    "token_ids": list(result.token_ids),
                    "finish_reason": result.finish_reason,
                }
            )
    with args.output.open("x") as destination:
        destination.write(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
