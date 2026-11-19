# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the original eager V-JEPA2 oracle in its declared interpreter."""

import json
import os
from pathlib import Path
import subprocess
import sys


def run_reference(model_dir, video, work, *, inference, reference_environment):
    import numpy as np

    from tools.ci.environment import prepared_python
    from tools.e2e_evidence import record_evidence

    if reference_environment != "isolated":
        raise ValueError("This reference requires its declared isolated environment")
    source = Path(__file__).resolve().parents[4]
    python = prepared_python(source, "vjepa2", dict(os.environ))
    work.mkdir()
    np.save(work / "video.npy", video, allow_pickle=False)
    job = work / "request.json"
    job.write_text(json.dumps({"model_dir": str(model_dir), "inference": inference}))
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
        return {key: archive[key] for key in archive.files}, metadata


def main():
    import numpy as np
    import torch
    import transformers
    from transformers import AutoVideoProcessor, VJEPA2Model

    assert transformers.__version__ == "5.20.0"
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    work = job_path.parent
    video = np.load(work / "video.npy", allow_pickle=False)
    processor = AutoVideoProcessor.from_pretrained(job["model_dir"])
    inputs = processor(
        videos=[video],
        input_data_format="channels_last",
        do_sample_frames=False,
        return_tensors="pt",
    )
    metadata = {
        "python": sys.executable,
        "transformers": transformers.__version__,
        "pixel_shape": list(inputs.pixel_values_videos.shape),
    }
    if job["inference"]:
        assert inputs.pixel_values_videos.shape == (1, 64, 3, 256, 256)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        reference = (
            VJEPA2Model.from_pretrained(
                job["model_dir"],
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .eval()
            .to("cuda")
        )
        with torch.no_grad():
            features = reference(**inputs.to("cuda"), skip_predictor=True).last_hidden_state[0]
        np.savez_compressed(work / "result.npz", features=features.cpu().numpy())
    else:
        np.savez_compressed(work / "result.npz", pixels=inputs.pixel_values_videos[0, 0].numpy())
    (work / "environment.json").write_text(json.dumps(metadata))


if __name__ == "__main__":
    main()
