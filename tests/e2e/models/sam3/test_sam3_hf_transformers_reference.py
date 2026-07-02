# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3-owned Hugging Face reference tests."""

from __future__ import annotations

import json
import subprocess

from tests.e2e.models.sam3.e2e_plugins import reference as sam3_reference
from tests.e2e.models.sam3.e2e_plugins.references import hf_transformers as sam3_hf_base
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_reference_uses_cached_model_ref(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        import numpy as np

        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        case_dir = tmp_path / "sam3"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "hf_sam3.json").write_text(
            json.dumps(
                {
                    "text_prompt": "ear",
                    "scores": [0.9],
                    "boxes": [[1.0, 2.0, 3.0, 4.0]],
                    "num_masks": 1,
                }
            ),
            encoding="utf-8",
        )
        np.save(case_dir / "hf_sam3_masks.npy", np.zeros((1, 2, 2), dtype=np.uint8))
        (case_dir / "hf_sam3_segmented.png").write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK masks=1\n", stderr="")

    monkeypatch.setattr(
        sam3_reference,
        "_resolve_cached_model_ref",
        lambda hf_id: "/cached/sam3",
    )
    monkeypatch.setattr(sam3_hf_base.subprocess, "run", _fake_run)

    case = E2ECase(
        name="sam3",
        hf_id="facebook/sam3",
        family="sam3",
        runtime_strategy="sam3_prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam3",
        inputs={"image": "data/test_img.jpeg", "prompt": "ear"},
        metadata={"trust_remote_code": False, "precision": "fp32"},
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/ref/python",
    )

    out = sam3_reference.Sam3HfTransformersReference().run_stage(
        case, StageSpec(name="full_inference"), ctx
    )

    cmd = captured["cmd"]
    assert cmd[:2] == ["/ref/python", "-c"]
    script = cmd[2]
    assert "model_ref = '/cached/sam3'" in script
    assert "def _load_sam3_processor(" in script
    assert "Sam3Processor.from_pretrained(" in script
    assert "except Exception as processor_error:" in script
    assert '"target_size": 1008' in script
    assert "Sam3ImageProcessorFast(**image_processor_kwargs)" in script
    assert "AutoTokenizer.from_pretrained(" in script
    assert "processor = _load_sam3_processor(model_ref, trust_remote_code)" in script
    assert "model_ref, trust_remote_code=trust_remote_code" in script
    assert "Sam3Model.from_pretrained(" in script
    assert "model_ref, torch_dtype=torch.float32" in script
    assert out.data["text_prompt"] == "ear"
    assert out.data["masks_path"].endswith("hf_sam3_masks.npy")
    assert out.metadata["returncode"] == 0
