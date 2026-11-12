# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from families.pp_doclayout_v3.model import build, resolve
from tensorrt_model_connect import BuildRequest


@pytest.mark.parametrize(
    "options",
    [
        {"precision": "fp16"},
        {"max_batch_size": 2},
        {"quantization": "fp8"},
        {"task": "object_detection"},
        {"backend": "trt_rtx"},
        {"image_width": 640},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
    ],
)
def test_unsupported_requests_fail_before_reading_weights(options):
    request = {
        "model_dir": Path("missing"),
        "output_path": Path("unused.bundle"),
        "family": "pp_doclayout_v3",
        "task": "document_layout",
        "precision": "fp32",
    }
    with pytest.raises((ValueError, NotImplementedError)):
        build(BuildRequest(**{**request, **options}), None)


def test_changed_mask_architecture_cannot_silently_use_default_graph(tmp_path):
    config = {"model_type": "pp_doclayout_v3", "mask_enhanced": False}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(NotImplementedError, match="mask_enhanced"):
        resolve(tmp_path)
