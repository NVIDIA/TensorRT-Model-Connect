# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generation controls must satisfy the native loader before engine compilation."""
import json
from pathlib import Path

import pytest

from families.diffusion_gemma.config import Config
from families.diffusion_gemma.model import _generation_config, build
from tensorrt_model_connect import BuildRequest

DATA = Path(__file__).parent / "data"


def documents():
    return (json.loads((DATA / "generation_config.json").read_text()),
            Config.from_dict(json.loads((DATA / "diffusiongemma-26b-a4b-config.json").read_text())))


def validate(tmp_path, document, config):
    (tmp_path / "generation_config.json").write_text(json.dumps(document))
    return _generation_config(tmp_path, config)


@pytest.mark.parametrize("field", ["max_new_tokens", "max_denoising_steps", "stability_threshold",
    "confidence_threshold", "t_min", "t_max", "sampler_config.entropy_bound"])
def test_missing_required_generation_control_is_rejected(tmp_path, field):
    document, config = documents()
    target, key = (document["sampler_config"], "entropy_bound") if "." in field else (document, field)
    del target[key]
    with pytest.raises(ValueError, match="diffusion_gemma"):
        validate(tmp_path, document, config)


@pytest.mark.parametrize(("field", "value"), [
    ("max_new_tokens", 0), ("max_new_tokens", True), ("max_new_tokens", 2**31),
    ("max_denoising_steps", -1), ("max_denoising_steps", 1.5),
    ("stability_threshold", -1), ("stability_threshold", 2**31),
    ("confidence_threshold", -0.1), ("confidence_threshold", float("nan")),
    ("confidence_threshold", 1e100), ("t_min", 0), ("t_min", 1e-50),
    ("t_min", 1.0), ("t_max", 0.39), ("t_max", float("inf")),
    ("t_max", True), ("sampler_config.entropy_bound", -1),
    ("sampler_config.entropy_bound", float("nan")), ("sampler_config", None),
])
def test_invalid_generation_controls_are_rejected(tmp_path, field, value):
    document, config = documents()
    target, key = (document["sampler_config"], "entropy_bound") if "." in field else (document, field)
    target[key] = value
    with pytest.raises(ValueError, match="diffusion_gemma"):
        validate(tmp_path, document, config)


def test_valid_controls_keep_bundle_field_values(tmp_path):
    document, config = documents()
    assert validate(tmp_path, document, config) == document
    document["stability_threshold"] = document["confidence_threshold"] = 0
    document["sampler_config"]["entropy_bound"] = 0
    assert validate(tmp_path, document, config) == document


def test_generation_validation_precedes_weights_and_compilation(tmp_path):
    document, _ = documents()
    del document["t_min"]
    (tmp_path / "config.json").write_bytes((DATA / "diffusiongemma-26b-a4b-config.json").read_bytes())
    (tmp_path / "generation_config.json").write_text(json.dumps(document))
    request = BuildRequest(tmp_path, tmp_path / "out.bundle", "diffusion_gemma", "text_continuation", "fp32")
    with pytest.raises(ValueError, match="t_min"):
        build(request, None)
