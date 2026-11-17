# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from families.flux3.support import describe
from tensorrt_model_connect.model_support import ModelMetadata, resolve_family

# Root config.json of black-forest-labs/flux-3-image-klein (revision 09561231), abridged to the identity fields.
_PACKAGE = {
    "format_version": 1,
    "model": "flux-3-image-klein",
    "quantization": None,
    "torch_dtype": "bfloat16",
    "dit": {"hidden": 3072, "depth": 5, "depth_single": 28, "flow_map": True},
    "sampler": {"steps": 16, "guidance": None, "alpha": 6.93, "size": [1024, 1024]},
}
_FILES = (
    "config.json",
    "modular_model_index.json",
    "text_encoder/config.json",
    "text_encoder/tokenizer.json",
    "transformer/config.json",
    "vae/config.json",
)


def test_flux3_klein_resolves_to_text_to_image_in_bf16() -> None:
    family, support = resolve_family(ModelMetadata(_PACKAGE, {}, _FILES))

    assert family == "flux3"
    assert support.tasks == ("text_to_image",)
    assert support.default_task == "text_to_image"
    assert support.default_precision == "bf16"


@pytest.mark.parametrize(
    ("package", "files"),
    [
        ({**_PACKAGE, "dit": {**_PACKAGE["dit"], "flow_map": False}}, _FILES),
        ({**_PACKAGE, "model": "flux-3-image-klein-base"}, _FILES),
        ({**_PACKAGE, "format_version": 2}, _FILES),
        (_PACKAGE, tuple(name for name in _FILES if name != "modular_model_index.json")),
    ],
    ids=["not-flow-map", "other-model", "other-format", "missing-sentinel"],
)
def test_other_packages_are_not_flux3(package, files) -> None:
    assert describe(ModelMetadata(package, {}, files)) is None


def test_text_encoder_config_is_not_claimed_by_flux3() -> None:
    family, _ = resolve_family(ModelMetadata({"model_type": "qwen3_vl"}, {}, ("config.json",)))

    assert family != "flux3"
