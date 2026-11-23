# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint precision and native task contracts."""

import numpy as np
import pytest


def test_weight_only_fp8_dequantization_preserves_bf16_rounding():
    import torch
    from torch import nn
    from ideogram4.quantized_loading import Fp8Linear
    from families.ideogram4.graphs import materialize_fp8

    layer = Fp8Linear(8, 4, bias=True, compute_dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(-16, 16).reshape(4, 8).to(torch.float8_e4m3fn))
        layer.weight_scale.copy_(torch.tensor([0.0173, 0.0127, 0.0321, 0.0024]))
        layer.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.bfloat16))
    values = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 17
    with torch.inference_mode():
        reference = layer(values).float().numpy()
        built = materialize_fp8(nn.Sequential(layer))(values).float().numpy()
    np.testing.assert_array_equal(built, reference)


@pytest.mark.parametrize("value", [0, -1, 255, 257])
def test_non_image_profiles_are_rejected_before_loading_weights(value, tmp_path):
    import json
    from tensorrt_model_connect import BuildRequest
    from families.ideogram4.model import build

    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "Ideogram4Pipeline"}))
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder/config.json").write_text(
        json.dumps({"model_type": "qwen3_vl", "ideogram_fp8_weight_only": True})
    )
    with pytest.raises(ValueError):
        request = BuildRequest(
            model_dir=tmp_path,
            output_path=tmp_path / "model.bundle",
            family="ideogram4",
            task="text_to_image",
            precision="bf16",
            image_height=value,
            image_width=512,
        )
        build(request, None)
