# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.nemotron_voicechat.native_codec import (
    CAUSAL_CACHE_WIDTH,
    CODEC_ENGINE_SECTION,
    CodecArchitecture,
    VOICECHAT_CODEC,
    cache_input_name,
    cache_output_name,
    dequantize_codes_numpy,
    expected_weight_shapes,
    validate_codec_weights,
)


def _tiny_architecture() -> CodecArchitecture:
    return CodecArchitecture(
        num_quantizers=2,
        codebook_size=4,
        latent_size=3,
        base_hidden_size=2,
        channel_mult=(2,),
        rates=(2,),
        num_blocks=1,
        convnext_kernel_size=7,
        n_fft=4,
        hop_length=2,
        samples_per_codec_frame=4,
    )


def _zero_weights(architecture: CodecArchitecture) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in expected_weight_shapes(architecture).items()
    }


def test_public_codec_contract_is_one_80ms_frame() -> None:
    assert VOICECHAT_CODEC.num_quantizers == 31
    assert VOICECHAT_CODEC.codebook_size == 1024
    assert VOICECHAT_CODEC.latent_size == 512
    assert VOICECHAT_CODEC.rates == (9, 7, 7)
    assert VOICECHAT_CODEC.spectral_frames_per_codec_frame == 441
    assert VOICECHAT_CODEC.spectral_channels == 18
    assert VOICECHAT_CODEC.samples_per_codec_frame == 1764
    assert VOICECHAT_CODEC.samples_per_codec_frame / 22050 == pytest.approx(0.08)
    assert VOICECHAT_CODEC.cache_channels == (
        1536,
        1536,
        1536,
        768,
        768,
        768,
        384,
        384,
        384,
    )
    assert CAUSAL_CACHE_WIDTH == 6
    assert CODEC_ENGINE_SECTION == "codec_engine_plan"
    assert cache_input_name(8) == "codec_cache_in_8"
    assert cache_output_name(8) == "codec_cache_out_8"


def test_expected_shapes_match_exact_checkpoint_layer_numbering() -> None:
    shapes = expected_weight_shapes()
    assert len(shapes) == 107
    assert shapes["prvq.mus_list.0"] == (1024, 512)
    assert shapes["prvq.mus_list.30"] == (1024, 512)
    assert shapes["decoder.layers.0.weight"] == (512, 1536, 9)
    assert shapes["decoder.layers.1.dwconv.weight"] == (1536, 1, 7)
    assert shapes["decoder.layers.4.weight"] == (1536, 768, 7)
    assert shapes["decoder.layers.8.weight"] == (768, 384, 7)
    assert shapes["decoder.layers.12.weight"] == (18, 384, 1)


def test_rvq_dequantization_adds_each_independent_codebook() -> None:
    architecture = _tiny_architecture()
    weights = _zero_weights(architecture)
    weights["prvq.mus_list.0"][:] = np.asarray(
        [[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]],
        dtype=np.float32,
    )
    weights["prvq.mus_list.1"][:] = np.asarray(
        [[100, 101, 102], [200, 201, 202], [300, 301, 302], [400, 401, 402]],
        dtype=np.float32,
    )

    codes = np.asarray([[1, 3], [2, 0]], dtype=np.int32)
    actual = dequantize_codes_numpy(codes, weights, architecture)

    np.testing.assert_array_equal(
        actual,
        np.asarray([[410, 412, 414], [120, 122, 124]], dtype=np.float32),
    )


def test_weight_validation_rejects_missing_or_wrong_checkpoint_tensor() -> None:
    architecture = _tiny_architecture()
    weights = _zero_weights(architecture)
    del weights["decoder.layers.2.weight"]
    with pytest.raises(KeyError, match="decoder.layers.2.weight"):
        validate_codec_weights(weights, architecture)

    weights = _zero_weights(architecture)
    weights["decoder.layers.0.weight"] = np.zeros((1,), dtype=np.float32)
    with pytest.raises(ValueError, match="decoder.layers.0.weight"):
        validate_codec_weights(weights, architecture)


def test_code_validation_rejects_non_integer_and_out_of_range_ids() -> None:
    architecture = _tiny_architecture()
    weights = _zero_weights(architecture)
    with pytest.raises(TypeError, match="must be integers"):
        dequantize_codes_numpy(np.zeros((1, 2), dtype=np.float32), weights, architecture)
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        dequantize_codes_numpy(np.asarray([[0, 4]], dtype=np.int32), weights, architecture)


def test_native_codec_has_no_framework_or_export_runtime_dependency() -> None:
    source = (Path(__file__).parents[1] / "native_codec.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import nemo" not in source
    assert "import onnx" not in source
    assert "NetworkDefinitionCreationFlag.STRONGLY_TYPED" in source
