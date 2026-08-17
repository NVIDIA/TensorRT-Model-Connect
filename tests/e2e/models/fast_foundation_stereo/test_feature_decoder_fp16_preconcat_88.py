# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


_ENV_88 = "TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_88"
_ENV_176 = "TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_176"
_TARGET_SHAPE = (2, 96, 88, 88)
_CONCAT_SHAPE = (2, 192, 88, 88)


class _Tensor:
    def __init__(self, name: str, shape: tuple[int, ...], dtype: str):
        self.name = name
        self.shape = shape
        self.dtype = dtype


def _decoder_block():
    return SimpleNamespace(
        conv1=SimpleNamespace(
            conv=SimpleNamespace(label="deconvolution"),
            IN=SimpleNamespace(label="instance-norm"),
            relu=SimpleNamespace(negative_slope=0.01),
        ),
        conv2=SimpleNamespace(label="residual-block"),
    )


def _run_decoder(
    monkeypatch,
    *,
    enabled_88: bool,
    enabled_176: bool = False,
    scope: str = "deconv16_8",
    fp16: bool = True,
    branch_shape: tuple[int, ...] = _TARGET_SHAPE,
    branch_dtype: str = "fp16",
    skip_shape: tuple[int, ...] = _TARGET_SHAPE,
    skip_dtype: str = "fp32",
    fail_on_concat: bool = False,
):
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    if enabled_88:
        monkeypatch.setenv(_ENV_88, "1")
    else:
        monkeypatch.setenv(_ENV_88, "0")
    if enabled_176:
        monkeypatch.setenv(_ENV_176, "1")
    else:
        monkeypatch.setenv(_ENV_176, "0")
    events = []

    def deconv2d(_graph, tensor, module, *, fp16):
        events.append(("deconv", tensor.shape, tensor.dtype, module.label, fp16))
        return _Tensor("branch", branch_shape, branch_dtype)

    def instance_norm(_graph, tensor, module, *, fp16):
        events.append(("norm", tensor.shape, tensor.dtype, module.label, fp16))
        return tensor

    def activation(_graph, tensor, kind, *, alpha=0.01):
        events.append(("activation", tensor.shape, tensor.dtype, kind, alpha))
        return tensor

    def cast(_graph, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        events.append(("cast", tensor.name, tensor.shape, tensor.dtype, dtype))
        return _Tensor(f"cast-{tensor.name}", tensor.shape, dtype)

    def concat(_graph, tensors, axis):
        if fail_on_concat:
            pytest.fail("branch metadata check must precede concat")
        tensors = tuple(tensors)
        events.append(
            (
                "concat",
                tuple(tensor.shape for tensor in tensors),
                tuple(tensor.dtype for tensor in tensors),
                axis,
            )
        )
        shape = list(tensors[0].shape)
        shape[axis] = sum(tensor.shape[axis] for tensor in tensors)
        return _Tensor("concat", tuple(shape), tensors[0].dtype)

    def residual(_graph, tensor, block, *, fp16):
        events.append(
            (
                "next-residual",
                tensor.shape,
                tensor.dtype,
                block.label,
                fp16,
            )
        )
        return tensor

    monkeypatch.setattr(native_feature, "_deconv2d", deconv2d)
    monkeypatch.setattr(native_feature, "_instance_norm_2d", instance_norm)
    monkeypatch.setattr(native_feature, "_activation", activation)
    monkeypatch.setattr(native_feature, "_cast", cast)
    monkeypatch.setattr(native_feature, "_concat", concat)
    monkeypatch.setattr(native_feature, "_resnet_instance_block", residual)

    graph = SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32"))
    output = native_feature._decoder_block(
        graph,
        _Tensor("input", (2, 320, 44, 44), "fp16" if fp16 else "fp32"),
        _Tensor("skip", skip_shape, skip_dtype),
        _decoder_block(),
        decoder_scope=scope,
        fp16=fp16,
    )
    return events, output


def test_decoder_fp16_preconcat_88_gate_is_default_on_scope_owned_and_independent(
    monkeypatch,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    scopes = ("deconv32_16", "deconv16_8", "deconv8_4")
    monkeypatch.delenv(_ENV_88, raising=False)
    monkeypatch.delenv(_ENV_176, raising=False)
    assert [
        native_feature._use_decoder_fp16_preconcat_88(decoder_scope=scope) for scope in scopes
    ] == [False, True, False]
    assert [
        native_feature._use_decoder_fp16_preconcat_176(decoder_scope=scope) for scope in scopes
    ] == [False, False, True]

    monkeypatch.setenv(_ENV_88, "0")
    assert not any(
        native_feature._use_decoder_fp16_preconcat_88(decoder_scope=scope) for scope in scopes
    )
    assert native_feature._use_decoder_fp16_preconcat_176(decoder_scope="deconv8_4")


@pytest.mark.parametrize("scope", ("", "decoder", "deconv16_8.extra", None))
def test_decoder_fp16_preconcat_88_rejects_unknown_decoder_scope(scope) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    with pytest.raises(RuntimeError, match="decoder scope must be one of"):
        native_feature._use_decoder_fp16_preconcat_88(decoder_scope=scope)


def test_decoder_fp16_preconcat_88_moves_exactly_one_cast_before_concat(monkeypatch) -> None:
    events, output = _run_decoder(monkeypatch, enabled_88=True)

    assert [event for event in events if event[0] == "cast"] == [
        ("cast", "skip", _TARGET_SHAPE, "fp32", "fp16")
    ]
    assert [event for event in events if event[0] == "concat"] == [
        (
            "concat",
            (_TARGET_SHAPE, _TARGET_SHAPE),
            ("fp16", "fp16"),
            1,
        )
    ]
    assert [event for event in events if event[0] == "next-residual"] == [
        ("next-residual", _CONCAT_SHAPE, "fp16", "residual-block", True)
    ]
    assert output.shape == _CONCAT_SHAPE
    assert output.dtype == "fp16"


def test_decoder_fp16_preconcat_88_gate_off_keeps_legacy_promotion(monkeypatch) -> None:
    events, output = _run_decoder(monkeypatch, enabled_88=False)

    assert [event for event in events if event[0] == "cast"] == [
        ("cast", "branch", _TARGET_SHAPE, "fp16", "fp32")
    ]
    assert [event for event in events if event[0] == "concat"] == [
        (
            "concat",
            (_TARGET_SHAPE, _TARGET_SHAPE),
            ("fp32", "fp32"),
            1,
        )
    ]
    assert output.shape == _CONCAT_SHAPE
    assert output.dtype == "fp32"


@pytest.mark.parametrize(
    ("scope", "shape"),
    (("deconv32_16", (2, 160, 44, 44)), ("deconv8_4", (2, 48, 176, 176))),
)
def test_decoder_fp16_preconcat_88_does_not_touch_other_scopes(
    monkeypatch,
    scope: str,
    shape: tuple[int, ...],
) -> None:
    enabled_events, enabled_output = _run_decoder(
        monkeypatch,
        enabled_88=True,
        scope=scope,
        branch_shape=shape,
        skip_shape=shape,
    )
    disabled_events, disabled_output = _run_decoder(
        monkeypatch,
        enabled_88=False,
        scope=scope,
        branch_shape=shape,
        skip_shape=shape,
    )

    assert enabled_events == disabled_events
    assert (enabled_output.shape, enabled_output.dtype) == (
        disabled_output.shape,
        disabled_output.dtype,
    )


def test_decoder_fp16_preconcat_88_and_176_gates_compose_without_interference(
    monkeypatch,
) -> None:
    events_88, output_88 = _run_decoder(
        monkeypatch,
        enabled_88=True,
        enabled_176=True,
    )
    shape_176 = (2, 48, 176, 176)
    events_176, output_176 = _run_decoder(
        monkeypatch,
        enabled_88=True,
        enabled_176=True,
        scope="deconv8_4",
        branch_shape=shape_176,
        skip_shape=shape_176,
    )

    assert [event for event in events_88 if event[0] == "cast"] == [
        ("cast", "skip", _TARGET_SHAPE, "fp32", "fp16")
    ]
    assert [event for event in events_176 if event[0] == "cast"] == [
        ("cast", "skip", shape_176, "fp32", "fp16")
    ]
    assert (output_88.shape, output_88.dtype) == (_CONCAT_SHAPE, "fp16")
    assert (output_176.shape, output_176.dtype) == ((2, 96, 176, 176), "fp16")


def test_decoder_fp16_preconcat_88_rejects_non_fp16_before_layers(monkeypatch) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    monkeypatch.setenv(_ENV_88, "1")
    monkeypatch.delenv(_ENV_176, raising=False)
    monkeypatch.setattr(
        native_feature,
        "_deconv2d",
        lambda *_args, **_kwargs: pytest.fail("precision check must precede decoder layers"),
    )
    with pytest.raises(RuntimeError, match="requires the FP16 feature graph"):
        native_feature._decoder_block(
            SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32")),
            _Tensor("input", (2, 320, 44, 44), "fp32"),
            _Tensor("skip", _TARGET_SHAPE, "fp32"),
            _decoder_block(),
            decoder_scope="deconv16_8",
            fp16=False,
        )


@pytest.mark.parametrize(
    ("skip_shape", "skip_dtype", "match"),
    (
        ((2, 95, 88, 88), "fp32", "skip shape"),
        (_TARGET_SHAPE, "fp16", "skip dtype"),
    ),
)
def test_decoder_fp16_preconcat_88_rejects_skip_drift_before_layers(
    monkeypatch,
    skip_shape: tuple[int, ...],
    skip_dtype: str,
    match: str,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    monkeypatch.setenv(_ENV_88, "1")
    monkeypatch.delenv(_ENV_176, raising=False)
    monkeypatch.setattr(
        native_feature,
        "_deconv2d",
        lambda *_args, **_kwargs: pytest.fail("skip check must precede decoder layers"),
    )
    with pytest.raises(RuntimeError, match=match):
        native_feature._decoder_block(
            SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32")),
            _Tensor("input", (2, 320, 44, 44), "fp16"),
            _Tensor("skip", skip_shape, skip_dtype),
            _decoder_block(),
            decoder_scope="deconv16_8",
            fp16=True,
        )


@pytest.mark.parametrize(
    ("branch_shape", "branch_dtype", "match"),
    (
        ((2, 95, 88, 88), "fp16", "branch shape"),
        (_TARGET_SHAPE, "fp32", "branch dtype"),
    ),
)
def test_decoder_fp16_preconcat_88_rejects_branch_drift_before_concat(
    monkeypatch,
    branch_shape: tuple[int, ...],
    branch_dtype: str,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _run_decoder(
            monkeypatch,
            enabled_88=True,
            branch_shape=branch_shape,
            branch_dtype=branch_dtype,
            fail_on_concat=True,
        )


def test_decoder_fp16_preconcat_88_is_mathematically_exact_at_next_fp16_boundary() -> None:
    generator = np.random.default_rng(88)
    branch = generator.normal(size=(2, 96, 5, 7)).astype(np.float16)
    skip = generator.normal(size=(2, 96, 5, 7)).astype(np.float32)

    legacy_next_conv_input = np.concatenate(
        (branch.astype(np.float32), skip),
        axis=1,
    ).astype(np.float16)
    candidate_next_conv_input = np.concatenate(
        (branch, skip.astype(np.float16)),
        axis=1,
    )

    np.testing.assert_array_equal(candidate_next_conv_input, legacy_next_conv_input)
    legacy_residual_identity = legacy_next_conv_input.astype(np.float16)
    candidate_residual_identity = candidate_next_conv_input
    np.testing.assert_array_equal(candidate_residual_identity, legacy_residual_identity)


def test_decoder_fp16_preconcat_88_uses_only_native_network_api_helpers() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    source = "\n".join(
        inspect.getsource(function)
        for function in (
            native_feature._use_decoder_fp16_preconcat_88,
            native_feature._validate_decoder_fp16_preconcat_88_skip,
            native_feature._validate_decoder_fp16_preconcat_88_branch,
            native_feature._decoder_block,
        )
    ).lower()
    assert "onnx" not in source
    assert "plugin" not in source
    assert "_cast(" in source
    assert "_concat(" in source
