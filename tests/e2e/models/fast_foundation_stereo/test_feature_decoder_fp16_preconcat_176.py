# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


_ENV = "TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_176"
_TARGET_SHAPE = (2, 48, 176, 176)


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
    enabled: bool,
    scope: str = "deconv8_4",
    fp16: bool = True,
    branch_shape: tuple[int, ...] = _TARGET_SHAPE,
    branch_dtype: str = "fp16",
    skip_shape: tuple[int, ...] = _TARGET_SHAPE,
    skip_dtype: str = "fp32",
):
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    if enabled:
        monkeypatch.setenv(_ENV, "1")
    else:
        monkeypatch.setenv(_ENV, "0")
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
        _Tensor("input", (2, 192, 88, 88), "fp16" if fp16 else "fp32"),
        _Tensor("skip", skip_shape, skip_dtype),
        _decoder_block(),
        decoder_scope=scope,
        fp16=fp16,
    )
    return events, output


def test_decoder_fp16_preconcat_176_gate_is_default_on_scope_owned(monkeypatch) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    monkeypatch.delenv(_ENV, raising=False)
    assert [
        native_feature._use_decoder_fp16_preconcat_176(decoder_scope=scope)
        for scope in ("deconv32_16", "deconv16_8", "deconv8_4")
    ] == [False, False, True]

    monkeypatch.setenv(_ENV, "0")
    assert not native_feature._use_decoder_fp16_preconcat_176(decoder_scope="deconv8_4")


@pytest.mark.parametrize("scope", ("", "decoder", "deconv8_4.extra", None))
def test_decoder_fp16_preconcat_176_rejects_unknown_decoder_scope(scope) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    with pytest.raises(RuntimeError, match="decoder scope must be one of"):
        native_feature._use_decoder_fp16_preconcat_176(decoder_scope=scope)


def test_decoder_fp16_preconcat_176_moves_exactly_one_cast_before_concat(monkeypatch) -> None:
    events, output = _run_decoder(monkeypatch, enabled=True)

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
        ("next-residual", (2, 96, 176, 176), "fp16", "residual-block", True)
    ]
    assert output.shape == (2, 96, 176, 176)
    assert output.dtype == "fp16"


def test_decoder_fp16_preconcat_176_gate_off_keeps_legacy_promotion(monkeypatch) -> None:
    events, output = _run_decoder(monkeypatch, enabled=False)

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
    assert output.shape == (2, 96, 176, 176)
    assert output.dtype == "fp32"


@pytest.mark.parametrize(
    ("scope", "shape"),
    (("deconv32_16", (2, 160, 44, 44)), ("deconv16_8", (2, 96, 88, 88))),
)
def test_decoder_fp16_preconcat_176_does_not_touch_other_scopes(
    monkeypatch,
    scope: str,
    shape: tuple[int, ...],
) -> None:
    enabled_events, enabled_output = _run_decoder(
        monkeypatch,
        enabled=True,
        scope=scope,
        branch_shape=shape,
        skip_shape=shape,
    )
    disabled_events, disabled_output = _run_decoder(
        monkeypatch,
        enabled=False,
        scope=scope,
        branch_shape=shape,
        skip_shape=shape,
    )

    assert enabled_events == disabled_events
    assert (enabled_output.shape, enabled_output.dtype) == (
        disabled_output.shape,
        disabled_output.dtype,
    )


def test_decoder_fp16_preconcat_176_rejects_non_fp16_before_layers(monkeypatch) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(
        native_feature,
        "_deconv2d",
        lambda *_args, **_kwargs: pytest.fail("precision check must precede decoder layers"),
    )
    with pytest.raises(RuntimeError, match="requires the FP16 feature graph"):
        native_feature._decoder_block(
            SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32")),
            _Tensor("input", (2, 192, 88, 88), "fp32"),
            _Tensor("skip", _TARGET_SHAPE, "fp32"),
            _decoder_block(),
            decoder_scope="deconv8_4",
            fp16=False,
        )


@pytest.mark.parametrize(
    ("skip_shape", "skip_dtype", "match"),
    (
        ((2, 49, 176, 176), "fp32", "skip shape"),
        (_TARGET_SHAPE, "fp16", "skip dtype"),
    ),
)
def test_decoder_fp16_preconcat_176_rejects_skip_drift_before_layers(
    monkeypatch,
    skip_shape: tuple[int, ...],
    skip_dtype: str,
    match: str,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(
        native_feature,
        "_deconv2d",
        lambda *_args, **_kwargs: pytest.fail("skip check must precede decoder layers"),
    )
    with pytest.raises(RuntimeError, match=match):
        native_feature._decoder_block(
            SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32")),
            _Tensor("input", (2, 192, 88, 88), "fp16"),
            _Tensor("skip", skip_shape, skip_dtype),
            _decoder_block(),
            decoder_scope="deconv8_4",
            fp16=True,
        )


@pytest.mark.parametrize(
    ("branch_shape", "branch_dtype", "match"),
    (
        ((2, 47, 176, 176), "fp16", "branch shape"),
        (_TARGET_SHAPE, "fp32", "branch dtype"),
    ),
)
def test_decoder_fp16_preconcat_176_rejects_branch_drift_before_concat(
    monkeypatch,
    branch_shape: tuple[int, ...],
    branch_dtype: str,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _run_decoder(
            monkeypatch,
            enabled=True,
            branch_shape=branch_shape,
            branch_dtype=branch_dtype,
        )


def test_decoder_fp16_preconcat_176_is_mathematically_exact_at_next_fp16_boundary() -> None:
    generator = np.random.default_rng(176)
    branch = generator.normal(size=(2, 48, 7, 9)).astype(np.float16)
    skip = generator.normal(size=(2, 48, 7, 9)).astype(np.float32)

    legacy_next_conv_input = np.concatenate(
        (branch.astype(np.float32), skip),
        axis=1,
    ).astype(np.float16)
    candidate_next_conv_input = np.concatenate(
        (branch, skip.astype(np.float16)),
        axis=1,
    )

    np.testing.assert_array_equal(candidate_next_conv_input, legacy_next_conv_input)
    # The decoder residual identity is cast to the convolution output dtype;
    # equality here also proves that identity input remains unchanged.
    legacy_residual_identity = legacy_next_conv_input.astype(np.float16)
    candidate_residual_identity = candidate_next_conv_input
    np.testing.assert_array_equal(candidate_residual_identity, legacy_residual_identity)


def test_decoder_fp16_preconcat_176_uses_only_native_network_api_helpers() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    source = inspect.getsource(native_feature._decoder_block).lower()
    assert "onnx" not in source
    assert "plugin" not in source
    assert "_cast(" in source
    assert "_concat(" in source
