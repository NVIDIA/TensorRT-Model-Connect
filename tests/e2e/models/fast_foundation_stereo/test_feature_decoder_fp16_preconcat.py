# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass(frozen=True)
class _DecoderCase:
    label: str
    env: str
    gate: str
    scope: str
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    other_scopes: tuple[tuple[str, tuple[int, ...]], ...]
    numeric_shape: tuple[int, ...]
    seed: int

    @property
    def concat_shape(self) -> tuple[int, ...]:
        batch, channels, height, width = self.target_shape
        return (batch, channels * 2, height, width)


_CASES = (
    _DecoderCase(
        label="88",
        env="TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_88",
        gate="_use_decoder_fp16_preconcat_88",
        scope="deconv16_8",
        input_shape=(2, 320, 44, 44),
        target_shape=(2, 96, 88, 88),
        other_scopes=(("deconv32_16", (2, 160, 44, 44)), ("deconv8_4", (2, 48, 176, 176))),
        numeric_shape=(2, 96, 5, 7),
        seed=88,
    ),
    _DecoderCase(
        label="176",
        env="TRTMC_FAST_FOUNDATION_STEREO_DECODER_FP16_PRECONCAT_176",
        gate="_use_decoder_fp16_preconcat_176",
        scope="deconv8_4",
        input_shape=(2, 192, 88, 88),
        target_shape=(2, 48, 176, 176),
        other_scopes=(("deconv32_16", (2, 160, 44, 44)), ("deconv16_8", (2, 96, 88, 88))),
        numeric_shape=(2, 48, 7, 9),
        seed=176,
    ),
)


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
    case: _DecoderCase,
    *,
    enabled: bool,
    other_enabled: bool = False,
    scope: str | None = None,
    fp16: bool = True,
    branch_shape: tuple[int, ...] | None = None,
    branch_dtype: str = "fp16",
    skip_shape: tuple[int, ...] | None = None,
    skip_dtype: str = "fp32",
    fail_on_concat: bool = False,
):
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    for candidate in _CASES:
        selected = enabled if candidate.label == case.label else other_enabled
        monkeypatch.setenv(candidate.env, "1" if selected else "0")
    branch_shape = case.target_shape if branch_shape is None else branch_shape
    skip_shape = case.target_shape if skip_shape is None else skip_shape
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
        events.append(("next-residual", tensor.shape, tensor.dtype, block.label, fp16))
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
        _Tensor("input", case.input_shape, "fp16" if fp16 else "fp32"),
        _Tensor("skip", skip_shape, skip_dtype),
        _decoder_block(),
        decoder_scope=case.scope if scope is None else scope,
        fp16=fp16,
    )
    return events, output


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_gate_defaults_on_for_only_its_scope(monkeypatch, case: _DecoderCase) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    scopes = ("deconv32_16", "deconv16_8", "deconv8_4")
    for candidate in _CASES:
        monkeypatch.delenv(candidate.env, raising=False)
    gate = getattr(native_feature, case.gate)
    assert [gate(decoder_scope=scope) for scope in scopes] == [
        scope == case.scope for scope in scopes
    ]

    monkeypatch.setenv(case.env, "0")
    assert not gate(decoder_scope=case.scope)
    other = next(candidate for candidate in _CASES if candidate.label != case.label)
    assert getattr(native_feature, other.gate)(decoder_scope=other.scope)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("scope", ("", "decoder", "owned.extra", None))
def test_decoder_gate_rejects_unknown_scope(case: _DecoderCase, scope: str | None) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    invalid = f"{case.scope}.extra" if scope == "owned.extra" else scope
    with pytest.raises(RuntimeError, match="decoder scope must be one of"):
        getattr(native_feature, case.gate)(decoder_scope=invalid)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_gate_moves_exactly_one_cast_before_concat(monkeypatch, case: _DecoderCase) -> None:
    events, output = _run_decoder(monkeypatch, case, enabled=True)

    assert [event for event in events if event[0] == "cast"] == [
        ("cast", "skip", case.target_shape, "fp32", "fp16")
    ]
    assert [event for event in events if event[0] == "concat"] == [
        ("concat", (case.target_shape, case.target_shape), ("fp16", "fp16"), 1)
    ]
    assert [event for event in events if event[0] == "next-residual"] == [
        ("next-residual", case.concat_shape, "fp16", "residual-block", True)
    ]
    assert (output.shape, output.dtype) == (case.concat_shape, "fp16")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_gate_off_keeps_legacy_promotion(monkeypatch, case: _DecoderCase) -> None:
    events, output = _run_decoder(monkeypatch, case, enabled=False)

    assert [event for event in events if event[0] == "cast"] == [
        ("cast", "branch", case.target_shape, "fp16", "fp32")
    ]
    assert [event for event in events if event[0] == "concat"] == [
        ("concat", (case.target_shape, case.target_shape), ("fp32", "fp32"), 1)
    ]
    assert (output.shape, output.dtype) == (case.concat_shape, "fp32")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_gate_does_not_touch_other_scopes(monkeypatch, case: _DecoderCase) -> None:
    for scope, shape in case.other_scopes:
        enabled_events, enabled_output = _run_decoder(
            monkeypatch,
            case,
            enabled=True,
            scope=scope,
            branch_shape=shape,
            skip_shape=shape,
        )
        disabled_events, disabled_output = _run_decoder(
            monkeypatch,
            case,
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


def test_decoder_gates_compose_without_interference(monkeypatch) -> None:
    for case in _CASES:
        events, output = _run_decoder(
            monkeypatch,
            case,
            enabled=True,
            other_enabled=True,
        )
        assert [event for event in events if event[0] == "cast"] == [
            ("cast", "skip", case.target_shape, "fp32", "fp16")
        ]
        assert (output.shape, output.dtype) == (case.concat_shape, "fp16")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_gate_rejects_non_fp16_before_layers(monkeypatch, case: _DecoderCase) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    for candidate in _CASES:
        monkeypatch.setenv(candidate.env, "1" if candidate.label == case.label else "0")
    monkeypatch.setattr(
        native_feature,
        "_deconv2d",
        lambda *_args, **_kwargs: pytest.fail("precision check must precede decoder layers"),
    )
    with pytest.raises(RuntimeError, match="requires the FP16 feature graph"):
        native_feature._decoder_block(
            SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32")),
            _Tensor("input", case.input_shape, "fp32"),
            _Tensor("skip", case.target_shape, "fp32"),
            _decoder_block(),
            decoder_scope=case.scope,
            fp16=False,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("drift", ("shape", "dtype"))
def test_decoder_gate_rejects_skip_drift_before_layers(
    monkeypatch,
    case: _DecoderCase,
    drift: str,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

    for candidate in _CASES:
        monkeypatch.setenv(candidate.env, "1" if candidate.label == case.label else "0")
    monkeypatch.setattr(
        native_feature,
        "_deconv2d",
        lambda *_args, **_kwargs: pytest.fail("skip check must precede decoder layers"),
    )
    bad_shape = (*case.target_shape[:1], case.target_shape[1] - 1, *case.target_shape[2:])
    with pytest.raises(RuntimeError, match=f"skip {drift}"):
        native_feature._decoder_block(
            SimpleNamespace(trt=SimpleNamespace(float16="fp16", float32="fp32")),
            _Tensor("input", case.input_shape, "fp16"),
            _Tensor(
                "skip",
                bad_shape if drift == "shape" else case.target_shape,
                "fp16" if drift == "dtype" else "fp32",
            ),
            _decoder_block(),
            decoder_scope=case.scope,
            fp16=True,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("drift", ("shape", "dtype"))
def test_decoder_gate_rejects_branch_drift_before_concat(
    monkeypatch,
    case: _DecoderCase,
    drift: str,
) -> None:
    bad_shape = (*case.target_shape[:1], case.target_shape[1] - 1, *case.target_shape[2:])
    with pytest.raises(RuntimeError, match=f"branch {drift}"):
        _run_decoder(
            monkeypatch,
            case,
            enabled=True,
            branch_shape=bad_shape if drift == "shape" else case.target_shape,
            branch_dtype="fp32" if drift == "dtype" else "fp16",
            fail_on_concat=True,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_gate_is_exact_at_next_fp16_boundary(case: _DecoderCase) -> None:
    generator = np.random.default_rng(case.seed)
    branch = generator.normal(size=case.numeric_shape).astype(np.float16)
    skip = generator.normal(size=case.numeric_shape).astype(np.float32)

    legacy = np.concatenate((branch.astype(np.float32), skip), axis=1).astype(np.float16)
    candidate = np.concatenate((branch, skip.astype(np.float16)), axis=1)

    np.testing.assert_array_equal(candidate, legacy)
    np.testing.assert_array_equal(candidate, legacy.astype(np.float16))
