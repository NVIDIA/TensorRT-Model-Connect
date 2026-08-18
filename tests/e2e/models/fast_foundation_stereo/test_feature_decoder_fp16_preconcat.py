# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


@dataclass(frozen=True)
class _DecoderCase:
    label: str
    scope: str
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    numeric_shape: tuple[int, ...]
    seed: int

    @property
    def concat_shape(self) -> tuple[int, ...]:
        batch, channels, height, width = self.target_shape
        return (batch, channels * 2, height, width)


_CASES = (
    _DecoderCase(
        label="88",
        scope="deconv16_8",
        input_shape=(2, 320, 44, 44),
        target_shape=(2, 96, 88, 88),
        numeric_shape=(2, 96, 5, 7),
        seed=88,
    ),
    _DecoderCase(
        label="176",
        scope="deconv8_4",
        input_shape=(2, 192, 88, 88),
        target_shape=(2, 48, 176, 176),
        numeric_shape=(2, 48, 7, 9),
        seed=176,
    ),
)
_DEFAULT_SCOPE = object()


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
    scope: str | None | object = _DEFAULT_SCOPE,
    fp16: bool = True,
    branch_shape: tuple[int, ...] | None = None,
    branch_dtype: str = "fp16",
    skip_shape: tuple[int, ...] | None = None,
    skip_dtype: str = "fp32",
    fail_on_concat: bool = False,
):
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

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
        decoder_scope=case.scope if scope is _DEFAULT_SCOPE else scope,
        fp16=fp16,
    )
    return events, output


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
@pytest.mark.parametrize("scope", ("", "decoder", "owned.extra", None))
def test_decoder_rejects_unknown_scope_before_layers(
    monkeypatch,
    case: _DecoderCase,
    scope: str | None,
) -> None:
    invalid = f"{case.scope}.extra" if scope == "owned.extra" else scope
    with pytest.raises(RuntimeError, match="decoder scope must be one of"):
        _run_decoder(monkeypatch, case, scope=invalid)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_default_winner_trace(monkeypatch, case: _DecoderCase) -> None:
    events, output = _run_decoder(monkeypatch, case)

    assert events == [
        ("deconv", case.input_shape, "fp16", "deconvolution", True),
        ("norm", case.target_shape, "fp16", "instance-norm", True),
        ("activation", case.target_shape, "fp16", "leaky_relu", 0.01),
        ("cast", "skip", case.target_shape, "fp32", "fp16"),
        ("concat", (case.target_shape, case.target_shape), ("fp16", "fp16"), 1),
        ("next-residual", case.concat_shape, "fp16", "residual-block", True),
    ]
    assert (output.shape, output.dtype) == (case.concat_shape, "fp16")


def test_decoder_44_scope_keeps_checkpoint_dtype_promotion(monkeypatch) -> None:
    shape = (2, 160, 44, 44)
    events, output = _run_decoder(
        monkeypatch,
        _CASES[0],
        scope="deconv32_16",
        branch_shape=shape,
        skip_shape=shape,
    )
    assert [event for event in events if event[0] == "cast"] == [
        ("cast", "branch", shape, "fp16", "fp32")
    ]
    assert [event for event in events if event[0] == "concat"] == [
        ("concat", (shape, shape), ("fp32", "fp32"), 1)
    ]
    assert (output.shape, output.dtype) == ((2, 320, 44, 44), "fp32")


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_rejects_non_fp16_before_layers(monkeypatch, case: _DecoderCase) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

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
def test_decoder_rejects_skip_drift_before_layers(
    monkeypatch,
    case: _DecoderCase,
    drift: str,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_feature

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
def test_decoder_rejects_branch_drift_before_concat(
    monkeypatch,
    case: _DecoderCase,
    drift: str,
) -> None:
    bad_shape = (*case.target_shape[:1], case.target_shape[1] - 1, *case.target_shape[2:])
    with pytest.raises(RuntimeError, match=f"branch {drift}"):
        _run_decoder(
            monkeypatch,
            case,
            branch_shape=bad_shape if drift == "shape" else case.target_shape,
            branch_dtype="fp32" if drift == "dtype" else "fp16",
            fail_on_concat=True,
        )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.label)
def test_decoder_is_exact_at_next_fp16_boundary(case: _DecoderCase) -> None:
    generator = np.random.default_rng(case.seed)
    branch = generator.normal(size=case.numeric_shape).astype(np.float16)
    skip = generator.normal(size=case.numeric_shape).astype(np.float32)

    legacy = np.concatenate((branch.astype(np.float32), skip), axis=1).astype(np.float16)
    candidate = np.concatenate((branch, skip.astype(np.float16)), axis=1)

    np.testing.assert_array_equal(candidate, legacy)
    np.testing.assert_array_equal(candidate, legacy.astype(np.float16))
