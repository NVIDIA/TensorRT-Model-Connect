# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from tools.reference import transformers_encoder


class _FakeModel:
    def __init__(self, model_type: str) -> None:
        self.config = SimpleNamespace(model_type=model_type)
        self.device = "mapped-device"
        self.to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self


def _load_runtime(model_type: str):
    requested_dtype = object()
    model = _FakeModel(model_type)
    arguments = SimpleNamespace(
        model="org/model",
        model_revision="",
        reference_family="encoder_base_features",
        trust_remote_code=False,
        local_files_only=True,
        dtype="float16",
        device="cuda",
        device_map="",
    )
    transformers_module = SimpleNamespace(
        logging=SimpleNamespace(set_verbosity_error=lambda: None),
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda _model, **_kwargs: object()
        ),
        AutoModel=SimpleNamespace(
            from_pretrained=lambda _model, **_kwargs: model
        ),
    )
    torch_module = SimpleNamespace(
        float16=requested_dtype,
        device=lambda name: f"device:{name}",
    )

    transformers_encoder._load_runtime(
        arguments,
        torch_module,
        transformers_module,
    )
    return model, requested_dtype


def test_xlnet_reference_casts_all_parameters_to_requested_dtype() -> None:
    model, requested_dtype = _load_runtime("xlnet")

    assert model.to_calls == [
        ((), {"dtype": requested_dtype}),
        (("device:cuda",), {}),
    ]


def test_non_xlnet_reference_keeps_existing_load_behavior() -> None:
    model, _requested_dtype = _load_runtime("bert")

    assert model.to_calls == [(("device:cuda",), {})]
