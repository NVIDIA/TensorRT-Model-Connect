# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for loaded-reference execution identity, without model inference."""

import json
import sys
from types import ModuleType, SimpleNamespace

from families.lfm2.tests.lfm25_contract_validation import capture_reference_execution_identity


def test_loaded_reference_path_and_enabled_flags(monkeypatch, tmp_path):
    source = tmp_path / "modeling_fixture.py"
    source.write_text("# reference source fixture\n")
    module = ModuleType("reference_identity_fixture")
    module.__file__ = str(source)
    module.is_fast_path_available = False
    module.causal_conv1d_fn = None
    module.causal_conv1d_update = None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = type("Model", (), {"__module__": module.__name__})()
    model.config = SimpleNamespace(
        _attn_implementation="sdpa", _experts_implementation="grouped_mm"
    )
    model.dtype = "torch.bfloat16"
    torch = SimpleNamespace(
        get_float32_matmul_precision=lambda: "highest",
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(
                    allow_tf32=False, allow_bf16_reduced_precision_reduction=True
                ),
                flash_sdp_enabled=lambda: True,
                math_sdp_enabled=lambda: True,
                mem_efficient_sdp_enabled=lambda: False,
            )
        ),
    )
    identity = capture_reference_execution_identity(model, torch)
    assert identity["attention_implementation"] == "sdpa"
    assert identity["experts_implementation"] == "grouped_mm"
    assert identity["model_dtype"] == "torch.bfloat16"
    assert identity["model_module"]["source"] == {
        "name": source.name,
        "bytes": source.stat().st_size,
    }
    assert identity["causal_conv"] == {
        "fast_path_available": False,
        "forward_function_available": False,
        "update_function_available": False,
    }
    flags = identity["torch_backend_flags"]
    assert flags["get_float32_matmul_precision"] == "highest"
    assert flags["backends.cuda.matmul.allow_tf32"] is False
    assert flags["backends.cuda.flash_sdp_enabled"] is True
    assert flags["backends.cuda.math_sdp_enabled"] is True
    assert flags["backends.cuda.mem_efficient_sdp_enabled"] is False
    assert flags["backends.cuda.cudnn_sdp_enabled"] is None
    assert str(tmp_path) not in json.dumps(identity)
    module.is_fast_path_available = True
    module.causal_conv1d_fn = lambda: None
    module.causal_conv1d_update = lambda: None
    model.config._experts_implementation = "eager"
    changed = capture_reference_execution_identity(model, torch)
    assert changed["causal_conv"]["fast_path_available"] is True
    assert changed["causal_conv"]["forward_function_available"] is True
    assert changed["causal_conv"]["update_function_available"] is True
    assert changed["experts_implementation"] == "eager"


def test_absent_optional_reference_metadata():
    model = type("UnavailableReference", (), {"__module__": "unavailable_reference_fixture"})()
    identity = capture_reference_execution_identity(model, SimpleNamespace())
    assert identity["attention_implementation"] is None
    assert identity["experts_implementation"] is None
    assert identity["model_dtype"] is None
    assert identity["model_module"]["source"] is None
    assert identity["causal_conv"]["fast_path_available"] is None
    assert all(value is None for value in identity["torch_backend_flags"].values())
