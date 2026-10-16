# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for loaded-reference execution identity, without model inference."""

import json
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from families.lfm2_moe.tests.lfm25_contract_validation import capture_reference_execution_identity


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
    model.named_parameters = lambda: iter(
        [
            ("model.layers.2.feed_forward.gate.weight", SimpleNamespace(dtype="torch.bfloat16")),
            (
                "model.layers.2.feed_forward.experts.gate_up_proj",
                SimpleNamespace(dtype="torch.bfloat16"),
            ),
            (
                "model.layers.2.feed_forward.experts.down_proj",
                SimpleNamespace(dtype="torch.bfloat16"),
            ),
        ]
    )
    model.named_buffers = lambda: iter(
        [
            ("model.layers.2.feed_forward.expert_bias", SimpleNamespace(dtype="torch.float32")),
        ]
    )
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
    assert identity["expert_tensor_dtypes"] == {
        "gate.weight": ["torch.bfloat16"],
        "expert_bias": ["torch.float32"],
        "experts.gate_up_proj": ["torch.bfloat16"],
        "experts.down_proj": ["torch.bfloat16"],
    }
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

def test_available_callables_are_described_without_invocation_or_lazy_resolution(monkeypatch):
    def forbidden_call(*args, **kwargs):
        raise AssertionError("metadata must not invoke a backend")

    class LazyModule(ModuleType):
        def __getattr__(self, name):
            raise AssertionError("metadata must not resolve a lazy attribute: " + name)

    module = LazyModule("reference_identity_fixture")
    module.__file__ = None
    module.is_fast_path_available = True
    module.causal_conv1d_fn = forbidden_call
    module.causal_conv1d_update = forbidden_call
    module.ALL_ATTENTION_FUNCTIONS = SimpleNamespace(get_interface=forbidden_call)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sdpa = LazyModule("transformers.integrations.sdpa_attention")
    sdpa.sdpa_attention_forward = forbidden_call
    monkeypatch.setitem(sys.modules, sdpa.__name__, sdpa)
    model = type("Model", (), {"__module__": module.__name__, "forward": forbidden_call})()
    model.config = SimpleNamespace(_attn_implementation="sdpa", _experts_implementation="grouped_mm")
    experts = LazyModule("transformers.integrations.moe")
    experts.grouped_mm_experts_forward = forbidden_call
    experts.ALL_EXPERTS_FUNCTIONS = SimpleNamespace(get_interface=forbidden_call)
    monkeypatch.setitem(sys.modules, experts.__name__, experts)
    # The missing functional namespace must not trigger LazyModule.__getattr__.
    torch = SimpleNamespace(nn=LazyModule("torch.nn"), _grouped_mm=forbidden_call)
    before = vars(model.config).copy()
    identity = capture_reference_execution_identity(model, torch)
    descriptor = {"module": forbidden_call.__module__, "qualname": forbidden_call.__qualname__}
    assert identity["available_callables"] == {
        "model_forward": descriptor,
        "eager_attention_forward": None,
        "sdpa_attention_forward": descriptor,
        "causal_conv1d_fn": descriptor,
        "causal_conv1d_update": descriptor,
        "torch.nn.functional.grouped_mm": None,
        "torch._grouped_mm": descriptor,
        "grouped_mm_experts_forward": descriptor,
        "batched_mm_experts_forward": None,
    }
    assert identity["grouped_mm_api_availability"] == {
        "torch.nn.functional.grouped_mm": False, "torch._grouped_mm": True
    }
    torch.nn.functional = SimpleNamespace(grouped_mm=forbidden_call)
    available = capture_reference_execution_identity(model, torch)
    assert available["grouped_mm_api_availability"] == {
        "torch.nn.functional.grouped_mm": True, "torch._grouped_mm": True
    }
    assert available["available_callables"]["torch.nn.functional.grouped_mm"] == descriptor
    assert identity["invocation_observed"] is False
    assert "no dispatch resolution or invocation observation" in identity["selection_scope"]
    assert vars(model.config) == before


@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_e2e_execution_metadata_survives_generate_failure_without_changing_generation(
    monkeypatch, tmp_path, sampled, fails
):
    import numpy as np

    from families.lfm2_moe.tests import test_e2e as e2e
    from tools import e2e_evidence

    source = tmp_path / "private-location" / "modeling_fixture.py"
    source.parent.mkdir()
    source.write_text("# reference fixture\n")
    module = ModuleType("reference_e2e_fixture")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = SimpleNamespace(_attn_implementation="sdpa", _experts_implementation="grouped_mm")
    calls, seeds, loads = [], [], []
    recorder = e2e_evidence.Evidence(
        tmp_path / "evidence" / "lfm2_moe" / "reference-identity",
        family="lfm2_moe", case="reference-identity", source_revision="test",
    )

    class Model:
        dtype, device = "torch.bfloat16", "cuda"

        def eval(self):
            return self

        def to(self, device):
            assert device == "cuda"
            return self

        def generate(self, **kwargs):
            calls.append(kwargs)
            assert recorder.data["reference_execution"]["attention_implementation"] == "sdpa"
            assert vars(self.config) == {
                "_attn_implementation": "sdpa", "_experts_implementation": "grouped_mm"
            }
            if fails:
                raise RuntimeError("injected reference failure")
            return np.array([[1, 2, 9]])

        def set_attn_implementation(self, *args, **kwargs):
            raise AssertionError("metadata must not select a backend")

    Model.__module__ = module.__name__
    model = Model()
    model.config = config

    class Inputs(dict):
        def to(self, device):
            assert device == "cuda"
            return self

    inputs = Inputs(input_ids=np.array([[1, 2]]), attention_mask=np.array([[1, 1]]))
    monkeypatch.setattr(e2e, "_render_prompt", lambda *args: inputs)

    def load_model(*args, **kwargs):
        loads.append(kwargs)
        return model

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(decode=lambda *args, **kwargs: "done")
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch = SimpleNamespace(
        float32="fp32", float16="fp16", bfloat16="bf16", inference_mode=nullcontext,
        manual_seed=seeds.append, cuda=SimpleNamespace(is_available=lambda: False),
    )
    case = {"max_new_tokens": 7, "temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0}
    if sampled:
        case.update(temperature=0.2, top_k=80, top_p=1.0, min_p=0.0,
                    repetition_penalty=1.05, seed=42)
    token = e2e_evidence._ACTIVE.set(recorder)
    try:
        if fails:
            with pytest.raises(RuntimeError, match="injected reference failure"):
                with e2e.evidence_stage("reference"):
                    e2e._hf_reference(tmp_path, {"precision": "bf16"}, case, "fixture", [9], torch)
        else:
            with e2e.evidence_stage("reference"):
                result = e2e._hf_reference(
                    tmp_path, {"precision": "bf16"}, case, "fixture", [9], torch
                )
            assert result == ([9], "done", "done")
        recorder.finish("failed" if fails else "passed",
                        failure="injected reference failure" if fails else "")
    finally:
        e2e_evidence._ACTIVE.reset(token)
    saved = json.loads((recorder.directory / "evidence.json").read_text())
    metadata = saved["reference_execution"]
    assert saved["evidence_status"] == "recorded"
    assert saved["issues"] == []
    assert saved["status"] == ("failed" if fails else "passed")
    if fails:
        assert saved["failure_stage"] == "reference"
    assert metadata["attention_implementation"] == "sdpa"
    assert metadata["invocation_observed"] is False
    assert metadata["model_module"]["source"]["name"] == source.name
    assert str(tmp_path) not in json.dumps(metadata)
    assert saved["observations"][0] == {"name": "reference_execution", "value": metadata}
    assert loads == [{"local_files_only": True, "trust_remote_code": False, "dtype": "bf16"}]
    assert seeds == ([42] if sampled else [])
    assert len(calls) == 1
    assert calls[0]["input_ids"] is inputs["input_ids"]
    assert calls[0]["attention_mask"] is inputs["attention_mask"]
    expected = {key: value for key, value in case.items() if key != "seed"}
    expected["do_sample"] = sampled
    assert {key: value for key, value in calls[0].items() if key not in inputs} == expected
