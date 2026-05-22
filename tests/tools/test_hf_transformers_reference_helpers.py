"""Tests for Hugging Face reference helper logic."""

from __future__ import annotations

import subprocess

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.references import hf_transformers
from tests.e2e_harness.references.hf_transformers import (
    HfTransformersReference,
    _decode_vl_generated_text,
    _normalize_nemotron_labs_diffusion_mode,
    _vl_fallback_prompt,
    _vl_prompt_has_image_placeholder,
)


class _FakeProcessor:
    def __init__(self, mapping: dict[tuple[int, ...], str]) -> None:
        self.mapping = mapping

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return self.mapping[tuple(int(token) for token in token_ids)]


def test_qwen_vl_fallback_prompt_includes_image_pad() -> None:
    assert _vl_fallback_prompt("Qwen/Qwen3-VL-2B-Instruct", "Describe it") == (
        "<|vision_start|><|image_pad|><|vision_end|>Describe it"
    )


def test_internvl_fallback_prompt_includes_image_placeholder() -> None:
    assert _vl_fallback_prompt("OpenGVLab/InternVL3-8B-hf", "Describe it") == (
        "<IMG_CONTEXT>\nDescribe it"
    )


def test_non_vl_fallback_prompt_is_unchanged() -> None:
    assert _vl_fallback_prompt("Qwen/Qwen3-0.6B", "Hello") == "Hello"


def test_vl_prompt_placeholder_guard_accepts_internvl_marker() -> None:
    assert _vl_prompt_has_image_placeholder(
        "<|im_start|>user\n<IMG_CONTEXT>\nDescribe it<|im_end|>"
    )


def test_vl_decode_uses_generated_suffix_for_full_sequences() -> None:
    processor = _FakeProcessor({
        (101, 102): "prompt",
        (201, 202): "blue",
    })

    assert _decode_vl_generated_text(processor, [101, 102, 201, 202], 2) == "blue"


def test_vl_decode_falls_back_when_model_returns_generated_only_ids() -> None:
    processor = _FakeProcessor({
        (): "",
        (201, 202): "blue",
    })

    assert _decode_vl_generated_text(processor, [201, 202], 4) == "blue"


def test_nemotron_labs_diffusion_mode_aliases() -> None:
    assert _normalize_nemotron_labs_diffusion_mode("auto") == "diffusion"
    assert _normalize_nemotron_labs_diffusion_mode("dlm") == "diffusion"
    assert _normalize_nemotron_labs_diffusion_mode("autoregressive") == "ar"
    assert (
        _normalize_nemotron_labs_diffusion_mode("linear-speculation-lora")
        == "linear_spec_lora"
    )


def test_nemotron_labs_diffusion_reference_uses_custom_auto_model_path(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            cmd, 0, stdout="OK mode=linear_spec_lora tokens=1\n", stderr=""
        )

    monkeypatch.setattr(
        hf_transformers, "_resolve_cached_model_ref", lambda hf_id: "/cached/model"
    )
    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    case = E2ECase(
        name="nemotron-labs-diffusion",
        hf_id="nvidia/Nemotron-Labs-Diffusion-8B",
        family="nemotron_labs_diffusion",
        runtime_strategy="nemotron_labs_diffusion",
        task_strategy="text_generation_causal",
        inputs={
            "prompt": "hello",
            "max_new_tokens": 64,
            "generation_mode": "linear_spec_adapter",
            "block_length": 16,
            "threshold": 0.75,
            "temperature": 0.0,
        },
        metadata={
            "precision": "bf16",
            "contract_config": {
                "use_chat_template": True,
                "enable_thinking": False,
            },
        },
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/ref/python",
    )

    out = HfTransformersReference().run_stage(
        case, StageSpec(name="full_generation"), ctx
    )

    cmd = captured["cmd"]
    assert cmd[:2] == ["/ref/python", "-c"]
    script = cmd[2]
    assert "from transformers import AutoModel, AutoTokenizer" in script
    assert "AutoModelForCausalLM" not in script
    assert 'template_path = Path(model_ref) / "chat_template.jinja"' in script
    assert "fallback_chat_template" in script
    assert "<|im_start|>system\\\\n<|im_end|>" in script
    assert 'adapter_path = Path(model_ref) / "linear_spec_lora"' in script
    assert "str(adapter_path)" in script
    assert '_tf_generic.check_model_inputs = lambda fn: fn' in script
    assert "PeftModel.from_pretrained" in script
    assert "generation_model = model.model" in script
    assert "def _call_supported(fn, input_ids, **kwargs):" in script
    assert "return fn(input_ids, **filtered_kwargs)" in script
    assert "generation_model.linear_spec_generate" in script
    assert "ids_tensor," in script
    assert 'subfolder="linear_spec_lora"' in script
    assert "generation_mode = 'linear_spec_lora'" in script
    assert "block_length = 16" in script
    assert "threshold = 0.75" in script
    assert "enable_thinking = False" in script
    assert out.metadata["generation_mode"] == "linear_spec_lora"
    assert out.logits is None
