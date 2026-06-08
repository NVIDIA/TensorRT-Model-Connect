"""Tests for Hugging Face reference helper logic."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec
from tests.e2e_harness.references import hf_transformers
from tests.e2e_harness.references.hf_transformers import (
    HfTransformersReference,
    _decode_vl_generated_text,
    _json_output_reader,
    _npy_output_reader,
    _normalize_nemotron_labs_diffusion_mode,
    _read_text_artifact,
    _vl_fallback_prompt,
    _vl_prompt_has_image_placeholder,
    run_reference_subprocess,
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


def test_run_reference_subprocess_loads_artifacts(monkeypatch, tmp_path) -> None:
    json_path = tmp_path / "out.json"
    npy_path = tmp_path / "out.npy"
    text_path = tmp_path / "out.txt"

    def _fake_run(cmd, **kwargs):
        import numpy as np

        assert cmd == ["/ref/python", "-c", "print('ok')"]
        assert kwargs["timeout"] == 5
        assert kwargs["env"]["LD_LIBRARY_PATH"] == "/libs"
        json_path.write_text(json.dumps({"answer": 42}), encoding="utf-8")
        np.save(npy_path, np.array([1, 2, 3], dtype=np.int32))
        text_path.write_text("done", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK\n", stderr="warn\n")

    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    out = run_reference_subprocess(
        command=["/ref/python", "-c", "print('ok')"],
        timeout_s=5,
        label="hf_helper",
        artifact_dir=str(tmp_path),
        case_name="case-a",
        stage_name="full_generation",
        env={"LD_LIBRARY_PATH": "/libs"},
        output_readers=(
            _json_output_reader(str(json_path)),
            _npy_output_reader(str(npy_path), "values", path_key="values_path"),
        ),
        text_reader=lambda: _read_text_artifact(str(text_path)),
        include_stdio_metadata=True,
        metadata={"trust_remote_code": False},
        failure_label="HF helper",
    )

    assert out.stage_name == "full_generation"
    assert out.data["answer"] == 42
    assert out.data["values_path"] == str(npy_path)
    assert out.data["values"].tolist() == [1, 2, 3]
    assert out.text == "done"
    assert out.metadata == {
        "returncode": 0,
        "stdout": "OK\n",
        "stderr": "warn\n",
        "trust_remote_code": False,
    }


def test_run_reference_subprocess_nonzero_saves_full_stderr(
    monkeypatch, tmp_path
) -> None:
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 7, stdout="", stderr="bad stderr")

    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        run_reference_subprocess(
            command=["/ref/python", "-c", "raise SystemExit(7)"],
            timeout_s=5,
            label="hf_helper",
            artifact_dir=str(tmp_path),
            case_name="case-a",
            stage_name="full_generation",
            failure_label="HF helper",
        )

    assert "HF helper failed for case-a (rc=7)" in str(excinfo.value)
    assert "bad stderr" in str(excinfo.value)
    log_path = tmp_path / "case-a" / "hf_helper_stderr.log"
    assert log_path.read_text(encoding="utf-8") == "bad stderr"


def test_run_reference_subprocess_timeout_saves_stderr(monkeypatch, tmp_path) -> None:
    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"], stderr="late stderr")

    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        run_reference_subprocess(
            command=["/ref/python", "-c", "while True: pass"],
            timeout_s=5,
            label="hf_helper",
            artifact_dir=str(tmp_path),
            case_name="case-a",
            stage_name="full_generation",
            failure_label="HF helper",
        )

    assert "HF helper timed out for case-a" in str(excinfo.value)
    assert "late stderr" in str(excinfo.value)
    log_path = tmp_path / "case-a" / "hf_helper_stderr.log"
    assert log_path.read_text(encoding="utf-8") == "late stderr"


def test_locateanything_vl_reference_uses_manual_processor(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        text_path = tmp_path / "locateanything-3b" / "hf_vl_text.txt"
        text_path.write_text("found", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK text='found'\n", stderr="")

    monkeypatch.setattr(
        hf_transformers, "_resolve_cached_model_ref", lambda hf_id: "/cached/model"
    )
    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    case = E2ECase(
        name="locateanything-3b",
        hf_id="nvidia/LocateAnything-3B",
        family="locateanything",
        runtime_strategy="vision_language",
        task_strategy="vision_language_generation",
        inputs={
            "prompt": "Find the red vehicle in this image.",
            "max_new_tokens": 1,
            "image": "tests/e2e/data/test_img.jpeg",
        },
        metadata={"precision": "bf16", "trust_remote_code": True},
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
    assert "AutoProcessor" not in script
    assert "from transformers import AutoConfig, AutoModel, AutoTokenizer" in script
    assert "def _install_tied_weight_compat" in script
    assert "all_tied_weights_keys" in script
    assert "DynamicCache.to_legacy_cache" in script
    assert "DynamicCache.from_legacy_cache" in script
    assert "get_expanded_tied_weights_keys" in script
    assert "model.embed_tokens.weight" in script
    assert "def _load_locateanything_config" in script
    assert "config.text_config.rope_theta" in script
    assert "AutoModel.from_pretrained" in script
    assert "config=config" in script
    assert "def _load_locateanything_tokenizer" in script
    assert "Tokenizer.from_file" in script
    assert "model_max_length" in script
    assert "def batch_decode" in script
    assert "preprocess_image_inputs_for_trt" in script
    assert 'preprocessor_type="locateanything_patchify"' in script
    assert 'image_pads = "<IMG_CONTEXT>" * 256' in script
    assert 'f"<img>{image_pads}</img>{prompt}<|im_end|>\\n"' in script
    assert '"generation_mode": "slow"' in script
    assert '"do_sample": False' in script
    assert "torch_dtype=torch.bfloat16" in script
    assert out.text == "found"
    assert out.metadata["reference_variant"] == "locateanything_manual_processor"
    assert out.logits is None


def test_sam3_reference_uses_cached_model_ref(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        import numpy as np

        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        case_dir = tmp_path / "sam3"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "hf_sam3.json").write_text(
            json.dumps(
                {
                    "text_prompt": "ear",
                    "scores": [0.9],
                    "boxes": [[1.0, 2.0, 3.0, 4.0]],
                    "num_masks": 1,
                }
            ),
            encoding="utf-8",
        )
        np.save(case_dir / "hf_sam3_masks.npy", np.zeros((1, 2, 2), dtype=np.uint8))
        (case_dir / "hf_sam3_segmented.png").write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK masks=1\n", stderr="")

    monkeypatch.setattr(
        hf_transformers, "_resolve_cached_model_ref", lambda hf_id: "/cached/sam3"
    )
    monkeypatch.setattr(hf_transformers.subprocess, "run", _fake_run)

    case = E2ECase(
        name="sam3",
        hf_id="facebook/sam3",
        family="sam3",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        reference_family="prompted_segmentation_sam3",
        inputs={"image_url": "https://example.com/sam3.jpg", "prompt": "ear"},
        metadata={"trust_remote_code": False, "precision": "fp32"},
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/ref/python",
    )

    out = HfTransformersReference().run_stage(
        case, StageSpec(name="full_inference"), ctx
    )

    cmd = captured["cmd"]
    assert cmd[:2] == ["/ref/python", "-c"]
    script = cmd[2]
    assert "model_ref = '/cached/sam3'" in script
    assert "Sam3Processor.from_pretrained(" in script
    assert "model_ref, trust_remote_code=trust_remote_code" in script
    assert "Sam3Model.from_pretrained(" in script
    assert "model_ref, torch_dtype=torch.float32" in script
    assert out.data["text_prompt"] == "ear"
    assert out.data["masks_path"].endswith("hf_sam3_masks.npy")
    assert out.metadata["returncode"] == 0


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
