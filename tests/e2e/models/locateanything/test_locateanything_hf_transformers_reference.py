"""LocateAnything-owned Hugging Face reference tests."""

from __future__ import annotations

import subprocess

from tests.e2e.models.locateanything.e2e_plugins.references import (
    hf_transformers as locateanything_hf_transformers,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_vl_reference_uses_manual_processor(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        text_path = tmp_path / "locateanything-3b" / "hf_vl_text.txt"
        text_path.write_text("found", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="OK text='found'\n", stderr="")

    monkeypatch.setattr(
        locateanything_hf_transformers,
        "_resolve_cached_model_ref",
        lambda hf_id: "/cached/model",
    )
    monkeypatch.setattr(locateanything_hf_transformers.subprocess, "run", _fake_run)

    case = E2ECase(
        name="locateanything-3b",
        hf_id="nvidia/LocateAnything-3B",
        family="locateanything",
        runtime_strategy="vision_language",
        task_strategy="vision_language_generation",
        inputs={
            "prompt": "Find the red vehicle in this image.",
            "max_new_tokens": 1,
            "image": "tests/e2e/models/locateanything/data/test_img.jpeg",
        },
        metadata={"precision": "bf16", "trust_remote_code": True},
    )
    ctx = RunContext(
        case=case,
        artifacts_dir=str(tmp_path),
        reference_python="/ref/python",
    )

    out = locateanything_hf_transformers.HfTransformersReference().run_stage(
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
    assert 'preprocessor_type="patchify_chw"' in script
    assert 'image_pads = "<IMG_CONTEXT>" * 256' in script
    assert 'f"<img>{image_pads}</img>{prompt}<|im_end|>\\n"' in script
    assert '"generation_mode": "slow"' in script
    assert '"do_sample": False' in script
    assert "torch_dtype=torch.bfloat16" in script
    assert out.text == "found"
    assert out.metadata["reference_variant"] == "locateanything_manual_processor"
    assert out.logits is None
