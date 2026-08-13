# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything-owned Hugging Face reference tests."""

from __future__ import annotations

import subprocess
import sys
import types

import huggingface_hub

from tensorrt_model_connect.families.locateanything.transformers_compat import (
    install_remote_attention_compat,
)
from tests.e2e.models.locateanything.e2e_plugins.references import (
    hf_transformers as locateanything_hf_transformers,
)
from tests.e2e_harness.contracts import E2ECase, RunContext, StageSpec


def test_cached_model_ref_uses_selective_snapshot_contract(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_snapshot_download(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return "/cached/selective-snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    resolved = locateanything_hf_transformers._resolve_cached_model_ref(
        "nvidia/LocateAnything-3B"
    )

    from tensorrt_model_connect.hf_snapshot import hf_snapshot_allow_patterns

    assert resolved == "/cached/selective-snapshot"
    assert captured["repo_id"] == "nvidia/LocateAnything-3B"
    assert captured["kwargs"] == {
        "allow_patterns": hf_snapshot_allow_patterns(),
        "local_files_only": True,
    }


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
        runtime_strategy="locateanything_vision_language",
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
    assert (
        "from transformers import AutoConfig, AutoModel, AutoTokenizer, "
        "PretrainedConfig"
    ) in script
    assert "def _install_tied_weight_compat" in script
    assert "all_tied_weights_keys" in script
    assert "DynamicCache.to_legacy_cache" in script
    assert "DynamicCache.from_legacy_cache" in script
    assert "get_expanded_tied_weights_keys" in script
    assert "model.embed_tokens.weight" in script
    assert "get_class_from_dynamic_module" in script
    assert "install_remote_attention_compat" in script
    assert "torch.backends.cudnn.enabled = False" in script
    assert "def _load_locateanything_config" in script
    assert "def _load_locateanything_raw_config" in script
    assert "PretrainedConfig.get_config_dict" in script
    assert "config.text_config.rope_theta" in script
    assert "AutoModel.from_pretrained" in script
    assert "config=config" in script
    assert "def _load_locateanything_tokenizer" in script
    assert "hf_hub_download" in script
    assert "Tokenizer.from_file" in script
    assert "model_max_length" in script
    assert "def batch_decode" in script
    assert "tensorrt_model_connect.families.locateanything.vl_debug_runner" in script
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


def test_remote_attention_compat_accepts_transformers_55_keyword(monkeypatch) -> None:
    calls: list[tuple[str | None, bool, bool]] = []

    class NativePreTrainedModel:
        def _check_and_adjust_attn_implementation(
            self,
            attn_implementation,
            is_init_check=False,
            allow_all_kernels=False,
        ):
            calls.append((attn_implementation, is_init_check, allow_all_kernels))
            return attn_implementation

    class LocateAnythingPreTrainedModel(NativePreTrainedModel):
        def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):
            if attn_implementation == "magi":
                return "magi"
            return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)

    class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel):
        pass

    class Qwen2PreTrainedModel(NativePreTrainedModel):
        def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False):
            return super()._check_and_adjust_attn_implementation(attn_implementation, is_init_check)

    class Qwen2ForCausalLM(Qwen2PreTrainedModel):
        pass

    module_name = "transformers_modules.test_locateanything.modeling_locateanything"
    for remote_class in (
        LocateAnythingPreTrainedModel,
        LocateAnythingForConditionalGeneration,
        Qwen2PreTrainedModel,
        Qwen2ForCausalLM,
    ):
        remote_class.__module__ = module_name
    remote_module = types.ModuleType(module_name)
    remote_module.Qwen2ForCausalLM = Qwen2ForCausalLM
    monkeypatch.setitem(sys.modules, module_name, remote_module)

    assert install_remote_attention_compat(LocateAnythingForConditionalGeneration) == 2
    assert (
        LocateAnythingForConditionalGeneration()._check_and_adjust_attn_implementation(
            "sdpa", is_init_check=True, allow_all_kernels=True
        )
        == "sdpa"
    )
    assert (
        Qwen2ForCausalLM()._check_and_adjust_attn_implementation("eager", allow_all_kernels=True)
        == "eager"
    )
    assert (
        LocateAnythingForConditionalGeneration()._check_and_adjust_attn_implementation(
            "magi", allow_all_kernels=True
        )
        == "magi"
    )
    assert calls == [("sdpa", True, False), ("eager", False, False)]
