# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject inert kernel selectors before reference loaders or sampling can run."""

import copy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from . import chat_rendering_reference, preprocessing_reference, sampled_parity, test_e2e
from .reference_source import guard_reference_config, guard_reference_model, reference_source

PINNED = Path(__file__).parent / "fixtures/config.json"
FIELDS = (
    "attn_implementation",
    "_attn_implementation",
    "_attn_implementation_internal",
    "experts_implementation",
    "_experts_implementation",
    "_experts_implementation_internal",
)


@pytest.fixture
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reference input guard attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    return forbidden


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("component", [None, "text_config", "vision_config"])
def test_inert_hub_selector_rejected_in_every_config_scope(tmp_path, no_network, field, component):
    config = json.loads(PINNED.read_text())
    target = config if component is None else config[component]
    target[field] = "inert-owner/inert-kernel"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unsupported reference kernel selector"):
        reference_source(str(tmp_path))


@pytest.mark.parametrize(
    "selector", ["flash_attention_2", "flash_attention_3", "paged|flash_attention_2"]
)
def test_flash_attention_hub_fallback_is_not_admitted(selector):
    with pytest.raises(ValueError, match="unsupported reference kernel selector"):
        guard_reference_config({"attn_implementation": {"text_config": selector}})


def test_component_mapping_cannot_hide_a_hub_selector():
    with pytest.raises(ValueError, match="unsupported reference kernel selector"):
        guard_reference_config(
            {"attn_implementation": {"": "sdpa", "vision_config": "inert-owner/inert-kernel"}}
        )


def test_reference_redirect_is_rejected_before_alternate_config_loading(tmp_path, no_network):
    (tmp_path / "config.json").write_text(
        json.dumps({"configuration_files": ["config.5.2.0.json"]})
    )
    (tmp_path / "config.5.2.0.json").write_text(
        json.dumps({"_attn_implementation_internal": "inert-owner/inert-kernel"})
    )
    with pytest.raises(ValueError, match="configuration-file redirection"):
        reference_source(str(tmp_path))


def test_adapter_cannot_redirect_the_original_model_loader(tmp_path, no_network):
    (tmp_path / "config.json").write_bytes(PINNED.read_bytes())
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "inert-owner/inert-model"})
    )
    with pytest.raises(ValueError, match="full checkpoint, not an adapter"):
        reference_source(str(tmp_path))


def test_downloaded_config_is_guarded_before_reference_loading(monkeypatch, tmp_path, no_network):
    import huggingface_hub
    from .reference_source import CHECKPOINT

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"experts_implementation": "inert-owner/inert-kernel"}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: str(path))
    with pytest.raises(ValueError, match="unsupported reference kernel selector"):
        reference_source(CHECKPOINT)


def test_pinned_and_builtin_configurations_are_accepted_without_changes(tmp_path, no_network):
    original = PINNED.read_bytes()
    (tmp_path / "config.json").write_bytes(original)
    reference_source(str(tmp_path))
    assert (tmp_path / "config.json").read_bytes() == original
    config = json.loads(original)
    config["attn_implementation"] = {"": None, "text_config": "sdpa", "vision_config": "eager"}
    config["experts_implementation"] = {"": "eager", "text_config": "grouped_mm"}
    config["vision_config"]["_experts_implementation_internal"] = "batched_mm"
    config["_name_or_path"] = "an-ordinary/checkpoint-name"
    before = copy.deepcopy(config)
    guard_reference_config(config)
    assert config == before


@pytest.mark.parametrize("entrypoint", ["e2e", "preprocessing", "chat"])
def test_original_loaders_are_never_called_for_rejected_input(
    monkeypatch, tmp_path, no_network, entrypoint
):
    import torch
    import transformers

    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"attn_implementation": "inert-owner/inert-kernel"}})
    )
    for loader in (
        transformers.AutoConfig,
        transformers.AutoProcessor,
        transformers.AutoTokenizer,
        transformers.AutoModelForImageTextToText,
    ):
        monkeypatch.setattr(loader, "from_pretrained", no_network)
    with pytest.raises(ValueError, match="unsupported reference kernel selector"):
        if entrypoint == "e2e":
            monkeypatch.setenv("TRTMC_LFM25_VL_CHECKPOINT", str(tmp_path))
            monkeypatch.setenv("TRTMC_RUNTIME_ROOT", str(tmp_path))
            monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
            monkeypatch.setattr(test_e2e, "build_native_probe", lambda *args: tmp_path / "unused")
            options = SimpleNamespace(
                getoption=lambda name, default: ["lfm2_vl"] if name == "--e2e-model" else []
            )
            test_e2e.test_e2e(test_e2e.CASE, tmp_path, SimpleNamespace(config=options))
        elif entrypoint == "preprocessing":
            monkeypatch.setattr(sys, "argv", ["preprocessing_reference", str(tmp_path)])
            preprocessing_reference.main()
        else:
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "chat_rendering_reference",
                    "--model-dir",
                    str(tmp_path),
                    "--output",
                    str(tmp_path / "unused"),
                ],
            )
            chat_rendering_reference.main()


@pytest.mark.parametrize("kernelized", [False, True])
def test_sampling_guard_runs_before_forward_or_native_processes(
    monkeypatch, no_network, kernelized
):
    config = SimpleNamespace(
        sub_configs={"text_config": object},
        text_config=SimpleNamespace(_attn_implementation_internal="inert-owner/inert-kernel"),
    )
    model = SimpleNamespace(config=config, use_kernels=kernelized)
    monkeypatch.setattr(sampled_parity.subprocess, "run", no_network)
    with pytest.raises(ValueError, match="use_kernels|unsupported reference kernel selector"):
        sampled_parity.compare_sampled(model, None, None, None, None, None)


def test_actual_original_config_is_accepted_for_sampling(no_network):
    from transformers import AutoConfig

    options, _ = reference_source(str(PINNED.parent))
    config = AutoConfig.from_pretrained(PINNED.parent, trust_remote_code=False, **options)
    config._attn_implementation = "sdpa"
    before = copy.deepcopy(vars(config))
    guard_reference_model(SimpleNamespace(config=config, use_kernels=False))
    assert vars(config) == before
