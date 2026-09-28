# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turbo REF routing keeps distinct weights, partitioned plans and honest provenance."""

import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from families.minimax_h3 import delivery, model, staged_build, turbo_checkpoint
from families.minimax_h3.runtime_config_schema import normalize_build_options


@pytest.mark.parametrize("component,filename,_section", staged_build._TURBO_TEXT_COMPONENTS)
def test_ref_text_segments_share_full_sequence_workspace(component, filename, _section):
    assert staged_build._component_workspace_bytes(component, ref2va=True) == 96 << 30
    assert (
        staged_build._component_workspace_bytes(component, ref2va=False)
        == staged_build.RTX_STAGED_WORKSPACE_BYTES
    )
    assert (
        staged_build._workspace_limits(staged_build._TURBO_COMPONENTS, ref2va=True)[filename]
        == 96 << 30
    )


def test_ref_override_requires_explicit_turbo():
    with pytest.raises(ValueError, match="require turbo=true"):
        normalize_build_options({"turbo_ref_transformer": "ref.safetensors"})
    for invalid in ("", None, True, 1):
        with pytest.raises(ValueError, match="invalid MiniMax-H3 option"):
            normalize_build_options({"turbo": True, "turbo_ref_transformer": invalid})


@pytest.mark.parametrize("precision", ("bf16", "int8"))
def test_public_build_routes_explicit_ref_override_without_downloading(
    tmp_path, monkeypatch, precision
):
    import huggingface_hub

    paths = {name: tmp_path / name for name in (*delivery.TURBO_SOURCES, "turbo_ref_transformer")}
    for path in paths.values():
        path.write_bytes(b"fixture")
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *_args, **_kwargs: pytest.fail("Unexpected download"),
    )
    observed = {}
    monkeypatch.setattr(
        staged_build, "build_staged_bundle", lambda *_args, **kwargs: observed.update(kwargs)
    )
    options = normalize_build_options(
        {"turbo": True, "turbo_base_precision": precision, **{k: str(v) for k, v in paths.items()}}
    )
    model.plugin.build_staged_bundle(
        str(tmp_path),
        object(),
        SimpleNamespace(raw={"_family_build_options": {"minimax_h3": options}}),
        {"_model_dir": str(tmp_path)},
        plans_dir=tmp_path / "plans",
        precision="bf16",
    )
    assert {key: observed[key] for key in paths} == paths
    assert observed["turbo_base_precision"] == precision


def test_ref_source_is_not_implicitly_selected_or_downloaded(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *_args, **_kwargs: pytest.fail("Unexpected download"),
    )
    for _repository, _revision, filename in delivery.TURBO_SOURCES.values():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    reference = tmp_path / turbo_checkpoint.REF2VA_BASE_FILENAME
    reference.write_bytes(b"reference fixture")
    assert "turbo_ref_transformer" not in delivery.resolve_turbo_sources(tmp_path, {})
    with pytest.raises(FileNotFoundError, match="turbo_ref_transformer checkpoint is missing"):
        delivery.resolve_turbo_sources(
            tmp_path, {"turbo_ref_transformer": str(tmp_path / "missing")}
        )


def test_turbo_ref_monolithic_component_is_rejected_before_weight_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(staged_build.trt_compat, "configure_backend", lambda **_kwargs: None)
    monkeypatch.setattr(
        turbo_checkpoint,
        "load_selected_turbo_transformer_weights",
        lambda *_args, **_kwargs: pytest.fail("Unexpected weight load"),
    )
    with pytest.raises(ValueError, match="requires split denoiser components"):
        staged_build._build_component(
            "ref2va_denoiser",
            tmp_path,
            tmp_path / "unused.plan",
            verbose=False,
            turbo_ref_transformer_path=tmp_path / "ref",
            turbo_lora_path=tmp_path / "lora",
        )


def test_ref_child_cli_preserves_explicit_reference_source(tmp_path, monkeypatch):
    observed = {}
    monkeypatch.setattr(
        staged_build, "_build_component", lambda *_args, **kwargs: observed.update(kwargs)
    )
    assert (
        staged_build._main(
            [
                "--child",
                "--component",
                "ref2va_dit_tail_1",
                "--model-dir",
                str(tmp_path),
                "--output",
                str(tmp_path / "tail.plan"),
                "--turbo-ref-transformer",
                str(tmp_path / "ref"),
                "--turbo-lora",
                str(tmp_path / "lora"),
                "--turbo-base-precision",
                "int8",
            ]
        )
        == 0
    )
    assert observed["turbo_ref_transformer_path"] == tmp_path / "ref"
    assert observed["turbo_transformer_path"] is None
    assert observed["turbo_base_precision"] == "int8"


def _identity():
    return {
        "base_model_id": turbo_checkpoint.BASE_MODEL_ID,
        "base_revision": turbo_checkpoint.BASE_REVISION,
        "base_filename": turbo_checkpoint.REF2VA_BASE_FILENAME,
        "base_size_bytes": turbo_checkpoint.REF2VA_BASE_BYTES,
        "base_tensor_count": 535,
        "base_workflow": "ref2va",
        "adapter_compatibility": "experimental_not_author_certified",
        "lora_strength": 1.0,
        "lora_merged": False,
    }


def test_turbo_ref_metadata_keeps_real_source_and_segment_bindings():
    metadata = staged_build._turbo_ref_runtime_config(_identity())
    assert metadata["ref2va_transformer_ref"]["filename"] == turbo_checkpoint.REF2VA_BASE_FILENAME
    assert (
        metadata["turbo_lora_ref"]["adapter_compatibility"] == "experimental_not_author_certified"
    )
    assert metadata["ref2va_first_block_cache"] == {"enabled": False, "threshold": 0.0}
    assert metadata["ref2va_scheduler"]["sigma_grid_points"] == 9
    assert metadata["ref2va_scheduler"]["transformer_forwards"] == 8
    assert len(metadata["ref2va"]["text_encoder_sections"]) == 5
    assert metadata["ref2va"]["denoiser_tail_layer_ranges"] == [[1, 26], [26, 50]]
    plans = metadata["ref2va_plan_abis"]
    for section, kind, start, end in (
        ("ref2va_dit_tail_plan", "inputs", 1, 26),
        ("ref2va_dit_tail_1_plan", "inputs", 26, 50),
        ("ref2va_adaln_precompute_plan", "outputs", 0, 25),
        ("ref2va_adaln_precompute_1_plan", "outputs", 25, 50),
    ):
        modulation_names = [
            b["name"] for b in plans[section][kind] if b["name"].startswith("block_modulation_")
        ]
        assert modulation_names == [f"block_modulation_{i}" for i in range(start, end)]
    assert "final_modulation" in [
        b["name"] for b in plans["ref2va_adaln_precompute_plan"]["outputs"]
    ]
    assert "final_modulation" not in [
        b["name"] for b in plans["ref2va_adaln_precompute_1_plan"]["outputs"]
    ]
    assert (
        metadata["ref2va_shared_qwen_profiles"]["text_encoder_plan"]["sequence_rows"][-1] == 262144
    )


def test_ref_child_command_accepts_ref_only_source_without_fl_fallback(tmp_path, monkeypatch):
    output = tmp_path / "tail.plan"
    observed = []

    def run(command, *, check):
        assert check
        observed.extend(command)
        output.write_bytes(b"fixture")

    monkeypatch.setattr(staged_build.subprocess, "run", run)
    staged_build._run_component(
        "ref2va_dit_tail_1",
        tmp_path,
        output,
        verbose=False,
        turbo_ref_transformer_path=tmp_path / "ref",
        turbo_lora_path=tmp_path / "lora",
        turbo_base_precision="int8",
    )
    assert observed[observed.index("--turbo-ref-transformer") + 1] == str(tmp_path / "ref")
    assert "--turbo-transformer" not in observed
    assert observed[observed.index("--turbo-base-precision") + 1] == "int8"


@pytest.mark.parametrize(
    "component,build_name,key_name,partition",
    [
        ("ref2va_dit_head", "build_ref2va_dit_head_engine", "head_checkpoint_keys", {}),
        (
            "ref2va_dit_tail",
            "build_ref2va_dit_tail_engine",
            "tail_checkpoint_keys",
            {"block_start": 1, "block_end": 26},
        ),
        (
            "ref2va_dit_tail_1",
            "build_ref2va_dit_tail_engine",
            "tail_checkpoint_keys",
            {"block_start": 26, "block_end": 50},
        ),
        ("ref2va_dit_finish", "build_ref2va_dit_finish_engine", "finish_checkpoint_keys", {}),
        (
            "ref2va_adaln_precompute",
            "build_ref2va_adaln_precompute_engine",
            "adaln_checkpoint_keys",
            {"block_start": 0, "block_end": 25, "include_final": True},
        ),
        (
            "ref2va_adaln_precompute_1",
            "build_ref2va_adaln_precompute_engine",
            "adaln_checkpoint_keys",
            {"block_start": 25, "block_end": 50, "include_final": False},
        ),
    ],
)
def test_ref_child_reads_ref_weights_and_keeps_turbo_partition(
    tmp_path, monkeypatch, component, build_name, key_name, partition
):
    from families.minimax_h3 import checkpoint

    # Test routing without importing an SDK or creating a GPU builder.
    ref2va_dit_builder = ModuleType("families.minimax_h3.ref2va_dit_builder")
    for name in (
        "build_ref2va_dit_head_engine",
        "build_ref2va_dit_tail_engine",
        "build_ref2va_dit_finish_engine",
        "build_ref2va_adaln_precompute_engine",
        "head_checkpoint_keys",
        "tail_checkpoint_keys",
        "finish_checkpoint_keys",
        "adaln_checkpoint_keys",
    ):
        setattr(ref2va_dit_builder, name, lambda **_kwargs: pytest.fail("Unexpected graph call"))
    monkeypatch.setitem(sys.modules, ref2va_dit_builder.__name__, ref2va_dit_builder)

    output = tmp_path / "result.plan"
    observed = {}
    monkeypatch.setattr(staged_build.trt_compat, "configure_backend", lambda **_kwargs: None)
    monkeypatch.setattr(ref2va_dit_builder, key_name, lambda **_kwargs: ("requested.weight",))
    monkeypatch.setattr(
        checkpoint,
        "load_selected_component_state_dict",
        lambda *_args: pytest.fail("Legacy base fallback"),
    )

    def load(source, lora, names, **kwargs):
        observed["loader"] = (source, lora, tuple(names), kwargs)
        return {"requested.weight": "unmerged-reference-weight"}

    def build(weights, **kwargs):
        observed["graph"] = (weights, kwargs)
        output.write_bytes(b"plan")
        return {"bytes": 4}

    monkeypatch.setattr(turbo_checkpoint, "load_selected_turbo_transformer_weights", load)
    monkeypatch.setattr(ref2va_dit_builder, build_name, build)
    staged_build._build_component(
        component,
        tmp_path,
        output,
        verbose=False,
        turbo_ref_transformer_path=tmp_path / "ref",
        turbo_lora_path=tmp_path / "lora",
        turbo_base_precision="int8",
    )
    assert observed["loader"] == (
        tmp_path / "ref",
        tmp_path / "lora",
        ("requested.weight",),
        {"workflow": "ref2va", "base_precision": "int8"},
    )
    weights, options = observed["graph"]
    assert weights == {"requested.weight": "unmerged-reference-weight"}
    assert options["turbo"] is True
    assert {key: options[key] for key in partition} == partition


def test_unified_turbo_bundle_carries_ref_source_and_required_image_plans(tmp_path, monkeypatch):
    from families.minimax_h3 import turbo_text_checkpoint

    model = tmp_path / "model"
    for name in ("vae", "audio_vae", "tokenizer"):
        (model / name).mkdir(parents=True)
    (model / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "audio_vae/config.json").write_text(
        json.dumps(
            {
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "sampling_rate": 32000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )
    paths = {name: tmp_path / name for name in ("fl", "ref", "lora", "text")}
    for path in paths.values():
        path.write_bytes(b"fixture")
    calls, sections = [], {}

    def validate(source, _lora, *, workflow="fl2va", **_kwargs):
        if workflow == "ref2va":
            assert source == paths["ref"]
            return _identity()
        assert source == paths["fl"]
        return {"lora_strength": 1.0}

    def build(component, _model, path, **options):
        assert options["turbo_ref_transformer_path"] == paths["ref"]
        calls.append(component)
        path.write_bytes(b"plan")

    class Writer:
        def add_file(self, section, path):
            sections[section] = Path(path)

        def add_bytes(self, section, data):
            sections[section] = data

        def add_json(self, section, data):
            sections[section] = data

    monkeypatch.setattr(turbo_checkpoint, "validate_turbo_transformer_checkpoint", validate)
    monkeypatch.setattr(turbo_text_checkpoint, "validate_turbo_text_checkpoint", lambda _p: {})
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _v: "1_6")
    monkeypatch.setattr(staged_build, "_run_component", build)
    staged_build.build_staged_bundle(
        model,
        Writer(),
        plans_dir=tmp_path / "plans",
        turbo_transformer=paths["fl"],
        turbo_ref_transformer=paths["ref"],
        turbo_lora=paths["lora"],
        turbo_text_encoder=paths["text"],
    )
    assert "vision_encoder" in calls and "fl2va_keyframe_vae_encoder" in calls
    assert "ref2va_dit_tail_1" in calls and "ref2va_adaln_precompute_1" in calls
    runtime = sections["runtime.json"]
    assert runtime["public_workflows"] == ["t2va", "fl2va", "ref2va"]
    assert runtime["turbo_lora_ref"] == _identity()
    assert runtime["text_rows"] == runtime["text_rows_max"] == 2641
    assert runtime["conditioning"]["text_sequence_profile"][-1] == 262144
    for _component, filename, _section in staged_build._TURBO_TEXT_COMPONENTS:
        assert runtime["workspace_limit_bytes"][filename] == 96 << 30
    state = json.loads((tmp_path / "plans/build_state.json").read_text())
    assert state["turbo"]["reference"]["base_filename"] == turbo_checkpoint.REF2VA_BASE_FILENAME
