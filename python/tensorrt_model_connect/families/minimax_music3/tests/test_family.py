# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routing and pipeline-index tests for the MiniMax-Music3 family."""

from __future__ import annotations

import importlib
import json

import pytest

families = importlib.import_module("tensorrt_model_connect.families")
mod = importlib.import_module(
    "tensorrt_model_connect.families.minimax_music3.plugin"
)

plugin = mod.plugin

#: The published root ``config.json`` at revision
#: ``fbdf52fbaaca799592917417eb05f1899f1255ec``, verbatim.
PUBLISHED_CONFIG = {
    "architectures": ["MiniMaxMusic3ForConditionalGeneration"],
    "model_type": "minimax_music3",
}

#: Shape of the published ``modular_model_index.json`` entries.
PUBLISHED_INDEX = {
    "_blocks_class_name": "MiniMaxMusic3Blocks",
    "_class_name": "MiniMaxMusic3ModularPipeline",
    "condition_encoder": ["diffusers", "MiniMaxMusic3ConditionEncoder", {}],
    "language_model": ["transformers", "Qwen3ForCausalLM", {}],
    "rvq_depth_decoder": ["diffusers", "MiniMaxMusic3RVQDepthDecoder", {}],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler", {}],
    "tokenizer": ["transformers", "Qwen2Tokenizer", {}],
    "transformer": ["diffusers", "MiniMaxMusic3Transformer1DModel", {}],
    "vocoder": ["diffusers", "MiniMaxMusic3Vocoder", {}],
}


class _Config:
    def __init__(self, raw: dict) -> None:
        self.raw = raw


def _write_index(tmp_path, index: dict):
    (tmp_path / mod.MODULAR_INDEX_NAME).write_text(
        json.dumps(index), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "model_type", ["minimax_music3", "minimax-music3", "MiniMax-Music3", "MINIMAXMUSIC3"]
)
def test_matches_published_model_type(model_type: str) -> None:
    assert plugin.matches(model_type)


@pytest.mark.parametrize("model_type", ["minimax_h3", "minimaxh3", "qwen3", "acestep"])
def test_does_not_claim_other_families(model_type: str) -> None:
    assert not plugin.matches(model_type)


def test_repository_routes_the_published_model_type() -> None:
    resolved = families.find_plugin("minimax_music3")

    assert resolved is not None
    assert resolved.name == "minimax_music3"
    assert resolved.runtime_strategy == "minimax_music3_text_to_music"


def test_matches_the_published_config() -> None:
    assert plugin.matches_config(_Config(dict(PUBLISHED_CONFIG)))


def test_matches_config_on_architecture_alone() -> None:
    raw = {"architectures": ["MiniMaxMusic3ForConditionalGeneration"]}

    assert plugin.matches_config(_Config(raw))


def test_matches_config_declines_a_plain_qwen3() -> None:
    raw = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}

    assert not plugin.matches_config(_Config(raw))


def test_matches_config_declines_minimax_h3() -> None:
    raw = {"_class_name": "MiniMaxH3ModularPipeline", "model_type": "minimax_h3"}

    assert not plugin.matches_config(_Config(raw))


def test_reads_every_published_component(tmp_path) -> None:
    components = mod.read_pipeline_components(_write_index(tmp_path, PUBLISHED_INDEX))

    assert set(components) == set(mod.REQUIRED_COMPONENTS)
    assert components["transformer"] == (
        "diffusers", "MiniMaxMusic3Transformer1DModel")
    assert components["language_model"] == ("transformers", "Qwen3ForCausalLM")


def test_underscore_keys_are_not_components(tmp_path) -> None:
    components = mod.read_pipeline_components(_write_index(tmp_path, PUBLISHED_INDEX))

    assert "_class_name" not in components
    assert "_blocks_class_name" not in components


def test_missing_index_yields_no_components(tmp_path) -> None:
    assert mod.read_pipeline_components(tmp_path) == {}


def test_unreadable_index_yields_no_components(tmp_path) -> None:
    (tmp_path / mod.MODULAR_INDEX_NAME).write_text("{not json", encoding="utf-8")

    assert mod.read_pipeline_components(tmp_path) == {}


def test_published_pipeline_class_is_only_in_the_modular_index() -> None:
    """The root config carries no `_class_name`, which is why routing is by
    `model_type`; guard that assumption so a future config change is noticed."""

    assert "_class_name" not in PUBLISHED_CONFIG
    assert PUBLISHED_INDEX["_class_name"] == mod.PIPELINE_CLASS


def test_task_strategy_is_text_to_audio() -> None:
    assert plugin.task_strategy == "text_to_audio"
    assert plugin.runtime_config_namespace == "music_minimax_music3"


def test_runtime_config_schema_registers() -> None:
    importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.runtime_config_schema"
    )
    from tensorrt_model_connect.runtime_config import lookup

    schema = lookup("music_minimax_music3")

    assert schema is not None
    assert {field.name for field in schema.fields} == {
        "caption", "max_frames", "seed", "top_k", "temperature"}


def _field(name: str):
    importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.runtime_config_schema"
    )
    from tensorrt_model_connect.runtime_config import lookup

    return next(f for f in lookup("music_minimax_music3").fields if f.name == name)


def test_caption_defaults_to_unconditioned() -> None:
    field = _field("caption")

    assert field.type_tag == "string"
    assert field.default == ""
    assert field.validator("upbeat electric blues, bpm 92, key E minor")


def test_caption_rejects_an_absurd_length() -> None:
    schema_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.runtime_config_schema"
    )

    assert not _field("caption").validator("x" * (schema_mod.MAX_CAPTION_CHARS + 1))
    assert not _field("caption").validator(None)


def test_max_frames_honours_the_upstream_limit() -> None:
    schema_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.runtime_config_schema"
    )
    field = _field("max_frames")

    assert field.default == schema_mod.MAX_AUDIO_FRAMES == 9000
    assert field.validator(9000)
    assert field.validator(1)
    assert not field.validator(0)
    assert not field.validator(9001)


def _fake_checkpoint(tmp_path):
    """A snapshot carrying the components' tensor *names*.

    The three callers assert routing, a missing name, and a missing directory
    -- none reads a shape or a value. Writing the real shapes serialises about
    3.25 GB per call (``_rvq_depth_decoder`` alone is 2.58 GB), so each name
    gets a scalar instead. ``test_checkpoint`` keeps the shape inventory.
    """

    import numpy as np
    from safetensors.numpy import save_file

    engines = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.engines"
    )
    ce_tests = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.tests.test_checkpoint"
    )
    builders = {
        "condition_encoder": ce_tests._condition_encoder,
        "rvq_depth_decoder": ce_tests._rvq_depth_decoder,
        "vocoder": ce_tests._vocoder,
    }
    for component, build in builders.items():
        directory = tmp_path / component
        directory.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: np.zeros(1, dtype=np.float32) for k in build()},
            str(directory / "model.safetensors"),
        )
    for name, tensor in (
        ("transformer", ("proj_in.weight", (2048, 2304))),
        ("language_model", ("model.norm.weight", (4096,))),
    ):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        save_file({tensor[0]: np.zeros(tensor[1], dtype=np.float32)},
                  str(directory / "model.safetensors"))
    for extra in ("scheduler", "tokenizer"):
        (tmp_path / extra).mkdir(parents=True, exist_ok=True)
    del engines
    return tmp_path


def test_load_weights_reads_the_four_buildable_components(tmp_path) -> None:
    pytest.importorskip("safetensors")
    engines = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.engines"
    )
    root = _fake_checkpoint(tmp_path)

    weights = plugin.load_weights(str(root), None)

    for engine in engines.ENGINE_NAMES:
        assert engine in weights, engine
    assert weights["_model_dir"] == str(root)
    assert engines.LANGUAGE_MODEL_ENGINE in weights


def test_load_weights_validates_the_component_inventories(tmp_path) -> None:
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    checkpoint = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.checkpoint"
    )
    root = _fake_checkpoint(tmp_path)
    save_file({"proj.weight": np.zeros((2048, 4096, 3), dtype=np.float32)},
              str(root / "condition_encoder" / "model.safetensors"))

    with pytest.raises(checkpoint.CheckpointError, match="condition_encoder is missing"):
        plugin.load_weights(str(root), None)


def test_load_weights_names_a_missing_component(tmp_path) -> None:
    pytest.importorskip("safetensors")
    root = _fake_checkpoint(tmp_path)
    for shard in (root / "vocoder").glob("*"):
        shard.unlink()
    (root / "vocoder").rmdir()

    with pytest.raises(FileNotFoundError, match="vocoder"):
        plugin.load_weights(str(root), None)


def test_bundle_overrides_reach_the_plugin() -> None:
    engines = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.engines"
    )

    assert plugin.get_bundle_config_overrides(None) == engines.bundle_config_overrides()


def test_bfloat16_shards_are_widened_at_load(tmp_path) -> None:
    """The depth decoder is stored in bfloat16; every engine builds in float32.

    Whether ``safetensors.numpy`` can open a bfloat16 shard depends on whether
    ``ml_dtypes`` is imported, which registers the dtype with numpy. So the
    loader widens unconditionally rather than falling back on an exception
    that fires or not depending on import order -- this test passed alone and
    failed in the suite until it did.
    """

    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    plugin_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.plugin"
    )
    directory = tmp_path / "bf16"
    directory.mkdir()
    save_file({"w": torch.zeros((4, 8), dtype=torch.bfloat16)},
              str(directory / "model.safetensors"))

    tensors = plugin_mod._read_component(directory)

    assert set(tensors) == {"w"}
    assert tensors["w"].dtype.name == "float32"
    assert tensors["w"].shape == (4, 8)


def test_float32_shards_take_the_numpy_path(tmp_path) -> None:
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import save_file

    plugin_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.plugin"
    )
    directory = tmp_path / "f32"
    directory.mkdir()
    save_file({"w": np.ones((2, 3), dtype=np.float32)},
              str(directory / "model.safetensors"))

    tensors = plugin_mod._read_component(directory)

    assert tensors["w"].dtype.name == "float32"
    assert float(tensors["w"][0, 0]) == 1.0


def test_read_component_names_an_empty_directory(tmp_path) -> None:
    plugin_mod = importlib.import_module(
        "tensorrt_model_connect.families.minimax_music3.plugin"
    )
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="no safetensors under"):
        plugin_mod._read_component(empty)
