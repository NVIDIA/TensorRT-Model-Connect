# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only graph and checkpoint contracts for native VoiceChat EAR-TTS."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tensorrt_model_connect.families.nemotron_voicechat import native_tts


def test_exact_checkpoint_and_runtime_contract() -> None:
    config = native_tts.EXACT_CONFIG
    shapes = native_tts.required_checkpoint_shapes()
    bindings = native_tts.runtime_binding_specs()

    assert len(shapes) == 418
    assert config.refinement_widths == (0, 0, 0, 1, 1, 3, 4, 22)
    assert config.layer_types[5::6] == ("full_attention",) * 4
    assert config.kv_width == 16 * 72 == 1152
    assert shapes["backbone.layers.27.self_attn.k_proj.weight"] == (1152, 1152)
    assert shapes["embed_subword.embed_tokens.weight"] == (257, 1152)
    assert shapes["mog_head.proj_mus.weight"] == (65536, 1152)
    assert shapes["rvq_embs"] == (31, 1024, 512)

    assert bindings["inputs"]["mixture_uniform"] == ("float32", (8, 1024))
    assert bindings["inputs"]["mog_noise"] == ("float32", (8, 512))
    assert bindings["inputs"]["audio_prompt_latent"] == ("float32", (1152,))
    assert bindings["inputs"]["cache_k_27"] == ("work", (2, -1, 1152))
    assert bindings["outputs"]["rvq_codes"] == ("int32", (31,))
    assert bindings["outputs"]["present_v_27"] == ("work", (2, 1, 1152))


def test_subword_tables_match_nemo_character_mapping(tmp_path: Path) -> None:
    payload = {"model": {"vocab": {"a": 0, "b": 1, "ab": 2, "ba": 3}}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")

    tables = native_tts.build_subword_tables(
        tmp_path,
        expected_vocab_size=4,
        expected_char_vocab_size=2,
    )

    assert tables.char_padding_id == 2
    assert tables.max_chars == 2
    np.testing.assert_array_equal(
        tables.char_ids,
        np.array([[0, 2], [1, 2], [0, 1], [1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(tables.char_lengths, np.array([1, 1, 2, 2]))


def test_weight_loader_requires_and_maps_every_tts_tensor(monkeypatch, tmp_path: Path) -> None:
    config = native_tts.NativeTTSConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        latent_size=2,
        codebook_size=4,
        num_quantizers=2,
        char_vocab_size=3,
        mog_num_layers=1,
        mog_num_predictions=4,
        mog_low_rank=1,
        sliding_window_pattern=1,
    )
    required = native_tts.required_checkpoint_shapes(config)

    class FakeReader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return [native_tts.CHECKPOINT_PREFIX + key for key in required]

        def get_tensor(self, name):
            relative = name.removeprefix(native_tts.CHECKPOINT_PREFIX)
            dtype = (
                np.int64
                if relative.endswith(("pad_tensor", "special_flags", "is_continuation"))
                else np.float32
            )
            return np.zeros(required[relative], dtype=dtype)

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(native_tts, "safe_open", lambda *args, **kwargs: FakeReader())

    weights = native_tts.load_native_tts_weights(tmp_path, config=config)

    assert set(weights) == set(required)
    assert weights["backbone.layers.0.self_attn.q_proj.weight"].dtype == np.float32
    assert weights["embed_subword.bos_eos_emb.special_flags"].dtype == np.int32


def test_builder_uses_strongly_typed_network_and_dynamic_compact_cache(monkeypatch) -> None:
    events = {}

    class FakeProfile:
        def __init__(self):
            self.shapes = {}

        def set_shape(self, name, minimum, optimum, maximum):
            self.shapes[name] = (minimum, optimum, maximum)

    class FakeBuilderConfig:
        def __init__(self):
            self.profile = None

        def clear_flag(self, flag):
            events["cleared"] = flag

        def add_optimization_profile(self, profile):
            self.profile = profile

    class FakeBuilder:
        def __init__(self, logger):
            self.logger = logger
            self.network = object()
            self.config = FakeBuilderConfig()

        def create_network(self, flags):
            events["flags"] = flags
            return self.network

        def create_builder_config(self):
            return self.config

        def create_optimization_profile(self):
            self.profile = FakeProfile()
            events["profile"] = self.profile
            return self.profile

        def build_serialized_network(self, network, config):
            events["build"] = (network, config)
            return b"tts-plan"

    class FakeLogger:
        WARNING = 1
        VERBOSE = 2

        def __init__(self, severity):
            self.severity = severity

    fake_trt = SimpleNamespace(
        Logger=FakeLogger,
        Builder=FakeBuilder,
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=3),
        BuilderFlag=SimpleNamespace(TF32="tf32"),
    )
    monkeypatch.setattr(native_tts.trt_compat, "get_trt", lambda: fake_trt)
    monkeypatch.setattr(
        native_tts,
        "add_native_tts_step_graph",
        lambda network, trt, weights, tables, **kwargs: events.setdefault(
            "graph", (network, trt, kwargs)
        ),
    )

    assert (
        native_tts.build_native_tts_engine_from_weights(
            native_tts.NativeTTSWeights(),
            SimpleNamespace(),
            max_cache_length=7500,
        )
        == b"tts-plan"
    )
    assert events["flags"] == 1 << 3
    assert events["graph"][2]["max_cache_length"] == 7500
    assert events["profile"].shapes["attention_mask"] == (
        (1, 1, 1, 2),
        (1, 1, 1, 257),
        (1, 1, 1, 7501),
    )
    assert events["profile"].shapes["cache_k_27"] == (
        (2, 1, 1152),
        (2, 256, 1152),
        (2, 7500, 1152),
    )


def test_aria_warmup_recipe_matches_reference_boundary(monkeypatch) -> None:
    prompt = np.arange(37 * 1152, dtype=np.float32).reshape(1, 37, 1152)

    class FakeReader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_tensor(self, name):
            assert name == "tts_model.audio_prompt_latents.Aria"
            return prompt

    monkeypatch.setattr(native_tts, "_resolve_safetensors", lambda model_dir: Path("model"))
    monkeypatch.setattr(native_tts, "safe_open", lambda *args, **kwargs: FakeReader())

    latents, recipe = native_tts._load_aria_warmup_assets("model")

    np.testing.assert_array_equal(latents, prompt[0])
    assert recipe["subword_ids"] == [12] * 36 + [2]
    assert recipe["subword_mask"] == [0] * 35 + [1, 1]
    assert recipe["audio_prompt_mode"] == [1] * 36 + [0]
    assert recipe["bos_flags"] == [0] * 36 + [1]
    assert recipe["position_ids"] == list(range(37))
    assert "tts_silence_codes_for_step_36" in recipe["prev_codes_policy"]


def test_model_integration_section_names(monkeypatch) -> None:
    monkeypatch.setattr(native_tts, "_resolve_tokenizer_snapshot", lambda value: Path("tokenizer"))
    monkeypatch.setattr(native_tts, "build_native_tts_engine", lambda *args, **kwargs: b"plan")
    monkeypatch.setattr(
        native_tts,
        "_load_runtime_code_assets",
        lambda model_dir: (
            np.arange(31, dtype="<i4"),
            np.array([1026, 1025, 1024], dtype="<i4"),
        ),
    )
    monkeypatch.setattr(
        native_tts,
        "_load_aria_warmup_assets",
        lambda model_dir: (
            np.zeros((37, 1152), dtype="<f4"),
            {"speaker": "Aria"},
        ),
    )

    sections = native_tts.build_tts_sections("model", {})

    assert [name for name, _ in sections] == [
        "tts_engine_plan",
        "tts_silence_codes",
        "tts_control_codes",
        "tts_first_code_input",
        "tts_aria_prompt_latents",
        "tts_prompt_config.json",
    ]
    assert len(sections[1][1]) == 31 * 4
    assert len(sections[2][1]) == 3 * 4
    assert np.frombuffer(sections[3][1], dtype="<i4").tolist() == [1024] * 31
    assert len(sections[4][1]) == 37 * 1152 * 4
    assert json.loads(sections[5][1])["tts_max_cache_length"] == 7500
