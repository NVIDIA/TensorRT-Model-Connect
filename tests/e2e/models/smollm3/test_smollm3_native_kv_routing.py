# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contract tests for SmolLM3's TensorRT native KV path."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import pytest

from tensorrt_model_connect.families.smollm3.build_routing import (
    native_kv_architecture_capability,
    native_kv_build_capability,
    native_kv_cache_geometry,
    prefer_native_default,
    resolved_head_dim,
)
from tensorrt_model_connect.families.smollm3.config import ModelConfig
from tensorrt_model_connect.families.smollm3.native_kv_contract import (
    validate_native_kv_weights,
)


_LLAMA3_ROPE = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _config(
    *,
    raw_updates: dict | None = None,
    llama3_rope: bool = True,
    **overrides,
) -> ModelConfig:
    values = {
        "model_type": "smollm3",
        "architectures": ["SmolLM3ForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500_000.0,
        "max_position_embeddings": 131072,
        "hidden_act": "silu",
        "_head_dim": 0,
    }
    values.update(overrides)
    raw = {
        "_decoder_engine_layout": "split",
        "rope_scaling": dict(_LLAMA3_ROPE) if llama3_rope else None,
    }
    raw.update(raw_updates or {})
    values["raw"] = raw
    return ModelConfig(**values)


@pytest.mark.parametrize(
    (
        "hidden",
        "mlp",
        "layers",
        "heads",
        "kv_heads",
        "context",
    ),
    [
        (4096, 14336, 32, 32, 8, 131072),
        (5120, 13824, 40, 40, 40, 4096),
        (8192, 28672, 80, 64, 8, 131072),
    ],
    ids=("smollm3-8b-shape", "smollm3-13b-shape", "smollm3-70b-shape"),
)
def test_dense_smollm3_sizes_share_one_native_contract(
    hidden, mlp, layers, heads, kv_heads, context,
):
    config = _config(
        hidden_size=hidden,
        intermediate_size=mlp,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=context,
    )

    architecture = native_kv_architecture_capability(config)
    build = native_kv_build_capability(config)
    row_bytes, cache_bytes = native_kv_cache_geometry(config, context)

    assert architecture.eligible, architecture.reason
    assert build.eligible, build.reason
    assert prefer_native_default(config)
    assert resolved_head_dim(config) == 128
    assert row_bytes == 2 * layers * kv_heads * 128 * 2
    assert cache_bytes == context * row_bytes


def test_route_uses_architecture_not_checkpoint_identity():
    config = _config(
        raw_updates={
            "_model_dir": "/models/renamed-checkpoint",
            "name_or_path": "any-owner/any-smollm3",
            "checkpoint_sha256": "a" * 64,
        }
    )

    assert native_kv_architecture_capability(config).eligible
    assert prefer_native_default(config)


def test_explicit_head_dim_is_supported_when_hidden_width_is_decoupled():
    config = _config(
        hidden_size=3072,
        num_attention_heads=32,
        _head_dim=128,
    )

    assert resolved_head_dim(config) == 128
    assert native_kv_architecture_capability(config).eligible


@pytest.mark.parametrize(
    ("overrides", "raw_updates", "reason"),
    [
        ({"model_type": "smollm34"}, {}, "model_type"),
        ({"architectures": ["OtherForCausalLM"]}, {}, "architectures"),
        ({"hidden_size": 4100}, {}, "divisible"),
        ({"_head_dim": 64}, {}, "head_dim=128"),
        ({"num_key_value_heads": 6}, {}, "divisible"),
        ({"hidden_act": "gelu"}, {}, "hidden_act"),
        ({}, {"sliding_window": 4096}, "unsupported SmolLM3 fields"),
        ({}, {"num_experts": 8}, "unsupported SmolLM3 fields"),
        (
            {},
            {"layer_types": ["full_attention", "linear_attention"]},
            "hybrid",
        ),
        (
            {},
            {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "rope_type",
        ),
    ],
)
def test_architecture_variants_fail_closed(overrides, raw_updates, reason):
    decision = native_kv_architecture_capability(
        _config(raw_updates=raw_updates, **overrides)
    )

    assert decision.applicable
    assert not decision.eligible
    assert reason in decision.reason


@pytest.mark.parametrize(
    ("kwargs", "raw_updates", "reason"),
    [
        ({"precision": "fp16"}, {}, "BF16"),
        ({"max_cache_length": 131071}, {}, "max_cache_length"),
        ({"parallel_enabled": True}, {}, "tensor parallel"),
        ({"dynamic_kv_cache": True}, {}, "fixed physical"),
        ({"quantized": True}, {}, "quantized"),
        ({"debug_layer_outputs": True}, {}, "debug"),
        ({}, {"_fp32_layers": ["layer.0"]}, "FP32 layer"),
        ({}, {"_decoder_engine_layout": "dual_profile"}, "split"),
        ({}, {"_rtx_build_requested": True}, "standard TensorRT"),
    ],
)
def test_unqualified_build_modes_fail_closed(kwargs, raw_updates, reason):
    decision = native_kv_build_capability(
        _config(raw_updates=raw_updates),
        **kwargs,
    )

    assert not decision.eligible
    assert reason in decision.reason


@dataclass
class _Tensor:
    shape: tuple[int, ...]


def _small_config(*, role: str = "prefill") -> ModelConfig:
    return _config(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=256,
        llama3_rope=False,
        raw_updates={"_decoder_engine_role": role},
    )


def _weights(config: ModelConfig) -> dict[str, object]:
    hidden = config.hidden_size
    attention = config.num_attention_heads * 128
    kv_attention = config.num_key_value_heads * 128
    mlp = config.intermediate_size
    weights: dict[str, object] = {
        "embedding": _Tensor((config.vocab_size, hidden)),
        "final_norm": _Tensor((hidden,)),
        "w_out": _Tensor((hidden, config.vocab_size)),
        "_attention_size": attention,
        "_kv_attention_size": kv_attention,
        "_mlp_size": mlp,
    }
    for name, shape in (
        ("input_norm", (hidden,)),
        ("w_q", (hidden, attention)),
        ("w_k", (hidden, kv_attention)),
        ("w_v", (hidden, kv_attention)),
        ("w_o", (attention, hidden)),
        ("post_attn_norm", (hidden,)),
        ("w_gate", (hidden, mlp)),
        ("w_up", (hidden, mlp)),
        ("w_down", (mlp, hidden)),
    ):
        weights[f"layer.0.{name}"] = _Tensor(shape)
    return weights


def test_weight_contract_rejects_missing_shape_and_bias():
    config = _small_config()
    weights = _weights(config)
    validate_native_kv_weights(config, weights)

    missing = dict(weights)
    missing.pop("layer.0.w_k")
    with pytest.raises(ValueError, match="missing.*w_k"):
        validate_native_kv_weights(config, missing)

    wrong_shape = dict(weights)
    wrong_shape["layer.0.w_q"] = _Tensor((127, 128))
    with pytest.raises(ValueError, match="must have shape"):
        validate_native_kv_weights(config, wrong_shape)

    biased = dict(weights)
    biased["layer.0.q_bias"] = _Tensor((128,))
    with pytest.raises(ValueError, match="bias"):
        validate_native_kv_weights(config, biased)


def test_plugin_builds_the_requested_split_role_directly(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.smollm3.plugin"
    )

    config = _small_config(role="prefill")
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"plan"

    monkeypatch.setattr(
        plugin_module,
        "build_dual_profile_decoder_engine",
        _build,
    )

    result = plugin_module.plugin.build_engine(
        config,
        _weights(config),
        256,
        precision="bf16",
    )

    assert result == b"plan"
    assert captured["kwargs"]["profile_mode"] == "prefill"
    assert captured["kwargs"]["native_kv_cache"] is True
    assert plugin_module.plugin.get_bundle_config_overrides(config) == {
        "native_kv_contract_version": 1,
        "native_kv_cache": True,
    }


def test_plugin_falls_back_for_explicit_legacy_build_options(monkeypatch):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.smollm3.plugin"
    )

    config = _small_config(role="decode")
    config.raw["_native_kv_cache_metadata"] = {"stale": True}
    quant_ctx = object()
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"legacy-plan"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        _build,
    )

    result = plugin_module.plugin.build_engine(
        config,
        _weights(config),
        128,
        precision="fp16",
        quant_ctx=quant_ctx,
    )

    assert result == b"legacy-plan"
    assert captured["args"][2] == 128
    assert captured["kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["quant_ctx"] is quant_ctx
    assert plugin_module.plugin.get_bundle_config_overrides(config) is None


def test_dynamic_kv_dual_profile_dispatches_bucket_rows(monkeypatch):
    pytest.importorskip("tensorrt")
    builder_module = importlib.import_module(
        "tensorrt_model_connect.families.smollm3.standard_decoder_builder"
    )
    config = _small_config(role="dual_profile")
    config.raw["dynamic_kv_cache"] = True
    config.raw["_dynamic_kv_profile_rows"] = [256, 131072]
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"dynamic-dual-profile-plan"

    monkeypatch.setattr(
        builder_module,
        "build_dual_profile_decoder_engine",
        _build,
    )

    plan = builder_module.build_standard_decoder_engine(
        config,
        _weights(config),
        131072,
        precision="fp16",
    )

    assert plan == b"dynamic-dual-profile-plan"
    assert captured["args"][2] == 131072
    assert captured["kwargs"]["precision"] == "fp16"
    assert captured["kwargs"]["dynamic_kv_profile_rows"] == [256, 131072]
    assert captured["kwargs"]["profile_mode"] == "dual_profile"

    config.raw.pop("_dynamic_kv_profile_rows")
    captured.clear()
    plan = builder_module.build_standard_decoder_engine(
        config,
        _weights(config),
        131072,
        precision="fp16",
    )

    assert plan == b"dynamic-dual-profile-plan"
    assert captured["kwargs"]["dynamic_kv_profile_rows"] == [131072]


def test_plugin_falls_back_outside_the_native_architecture_contract(
    monkeypatch,
):
    pytest.importorskip("tensorrt")
    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.smollm3.plugin"
    )
    config = _small_config()
    config._head_dim = 64
    captured: dict[str, object] = {}

    def _build(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return b"legacy-plan"

    monkeypatch.setattr(
        plugin_module,
        "build_standard_decoder_engine",
        _build,
    )

    assert not prefer_native_default(config)
    assert plugin_module.plugin.default_build_precision(config) == "fp32"
    assert plugin_module.plugin.default_max_cache_length(config) == 256
    assert plugin_module.plugin.build_engine(
        config,
        _weights(config),
        128,
        precision="fp16",
    ) == b"legacy-plan"
    assert captured["args"][2] == 128
    assert captured["kwargs"]["precision"] == "fp16"


def _shared_build_config(**raw_updates):
    """Build the config the engine builder actually constructs.

    ``engine_builder`` resolves a checkpoint through the shared
    ``tensorrt_model_connect.config.ModelConfig``, never this family's
    dataclass, so anything the builders read has to work on that object. The
    typed fields mirror what ``from_dir`` fills in for SmolLM3-3B.
    """
    from tensorrt_model_connect.config import ModelConfig as SharedModelConfig

    raw = {
        "_decoder_engine_layout": "split",
        "no_rope_layer_interval": 4,
        "rope_scaling": None,
    }
    raw.update(raw_updates)
    return SharedModelConfig(
        model_type="smollm3",
        architectures=["SmolLM3ForCausalLM"],
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=5_000_000.0,
        max_position_embeddings=65536,
        hidden_act="silu",
        tie_word_embeddings=True,
        raw=raw,
    )


def test_rope_layer_schedule_resolves_on_the_shared_build_config():
    from tensorrt_model_connect.families.smollm3.config import (
        resolve_rope_layer_schedule,
    )

    schedule = resolve_rope_layer_schedule(_shared_build_config())

    assert len(schedule) == 36
    assert [index for index, uses in enumerate(schedule) if not uses] == [
        3, 7, 11, 15, 19, 23, 27, 31, 35
    ]
    assert schedule == ModelConfig(
        model_type="smollm3",
        num_hidden_layers=36,
        raw=dict(_shared_build_config().raw),
    ).rope_layer_schedule(), "family and shared configs must agree"


def test_published_no_rope_layers_wins_over_the_interval():
    from tensorrt_model_connect.families.smollm3.config import (
        resolve_rope_layer_schedule,
    )

    published = [1] * 36
    published[5] = 0
    schedule = resolve_rope_layer_schedule(
        _shared_build_config(no_rope_layers=published)
    )

    assert [index for index, uses in enumerate(schedule) if not uses] == [5]


@pytest.mark.parametrize(
    "raw_updates, fragment",
    [
        ({"no_rope_layer_interval": 0}, "no_rope_layer_interval must be positive"),
        ({"no_rope_layers": [1, 1, 1]}, "no_rope_layers must be a sequence"),
    ],
)
def test_malformed_schedule_is_rejected_on_the_shared_build_config(
    raw_updates, fragment
):
    from tensorrt_model_connect.families.smollm3.config import (
        resolve_rope_layer_schedule,
    )

    with pytest.raises(ValueError, match=fragment):
        resolve_rope_layer_schedule(_shared_build_config(**raw_updates))


def test_routing_rejects_a_malformed_schedule_on_the_shared_build_config():
    """Routing must judge the schedule on the config the build path carries.

    Resolving through a family-local method left this check silently inert for
    the shared config, so a malformed schedule routed as eligible and only
    surfaced once the graph builder ran.
    """
    capability = native_kv_architecture_capability(
        _shared_build_config(no_rope_layer_interval=0)
    )

    assert not capability.eligible
    assert any("no_rope_layer_interval must be positive" in reason
               for reason in capability.reason.split("; "))


def test_routing_still_accepts_a_well_formed_schedule():
    assert native_kv_architecture_capability(_shared_build_config()).eligible


class _FakeTensor:
    """Stand-in for an ITensor; the graph is never realized in this test."""

    def __init__(self, name="t", shape=(1, 1, 64)):
        self.name = name
        self.shape = shape
        self.dtype = None

    def __getattr__(self, _name):
        return None


class _FakeLayer:
    def get_output(self, _index):
        return _FakeTensor("out")

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None

    def __setattr__(self, _name, _value):
        pass


class _FakeNetwork:
    """Accepts any add_* call and returns a layer with one output."""

    def __getattr__(self, name):
        if name.startswith("add_"):
            return lambda *args, **kwargs: _FakeLayer()
        raise AttributeError(name)


def _nope_block_weights(prefix, hidden, attention):
    import numpy as np

    return {
        f"{prefix}.input_norm": np.ones(hidden, dtype=np.float32),
        f"{prefix}.w_q": np.zeros((hidden, attention), dtype=np.float32),
        f"{prefix}.w_k": np.zeros((hidden, attention), dtype=np.float32),
        f"{prefix}.w_v": np.zeros((hidden, attention), dtype=np.float32),
        f"{prefix}.w_o": np.zeros((attention, hidden), dtype=np.float32),
    }


def _count_rope_insertions(monkeypatch, *, apply_rope):
    """Drive the attention block and count RoPE layer insertions.

    Spies on ``add_apply_rope_native``, which is what actually puts an
    IRotaryEmbeddingLayer into the graph, so this observes the gate rather than
    the source that contains it.
    """
    from tensorrt_model_connect.families.smollm3 import graph_blocks, graph_ops

    hidden = attention = 64
    calls: list[tuple] = []
    monkeypatch.setattr(
        graph_ops,
        "add_apply_rope_native",
        lambda *args, **kwargs: calls.append(args) or _FakeTensor("roped"),
    )
    graph_blocks.add_attention_block(
        _FakeNetwork(),
        _FakeTensor("hidden"),
        _FakeTensor("cache_k"),
        _FakeTensor("cache_v"),
        _FakeTensor("mask"),
        _FakeTensor("position"),
        weights=_nope_block_weights("layer.0", hidden, attention),
        prefix="layer.0",
        hidden_size=hidden,
        attention_size=attention,
        num_heads=2,
        head_dim=32,
        max_cache_length=16,
        eps_tensor=_FakeTensor("eps"),
        num_kv_heads=2,
        kv_attention_size=attention,
        apply_rope=apply_rope,
        cos_half_tensor=_FakeTensor("cos"),
        sin_half_tensor=_FakeTensor("sin"),
    )
    return calls


def test_rope_layer_inserts_rotary_embedding_for_query_and_key(monkeypatch):
    assert len(_count_rope_insertions(monkeypatch, apply_rope=True)) == 2


def test_nope_layer_inserts_no_rotary_embedding_at_all(monkeypatch):
    assert _count_rope_insertions(monkeypatch, apply_rope=False) == []


def _rope_guard_sources():
    """Return, per builder, the schedule subscripts guarding RoPE insertion.

    The builders need TensorRT to run, so this reads their parsed source rather
    than driving them. It checks the wiring the block-level tests above cannot
    reach: that each builder selects the flag with its own layer loop variable
    instead of a constant or the wrong index.
    """
    import ast
    import pathlib

    family = pathlib.Path(
        "python/tensorrt_model_connect/families/smollm3"
    )
    if not family.is_dir():
        import tensorrt_model_connect.families.smollm3 as package

        family = pathlib.Path(package.__file__).parent

    found = {}
    for name in ("standard_decoder_builder.py", "dual_profile_decoder_builder.py"):
        tree = ast.parse((family / name).read_text(encoding="utf-8"))
        subscripts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "rope_schedule"
        ]
        found[name] = [
            node.slice.id
            for node in subscripts
            if isinstance(node.slice, ast.Name)
        ]
    return found


def test_both_builders_select_the_flag_with_their_layer_index():
    guards = _rope_guard_sources()

    for name, indices in guards.items():
        assert indices, f"{name} does not subscript rope_schedule at all"
        assert set(indices) == {"layer_idx"}, (
            f"{name} indexes rope_schedule with {sorted(set(indices))}, "
            "which is not the layer loop variable"
        )


def test_standard_builder_forwards_the_flag_to_the_attention_block():
    """The standard builder reaches RoPE through ``apply_rope=``.

    A dropped keyword would silently restore RoPE on every NoPE layer, which
    the block-level tests cannot see because they pass the flag directly.
    """
    import ast
    import pathlib

    import tensorrt_model_connect.families.smollm3 as package

    path = pathlib.Path(package.__file__).parent / "standard_decoder_builder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forwarded = [
        keyword
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "apply_rope"
    ]

    assert forwarded, "standard builder never passes apply_rope"
    assert any(
        isinstance(keyword.value, ast.Subscript)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "rope_schedule"
        for keyword in forwarded
    ), "apply_rope is passed but not from the resolved schedule"


def _published_checkpoint_config():
    """The published SmolLM3-3B configuration, field for field.

    Taken from config.json at the revision the manifest pins. It is spelled out
    rather than trimmed so the routing contract is exercised against what users
    actually download, including the fields this family does not read.
    """
    from tensorrt_model_connect.config import ModelConfig as SharedModelConfig

    return SharedModelConfig(
        model_type="smollm3",
        architectures=["SmolLM3ForCausalLM"],
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        rms_norm_eps=1e-06,
        rope_theta=5000000.0,
        max_position_embeddings=65536,
        hidden_act="silu",
        tie_word_embeddings=True,
        bos_token_id=128000,
        eos_token_id=128012,
        pad_token_id=128004,
        raw={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "layer_types": ["full_attention"] * 36,
            "max_window_layers": 28,
            "mlp_bias": False,
            "no_rope_layer_interval": 4,
            "no_rope_layers": [
                int((index + 1) % 4 != 0) for index in range(36)
            ],
            "pretraining_tp": 2,
            "rope_scaling": None,
            "sliding_window": None,
            "torch_dtype": "bfloat16",
            "use_cache": False,
            "use_sliding_window": False,
        },
    )


def test_published_checkpoint_reaches_the_native_kv_path():
    """The checkpoint this family targets must route to its own runtime.

    The published config carries pretraining_tp=2, a field SmolLM3ForCausalLM
    neither defines nor reads. Gating on it sent the default build of the only
    supported checkpoint to the fallback decoder.
    """
    config = _published_checkpoint_config()

    decision = native_kv_architecture_capability(config)

    assert decision.applicable
    assert decision.eligible, decision.reason
    assert prefer_native_default(config)


def test_default_build_of_the_published_checkpoint_is_native():
    """`trtmc build HuggingFaceTB/SmolLM3-3B` with no flags takes the native path."""
    import importlib

    plugin_module = importlib.import_module(
        "tensorrt_model_connect.families.smollm3.plugin"
    )
    config = _published_checkpoint_config()

    precision = plugin_module.plugin.default_build_precision(config)
    cache_length = plugin_module.plugin.default_max_cache_length(config)

    assert precision == "bf16"
    assert cache_length == config.max_position_embeddings == 65536

    decision = native_kv_build_capability(
        config, precision=precision, max_cache_length=cache_length
    )
    assert decision.eligible, decision.reason


def _routing_loaded_as_production_does():
    """Load build_routing.py the way the family loader does.

    MODEL.toml points ``default_build_route`` at ``build_routing.py|
    prefer_native_default``, and families/__init__.py loads that with
    spec_from_file_location under a synthetic top-level name. The module
    therefore has no package context at runtime, while the tests above import
    it as part of the package, where a relative import would work. Load it
    both ways so the difference cannot hide a failure again.
    """
    import importlib.util
    import pathlib

    import tensorrt_model_connect.families.smollm3 as package

    path = pathlib.Path(package.__file__).parent / "build_routing.py"
    spec = importlib.util.spec_from_file_location(
        "_trtmc_family_smollm3_build_routing", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_routing_works_without_package_context():
    routing = _routing_loaded_as_production_does()
    config = _published_checkpoint_config()

    # prefer_native_default is the entry point MODEL.toml names.
    assert routing.prefer_native_default(config) is True
    assert routing.native_kv_architecture_capability(config).eligible
