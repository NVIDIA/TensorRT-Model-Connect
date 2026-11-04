# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of build memory lifetime and bundle publication."""

import gc
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tracemalloc
import types
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
from safetensors.numpy import save_file

from tensorrt_model_connect import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter

from .. import checkpoint_mapper
from ..config import ModelConfig


@pytest.fixture
def request_and_writer(tmp_path, monkeypatch):
    # Jiaxin Deng: stub only GPU compilation; exercise the real build and writer on CPU.
    with monkeypatch.context() as imports:
        for module_name, function_name in (
            ("dual_profile_decoder_builder", "build_dual_profile_decoder_engine"),
            ("standard_decoder_builder", "build_standard_decoder_engine"),
        ):
            module = types.ModuleType(f"families.llama.{module_name}")
            setattr(module, function_name, None)
            imports.setitem(sys.modules, module.__name__, module)
        spec = importlib.util.spec_from_file_location(
            "families.llama._plan_lifetime_model", Path(__file__).parents[1] / "model.py"
        )
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "max_position_embeddings": 32,
                "eos_token_id": 2,
            }
        )
    )
    monkeypatch.setattr(model, "load_standard_weights", lambda *args, **kwargs: {})
    destination = tmp_path / "model.bundle"
    request = BuildRequest(
        model_dir=checkpoint,
        output_path=destination,
        family="llama",
        task="text_generation",
        precision="fp16",
        max_sequence_length=16,
    )
    writer = BundleWriter(destination)
    yield model, request, writer
    writer.abort()


def test_prefill_plan_is_released_before_decode_build(request_and_writer, monkeypatch):
    model, request, writer = request_and_writer
    released = []

    class Plan(bytes):
        def __del__(self):
            released.append(True)

    def build_engine(config, *args, **kwargs):
        if config.raw["_decoder_engine_role"] == "prefill":
            return Plan(b"prefill bytes")
        gc.collect()
        assert released, "prefill plan is still resident during decode build"
        return b"decode bytes"

    monkeypatch.setattr(model, "_build_engine", build_engine)
    model.build(request, writer)
    writer.finish()
    data = request.output_path.read_bytes()
    header_size = struct.unpack("<Q", data[8:16])[0]
    header = json.loads(data[16 : 16 + header_size])
    payload = data[16 + header_size :]
    sections = {
        name: payload[entry["offset"] : entry["offset"] + entry["length"]]
        for name, entry in header["sections"].items()
    }
    assert sections["engine.plan"] == b"decode bytes"
    assert sections["prefill.plan"] == b"prefill bytes"
    assert json.loads(sections["runtime.json"])["decoder_engine_layout"] == "split"


def test_decode_failure_preserves_published_bundle(request_and_writer, monkeypatch):
    model, request, writer = request_and_writer
    request.output_path.write_bytes(b"previous bundle")

    def build_engine(config, *args, **kwargs):
        if config.raw["_decoder_engine_role"] == "prefill":
            return b"prefill bytes"
        raise RuntimeError("decode build failed")

    monkeypatch.setattr(model, "_build_engine", build_engine)
    with pytest.raises(RuntimeError, match="decode build failed"):
        model.build(request, writer)
    writer.abort()
    assert request.output_path.read_bytes() == b"previous bundle"
    assert not list(request.output_path.parent.glob(".model.bundle.sections.*"))


@pytest.mark.parametrize("precision,fp32_layers", [("fp16", ()), ("fp32", ()), ("fp16", (1,))])
@pytest.mark.parametrize("tied", [False, True])
def test_projection_scratch_is_released_without_changing_weights(
    tmp_path, monkeypatch, precision, fp32_layers, tied
):
    config = ModelConfig(
        model_type="llama",
        hidden_size=4,
        vocab_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=6,
    )
    embedding = np.arange(32, dtype=np.float32).reshape(8, 4) / 7 - 2
    tensors = {"model.embed_tokens.weight": embedding}
    if not tied:
        tensors["lm_head.weight"] = embedding + 3
    shapes = {
        "self_attn.q_proj": (4, 4),
        "self_attn.k_proj": (2, 4),
        "self_attn.v_proj": (2, 4),
        "self_attn.o_proj": (4, 4),
        "mlp.gate_proj": (6, 4),
        "mlp.up_proj": (6, 4),
        "mlp.down_proj": (4, 6),
    }
    output_names = ("w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down")
    expected = {}
    for index in range(2):
        for projection_index, (source, output) in enumerate(zip(shapes, output_names)):
            shape = shapes[source]
            values = (
                np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 7
                + index * 3
                + projection_index / 9
                - 2
            ).astype(np.float16)
            tensors[f"model.layers.{index}.{source}.weight"] = values
            dtype = np.float32 if precision == "fp32" or index in fp32_layers else np.float16
            expected[f"layer.{index}.{output}"] = values.astype(np.float32).T.astype(dtype)
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"model.layers.{index}.{name}.weight"] = np.ones(4, dtype=np.float16)
    save_file(tensors, tmp_path / "model.safetensors")
    load_tensor = checkpoint_mapper._load_tensor
    previous = {}
    embedding_source = None

    def tracked_load(readers, name):
        nonlocal embedding_source
        if name == "model.embed_tokens.weight":
            value = load_tensor(readers, name)
            embedding_source = weakref.ref(value)
            return value
        assert embedding_source() is None, "embedding scratch is still live"
        if ".layers." in name and name.endswith("_proj.weight"):
            layer = name.split(".")[2]
            if layer in previous:
                assert previous[layer]() is None, "previous projection scratch is still live"
            value = load_tensor(readers, name)
            previous[layer] = weakref.ref(value)
            return value
        return load_tensor(readers, name)

    monkeypatch.setattr(checkpoint_mapper, "_load_tensor", tracked_load)
    weights = checkpoint_mapper.load_standard_weights(
        tmp_path, config, precision=precision, fp32_layers=fp32_layers
    )
    dtype = np.float32 if precision == "fp32" else np.float16
    np.testing.assert_array_equal(weights["embedding"], embedding.astype(dtype))
    output = embedding if tied else tensors["lm_head.weight"]
    np.testing.assert_array_equal(weights["w_out"], output.T.astype(dtype))
    assert weights["w_out"].dtype == dtype
    assert weights["w_out"].flags.c_contiguous
    for name, values in expected.items():
        np.testing.assert_array_equal(weights[name], values)
        assert weights[name].dtype == values.dtype
        assert weights[name].flags.c_contiguous
    assert weights["_attention_size"] == 4
    assert weights["_kv_attention_size"] == 2
    assert weights["_mlp_size"] == 6


@pytest.mark.parametrize("precision,fp32_layers", [("fp16", ()), ("fp32", ()), ("fp16", (1,))])
@pytest.mark.parametrize(
    "quantized", [("w_q", "w_gate"), ("w_q", "w_k", "w_v", "w_o", "w_gate", "w_up", "w_down")]
)
def test_quantized_placeholders_skip_projection_reads(tmp_path, precision, fp32_layers, quantized):
    config = ModelConfig(
        model_type="llama",
        hidden_size=4,
        vocab_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=6,
    )
    projection_shapes = {
        "self_attn.q_proj": ("w_q", 4, 4),
        "self_attn.k_proj": ("w_k", 2, 4),
        "self_attn.v_proj": ("w_v", 2, 4),
        "self_attn.o_proj": ("w_o", 4, 4),
        "mlp.gate_proj": ("w_gate", 6, 4),
        "mlp.up_proj": ("w_up", 6, 4),
        "mlp.down_proj": ("w_down", 4, 6),
    }
    embedding = np.arange(32, dtype=np.float32).reshape(8, 4)
    tensors = {"model.embed_tokens.weight": embedding, "model.norm.weight": np.ones(4)}
    scales = {}
    expected = {}
    for index in range(2):
        for norm in ("input_layernorm", "post_attention_layernorm"):
            tensors[f"model.layers.{index}.{norm}.weight"] = np.ones(4, dtype=np.float32)
        dtype = np.float32 if precision == "fp32" or index in fp32_layers else np.float16
        for source, (target, out_features, in_features) in projection_shapes.items():
            name = f"layer.{index}.{target}"
            if target in quantized:
                scales[name] = SimpleNamespace(
                    weight=SimpleNamespace(in_features=in_features, out_features=out_features)
                )
            else:
                values = (
                    np.arange(out_features * in_features, dtype=np.float32).reshape(
                        out_features, in_features
                    )
                    / 7
                    + index
                )
                tensors[f"model.layers.{index}.{source}.weight"] = values
                expected[name] = values.T.astype(dtype)
    save_file(tensors, tmp_path / "model.safetensors")
    context = SimpleNamespace(scales=scales, should_quantize=lambda name: name in scales)
    weights = checkpoint_mapper.load_standard_weights(
        tmp_path,
        config,
        precision=precision,
        fp32_layers=fp32_layers,
        quant_ctx=context,
    )
    for name, scale in scales.items():
        placeholder = weights[name]
        layer = int(name.split(".")[1])
        dtype = np.float32 if precision == "fp32" or layer in fp32_layers else np.float16
        assert placeholder.shape == (scale.weight.in_features, scale.weight.out_features)
        assert placeholder.dtype == dtype
        assert placeholder.strides == (0, 0)
        assert not placeholder.flags.writeable
        assert placeholder.base.nbytes == np.dtype(dtype).itemsize
        np.testing.assert_array_equal(placeholder, np.zeros(placeholder.shape, dtype=dtype))
    for name, values in expected.items():
        np.testing.assert_array_equal(weights[name], values)
        assert weights[name].dtype == values.dtype
        assert weights[name].flags.c_contiguous
    dtype = np.float32 if precision == "fp32" else np.float16
    np.testing.assert_array_equal(weights["embedding"], embedding.astype(dtype))
    np.testing.assert_array_equal(weights["w_out"], embedding.T.astype(dtype))
    assert (weights["_attention_size"], weights["_kv_attention_size"], weights["_mlp_size"]) == (
        4,
        2,
        6,
    )


def _write_tied_checkpoint(path, vocab, hidden):
    config = ModelConfig(
        model_type="llama",
        hidden_size=hidden,
        vocab_size=vocab,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=hidden,
    )
    embedding = (np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden) % 29 - 14) / 7
    tensors = {"model.embed_tokens.weight": embedding}
    for norm in ("input_layernorm", "post_attention_layernorm"):
        tensors[f"model.layers.0.{norm}.weight"] = np.ones(hidden, dtype=np.float32)
    for projection in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ):
        tensors[f"model.layers.0.{projection}.weight"] = np.zeros(
            (hidden, hidden), dtype=np.float32
        )
    save_file(tensors, path / "model.safetensors")
    return config, embedding


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("vocab,hidden", [(8, 4), (8, 1), (1, 4), (1, 1)])
def test_tied_output_owns_contiguous_storage(tmp_path, precision, vocab, hidden):
    config, source = _write_tied_checkpoint(tmp_path, vocab, hidden)
    weights = checkpoint_mapper.load_standard_weights(tmp_path, config, precision=precision)
    dtype = np.float32 if precision == "fp32" else np.float16
    np.testing.assert_array_equal(weights["w_out"], source.astype(dtype).T)
    assert weights["w_out"].dtype == dtype
    assert weights["w_out"].flags.c_contiguous
    assert not np.shares_memory(weights["w_out"], weights["embedding"])
    original = weights["embedding"].copy()
    weights["w_out"][0, 0] = 99
    np.testing.assert_array_equal(weights["embedding"], original)


def test_tied_output_allocation_does_not_copy_input(tmp_path, monkeypatch):
    config, _ = _write_tied_checkpoint(tmp_path, 4096, 256)
    has_tensor = checkpoint_mapper._has_tensor

    def begin_tracking(readers, name):
        found = has_tensor(readers, name)
        if name == "lm_head.weight":
            assert not found
            tracemalloc.start()
        return found

    monkeypatch.setattr(checkpoint_mapper, "_has_tensor", begin_tracking)
    try:
        weights = checkpoint_mapper.load_standard_weights(tmp_path, config, precision="fp16")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert weights["w_out"].nbytes == 2 * 4096 * 256
    assert peak < weights["w_out"].nbytes * 1.5


# Issue #1414: default precision and length must resolve to a loadable route.
_NATIVE_ELIGIBLE_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 256,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 32,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "max_position_embeddings": 131072,
    "eos_token_id": 2,
}


def _resolve_build(monkeypatch, model, request, writer, config, loads=None, **overrides):
    """Run the real family build with compilation stubbed; return the routes used."""
    from dataclasses import replace

    (request.model_dir / "config.json").write_text(json.dumps(config))
    loads = [] if loads is None else loads
    engines = []

    def load_weights(*args, **kwargs):
        loads.append(kwargs["precision"])
        return {}

    def build_engine(config, weights, max_sequence_length, *, precision, **kwargs):
        engines.append((config.raw["_decoder_engine_role"], max_sequence_length, precision))
        return b"plan"

    # Quantized builds calibrate before loading weights; the route is decided before that.
    calibration = types.ModuleType("families.llama.quantization")
    calibration.calibrate_llama = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, calibration.__name__, calibration)
    monkeypatch.setattr(model, "load_standard_weights", load_weights)
    monkeypatch.setattr(model, "_build_engine", build_engine)
    model.build(replace(request, **overrides), writer)
    return engines


def test_support_defaults_llama_to_native_precision():
    from tensorrt_model_connect.model_support import ModelMetadata

    from ..support import describe

    support = describe(ModelMetadata(config={"model_type": "llama"}, model_index={}))
    assert support.default_task == "text_generation"
    assert support.default_precision == "bf16"


@pytest.mark.parametrize(
    "overrides,config_updates,expected_length",
    [
        # BF16 takes the native KV route: full context, unchanged from before.
        ({"precision": "bf16"}, {}, 131072),
        # FP16 (and FP32) take the dense route: a full-context mask cannot load.
        ({"precision": "fp16"}, {}, 256),
        # Quantized builds take the dense route too; before #1414 they got 131072.
        ({"precision": "bf16", "quantization": "fp8"}, {}, 256),
        # Runtime-sized KV and FP32 layer overrides bound the prefill mask.
        ({"precision": "fp16", "dynamic_kv_cache": True}, {}, 131072),
        ({"precision": "fp16", "fp32_layers": (0,)}, {}, 131072),
        # A dense full-context mask that fits keeps the full-context default.
        ({"precision": "fp16"}, {"max_position_embeddings": 8192}, 8192),
        # Architectures outside the native contract keep the short fallback.
        ({"precision": "bf16"}, {"num_attention_heads": 4}, 256),
    ],
)
def test_default_length_follows_selected_route(
    monkeypatch, request_and_writer, overrides, config_updates, expected_length
):
    model, request, writer = request_and_writer
    config = dict(_NATIVE_ELIGIBLE_CONFIG, **config_updates)
    engines = _resolve_build(
        monkeypatch, model, request, writer, config, max_sequence_length=None, **overrides
    )
    assert engines
    assert {length for _, length, _ in engines} == {expected_length}


@pytest.mark.parametrize(
    "overrides,config_updates,suggests_bf16,suggests_dynamic",
    [
        ({"precision": "fp16", "max_sequence_length": 131072}, {}, True, True),
        ({"precision": "bf16", "max_sequence_length": 65536}, {}, True, True),
        ({"precision": "fp16", "max_sequence_length": 32769}, {}, True, True),
        # Quantized builds take neither the native route nor a runtime-sized KV cache.
        (
            {"precision": "bf16", "quantization": "fp8", "max_sequence_length": 131072},
            {},
            False,
            False,
        ),
        # No native route for this architecture, so BF16 is not offered as a fix.
        (
            {"precision": "fp16", "max_sequence_length": 131072},
            {"num_attention_heads": 4},
            False,
            True,
        ),
    ],
)
def test_unloadable_dense_mask_is_rejected_before_weights_load(
    monkeypatch, request_and_writer, overrides, config_updates, suggests_bf16, suggests_dynamic
):
    model, request, writer = request_and_writer
    config = dict(_NATIVE_ELIGIBLE_CONFIG, **config_updates)
    loads = []
    with pytest.raises(ValueError, match="attention_mask profile maximum") as error:
        _resolve_build(monkeypatch, model, request, writer, config, loads=loads, **overrides)
    assert loads == [], "weights must not load before the rejection"
    message = str(error.value)
    assert f"max_sequence_length={overrides['max_sequence_length']}" in message
    assert ("--precision bf16" in message) == suggests_bf16
    assert ("--dynamic-kv-cache" in message) == suggests_dynamic
    assert "--max-sequence-length" in message


@pytest.mark.parametrize(
    "overrides",
    [
        {"precision": "fp16", "max_sequence_length": 32768},
        {"precision": "bf16", "max_sequence_length": 131072},
        {"precision": "fp16", "max_sequence_length": 131072, "dynamic_kv_cache": True},
    ],
)
def test_bounded_explicit_lengths_still_build(monkeypatch, request_and_writer, overrides):
    model, request, writer = request_and_writer
    engines = _resolve_build(
        monkeypatch, model, request, writer, _NATIVE_ELIGIBLE_CONFIG, **overrides
    )
    assert {length for _, length, _ in engines} == {overrides["max_sequence_length"]}


def test_prepared_checkpoint_sidecar_takes_the_dense_route(monkeypatch, request_and_writer):
    """A ModelOpt checkpoint that declares FP8 only in hf_quant_config.json is not native."""
    model, request, writer = request_and_writer
    sidecar = {"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "FP8"}}
    (request.model_dir / "hf_quant_config.json").write_text(json.dumps(sidecar))
    engines = _resolve_build(
        monkeypatch, model, request, writer, _NATIVE_ELIGIBLE_CONFIG, max_sequence_length=None
    )
    assert {length for _, length, _ in engines} == {256}
    loads = []
    with pytest.raises(ValueError, match="attention_mask profile maximum"):
        _resolve_build(
            monkeypatch,
            model,
            request,
            writer,
            _NATIVE_ELIGIBLE_CONFIG,
            loads=loads,
            max_sequence_length=131072,
        )
    assert loads == []


def test_runtime_sized_kv_cache_is_never_native():
    """BF16 with --dynamic-kv-cache (now the default precision) builds the runtime-sized route."""
    from ..build_routing import native_kv_build_capability

    config = ModelConfig.from_json(json.dumps(_NATIVE_ELIGIBLE_CONFIG))
    assert native_kv_build_capability(config, precision="bf16").eligible
    config.raw["dynamic_kv_cache"] = True
    capability = native_kv_build_capability(config, precision="bf16")
    assert not capability.eligible
    assert "runtime-sized KV cache" in capability.reason


def test_dense_attention_mask_bound():
    from ..build_routing import dense_attention_mask_profile_error

    assert dense_attention_mask_profile_error(32768, 32768) is None
    assert dense_attention_mask_profile_error(64, 131072) is None
    assert "[131072, 262144]" in dense_attention_mask_profile_error(131072, 131072)
    assert "128.0 GiB" in dense_attention_mask_profile_error(131072, 131072)
    assert dense_attention_mask_profile_error(32769, 32769) is not None
    with pytest.raises(ValueError):
        dense_attention_mask_profile_error(0, 16)


def test_dense_builder_rejects_unbounded_prefill_profile():
    pytest.importorskip("tensorrt")
    from ..dual_profile_decoder_builder import build_dual_profile_decoder_engine

    config = ModelConfig.from_json(json.dumps(_NATIVE_ELIGIBLE_CONFIG))
    weights = {"embedding": np.zeros((32, 256), np.float16), "final_norm": np.ones(256)}
    with pytest.raises(ValueError, match="attention_mask profile maximum"):
        build_dual_profile_decoder_engine(config, weights, 131072, precision="fp16")


@pytest.mark.parametrize("entry", ["family", "shared"])
def test_cli_entry_points_default_to_bf16(tmp_path, monkeypatch, entry):
    from families.llama import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_NATIVE_ELIGIBLE_CONFIG))
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(tmp_path / "model.bundle")]
    if entry == "family":
        assert family_cli.main(["llama", "build", *args]) == 0
    else:
        assert build_cli.main(["build", *args, "--family", "llama"]) == 0
    assert [request.precision for request in captured] == ["bf16"]
    assert captured[0].max_sequence_length is None
