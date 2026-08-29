# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.minimax_h3 import audio_vae_builder
from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (
    AUDIO_REFERENCE_MAX_SAMPLES,
    AUDIO_REFERENCE_MIN_SAMPLES,
    AUDIO_REFERENCE_OPT_SAMPLES,
    AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES,
    AudioVaeDecoderConfig,
    _build_serialized_encoder_engine,
    _load_encoder_weights,
    _make_encoder_module,
    _remove_weight_normalization,
    build_audio_vae_encoder_engine,
    validate_audio_reference_samples,
)


def _official_config() -> dict:
    return {
        "encoder_dim": 64,
        "encoder_rates": [2, 4, 4, 5, 5],
        "latent_dim": 2048,
        "latent_channels": 32,
        "decoder_dim": 1024,
        "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
        "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
        "num_attention_heads": 8,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "sampling_rate": 32000,
        "latents_mean": [float(index) / 32 for index in range(32)],
        "latents_std": [1.0 + float(index) / 32 for index in range(32)],
    }


def _tiny_config() -> AudioVaeDecoderConfig:
    return AudioVaeDecoderConfig(
        latent_dim=4,
        latent_channels=2,
        decoder_dim=4,
        decoder_rates=(2,),
        decoder_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1,),),
        sampling_rate=32000,
        latents_mean=(1.0, -2.0),
        latents_std=(3.0, 4.0),
        encoder_dim=2,
        encoder_rates=(2,),
        num_attention_heads=1,
    )


def test_reference_audio_validation_accepts_exact_model_card_bounds() -> None:
    for samples, expected_latents in (
        (AUDIO_REFERENCE_MIN_SAMPLES, 80),
        (AUDIO_REFERENCE_OPT_SAMPLES, 207),
        (AUDIO_REFERENCE_MAX_SAMPLES, 600),
    ):
        waveform = np.zeros((2, 1, samples), dtype=np.float32)
        assert validate_audio_reference_samples(waveform, sample_rate=32000) == expected_latents


@pytest.mark.parametrize(
    ("samples", "sample_rate", "message"),
    [
        (np.zeros((2, 1, 64000), np.float64), 32000, "float32"),
        (np.zeros((1, 2, 64000), np.float32), 32000, "shape"),
        (np.zeros((2, 1, 64000), np.float32), 16000, "32000 Hz"),
        (np.zeros((2, 1, 63999), np.float32), 32000, "between 2 and 15"),
        (np.zeros((2, 1, 480001), np.float32), 32000, "between 2 and 15"),
        (np.zeros((2, 1, 64001), np.float32), 32000, "aligned to 800"),
    ],
)
def test_reference_audio_validation_rejects_wrong_abi(
    samples: np.ndarray, sample_rate: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_audio_reference_samples(samples, sample_rate=sample_rate)


def test_reference_audio_validation_rejects_nonfinite_or_out_of_range() -> None:
    waveform = np.zeros((2, 1, AUDIO_REFERENCE_MIN_SAMPLES), dtype=np.float32)
    waveform[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_audio_reference_samples(waveform, sample_rate=32000)

    waveform[0, 0, 0] = 1.01
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        validate_audio_reference_samples(waveform, sample_rate=32000)


def test_decoder_export_disables_mkldnn_only_while_tracing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    module = object()
    observed = {}

    monkeypatch.setattr(audio_vae_builder, "_make_decoder_module", lambda *_args: module)
    monkeypatch.setattr(audio_vae_builder, "_load_decoder_weights", lambda *_args: None)
    monkeypatch.setattr(audio_vae_builder, "_remove_weight_normalization", lambda *_args: None)

    def export(observed_module, dummy, output, **kwargs):
        observed.update(
            module=observed_module,
            shape=tuple(dummy.shape),
            mkldnn={
                name: getattr(torch.backends.mkldnn, name)
                for name in ("enabled", "deterministic", "allow_tf32", "fp32_precision")
            },
            kwargs=kwargs,
        )
        output.write(b"decoder-onnx")

    monkeypatch.setattr(torch.onnx, "export", export)
    original = {
        name: getattr(torch.backends.mkldnn, name)
        for name in ("enabled", "deterministic", "allow_tf32", "fp32_precision")
    }

    assert audio_vae_builder._export_decoder_onnx(tmp_path, _tiny_config(), False) == (
        b"decoder-onnx"
    )
    assert observed["module"] is module
    assert observed["shape"] == (2, 32, 207)
    assert observed["mkldnn"] == {**original, "enabled": False}
    assert observed["kwargs"]["opset_version"] == 17
    assert observed["kwargs"]["dynamo"] is False
    assert {
        name: getattr(torch.backends.mkldnn, name)
        for name in ("enabled", "deterministic", "allow_tf32", "fp32_precision")
    } == original


def test_encoder_reconstructs_only_posterior_mean_path_and_normalizes() -> None:
    torch = pytest.importorskip("torch")
    config = _tiny_config()
    module = _make_encoder_module(torch, config)
    state_keys = tuple(module.state_dict())
    assert state_keys
    assert all(name.startswith(("encoder.", "pre_block.", "mean_proj.")) for name in state_keys)
    assert not any("logs_proj" in name for name in state_keys)

    encoded = module(torch.zeros((2, 1, 8), dtype=torch.float32))
    assert tuple(encoded.shape) == (2, 4, 2)
    assert encoded.dtype == torch.float32

    mean = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

    class FixedEncoder(torch.nn.Module):
        def forward(self, _samples):
            return torch.zeros((2, 4, 4), dtype=torch.float32)

    class FixedMean(torch.nn.Module):
        def forward(self, _hidden_states):
            return mean

    module.encoder = FixedEncoder()
    module.pre_block = torch.nn.Identity()
    module.mean_proj = FixedMean()
    expected = (mean.transpose(1, 2) - torch.tensor([[[1.0, -2.0]]])) / torch.tensor([[[3.0, 4.0]]])
    assert torch.equal(module(torch.zeros((2, 1, 8), dtype=torch.float32)), expected)


def test_encoder_weight_selection_is_exact_and_fp32(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    module = _make_encoder_module(torch, _tiny_config())
    state = {name: value.detach().clone() for name, value in module.state_dict().items()}
    observed = {}

    def load(root, names):
        observed.update(root=root, names=tuple(names))
        return state

    monkeypatch.setattr(audio_vae_builder, "load_selected_component_state_dict", load)
    _load_encoder_weights(torch, module, tmp_path)
    assert observed["root"] == tmp_path
    assert set(observed["names"]) == set(state)
    assert not any("logs_proj" in name for name in observed["names"])

    first = next(iter(state))
    state[first] = state[first].to(torch.float16)
    with pytest.raises(ValueError, match="must be float32"):
        _load_encoder_weights(torch, module, tmp_path)


def test_tiny_encoder_exports_one_genuine_dynamic_onnx_graph() -> None:
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    reference = pytest.importorskip("onnx.reference")
    config = _tiny_config()
    torch.manual_seed(7)
    module = _make_encoder_module(torch, config)
    _remove_weight_normalization(torch, module)
    buffer = io.BytesIO()
    torch.onnx.export(
        module,
        torch.zeros((2, 1, 8), dtype=torch.float32),
        buffer,
        opset_version=17,
        input_names=["audio_samples"],
        output_names=["audio_condition_rows"],
        dynamic_axes={
            "audio_samples": {2: "num_samples"},
            "audio_condition_rows": {1: "num_audio_latents"},
        },
        dynamo=False,
    )
    model = onnx.load_model_from_string(buffer.getvalue())
    onnx.checker.check_model(model)
    assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    assert "Trilu" in {node.op_type for node in model.graph.node}
    input_dims = model.graph.input[0].type.tensor_type.shape.dim
    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    assert [input_dims[0].dim_value, input_dims[1].dim_value, input_dims[2].dim_param] == [
        2,
        1,
        "num_samples",
    ]
    assert [output_dims[0].dim_value, output_dims[1].dim_param, output_dims[2].dim_value] == [
        2,
        "num_audio_latents",
        2,
    ]

    evaluator = reference.ReferenceEvaluator(model)
    for sample_count in (8, 12):
        samples = np.linspace(-1.0, 1.0, 2 * sample_count, dtype=np.float32).reshape(
            2, 1, sample_count
        )
        with torch.inference_mode():
            expected = module(torch.from_numpy(samples)).numpy()
        (actual,) = evaluator.run(None, {"audio_samples": samples})
        assert actual.shape == (2, sample_count // 2, 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_encoder_builder_traces_then_builds_dynamic_plan(tmp_path: Path, monkeypatch) -> None:
    audio_vae_dir = tmp_path / "audio_vae"
    audio_vae_dir.mkdir()
    (audio_vae_dir / "config.json").write_text(json.dumps(_official_config()))
    observed = {}

    def export(root, config, verbose):
        observed.update(root=root, config=config, export_verbose=verbose)
        return b"encoder-torchscript"

    def build(module_bytes, *, verbose, workspace_bytes, metadata_out):
        observed.update(
            module_bytes=module_bytes,
            build_verbose=verbose,
            workspace_bytes=workspace_bytes,
            metadata_out=metadata_out,
        )
        return b"encoder-plan"

    monkeypatch.setattr(audio_vae_builder, "_export_encoder_torchscript", export)
    monkeypatch.setattr(audio_vae_builder, "_build_serialized_encoder_engine", build)
    assert (
        build_audio_vae_encoder_engine(audio_vae_dir, verbose=True, workspace_bytes=8 << 30)
        == b"encoder-plan"
    )
    assert observed["root"] == audio_vae_dir
    assert observed["config"].encoder_hop_length == 800
    assert observed["export_verbose"] is True
    assert observed["module_bytes"] == b"encoder-torchscript"
    assert observed["build_verbose"] is True
    assert observed["workspace_bytes"] == 8 << 30
    assert observed["metadata_out"] is None


def test_encoder_torchscript_graph_requires_38_bound_cudnn_tf32_convolutions() -> None:
    class Value:
        def __init__(self, value=None):
            self.value = value

        def toIValue(self):
            return self.value

    class Node:
        def __init__(self, kind="aten::_convolution", allow_tf32=True):
            self._kind = kind
            self._inputs = [Value() for _ in range(9)] + [
                Value(False),
                Value(False),
                Value(True),
                Value(allow_tf32),
            ]

        def kind(self):
            return self._kind

        def inputs(self):
            return iter(self._inputs)

        def blocks(self):
            return ()

    class Graph:
        def __init__(self, selected):
            self.selected = selected

        def nodes(self):
            return iter(self.selected)

    valid = SimpleNamespace(inlined_graph=Graph([Node() for _ in range(38)]))
    audio_vae_builder._validate_encoder_torchscript_graph(valid)

    for invalid in (
        SimpleNamespace(inlined_graph=Graph([Node() for _ in range(37)])),
        SimpleNamespace(inlined_graph=Graph([Node() for _ in range(37)] + [Node("aten::conv1d")])),
        SimpleNamespace(
            inlined_graph=Graph([Node() for _ in range(37)] + [Node(allow_tf32=False)])
        ),
    ):
        with pytest.raises(RuntimeError, match="convolution"):
            audio_vae_builder._validate_encoder_torchscript_graph(invalid)


def test_encoder_production_plan_source_has_no_onnx_or_legacy_audio_subgraph() -> None:
    build_source = inspect.getsource(_build_serialized_encoder_engine)
    finish_source = inspect.getsource(audio_vae_builder._finish_serialized_encoder_engine)
    assert "OnnxParser" not in build_source + finish_source
    assert "add_audio_encoder_plugin" in build_source
    assert "finally:" in build_source
    assert "release_audio_encoder_module_storage(network)" in build_source
    assert build_source.index("try:") < build_source.index(
        "output_tensor = add_audio_encoder_plugin"
    )
    assert '"CONSTANT"' in finish_source
    assert '"PLUGIN_V3"' in finish_source
    assert "len(layers) != 2" in finish_source
    assert 'getattr(plugin_layer, "num_inputs", -1) != 2' in finish_source


def test_encoder_builder_releases_storage_when_plugin_add_raises(monkeypatch) -> None:
    network = SimpleNamespace(
        add_input=lambda *_args: object(),
    )

    class Builder:
        def __init__(self, _logger):
            pass

        def create_network(self, _flags):
            return network

    class Logger:
        INFO = 1
        WARNING = 2

        def __init__(self, _severity):
            pass

    fake_trt = SimpleNamespace(
        Builder=Builder,
        Logger=Logger,
        float32=object(),
    )
    monkeypatch.setattr(audio_vae_builder.trt_compat, "get_trt", lambda: fake_trt)
    monkeypatch.setattr(audio_vae_builder.trt_compat, "network_creation_flags", lambda **_kw: 1)
    from tensorrt_model_connect.families.minimax_h3 import native_plugin_builder

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda **_kwargs: None)
    monkeypatch.setattr(
        native_plugin_builder,
        "add_audio_encoder_plugin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plugin add failed")),
    )
    released = []
    monkeypatch.setattr(
        native_plugin_builder,
        "release_audio_encoder_module_storage",
        lambda selected: released.append(selected),
    )
    with pytest.raises(RuntimeError, match="plugin add failed"):
        _build_serialized_encoder_engine(b"module", verbose=False, workspace_bytes=None)
    assert released == [network]


def test_encoder_export_releases_cuda_trace_references_before_emptying_cache() -> None:
    source = inspect.getsource(audio_vae_builder._export_encoder_torchscript)
    assert "torch.linspace(-0.75, 0.75" in source
    assert "torch.backends.cudnn.allow_tf32 = True" in source
    assert "torch.backends.cudnn.enabled = True" in source
    assert "torch.backends.cudnn.benchmark = False" in source
    assert "torch.backends.cudnn.deterministic = False" in source
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "with torch.jit.optimized_execution(False)" in source
    assert "for repeat in range(2)" in source
    release = "del output, traced, frozen_outputs, live_outputs, samples, module"
    assert release in source
    assert source.index(release) < source.index("torch.cuda.empty_cache()")


def test_encoder_plugin_contract_clears_tf32_and_adds_exact_dynamic_profile(monkeypatch) -> None:
    observed = {}
    fp32 = object()
    int8 = object()
    module_bytes = b"torchscript"

    class Tensor:
        def __init__(self, name, shape, dtype=fp32):
            self.name = name
            self.shape = shape
            self.dtype = dtype

    class Network:
        num_inputs = 1
        num_outputs = 1
        num_layers = 2

        def __init__(self):
            self.input = None
            self.output = None

        def add_input(self, name, dtype, shape):
            self.input = Tensor(name, shape)
            self.input.dtype = dtype
            return self.input

        def mark_output(self, output):
            self.output = output

        def get_input(self, index):
            assert index == 0
            return self.input

        def get_output(self, index):
            assert index == 0
            return self.output

        def get_layer(self, index):
            if index == 0:
                return SimpleNamespace(
                    type="constant",
                    num_inputs=0,
                    get_output=lambda output_index: (
                        Tensor("audio_encoder_module", (len(module_bytes),), int8)
                        if output_index == 0
                        else None
                    ),
                )
            assert index == 1
            return SimpleNamespace(
                type="plugin-v3",
                num_inputs=2,
                metadata=(
                    "trtmc.native_op=MiniMaxH3AudioEncoder;source=audio_encoder;"
                    f"module_bytes={len(module_bytes)};"
                    f"module_sha256={hashlib.sha256(module_bytes).hexdigest()}"
                ),
            )

    class Profile:
        def set_shape(self, name, minimum, optimum, maximum):
            observed["profile"] = (name, minimum, optimum, maximum)
            return True

    class BuildConfig:
        def set_memory_pool_limit(self, pool, size):
            observed.update(pool=pool, workspace=size)

        def get_memory_pool_limit(self, pool):
            assert pool == "workspace"
            return observed["workspace"]

        def clear_flag(self, flag):
            observed.setdefault("cleared_flags", []).append(flag)

        def add_optimization_profile(self, profile):
            observed["added_profile"] = profile
            return 0

    class Builder:
        def __init__(self, _logger):
            pass

        def create_network(self, flags):
            observed["flags"] = flags
            return Network()

        def create_builder_config(self):
            return BuildConfig()

        def create_optimization_profile(self):
            return Profile()

        def build_serialized_network(self, _network, _config):
            return b"encoder-plan"

    class Logger:
        INFO = "info"
        WARNING = "warning"

        def __init__(self, severity):
            observed["severity"] = severity

    fake_trt = SimpleNamespace(
        Logger=Logger,
        Builder=Builder,
        BuilderFlag=SimpleNamespace(TF32="tf32"),
        LayerType=SimpleNamespace(PLUGIN_V3="plugin-v3"),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
        float32=fp32,
        int8=int8,
    )
    fake_trt.LayerType.CONSTANT = "constant"
    monkeypatch.setattr(audio_vae_builder.trt_compat, "get_trt", lambda: fake_trt)
    monkeypatch.setattr(audio_vae_builder.trt_compat, "network_creation_flags", lambda **_kw: 9)
    from tensorrt_model_connect.families.minimax_h3 import native_plugin_builder

    monkeypatch.setattr(native_plugin_builder, "load_native_plugin", lambda **_kwargs: None)
    monkeypatch.setattr(
        native_plugin_builder,
        "release_audio_encoder_module_storage",
        lambda network: observed.update(released_network=network),
    )

    def add_plugin(network, audio_samples, module_bytes, **_kwargs):
        observed.update(audio_samples=audio_samples, module_bytes=module_bytes)
        return Tensor("audio_encoder", (2, -1, 32))

    monkeypatch.setattr(native_plugin_builder, "add_audio_encoder_plugin", add_plugin)

    assert (
        _build_serialized_encoder_engine(
            module_bytes,
            verbose=False,
            workspace_bytes=None,
            metadata_out=observed.setdefault("metadata", {}),
        )
        == b"encoder-plan"
    )
    assert observed["profile"] == (
        "audio_samples",
        (2, 1, AUDIO_REFERENCE_MIN_SAMPLES),
        (2, 1, AUDIO_REFERENCE_OPT_SAMPLES),
        (2, 1, AUDIO_REFERENCE_MAX_SAMPLES),
    )
    assert observed["workspace"] == AUDIO_VAE_ENCODER_DEFAULT_WORKSPACE_BYTES
    assert observed["cleared_flags"] == ["tf32"]
    assert observed["added_profile"].__class__ is Profile
    assert observed["module_bytes"] == b"torchscript"
    assert observed["audio_samples"].name == "audio_samples"
    assert observed["flags"] == 9
    assert observed["released_network"].num_layers == 2
    assert observed["metadata"] == {
        "implementation": "aten-torchscript-fp32-v1",
        "plugin_count": 1,
        "module_format": "torchscript-plan-constant-v1",
        "module_bytes": len(module_bytes),
        "module_sha256": hashlib.sha256(module_bytes).hexdigest(),
        "weight_norm": "cuda-frozen-effective-v1",
        "input_profile": [
            AUDIO_REFERENCE_MIN_SAMPLES,
            AUDIO_REFERENCE_OPT_SAMPLES,
            AUDIO_REFERENCE_MAX_SAMPLES,
        ],
        "network_layer_counts": {"constant": 1, "plugin_v3": 1},
        "plugin_inputs": 2,
        "python_runtime": False,
        "cuda_graphs": False,
        "cudnn_tf32": True,
        "matmul_tf32": False,
        "graph_optimizer": False,
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": False,
        "plan_bytes": len(b"encoder-plan"),
    }
