# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.models.fast_foundation_stereo.e2e_plugins.comparator import (
    StereoDisparityComparator,
)
from tests.e2e_harness.contracts import StageOutput, StageSpec, ThresholdProfile
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.fast_foundation_stereo import (
    FastFoundationStereoPlugin,
)
from tensorrt_model_connect.families.fast_foundation_stereo.model_config import (
    config_from_dir,
)
from tensorrt_model_connect.families.fast_foundation_stereo.prepare_model import (
    configure_official_model_args,
    install_official_io_import_shims,
    resolve_model_dir,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_FAMILY_SOURCE_DIR = (
    _REPOSITORY_ROOT / "python/tensorrt_model_connect/families/fast_foundation_stereo"
)
_RUNTIME_SOURCE_DIR = _REPOSITORY_ROOT / "src/runtime/models/fast_foundation_stereo"


def _model_dir(tmp_path: Path) -> Path:
    (tmp_path / "core").mkdir()
    (tmp_path / "core/foundation_stereo.py").write_text("# source\n")
    (tmp_path / "core/submodule.py").write_text("# source\n")
    checkpoint = tmp_path / "weights/23-36-37/model_best_bp2_serialize.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "benchmark_result.json").write_text(
        json.dumps({"max_disp": 192, "valid_iters": 8}), encoding="utf-8"
    )
    return tmp_path


def test_config_adapter_claims_only_complete_source_package(tmp_path: Path) -> None:
    assert config_from_dir(tmp_path) is None
    model_dir = _model_dir(tmp_path)
    config = config_from_dir(model_dir)
    assert config is not None
    assert config["model_type"] == "fast_foundation_stereo"
    assert config["runtime_strategy"] == "fast_foundation_stereo_disparity"
    assert config["stereo_engine_height"] == 704
    assert config["stereo_engine_width"] == 704
    assert config["stereo_accuracy_metric"] == "cosine_epe_bad2"
    assert config["stereo_min_cosine"] == 0.999
    assert config["stereo_max_mean_abs_error"] == 0.5
    assert config["stereo_max_bad_2px_fraction"] == 0.02


def test_plugin_owns_unique_strategy_and_skips_tokenizer() -> None:
    plugin = FastFoundationStereoPlugin()
    assert plugin.name == "fast_foundation_stereo"
    assert plugin.runtime_strategy == "fast_foundation_stereo_disparity"
    assert plugin.requires_tokenizer is False
    assert plugin.matches("fast-foundation-stereo")
    assert not plugin.matches("segformer")


def test_official_io_import_shims_cover_only_missing_optional_modules(
    monkeypatch,
) -> None:
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in ("cv2", "imageio")}
    for name in previous:
        sys.modules.pop(name, None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name in previous else real_find_spec(name),
    )

    try:
        install_official_io_import_shims()
        assert sys.modules["cv2"].COLORMAP_TURBO == 20
        assert sys.modules["imageio"].__name__ == "imageio"
    finally:
        for name, module in previous.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_official_model_args_include_normalized_gwc_contract() -> None:
    model = SimpleNamespace(args=SimpleNamespace())

    configure_official_model_args(
        model,
        max_disparity=192,
        valid_iters=8,
    )

    assert model.args.max_disp == 192
    assert model.args.valid_iters == 8
    assert model.args.normalize is True


def test_flat_hf_checkpoint_is_staged_with_pinned_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    for relative in (
        "Utils.py",
        "core/foundation_stereo.py",
        "core/submodule.py",
        "core/geometry.py",
        "core/extractor.py",
        "core/update.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_best_bp2_serialize.pth").write_bytes(b"checkpoint")
    (snapshot / "cfg.yaml").write_text("valid_iters: 8\n", encoding="utf-8")
    monkeypatch.setenv("TRTMC_FAST_FOUNDATION_STEREO_SOURCE_DIR", str(source))
    monkeypatch.setenv("TRTMC_FAST_FOUNDATION_STEREO_CACHE", str(tmp_path / "cache"))

    staged = resolve_model_dir(snapshot)
    assert staged is not None
    assert (staged / "core/foundation_stereo.py").is_file()
    assert (staged / "weights/23-36-37/model_best_bp2_serialize.pth").read_bytes() == b"checkpoint"
    assert config_from_dir(staged) is not None


def test_local_only_staging_requires_cached_pinned_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model_best_bp2_serialize.pth").write_bytes(b"checkpoint")
    monkeypatch.delenv("TRTMC_FAST_FOUNDATION_STEREO_SOURCE_DIR", raising=False)
    monkeypatch.setenv("TRTMC_FAST_FOUNDATION_STEREO_CACHE", str(tmp_path / "cache"))

    with pytest.raises(FileNotFoundError, match="not present in the local cache"):
        resolve_model_dir(snapshot, local_files_only=True)


def test_plugin_builds_feature_and_family_owned_post_section(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    raw = config_from_dir(model_dir)
    assert raw is not None
    config = ModelConfig.from_json(json.dumps(raw))
    plugin = FastFoundationStereoPlugin()
    weights = plugin.load_weights(str(model_dir), config, precision="fp16")

    from tensorrt_model_connect.families.fast_foundation_stereo import builder

    calls = []

    def feature(model_dir, **kwargs):
        calls.append(("feature", model_dir, kwargs))
        return b"feature-plan"

    def post(model_dir, **kwargs):
        calls.append(("post", model_dir, kwargs))
        return b"post-plan"

    monkeypatch.setattr(builder, "build_feature_engine", feature)
    monkeypatch.setattr(builder, "build_post_engine", post)
    from tensorrt_model_connect.families.fast_foundation_stereo import native_plugin_builder

    native_plugin = tmp_path / "native-plugin.so"
    native_plugin.write_bytes(b"native-plugin")
    monkeypatch.setattr(
        native_plugin_builder,
        "ensure_native_plugin",
        lambda **kwargs: native_plugin,
    )
    assert plugin.build_engine(config, weights, 1, precision="fp16") == b"feature-plan"
    assert plugin.build_extra_engines(config, weights, 1, precision="fp16") == {
        "fast_foundation_stereo_post_engine_plan": b"post-plan",
        "fast_foundation_stereo_native_plugin_so": b"native-plugin",
    }
    assert [call[0] for call in calls] == ["feature", "post"]
    assert all(call[2]["max_disparity"] == 192 for call in calls)
    assert all(call[2]["valid_iters"] == 8 for call in calls)


def test_native_builder_rejects_precision_not_supported_by_combined_volume_plugin() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import builder

    assert builder._validate_precision("fp16") is True
    with pytest.raises(ValueError, match="supports precision='fp16' only"):
        builder._validate_precision("fp32")


def test_native_builder_defaults_keep_feature_strong_and_tune_post_weak(
    monkeypatch,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import builder

    monkeypatch.delenv("TRTMC_FAST_FOUNDATION_STEREO_OPT_LEVEL", raising=False)
    monkeypatch.delenv("TRTMC_FAST_FOUNDATION_STEREO_AUX_STREAMS", raising=False)

    class Config:
        def __init__(self) -> None:
            self.flags = []
            self.builder_optimization_level = -1
            self.max_aux_streams = -1
            self.workspace = None

        def set_flag(self, flag) -> None:
            self.flags.append(flag)

        def set_memory_pool_limit(self, pool, size) -> None:
            self.workspace = (pool, size)

    class Builder:
        def __init__(self) -> None:
            self.configs = []

        def create_builder_config(self):
            config = Config()
            self.configs.append(config)
            return config

        @staticmethod
        def build_serialized_network(_network, _config):
            return b"plan"

    trt = SimpleNamespace(
        BuilderFlag=SimpleNamespace(FP16="fp16"),
        MemoryPoolType=SimpleNamespace(WORKSPACE="workspace"),
    )
    fake_builder = Builder()

    assert (
        builder._serialize_network(
            trt,
            fake_builder,
            object(),
            fp16=True,
            strongly_typed=True,
            default_optimization_level=5,
            default_aux_streams=2,
            verbose=False,
        )
        == b"plan"
    )
    assert (
        builder._serialize_network(
            trt,
            fake_builder,
            object(),
            fp16=True,
            strongly_typed=False,
            default_optimization_level=4,
            default_aux_streams=0,
            verbose=False,
        )
        == b"plan"
    )

    feature, post = fake_builder.configs
    assert feature.flags == []
    assert feature.builder_optimization_level == 5
    assert feature.max_aux_streams == 2
    assert post.flags == ["fp16"]
    assert post.builder_optimization_level == 4
    assert post.max_aux_streams == 0
    assert feature.workspace == post.workspace == ("workspace", 8 << 30)


def test_native_builder_uses_strong_post_when_weak_fp16_was_removed() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import builder

    trt10 = SimpleNamespace(BuilderFlag=SimpleNamespace(FP16="fp16"))
    trt11 = SimpleNamespace(BuilderFlag=SimpleNamespace())

    assert builder._post_network_strongly_typed(trt10) is False
    assert builder._post_network_strongly_typed(trt11) is True


def test_production_family_uses_only_native_tensorrt_network_definition() -> None:
    sources = sorted(
        path
        for root in (_FAMILY_SOURCE_DIR, _RUNTIME_SOURCE_DIR)
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in {".cpp", ".cu", ".h", ".py"} or path.name == "CMakeLists.txt")
    )
    assert sources
    source_by_path = {
        path.relative_to(_REPOSITORY_ROOT): path.read_text(encoding="utf-8") for path in sources
    }
    forbidden = {
        path: [
            (line_number, line.strip())
            for line_number, line in enumerate(text.splitlines(), start=1)
            if "onnx" in line.casefold()
        ]
        for path, text in source_by_path.items()
    }
    forbidden = {path: lines for path, lines in forbidden.items() if lines}
    assert forbidden == {}

    combined = "\n".join(text for path, text in source_by_path.items() if path.suffix == ".py")
    assert ".create_network(" in combined
    native_layer = re.compile(
        r"\.add_(?:activation|concatenation|constant|convolution_nd|"
        r"deconvolution_nd|elementwise|grid_sample|matrix_multiply|"
        r"normalization|plugin_v[23]|resize|shuffle|slice|softmax)\("
    )
    assert native_layer.search(combined), "no native TensorRT layer construction found"


def test_post_engine_consumes_only_feature_engine_outputs() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import builder

    assert builder.POST_INPUT_NAMES == builder.FEATURE_OUTPUT_NAMES
    assert "gwc_volume" not in builder.POST_INPUT_NAMES


def test_native_basic_conv_preserves_distilled_boolean_activation_semantics() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    class Conv2d:
        kernel_size = (3, 3)

    class Identity:
        pass

    graph = object.__new__(NativeGraph)
    graph.conv2d = lambda _tensor, _module: "conv"
    graph.activation = lambda tensor, kind, alpha=None: (tensor, kind, alpha)

    disabled = SimpleNamespace(conv=Conv2d(), bn=Identity(), relu=False)
    enabled = SimpleNamespace(conv=Conv2d(), bn=Identity(), relu=True)

    assert NativeGraph.basic_conv(graph, "input", disabled) == "conv"
    assert NativeGraph.basic_conv(graph, "input", enabled) == (
        "conv",
        "leaky_relu",
        0.01,
    )


def test_spatial_attention_uses_native_two_output_reduction_plugin(monkeypatch) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _spatial_attention,
    )
    from tensorrt_model_connect.families.fast_foundation_stereo import native_plugin_builder

    class Tensor:
        def __init__(self, shape, name, dtype="float16") -> None:
            self.shape = shape
            self.name = name
            self.dtype = dtype

    class Graph:
        def __init__(self) -> None:
            self.network = object()
            self.trt = SimpleNamespace(float16="float16")
            self.cast_calls = []

        def cast(self, tensor, dtype):
            self.cast_calls.append((tensor, dtype))
            return tensor

        @staticmethod
        def concat(tensors, axis):
            assert axis == 1
            assert tuple(tensor.shape for tensor in tensors) == (
                (1, 1, 176, 176),
                (1, 1, 176, 176),
            )
            return Tensor((1, 2, 176, 176), "concat")

        @staticmethod
        def conv2d(tensor, module):
            assert tensor.shape == (1, 2, 176, 176)
            return Tensor((1, 1, 176, 176), str(module))

        @staticmethod
        def activation(tensor, kind):
            return tensor, kind

    graph = Graph()
    plugin_calls = []

    def add_spatial_attention_reduce_plugin(network, tensor, *, trt_module):
        plugin_calls.append((network, tensor, trt_module))
        return (
            Tensor((1, 1, 176, 176), "average"),
            Tensor((1, 1, 176, 176), "maximum"),
        )

    monkeypatch.setattr(
        native_plugin_builder,
        "add_spatial_attention_reduce_plugin",
        add_spatial_attention_reduce_plugin,
    )
    tensor = Tensor((1, 48, 176, 176), "input")
    output = _spatial_attention(
        graph,
        tensor,
        SimpleNamespace(samconv="samconv"),
    )

    assert graph.cast_calls == [(tensor, "float16")]
    assert plugin_calls == [(graph.network, tensor, graph.trt)]
    assert output[0].shape == (1, 1, 176, 176)
    assert output[1] == "sigmoid"


def test_native_graph_retains_weight_buffers_until_engine_build() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    class Weights:
        def __init__(self, values=None) -> None:
            self.values = values

    class Tensor:
        dtype = "float32"
        shape = (1, 1, 2, 2)

    class Layer:
        def __init__(self) -> None:
            self.output = Tensor()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def add_constant(self, _shape, _weights):
            return Layer()

        def add_convolution_nd(self, *_args):
            return Layer()

    trt = SimpleNamespace(float16="float16", float32="float32", Weights=Weights)
    graph = NativeGraph(Network(), trt, fp16=False)
    graph.constant(np.ones((1,), dtype=np.float32))
    convolution = SimpleNamespace(
        weight=np.ones((1, 1, 1, 1), dtype=np.float32),
        bias=np.ones((1,), dtype=np.float32),
        out_channels=1,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    graph.conv2d(Tensor(), convolution)

    assert len(graph._weight_buffers) == 3
    assert all(buffer.flags.c_contiguous for buffer in graph._weight_buffers)


def test_native_graph_stacks_compatible_convolution_outputs() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    graph = object.__new__(NativeGraph)
    captured = []
    graph.conv2d = lambda tensor, module: captured.append((tensor, module)) or "output"
    first = SimpleNamespace(
        weight=np.full((2, 3, 1, 1), 1.0, dtype=np.float32),
        bias=np.full((2,), 2.0, dtype=np.float32),
        out_channels=2,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    second = SimpleNamespace(
        weight=np.full((4, 3, 1, 1), 3.0, dtype=np.float32),
        bias=np.full((4,), 4.0, dtype=np.float32),
        out_channels=4,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )

    assert NativeGraph.stacked_conv2d(graph, "input", (first, second)) == "output"
    assert len(captured) == 1
    tensor, combined = captured[0]
    assert tensor == "input"
    assert combined.out_channels == 6
    np.testing.assert_array_equal(combined.weight[:2], first.weight)
    np.testing.assert_array_equal(combined.weight[2:], second.weight)
    np.testing.assert_array_equal(combined.bias[:2], first.bias)
    np.testing.assert_array_equal(combined.bias[2:], second.bias)

    first.groups = 2
    second.groups = 2
    with pytest.raises(ValueError, match="ungrouped convolutions only"):
        NativeGraph.stacked_conv2d(graph, "input", (first, second))


def test_raft_gru_stacks_update_and_reset_convolutions() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import _raft_gru

    class Tensor:
        shape = (1, 60, 176, 176)

    class Graph:
        def __init__(self) -> None:
            self.stacked = []
            self.slices = []

        def stacked_conv2d(self, tensor, modules):
            self.stacked.append((tensor, modules))
            return "gates"

        def slice(self, tensor, start, shape):
            self.slices.append((tensor, start, shape))
            return Tensor()

        @staticmethod
        def activation(_tensor, _kind):
            return Tensor()

        @staticmethod
        def mul(_lhs, _rhs):
            return Tensor()

        @staticmethod
        def concat(_tensors, _axis):
            return Tensor()

        @staticmethod
        def conv2d(_tensor, _module):
            return Tensor()

        @staticmethod
        def scalar(_value, _rank, *, like):
            return like

        @staticmethod
        def sub(_lhs, _rhs):
            return Tensor()

        @staticmethod
        def add(_lhs, _rhs):
            return Tensor()

    graph = Graph()
    module = SimpleNamespace(convz="update", convr="reset", convq="proposal")
    hidden = Tensor()
    _raft_gru(graph, hidden, "x", "hx", module)

    assert graph.stacked == [("hx", ("update", "reset"))]
    assert graph.slices == [
        ("gates", (0, 0, 0, 0), (1, 60, 176, 176)),
        ("gates", (0, 60, 0, 0), (1, 60, 176, 176)),
    ]


def test_performance_tools_do_not_build_framework_side_gwc() -> None:
    for name in ("benchmark.py", "profile_ncu.py", "trt_runner.py"):
        source = (Path(__file__).parent / name).read_text(encoding="utf-8")
        assert "build_gwc_volume" not in source
        assert "gwc_volume" not in source


def test_requested_native_plugin_is_loaded_globally(tmp_path: Path, monkeypatch) -> None:
    from tests.e2e.models.fast_foundation_stereo import trt_runner

    library = tmp_path / "libstereo_plugin.so"
    library.write_bytes(b"test")
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        trt_runner.ctypes,
        "CDLL",
        lambda path, mode: calls.append((path, mode)) or sentinel,
    )
    trt_runner._PLUGIN_HANDLES.clear()

    loaded = trt_runner.load_native_plugin_libraries([library])

    assert loaded == [str(library.resolve())]
    assert calls == [(str(library.resolve()), trt_runner.ctypes.RTLD_GLOBAL)]
    assert trt_runner._PLUGIN_HANDLES == [sentinel]
    trt_runner._PLUGIN_HANDLES.clear()


def test_disparity_reference_is_bound_to_selected_pair_names(tmp_path: Path) -> None:
    from tests.e2e.models.fast_foundation_stereo.trt_runner import (
        load_named_disparity_reference,
    )

    reference = tmp_path / "reference.npz"
    disparity = np.zeros((2, 4, 4), dtype=np.float32)
    np.savez(reference, names=np.asarray(["right.png", "left.png"]), disparity=disparity)

    with pytest.raises(RuntimeError, match="do not match selected pairs"):
        load_named_disparity_reference(
            reference,
            ["left.png", "right.png"],
            disparity.shape,
        )

    np.savez(reference, names=np.asarray(["left.png", "right.png"]), disparity=disparity)
    loaded = load_named_disparity_reference(
        reference,
        ["left.png", "right.png"],
        disparity.shape,
    )
    np.testing.assert_array_equal(loaded, disparity)


def test_l4_receipt_preserves_native_measurement_boundaries() -> None:
    receipt = json.loads((Path(__file__).parent / "performance/l4.json").read_text())

    assert receipt["schema_version"] == 2
    assert receipt["measurement_status"] == "current_native_l4_latency_and_accuracy_qualified"
    assert receipt["hardware"]["gpu"] == "NVIDIA L4"
    assert receipt["hardware"]["compute_capability"] == [8, 9]

    implementation = receipt["implementation"]
    assert implementation["backend"] == "tensorrt"
    assert implementation["network_definition"] == "native_api"
    assert implementation["onnx_used"] is False
    assert implementation["builder"] == {
        "feature_optimization_level": 5,
        "feature_auxiliary_streams": 2,
        "post_optimization_level": 4,
        "post_auxiliary_streams": 0,
        "workspace_gib": 8,
    }

    protocol = receipt["protocol"]
    assert protocol["pairs_per_iteration"] == 5
    assert protocol["independent_processes"] == 5
    assert protocol["warmup_iterations_per_process"] == 20
    assert protocol["timed_iterations_per_process"] == 100
    assert protocol["cuda_graphs"] is True

    accuracy = receipt["accuracy"]
    thresholds = accuracy["thresholds"]
    observed = accuracy["observed"]
    assert observed["global_cosine"] >= thresholds["minimum_global_cosine"]
    assert observed["mean_abs_error_px"] <= thresholds["maximum_mean_abs_error_px"]
    assert observed["bad_2px_fraction"] <= thresholds["maximum_bad_2px_fraction"]
    assert accuracy["identical_output_across_runs"] is True
    assert accuracy["all_runs_passed"] is True

    latency = receipt["latency"]
    runs = latency["sustained_runs"]
    assert [run["run"] for run in runs] == [1, 2, 3, 4, 5]
    assert all(run["accuracy_passed"] for run in runs)
    assert all(run["inference_5_pairs"]["count"] == 100 for run in runs)
    output_hashes = {run["output"]["sha256"] for run in runs}
    assert output_hashes == {receipt["artifacts"]["shared_output"]["sha256"]}

    run_means = np.asarray([run["inference_5_pairs"]["mean_ms"] for run in runs], dtype=np.float64)
    cross_process = latency["cross_process_run_means"]
    assert cross_process["count"] == len(runs)
    assert cross_process["mean_ms"] == pytest.approx(float(np.mean(run_means)))
    assert cross_process["median_ms"] == pytest.approx(float(np.median(run_means)))
    assert cross_process["p90_ms"] == pytest.approx(float(np.percentile(run_means, 90)))
    assert cross_process["p95_ms"] == pytest.approx(float(np.percentile(run_means, 95)))
    assert cross_process["p99_ms"] == pytest.approx(float(np.percentile(run_means, 99)))
    assert cross_process["min_ms"] == pytest.approx(float(np.min(run_means)))
    assert cross_process["max_ms"] == pytest.approx(float(np.max(run_means)))
    assert cross_process["stddev_ms"] == pytest.approx(float(np.std(run_means)))
    assert cross_process["coefficient_of_variation"] == pytest.approx(
        float(np.std(run_means) / np.mean(run_means))
    )

    aggregate = latency["aggregate_500_samples"]
    assert aggregate["count"] == sum(run["inference_5_pairs"]["count"] for run in runs) == 500
    assert aggregate["mean_ms"] == pytest.approx(float(np.mean(run_means)))
    assert aggregate["min_ms"] == min(run["inference_5_pairs"]["min_ms"] for run in runs)
    assert aggregate["max_ms"] == max(run["inference_5_pairs"]["max_ms"] for run in runs)
    pooled_variance = (
        sum(
            run["inference_5_pairs"]["count"]
            * (
                run["inference_5_pairs"]["stddev_ms"] ** 2
                + (run["inference_5_pairs"]["mean_ms"] - aggregate["mean_ms"]) ** 2
            )
            for run in runs
        )
        / aggregate["count"]
    )
    assert aggregate["stddev_ms"] == pytest.approx(pooled_variance**0.5)
    assert aggregate["coefficient_of_variation"] == pytest.approx(
        aggregate["stddev_ms"] / aggregate["mean_ms"]
    )
    assert (
        aggregate["min_ms"]
        <= aggregate["median_ms"]
        <= aggregate["p90_ms"]
        <= aggregate["p95_ms"]
        <= aggregate["p99_ms"]
        <= aggregate["max_ms"]
    )

    recorded_baseline = latency["recorded_baseline"]
    baseline = recorded_baseline["inference_5_pairs_mean_ms"]
    baseline_source = recorded_baseline["source_receipt"]
    assert re.fullmatch(r"[0-9a-f]{40}", baseline_source["repository_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", baseline_source["sha256"])
    assert baseline_source["bytes"] > 0
    assert baseline_source["measurement_status"].startswith("superseded_")
    improvement = latency["improvement_from_recorded_baseline"]
    assert aggregate["mean_ms"] < baseline
    assert improvement["inference_throughput_speedup"] == pytest.approx(
        baseline / aggregate["mean_ms"]
    )
    assert improvement["inference_latency_reduction_percent"] == pytest.approx(
        (baseline - aggregate["mean_ms"]) / baseline * 100.0
    )

    roofline = receipt["roofline"]
    coverage = roofline["coverage"]
    assert coverage["exhaustive"] is True
    assert coverage["first_launch_id"] == 0
    assert coverage["last_launch_id"] + 1 == coverage["launch_count"]
    assert coverage["gaps"] == coverage["overlaps"] == []
    assert coverage["incomplete_launches"] == 0
    assert roofline["capture_protocol"]["launch_filters"] == []
    metrics = roofline["metrics"]
    gate = roofline["strict_gate"]
    assert metrics["sampled_duration_us"] > 0
    assert metrics["near_ceiling_duration_us"] == pytest.approx(
        metrics["sampled_duration_us"] * metrics["near_ceiling_duration_fraction"]
    )
    assert (
        metrics["compute_limited_launches"] + metrics["memory_limited_launches"]
        == coverage["launch_count"]
    )
    assert metrics["compute_limited_duration_fraction"] + metrics[
        "memory_limited_duration_fraction"
    ] == pytest.approx(1.0)
    assert metrics["near_ceiling_threshold_pct"] == gate["minimum_duration_weighted_limiter_pct"]
    expected_roofline_pass = (
        metrics["duration_weighted_limiter_pct"] >= gate["minimum_duration_weighted_limiter_pct"]
        and metrics["near_ceiling_duration_fraction"]
        >= gate["minimum_near_ceiling_duration_fraction"]
    )
    assert gate["roofline_passed"] == expected_roofline_pass
    assert gate["roofline_passed"] is False
    qualification = receipt["qualification"]
    expected_combined_pass = gate["latency_beats_recorded_baseline"] and expected_roofline_pass
    assert (
        gate["latency_beats_recorded_baseline"] is qualification["latency_beats_recorded_baseline"]
    )
    assert gate["combined_gate_passed"] == expected_combined_pass
    assert gate["combined_gate_passed"] is False
    assert "does not claim" in roofline["interpretation"]

    assert qualification == {
        "latency_beats_recorded_baseline": True,
        "accuracy_passed": True,
        "strict_roofline_passed": False,
        "combined_latency_and_roofline_gate_passed": False,
    }

    digest = re.compile(r"[0-9a-f]{64}")
    evidence = [
        receipt["artifacts"][name]
        for name in (
            "feature_engine",
            "post_engine",
            "native_plugin",
            "checkpoint",
            "accuracy_reference",
            "shared_output",
        )
    ]
    evidence.extend(run["benchmark_receipt"] for run in runs)
    evidence.extend(receipt["artifacts"]["tools"].values())
    evidence.extend(receipt["artifacts"]["roofline_evidence"].values())
    assert all(
        digest.fullmatch(value)
        for artifact in evidence
        for key, value in artifact.items()
        if key.endswith("sha256")
    )
    assert all(artifact["bytes"] > 0 for artifact in evidence)

    for input_pair in receipt["artifacts"]["inputs"]:
        assert digest.fullmatch(input_pair["left_sha256"])
        assert digest.fullmatch(input_pair["right_sha256"])
        assert input_pair["left_bytes"] > 0
        assert input_pair["right_bytes"] > 0

    tool_paths = {
        "benchmark": Path(__file__).with_name("benchmark.py"),
        "runner": Path(__file__).with_name("trt_runner.py"),
        "profiler": Path(__file__).with_name("profile_ncu.py"),
        "analyzer": Path(__file__).with_name("analyze_ncu.py"),
    }
    for name, path in tool_paths.items():
        recorded = receipt["artifacts"]["tools"][name]
        content = path.read_bytes()
        assert recorded["bytes"] == len(content)
        assert recorded["sha256"] == hashlib.sha256(content).hexdigest()


def test_disparity_comparator_gates_structure_and_pixel_error() -> None:
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    comparator = StereoDisparityComparator()
    threshold = ThresholdProfile(
        task_strategy="stereo_disparity",
        metrics={
            "finite_fraction": 1.0,
            "global_cosine": 0.999,
            "mean_abs_error": 0.5,
            "bad_2px_fraction": 0.02,
            "nonnegative_fraction": 1.0,
        },
    )
    stage = StageSpec(name="full_inference")
    reference = StageOutput(
        stage_name=stage.name,
        data={"disparity": expected, "expected_shape": [4, 4]},
    )

    passed = comparator.compare(
        StageOutput(stage_name=stage.name, data={"disparity": expected.copy()}),
        reference,
        threshold,
        stage,
    )
    failed = comparator.compare(
        StageOutput(
            stage_name=stage.name,
            data={"disparity": np.flip(expected, axis=1).copy()},
        ),
        reference,
        threshold,
        stage,
    )
    rescaled = comparator.compare(
        StageOutput(stage_name=stage.name, data={"disparity": expected * 2.0}),
        reference,
        threshold,
        stage,
    )

    assert passed.passed
    assert passed.metrics["global_cosine"].value == pytest.approx(1.0)
    assert passed.metrics["mean_abs_error"].value == pytest.approx(0.0)
    assert not failed.passed
    assert not failed.metrics["global_cosine"].passed
    assert not rescaled.passed
    assert rescaled.metrics["global_cosine"].passed
    assert not rescaled.metrics["mean_abs_error"].passed
