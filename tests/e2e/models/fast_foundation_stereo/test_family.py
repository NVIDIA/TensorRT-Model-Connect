# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from tensorrt_model_connect.families.fast_foundation_stereo.plugin import (
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


@pytest.mark.parametrize(
    ("helper_name", "variable"),
    (
        (
            "_full_volume_conv_bn_folding_enabled",
            "TRTMC_FAST_FOUNDATION_STEREO_FOLD_FULL_VOLUME_BN",
        ),
        (
            "_post16_to_8_conv_bn_folding_enabled",
            "TRTMC_FAST_FOUNDATION_STEREO_FOLD_POST16_TO_8_BN",
        ),
        (
            "_feature_att_8_conv_bn_folding_enabled",
            "TRTMC_FAST_FOUNDATION_STEREO_FOLD_FEATURE_ATT_8_BN",
        ),
        (
            "_remaining_safe_conv_bn_folding_enabled",
            "TRTMC_FAST_FOUNDATION_STEREO_FOLD_REMAINING_SAFE_BN",
        ),
    ),
)
def test_conv_bn_folding_is_default_on_with_explicit_fallback(
    monkeypatch,
    helper_name: str,
    variable: str,
) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_post

    enabled = getattr(native_post, helper_name)
    monkeypatch.delenv(variable, raising=False)
    assert enabled()

    monkeypatch.setenv(variable, "1")
    assert enabled()

    monkeypatch.setenv(variable, "0")
    assert not enabled()


@pytest.mark.parametrize("work_dtype", [np.float16, np.float32])
def test_conv_bn_folding_quantizes_first_without_mutating_source(work_dtype) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    graph = object.__new__(NativeGraph)
    graph.work_np_dtype = work_dtype
    source_weight = np.asarray([0.33337, -0.27191, 1.23491, -2.71819], dtype=np.float32).reshape(
        2, 2, 1, 1, 1
    )
    source_bias = np.asarray([0.33337, -0.27191], dtype=np.float32)
    original_weight = source_weight.copy()
    original_bias = source_bias.copy()
    convolution = SimpleNamespace(
        weight=source_weight,
        bias=source_bias,
        in_channels=2,
        out_channels=2,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
        training=False,
    )
    scale = np.asarray([1.2345, 0.7123], dtype=np.float32)
    shift = np.asarray([0.13, -0.29], dtype=np.float32)
    batch_norm = SimpleNamespace(
        weight=scale,
        bias=shift,
        running_mean=np.zeros(2, dtype=np.float32),
        running_var=np.ones(2, dtype=np.float32),
        eps=0.0,
        affine=True,
        track_running_stats=True,
        training=False,
    )

    folded = NativeGraph._fold_batch_norm_into_convolution(
        graph,
        convolution,
        batch_norm,
        deconv=False,
    )

    expected_weight = (
        source_weight.astype(work_dtype).astype(np.float32) * scale.reshape(2, 1, 1, 1, 1)
    ).astype(work_dtype)
    expected_bias = (source_bias.astype(work_dtype).astype(np.float32) * scale + shift).astype(
        work_dtype
    )
    np.testing.assert_array_equal(folded.weight, expected_weight)
    np.testing.assert_array_equal(folded.bias, expected_bias)
    np.testing.assert_array_equal(source_weight, original_weight)
    np.testing.assert_array_equal(source_bias, original_bias)
    assert folded.weight.flags.c_contiguous
    assert folded.bias.flags.c_contiguous

    if work_dtype is np.float16:
        direct_fp32_fold = (source_weight * scale.reshape(2, 1, 1, 1, 1)).astype(np.float16)
        assert not np.array_equal(folded.weight, direct_fp32_fold)


def test_deconv_bn_folding_scales_output_axis_within_each_group() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    graph = object.__new__(NativeGraph)
    graph.work_np_dtype = np.float16
    source_weight = np.ones((4, 3, 1, 1, 1), dtype=np.float32)
    scale = np.asarray([2.0, 3.0, 5.0, 7.0, 11.0, 13.0], dtype=np.float32)
    convolution = SimpleNamespace(
        weight=source_weight,
        bias=None,
        in_channels=4,
        out_channels=6,
        kernel_size=(1, 1, 1),
        stride=(2, 2, 2),
        padding=(0, 0, 0),
        output_padding=(1, 1, 1),
        dilation=(1, 1, 1),
        groups=2,
        training=False,
    )
    batch_norm = SimpleNamespace(
        weight=scale,
        bias=np.zeros(6, dtype=np.float32),
        running_mean=np.zeros(6, dtype=np.float32),
        running_var=np.ones(6, dtype=np.float32),
        eps=0.0,
        affine=True,
        track_running_stats=True,
        training=False,
    )

    folded = NativeGraph._fold_batch_norm_into_convolution(
        graph,
        convolution,
        batch_norm,
        deconv=True,
    )

    expected = np.ones((2, 2, 3, 1, 1, 1), dtype=np.float32)
    expected *= scale.reshape(2, 1, 3, 1, 1, 1)
    np.testing.assert_array_equal(folded.weight, expected.reshape(source_weight.shape))
    np.testing.assert_array_equal(folded.bias, np.zeros(6, dtype=np.float16))
    np.testing.assert_array_equal(source_weight, np.ones_like(source_weight))
    assert folded.output_padding == convolution.output_padding


def test_folded_resnet_keeps_second_activation_after_residual_add() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    graph = object.__new__(NativeGraph)
    calls = []

    def fold(tensor, convolution, batch_norm, *, dimensions, deconv):
        calls.append((tensor, convolution, batch_norm, dimensions, deconv))
        return f"folded-{len(calls)}"

    graph._convolution_batch_norm = fold
    graph.activation = lambda tensor, kind: ("activation", tensor, kind)
    graph.add = lambda lhs, rhs: ("add", lhs, rhs)
    convolution_1 = SimpleNamespace(kernel_size=(3, 3, 3))
    convolution_2 = SimpleNamespace(kernel_size=(3, 3, 3))
    batch_norm_1 = SimpleNamespace()
    batch_norm_2 = SimpleNamespace()
    module = SimpleNamespace(
        conv1=convolution_1,
        bn1=batch_norm_1,
        conv2=convolution_2,
        bn2=batch_norm_2,
        downsample=None,
    )

    output = NativeGraph.resnet(graph, "identity", module, fold_batch_norm=True)

    first_activation = ("activation", "folded-1", "relu")
    assert calls == [
        ("identity", convolution_1, batch_norm_1, 3, False),
        (first_activation, convolution_2, batch_norm_2, 3, False),
    ]
    assert output == (
        "activation",
        ("add", "folded-2", "identity"),
        "relu",
    )


def test_forward_helper_folds_only_its_basic_conv_layers() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    BasicConv = type("BasicConv", (), {})
    FeatureAtt = type("FeatureAtt", (), {})
    first = BasicConv()
    second = BasicConv()
    attention = FeatureAtt()
    graph = object.__new__(NativeGraph)
    calls = []

    def basic_conv(tensor, module, *, fold_batch_norm=False):
        calls.append(("basic_conv", tensor, module, fold_batch_norm))
        return f"conv-{len(calls)}"

    def feature_attention(volume, feature, module):
        calls.append(("feature_attention", volume, feature, module))
        return "attended"

    graph.basic_conv = basic_conv
    graph.feature_attention = feature_attention
    graph.module = lambda *_args: pytest.fail("folded helper must not use generic module routing")

    output = NativeGraph.forward_helper(
        graph,
        "input",
        "feature",
        SimpleNamespace(layers=(first, second, attention)),
        fold_batch_norm=True,
    )

    assert output == "attended"
    assert calls == [
        ("basic_conv", "input", first, True),
        ("basic_conv", "conv-1", second, True),
        ("feature_attention", "conv-2", "feature", attention),
    ]


def test_feature_att_8_folds_exact_three_bn_and_preserves_relu_gate_and_checkpoint() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _folded_feature_att_8_forward_helper,
    )

    BasicConv = type("BasicConv", (), {})
    Conv3d = type("Conv3d", (), {})
    Conv3dNormActReduced = type("Conv3dNormActReduced", (), {})
    FeatureAtt = type("FeatureAtt", (), {})
    ForwardHelper = type("ForwardHelper", (), {})
    ReLU = type("ReLU", (), {})
    SyncBatchNorm = type("SyncBatchNorm", (), {})

    def convolution(seed: float):
        module = Conv3d()
        module.weight = np.linspace(seed, seed + 0.7, 4, dtype=np.float32).reshape(2, 2, 1, 1, 1)
        module.bias = np.asarray([seed - 0.2, seed + 0.3], dtype=np.float32)
        module.in_channels = 2
        module.out_channels = 2
        module.kernel_size = (1, 1, 1)
        module.stride = (1, 1, 1)
        module.padding = (0, 0, 0)
        module.dilation = (1, 1, 1)
        module.groups = 1
        module.training = False
        return module

    def batch_norm(seed: float):
        module = SyncBatchNorm()
        module.weight = np.asarray([seed + 0.8, seed + 1.1], dtype=np.float32)
        module.bias = np.asarray([seed - 0.1, seed + 0.2], dtype=np.float32)
        module.running_mean = np.asarray([seed - 0.3, seed + 0.4], dtype=np.float32)
        module.running_var = np.asarray([seed + 0.9, seed + 1.2], dtype=np.float32)
        module.eps = 1.0e-5
        module.affine = True
        module.track_running_stats = True
        module.training = False
        return module

    convolutions = [convolution(0.1), convolution(0.4), convolution(0.7)]
    batch_norms = [batch_norm(0.1), batch_norm(0.4), batch_norm(0.7)]
    basic = BasicConv()
    basic.conv = convolutions[0]
    basic.bn = batch_norms[0]
    basic.relu = ReLU()
    reduced = Conv3dNormActReduced()
    reduced.conv1 = (convolutions[1], batch_norms[1], ReLU())
    reduced.conv2 = (convolutions[2], batch_norms[2], ReLU())
    gate = FeatureAtt()
    gate_batch_norm = batch_norm(1.0)
    gate.feat_att = (object(), gate_batch_norm, ReLU())
    helper = ForwardHelper()
    helper.layers = (basic, reduced, gate)

    source_arrays = []
    for module in (*convolutions, *batch_norms, gate_batch_norm):
        for name in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(module, name, None)
            if value is not None:
                source_arrays.append(value)
    snapshots = [value.copy() for value in source_arrays]

    graph = object.__new__(NativeGraph)
    graph.work_np_dtype = np.float16
    folded = []
    activations = []
    gate_calls = []

    def add_convolution(tensor, module, *, dimensions, deconv):
        folded.append((tensor, module, dimensions, deconv))
        return f"folded-{len(folded)}"

    def activate(tensor, kind, alpha=None):
        activations.append((tensor, kind, alpha))
        return f"relu-{len(activations)}"

    def feature_attention(volume, feature, module):
        gate_calls.append((volume, feature, module))
        return "gated"

    graph._convolution = add_convolution
    graph.activation = activate
    graph.feature_attention = feature_attention
    graph.module = lambda *_args: pytest.fail("feature_att_8 folded path must be explicit")

    output = _folded_feature_att_8_forward_helper(graph, "input", "feature", helper)

    assert output == "gated"
    assert [(tensor, dimensions, deconv) for tensor, _module, dimensions, deconv in folded] == [
        ("input", 3, False),
        ("relu-1", 3, False),
        ("relu-2", 3, False),
    ]
    assert all(module.weight.dtype == np.float16 for _tensor, module, _dim, _deconv in folded)
    assert activations == [
        ("folded-1", "relu", None),
        ("folded-2", "relu", None),
        ("folded-3", "relu", None),
    ]
    assert gate_calls == [("relu-3", "feature", gate)]
    assert len(folded) == 3
    for source, snapshot in zip(source_arrays, snapshots):
        np.testing.assert_array_equal(source, snapshot)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_feature_att_8_folding_rejects_non_distilled_top_level_class(index: int) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _folded_feature_att_8_forward_helper,
    )

    ForwardHelper = type("ForwardHelper", (), {})
    helper = ForwardHelper()
    helper.layers = [
        type("BasicConv", (), {})(),
        type("Conv3dNormActReduced", (), {})(),
        type("FeatureAtt", (), {})(),
    ]
    helper.layers[index] = type("UnexpectedLayer", (), {})()

    with pytest.raises(RuntimeError, match="feature_att_8.*distilled topology") as error:
        _folded_feature_att_8_forward_helper(
            object(),
            "input",
            "feature",
            helper,
        )

    assert "('BasicConv', 'Conv3dNormActReduced', 'FeatureAtt')" in str(error.value)


def test_feature_att_8_folding_rejects_non_forward_helper() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _folded_feature_att_8_forward_helper,
    )

    with pytest.raises(RuntimeError, match="requires a ForwardHelper"):
        _folded_feature_att_8_forward_helper(
            object(),
            "input",
            "feature",
            SimpleNamespace(layers=()),
        )


def test_post_forward_helper_folds_basic_conv_and_resnet_only_after_skip_add() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _post_forward_helper,
    )

    BasicConv = type("BasicConv", (), {})
    ResnetBasicBlock3D = type("ResnetBasicBlock3D", (), {})
    basic = BasicConv()
    resnet = ResnetBasicBlock3D()
    calls = []

    class Graph:
        @staticmethod
        def add(lhs, rhs):
            return "skip-add", lhs, rhs

        @staticmethod
        def basic_conv(tensor, module, *, fold_batch_norm=False):
            calls.append(("basic_conv", tensor, module, fold_batch_norm))
            return "basic-output"

        @staticmethod
        def resnet(tensor, module, *, fold_batch_norm=False):
            calls.append(("resnet", tensor, module, fold_batch_norm))
            return "resnet-output"

        @staticmethod
        def module(*_args):
            pytest.fail("folded post helper must not use generic module routing")

    output = _post_forward_helper(
        Graph(),
        "skip",
        "lower",
        "feature",
        SimpleNamespace(upsample=(), op="sum", out=(basic, resnet)),
        fold_batch_norm=True,
    )

    assert output == "resnet-output"
    assert calls == [
        ("basic_conv", ("skip-add", "lower", "skip"), basic, True),
        ("resnet", "basic-output", resnet, True),
    ]


def test_folded_conv3d_norm_act_reduced_keeps_both_relu_boundaries() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    Conv3d = type("Conv3d", (), {})
    SyncBatchNorm = type("SyncBatchNorm", (), {})
    ReLU = type("ReLU", (), {})
    conv1 = Conv3d()
    bn1 = SyncBatchNorm()
    relu1 = ReLU()
    conv2 = Conv3d()
    bn2 = SyncBatchNorm()
    relu2 = ReLU()
    module = SimpleNamespace(
        conv1=(conv1, bn1, relu1),
        conv2=(conv2, bn2, relu2),
    )
    graph = object.__new__(NativeGraph)
    calls = []

    def fold(tensor, convolution, batch_norm, *, dimensions, deconv):
        calls.append(("fold", tensor, convolution, batch_norm, dimensions, deconv))
        return f"folded-{len(calls)}"

    def activate(tensor, kind):
        calls.append(("activation", tensor, kind))
        return f"relu-{len(calls)}"

    graph._convolution_batch_norm = fold
    graph.activation = activate

    output = NativeGraph.conv3d_reduced(graph, "input", module, fold_batch_norm=True)

    assert output == "relu-4"
    assert calls == [
        ("fold", "input", conv1, bn1, 3, False),
        ("activation", "folded-1", "relu"),
        ("fold", "relu-2", conv2, bn2, 3, False),
        ("activation", "folded-3", "relu"),
    ]


def test_folded_conv3d_norm_act_reduced_rejects_non_sync_batch_norm() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    Conv3d = type("Conv3d", (), {})
    BatchNorm3d = type("BatchNorm3d", (), {})
    ReLU = type("ReLU", (), {})
    module = SimpleNamespace(
        conv1=(Conv3d(), BatchNorm3d(), ReLU()),
        conv2=(),
    )
    graph = object.__new__(NativeGraph)

    with pytest.raises(RuntimeError, match="expected.*SyncBatchNorm"):
        NativeGraph.conv3d_reduced(graph, "input", module, fold_batch_norm=True)


def test_sync_batch_norm_satisfies_fp16_first_folding_contract() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph

    Conv3d = type("Conv3d", (), {})
    SyncBatchNorm = type("SyncBatchNorm", (), {})
    convolution = Conv3d()
    convolution.weight = np.linspace(-0.71, 0.83, 56, dtype=np.float32).reshape(56, 1, 1, 1, 1)
    convolution.bias = None
    convolution.in_channels = 1
    convolution.out_channels = 56
    convolution.kernel_size = (1, 1, 1)
    convolution.stride = (1, 1, 1)
    convolution.padding = (0, 0, 0)
    convolution.dilation = (1, 1, 1)
    convolution.groups = 1
    convolution.training = False

    batch_norm = SyncBatchNorm()
    batch_norm.num_features = 56
    batch_norm.weight = np.linspace(0.71, 1.23, 56, dtype=np.float32)
    batch_norm.bias = np.linspace(-0.29, 0.13, 56, dtype=np.float32)
    batch_norm.running_mean = np.linspace(-0.4, 0.2, 56, dtype=np.float32)
    batch_norm.running_var = np.linspace(0.9, 1.1, 56, dtype=np.float32)
    batch_norm.eps = 1.0e-5
    batch_norm.affine = True
    batch_norm.track_running_stats = True
    batch_norm.training = False

    graph = object.__new__(NativeGraph)
    graph.work_np_dtype = np.float16
    folded = NativeGraph._fold_batch_norm_into_convolution(
        graph,
        convolution,
        batch_norm,
        deconv=False,
    )

    scale = batch_norm.weight / np.sqrt(batch_norm.running_var + batch_norm.eps)
    shift = batch_norm.bias - batch_norm.running_mean * scale
    expected_weight = (
        convolution.weight.astype(np.float16).astype(np.float32) * scale.reshape(56, 1, 1, 1, 1)
    ).astype(np.float16)
    expected_bias = shift.astype(np.float16)
    np.testing.assert_array_equal(folded.weight, expected_weight)
    np.testing.assert_array_equal(folded.bias, expected_bias)


def test_post16_to_8_folds_exact_four_bn_paths_without_moving_relu_boundaries() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _post_forward_helper,
    )

    BasicConv = type("BasicConv", (), {})
    Conv3d = type("Conv3d", (), {})
    ConvTranspose3d = type("ConvTranspose3d", (), {})
    FeatureAtt = type("FeatureAtt", (), {})
    Conv3dNormActReduced = type("Conv3dNormActReduced", (), {})
    SyncBatchNorm = type("SyncBatchNorm", (), {})
    upsample = BasicConv()
    upsample.conv = ConvTranspose3d()
    upsample.bn = SyncBatchNorm()
    out_conv = BasicConv()
    out_conv.conv = Conv3d()
    out_conv.bn = SyncBatchNorm()
    feature_attention = FeatureAtt()
    out_reduced = Conv3dNormActReduced()
    out_reduced.conv1 = (Conv3d(), SyncBatchNorm(), type("ReLU", (), {})())
    out_reduced.conv2 = (Conv3d(), SyncBatchNorm(), type("ReLU", (), {})())
    calls = []

    class Graph:
        @staticmethod
        def basic_conv(tensor, module, *, fold_batch_norm=False):
            calls.append(("basic_conv", tensor, module, fold_batch_norm))
            return "upsampled" if module is upsample else "out-conv"

        @staticmethod
        def add(lhs, rhs):
            calls.append(("add", lhs, rhs))
            return "skip-add"

        @staticmethod
        def feature_attention(volume, feature, module):
            calls.append(("feature_attention", volume, feature, module))
            return "attended"

        @staticmethod
        def conv3d_reduced(tensor, module, *, fold_batch_norm=False):
            calls.append(("conv3d_reduced", tensor, module, fold_batch_norm))
            return "reduced-output"

        @staticmethod
        def module(*_args):
            pytest.fail("post16_to_8 foldable paths must not use generic module routing")

    output = _post_forward_helper(
        Graph(),
        "skip",
        "lower",
        "feature",
        SimpleNamespace(
            upsample=(upsample,),
            op="sum",
            out=(out_conv, feature_attention, out_reduced),
        ),
        fold_post16_to_8_batch_norm=True,
    )

    assert output == "reduced-output"
    assert calls == [
        ("basic_conv", "lower", upsample, True),
        ("add", "upsampled", "skip"),
        ("basic_conv", "skip-add", out_conv, True),
        ("feature_attention", "out-conv", "feature", feature_attention),
        ("conv3d_reduced", "attended", out_reduced, True),
    ]
    # One BN in each BasicConv plus both reduced Conv3d-BN pairs: exactly four.
    assert (
        sum(
            2 if call[0] == "conv3d_reduced" else 1
            for call in calls
            if call[0] in {"basic_conv", "conv3d_reduced"}
        )
        == 4
    )


def test_remaining_safe_folding_covers_exact_14_pairs_and_preserves_checkpoint() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _folded_remaining_safe_context,
        _folded_remaining_safe_conv3,
        _folded_remaining_safe_feature_att_16,
        _post_forward_helper,
    )

    class BasicConv:
        pass

    class Conv2d:
        pass

    class Conv3d:
        pass

    class ConvTranspose3d:
        pass

    class Conv3dNormActReduced:
        pass

    class FeatureAtt:
        pass

    class ForwardHelper:
        pass

    class ModuleList(tuple):
        pass

    class PostForwardHelper:
        pass

    class ReLU:
        pass

    class Sequential(tuple):
        pass

    class SyncBatchNorm:
        pass

    source_arrays = []

    def convolution(kind, dimensions: int, seed: float):
        module = kind()
        kernel = (1,) * dimensions
        module.weight = np.full((2, 2, *kernel), seed, dtype=np.float32)
        module.bias = np.asarray([seed - 0.25, seed + 0.25], dtype=np.float32)
        module.in_channels = 2
        module.out_channels = 2
        module.kernel_size = kernel
        module.stride = (1,) * dimensions
        module.padding = (0,) * dimensions
        module.dilation = (1,) * dimensions
        module.groups = 1
        module.training = False
        if kind is ConvTranspose3d:
            module.output_padding = (0,) * dimensions
        source_arrays.extend((module.weight, module.bias))
        return module

    def batch_norm(seed: float):
        module = SyncBatchNorm()
        module.weight = np.asarray([seed + 1.0, seed + 1.25], dtype=np.float32)
        module.bias = np.asarray([seed - 0.1, seed + 0.1], dtype=np.float32)
        module.running_mean = np.asarray([seed - 0.2, seed + 0.2], dtype=np.float32)
        module.running_var = np.asarray([seed + 0.5, seed + 0.75], dtype=np.float32)
        module.eps = 1.0e-5
        module.affine = True
        module.track_running_stats = True
        module.training = False
        source_arrays.extend((module.weight, module.bias, module.running_mean, module.running_var))
        return module

    def basic(dimensions: int, seed: float, *, deconv: bool = False):
        module = BasicConv()
        kind = ConvTranspose3d if deconv else (Conv2d if dimensions == 2 else Conv3d)
        module.conv = convolution(kind, dimensions, seed)
        module.bn = batch_norm(seed)
        module.relu = True
        module.use_bn = True
        return module

    def reduced(seed: float):
        module = Conv3dNormActReduced()
        module.conv1 = Sequential((convolution(Conv3d, 3, seed), batch_norm(seed), ReLU()))
        module.conv2 = Sequential(
            (convolution(Conv3d, 3, seed + 0.05), batch_norm(seed + 0.05), ReLU())
        )
        return module

    def direct_pairs(module):
        if module.__class__.__name__ == "BasicConv":
            return [(module.conv, module.bn)]
        return [
            (module.conv1[0], module.conv1[1]),
            (module.conv2[0], module.conv2[1]),
        ]

    feature_layers = (basic(3, 0.1), reduced(0.2), reduced(0.3))
    feature_helper = ForwardHelper()
    feature_helper.layers = ModuleList(feature_layers)

    conv3_layers = (basic(3, 0.4), reduced(0.5))
    conv3 = Sequential(conv3_layers)

    excluded_upsample = basic(3, 0.6, deconv=True)
    excluded_gate_basic = basic(2, 0.7)
    gate = FeatureAtt()
    gate.feat_att = Sequential((excluded_gate_basic, convolution(Conv2d, 2, 0.75)))
    post_out = (gate, basic(3, 0.8), reduced(0.9), basic(3, 1.0))
    post = PostForwardHelper()
    post.upsample = Sequential((excluded_upsample,))
    post.op = "sum"
    post.out = ModuleList(post_out)

    context_layers = (basic(2, 1.1), basic(2, 1.2))
    context = ModuleList(context_layers)

    expected_pairs = []
    for module in (*feature_layers, *conv3_layers, *post_out[1:], *context_layers):
        expected_pairs.extend(direct_pairs(module))
    assert len(expected_pairs) == 14
    snapshots = [value.copy() for value in source_arrays]

    graph = object.__new__(NativeGraph)
    graph.work_np_dtype = np.float16
    folded_sources = []
    folded_dimensions = []
    events = []
    original_fold = NativeGraph._fold_batch_norm_into_convolution.__get__(graph, NativeGraph)

    def fold(convolution_module, batch_norm_module, *, deconv):
        folded_sources.append((convolution_module, batch_norm_module, deconv))
        return original_fold(convolution_module, batch_norm_module, deconv=deconv)

    def add_convolution(tensor, _module, *, dimensions, deconv):
        folded_dimensions.append((tensor, dimensions, deconv))
        events.append("fold")
        return f"folded-{len(folded_dimensions)}"

    graph._fold_batch_norm_into_convolution = fold
    graph._convolution = add_convolution
    graph.activation = lambda tensor, kind, alpha=None: (
        events.append("activation")
        or (
            "activation",
            tensor,
            kind,
            alpha,
        )
    )
    graph.module = lambda tensor, module: (
        events.append("module")
        or (
            "module",
            tensor,
            module,
        )
    )
    graph.add = lambda lhs, rhs: events.append("add") or ("add", lhs, rhs)
    graph.feature_attention = lambda volume, feature, module: (
        events.append("feature")
        or (
            "feature",
            volume,
            feature,
            module,
        )
    )

    _folded_remaining_safe_feature_att_16(graph, "feature16-input", feature_helper)
    _folded_remaining_safe_conv3(graph, "conv3-input", conv3)
    post_start = len(events)
    _post_forward_helper(
        graph,
        "skip",
        "lower",
        "feature",
        post,
        fold_remaining_safe_batch_norm=True,
    )
    post_events = events[post_start:]
    context_outputs = _folded_remaining_safe_context(graph, "feature04", context)

    assert [(conv, bn) for conv, bn, _deconv in folded_sources] == expected_pairs
    assert all(not deconv for _conv, _bn, deconv in folded_sources)
    assert [dimensions for _tensor, dimensions, _deconv in folded_dimensions] == [3] * 12 + [
        2,
        2,
    ]
    assert post_events == [
        "module",  # Excluded upsample BN remains before the sum skip.
        "add",
        "feature",  # Excluded FeatureAtt BN remains on sigmoid/multiply gate path.
        "fold",
        "activation",
        "fold",
        "activation",
        "fold",
        "activation",
        "fold",
        "activation",
    ]
    assert [tensor for tensor, _dimensions, _deconv in folded_dimensions[-2:]] == [
        "feature04",
        "feature04",
    ]
    assert len(context_outputs) == 2
    assert all(
        excluded is not convolution_module
        for excluded in (excluded_upsample.conv, excluded_gate_basic.conv)
        for convolution_module, _batch_norm, _deconv in folded_sources
    )
    for source, snapshot in zip(source_arrays, snapshots):
        np.testing.assert_array_equal(source, snapshot)


def test_remaining_safe_folding_rejects_all_audited_topology_drift() -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo.native_post import (
        _folded_remaining_safe_context,
        _folded_remaining_safe_conv3,
        _folded_remaining_safe_feature_att_16,
        _post_forward_helper,
    )

    BasicConv = type("BasicConv", (), {})
    Conv3dNormActReduced = type("Conv3dNormActReduced", (), {})
    FeatureAtt = type("FeatureAtt", (), {})
    ForwardHelper = type("ForwardHelper", (), {})
    ModuleList = type("ModuleList", (tuple,), {})
    PostForwardHelper = type("PostForwardHelper", (), {})
    Sequential = type("Sequential", (tuple,), {})

    feature = ForwardHelper()
    feature.layers = ModuleList((BasicConv(), FeatureAtt()))
    with pytest.raises(RuntimeError, match="feature_att_16 topology"):
        _folded_remaining_safe_feature_att_16(object(), "input", feature)

    feature.layers = ModuleList((BasicConv(), Conv3dNormActReduced(), Conv3dNormActReduced()))
    with pytest.raises(RuntimeError, match="direct BasicConv topology"):
        _folded_remaining_safe_feature_att_16(object(), "input", feature)

    with pytest.raises(RuntimeError, match="conv3 topology"):
        _folded_remaining_safe_conv3(object(), "input", Sequential((BasicConv(),)))

    with pytest.raises(RuntimeError, match="cnet.conv04 topology"):
        _folded_remaining_safe_context(
            object(),
            "feature",
            ModuleList((BasicConv(), FeatureAtt())),
        )

    post = PostForwardHelper()
    post.upsample = Sequential((BasicConv(),))
    post.op = "sum"
    post.out = ModuleList((FeatureAtt(), BasicConv()))
    with pytest.raises(RuntimeError, match="post32_to_16 topology"):
        _post_forward_helper(
            object(),
            "skip",
            "lower",
            "feature",
            post,
            fold_remaining_safe_batch_norm=True,
        )


def test_cost_aggregation_limits_folding_to_conv1_up_and_post8(monkeypatch) -> None:
    from tensorrt_model_connect.families.fast_foundation_stereo import native_post

    FeatureAtt = type("FeatureAtt", (), {})
    post_calls = []
    remaining_calls = []

    def post_forward_helper(
        _graph,
        skip,
        lower,
        feature,
        module,
        *,
        fold_batch_norm=False,
        fold_post16_to_8_batch_norm=False,
        fold_remaining_safe_batch_norm=False,
        post8_sum_plugin=False,
        post8_sum_tile_positions=32,
        full_volume_leaky_plugin=False,
    ):
        post_calls.append(
            (
                module,
                fold_batch_norm,
                fold_post16_to_8_batch_norm,
                fold_remaining_safe_batch_norm,
                post8_sum_plugin,
                post8_sum_tile_positions,
                full_volume_leaky_plugin,
                skip,
                lower,
                feature,
            )
        )
        return f"post-{module}"

    monkeypatch.setattr(native_post, "_post_forward_helper", post_forward_helper)
    monkeypatch.setattr(
        native_post,
        "_folded_remaining_safe_feature_att_16",
        lambda _graph, tensor, feature_att: (
            remaining_calls.append(("feature_att_16", tensor, feature_att))
            or "folded-feature-att-16"
        ),
    )
    monkeypatch.setattr(
        native_post,
        "_folded_remaining_safe_conv3",
        lambda _graph, tensor, conv3: (
            remaining_calls.append(("conv3", tensor, conv3)) or "folded-conv3"
        ),
    )

    class Graph:
        def __init__(self) -> None:
            self.basic_calls = []

        @staticmethod
        def module(tensor, module):
            return f"module-{module}({tensor})"

        @staticmethod
        def feature_attention(volume, feature, _module):
            return f"attention({volume},{feature})"

        @staticmethod
        def sequential(tensor, module):
            return f"sequential-{module}({tensor})"

        def basic_conv(self, tensor, module, *, fold_batch_norm=False):
            self.basic_calls.append((module, fold_batch_norm, tensor))
            return f"basic-{module}({tensor})"

    graph = Graph()
    module = SimpleNamespace(
        conv1="conv1",
        feature_att_8=FeatureAtt(),
        conv2="conv2",
        feature_att_16=FeatureAtt(),
        conv3="conv3",
        feature_att_32=FeatureAtt(),
        post32_to_16="post32",
        post16_to_8="post16",
        conv1_up="conv1_up",
        post8_to_4="post8",
    )

    native_post._cost_aggregation(
        graph,
        "volume",
        ("feature4", "feature8", "feature16", "feature32"),
        module,
        fold_full_volume_batch_norm=True,
    )

    assert [call[:2] for call in graph.basic_calls] == [("conv1_up", True)]
    assert [
        (module, full, post16, remaining) for module, full, post16, remaining, *_rest in post_calls
    ] == [
        ("post32", False, False, False),
        ("post16", False, False, False),
        ("post8", True, False, False),
    ]

    graph.basic_calls.clear()
    post_calls.clear()
    native_post._cost_aggregation(
        graph,
        "volume",
        ("feature4", "feature8", "feature16", "feature32"),
        module,
        fold_post16_to_8_batch_norm=True,
    )

    assert [call[:2] for call in graph.basic_calls] == [("conv1_up", False)]
    assert [
        (module, full, post16, remaining) for module, full, post16, remaining, *_rest in post_calls
    ] == [
        ("post32", False, False, False),
        ("post16", False, True, False),
        ("post8", False, False, False),
    ]

    graph.basic_calls.clear()
    post_calls.clear()
    native_post._cost_aggregation(
        graph,
        "volume",
        ("feature4", "feature8", "feature16", "feature32"),
        module,
        post8_sum_plugin=True,
        post8_sum_tile_positions=128,
    )

    assert [
        (module, plugin, tile)
        for module, _full, _post16, _remaining, plugin, tile, *_rest in post_calls
    ] == [
        ("post32", False, 32),
        ("post16", False, 32),
        ("post8", True, 128),
    ]

    graph.basic_calls.clear()
    post_calls.clear()
    native_post._cost_aggregation(
        graph,
        "volume",
        ("feature4", "feature8", "feature16", "feature32"),
        module,
        fold_remaining_safe_batch_norm=True,
    )

    assert remaining_calls == [
        (
            "feature_att_16",
            "module-conv2(attention(module-conv1(volume),feature8))",
            module.feature_att_16,
        ),
        ("conv3", "folded-feature-att-16", "conv3"),
    ]
    assert [
        (module, full, post16, remaining) for module, full, post16, remaining, *_rest in post_calls
    ] == [
        ("post32", False, False, True),
        ("post16", False, False, False),
        ("post8", False, False, False),
    ]


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
