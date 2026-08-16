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
    assert config["stereo_min_cosine"] == 0.999


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


def test_production_family_uses_only_native_tensorrt_network_definition() -> None:
    sources = sorted(_FAMILY_SOURCE_DIR.rglob("*.py"))
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

    combined = "\n".join(source_by_path.values())
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


def test_l4_performance_receipt_beats_baseline_and_passes_accuracy() -> None:
    receipt = json.loads((Path(__file__).parent / "performance/l4.json").read_text())
    baseline = receipt["recorded_baseline"]
    selected = receipt["selected_tensorrt_sustained_runs"][-1]
    protocol = receipt["protocol"]
    roofline = receipt["roofline_sample"]

    assert selected["inference_5_pairs_mean_ms"] < baseline["inference_5_pairs_mean_ms"]
    assert selected["total_5_pairs_mean_ms"] < baseline["total_5_pairs_mean_ms"]
    assert selected["global_cosine"] >= protocol["minimum_cosine"]
    assert selected["accuracy_passed"] is True
    assert (
        max(
            roofline["maximum_compute_percent_of_peak"],
            roofline["maximum_memory_percent_of_peak"],
        )
        >= 80.0
    )


def test_disparity_comparator_gates_global_fp32_cosine() -> None:
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    comparator = StereoDisparityComparator()
    threshold = ThresholdProfile(
        task_strategy="stereo_disparity",
        metrics={
            "finite_fraction": 1.0,
            "global_cosine": 0.999,
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

    assert passed.passed
    assert passed.metrics["global_cosine"].value == pytest.approx(1.0)
    assert not failed.passed
    assert not failed.metrics["global_cosine"].passed
