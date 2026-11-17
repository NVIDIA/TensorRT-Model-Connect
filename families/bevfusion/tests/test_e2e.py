# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public build/bundle/Task parity against the pinned upstream BEV implementation."""

import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from tensorrt_model_connect import BuildRequest, build
from tools.e2e_evidence import evidence_stage, record_evidence

from .reference import Reference
from .inputs import prepare


ROOT = Path(__file__).parent
CASES = {
    manifest["name"]: manifest
    for path in sorted((ROOT / "manifests").glob("*.json"))
    for manifest in [json.loads(path.read_text())]
}


def _enabled(config, name):
    models = set()
    for value in config.getoption("--e2e-model") or []:
        models.update(value.split(","))
    models_file = config.getoption("--e2e-models-file")
    if models_file:
        models.update(
            line.strip()
            for line in Path(models_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    cases = set()
    for value in config.getoption("--e2e-testcase") or []:
        cases.update(value.split(","))
    return (
        bool(models or cases)
        and (not models or "bevfusion" in models or name in models)
        and (not cases or name in cases)
    )


def _inspect_fp8_engine(bundle, output):
    """Require engine evidence of FP8 activation and convolution-weight execution."""
    import tensorrt as trt

    with bundle.open("rb") as source:
        assert source.read(8) == b"BUNDLE\x01\x00"
        size = struct.unpack("<Q", source.read(8))[0]
        section = json.loads(source.read(size))["sections"]["engine.plan"]
        source.seek(16 + size + section["offset"])
        plan = source.read(section["length"])
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    inspector = engine.create_engine_inspector()
    information = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    output.write_text(json.dumps(information, indent=2) + "\n")
    layers = [
        layer["Name"]
        for layer in information["Layers"]
        if any(value.get("Datatype") == "FP8" for value in layer.get("Inputs", []))
        and any(value.get("Datatype") == "FP8" for value in layer.get("Constants", []))
    ]
    assert layers, "FP8 build must execute convolutions with FP8 activations and weights"
    record_evidence("fp8_execution", {"layer_count": len(layers), "layers": layers})


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("name", sorted(CASES))
def test_official_bev_checkpoint(name, request, tmp_path):
    if not _enabled(request.config, name):
        pytest.skip("BEVFusion E2E requires explicit --e2e-model or --e2e-testcase selection")
    import torch

    manifest = CASES[name]
    for variable in (
        "TRTMC_BEVFUSION_MODEL_DIR",
        "TRTMC_RUNTIME_ROOT",
        "TRTMC_NATIVE_BUILD_DIR",
    ):
        assert os.environ.get(variable), f"selected BEVFusion E2E requires {variable}"
    model_dir = Path(os.environ["TRTMC_BEVFUSION_MODEL_DIR"])
    source = model_dir / "reference"
    runtime = Path(os.environ["TRTMC_RUNTIME_ROOT"])
    native = Path(os.environ["TRTMC_NATIVE_BUILD_DIR"])
    qualifier = native / "test_bevfusion_sdk_cpp"
    c_consumer = native / "test_bevfusion_sdk_c"
    assert c_consumer.is_file(), "build target test_bevfusion_sdk_c"
    assert qualifier.is_file(), (
        "build target test_bevfusion_pipeline, including its public Task qualifier"
    )
    revision = manifest["source_revision"]
    for dependency in manifest["external_files"]:
        assert (model_dir / dependency["path"]).is_file(), (
            f"prepare declared BEVFusion input: {dependency['path']}"
        )
        if dependency["path"].startswith("reference/"):
            assert f"/{revision}/" in dependency["url"]
    checkpoint = model_dir / "bevfusion-seg.pth"
    assert checkpoint.is_file()
    record_evidence(
        "checkpoint",
        {
            "source_revision": revision,
            "checkpoint_file": str(checkpoint),
            "declared_checkpoint": manifest["external_files"][0]["url"],
        },
    )
    case = manifest["testcases"][0]
    limits = json.loads((ROOT / "thresholds" / f"{name}.json").read_text())["threshold_overrides"]
    record_evidence("thresholds", limits)
    bundle = tmp_path / manifest["bundle"]
    with evidence_stage("build"):
        build(
            BuildRequest(
                model_dir=model_dir,
                output_path=bundle,
                family="bevfusion",
                task=manifest["task"],
                precision=manifest["precision"],
                quantization=manifest.get("quantization"),
                tensor_parallel_size=manifest["tensor_parallel_size"],
            )
        )
    if manifest.get("quantization") == "fp8":
        _inspect_fp8_engine(bundle, tmp_path / "engine-layers.json")
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.random.fork_rng(devices=[]):
            reference = Reference(source, checkpoint).cuda().eval()
        errors = []
        for seed in case["seeds"]:
            specs = {item["name"]: item for item in case["feature_maps"]}
            assert set(specs) == {"camera_bev", "lidar_bev"}
            fixture = prepare(tmp_path / f"seed{seed}", seed, case["feature_maps"])
            paths = [fixture / specs[key]["values_path"] for key in ("camera_bev", "lidar_bev")]
            paths.append(tmp_path / f"{seed}-output.f32")
            shapes = [
                (1, *(specs[key][axis] for axis in ("channels", "height", "width")))
                for key in ("camera_bev", "lidar_bev")
            ]
            camera, lidar = [
                np.fromfile(path, np.float32).reshape(shape)
                for path, shape in zip(paths[:2], shapes)
            ]
            with evidence_stage("reference"), torch.inference_mode():
                expected = (
                    reference(torch.from_numpy(camera).cuda(), torch.from_numpy(lidar).cuda())
                    .cpu()
                    .numpy()
                )
            argv = [str(qualifier), str(bundle), str(runtime), *(str(path) for path in paths)]
            record_evidence("commands", {"argv": argv})
            with evidence_stage("native"):
                subprocess.run(argv, check=True, capture_output=True, text=True)
                actual = np.fromfile(paths[2], np.float32).reshape(1, 6, 200, 200)
                c_output = tmp_path / f"{seed}-c-output.f32"
                subprocess.run(
                    [str(c_consumer), *argv[1:5], str(c_output)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                c_actual = np.fromfile(c_output, np.float32).reshape(actual.shape)
                np.testing.assert_array_equal(c_actual, actual)
            with evidence_stage("compare"):
                assert np.isfinite(actual).all() and (actual >= 0).all() and (actual <= 1).all()
                np.testing.assert_allclose(
                    actual, expected, atol=limits["atol"], rtol=limits["rtol"]
                )
                errors.append(float(np.max(np.abs(actual - expected))))
            if seed == case["seeds"][0]:
                performance = tmp_path / "engine-only.json"
                with evidence_stage("engine_performance"):
                    subprocess.run(
                        [*argv[:5], str(performance), "--engine-only"],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    report = json.loads(performance.read_text())
                    engine_values = np.fromfile(report["output_values_file"], np.float32).reshape(
                        actual.shape
                    )
                    np.testing.assert_array_equal(engine_values, actual)
                    assert report["timing_boundary"] == "trt_engine_cuda_graph_gpu"
                    assert report["engine_forward_calls_per_iteration"] == 1
                    assert report["transfers_included"] is False
                    assert np.isfinite(report["median_ms"]) and report["median_ms"] > 0
                    if manifest.get("quantization") == "fp8":
                        metadata = report["runtime_metadata"]
                        assert metadata["quantization"] == "fp8" and metadata["fp8_layers"]
                        calibration = metadata["calibration"]
                        calibration_seeds = range(
                            calibration["first_seed"],
                            calibration["first_seed"] + calibration["samples"],
                        )
                        assert set(case["seeds"]).isdisjoint(calibration_seeds)
                record_evidence("engine_performance", report)
        record_evidence(
            "metrics",
            {
                "maximum_absolute_errors": errors,
                "seeds": case["seeds"],
                "output_shape": [1, 6, 200, 200],
            },
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
