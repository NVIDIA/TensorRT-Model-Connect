# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tensorrt_model_connect.families.sam3 import (
    hard_mask_resize_aoti_exporter,
    native_plugin_builder,
    tracker_memory_aoti_exporter,
    tracker_step_aoti_exporter,
)


def _inputs(*, torch_version: str = "2.9.0") -> native_plugin_builder._BuildInputs:
    return native_plugin_builder._BuildInputs(
        torch_root=Path("/packages/torch"),
        torch_cmake_prefix=Path("/packages/torch/share/cmake"),
        tvm_ffi_root=Path("/packages/tvm_ffi"),
        tensorrt_root=Path("/packages"),
        torch_version=torch_version,
        tvm_ffi_version="0.1.6",
        tensorrt_version="11.2.0",
        host_architecture="x86_64",
        torch_cxx11_abi=False,
    )


def test_source_digest_tracks_source_and_dependency_abi(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source = source_dir / "bridge.cpp"
    source.write_text("first", encoding="utf-8")
    initial = native_plugin_builder._source_digest(source_dir, _inputs())

    source.write_text("second", encoding="utf-8")
    source_changed = native_plugin_builder._source_digest(source_dir, _inputs())
    dependency_changed = native_plugin_builder._source_digest(
        source_dir, _inputs(torch_version="2.10.0")
    )

    assert initial != source_changed
    assert source_changed != dependency_changed


def test_configure_command_passes_all_dependency_roots() -> None:
    command = native_plugin_builder._configure_command(Path("/source"), Path("/build"), _inputs())

    assert "-DCMAKE_PREFIX_PATH=/packages/torch/share/cmake" in command
    assert "-DSAM3_TVM_FFI_ROOT=/packages/tvm_ffi" in command
    assert "-DSAM3_TENSORRT_ROOT=/packages" in command
    assert "-DSAM3_TORCH_VERSION=2.9.0" in command
    assert "-DSAM3_TVM_FFI_VERSION=0.1.6" in command
    assert "-DSAM3_TENSORRT_VERSION=11.2.0" in command
    assert "-DSAM3_TORCH_CXX11_ABI=0" in command


def test_native_plugin_build_is_content_addressed_and_cached(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess:
        calls.append(command)
        if command[1] == "--build":
            output = Path(command[2]) / native_plugin_builder._PLUGIN_NAME
            output.write_bytes(b"plugin")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(native_plugin_builder, "_BUILD_ROOT", tmp_path)
    monkeypatch.setattr(native_plugin_builder, "_discover_build_inputs", _inputs)
    monkeypatch.setattr(native_plugin_builder.subprocess, "run", run)

    first = native_plugin_builder.ensure_native_plugin()
    second = native_plugin_builder.ensure_native_plugin()

    assert first == second
    assert first.read_bytes() == b"plugin"
    assert len(calls) == 2
    assert not list(tmp_path.glob(".*.build-*"))


def test_native_bridge_preserves_tensorrt_stream_without_host_sync() -> None:
    source_dir = Path(native_plugin_builder.__file__).with_name("native_plugins")
    for filename in (
        "CMakeLists.txt",
        "sam3_tracker_step_aoti_bridge.cpp",
        "sam3_tracker_step_ffi_plugin.cpp",
        "sam3_tracker_step_ffi_plugin.h",
    ):
        assert (source_dir / filename).is_file(), filename
    plugin_source = (source_dir / "sam3_tracker_step_ffi_plugin.cpp").read_text(encoding="utf-8")
    aoti_source = (source_dir / "sam3_tracker_step_aoti_bridge.cpp").read_text(encoding="utf-8")

    assert "TVMFFIEnvSetStream" in plugin_source
    assert "return resolve_kernel() ? 0 : -1;" in plugin_source
    assert "descriptor_shapes_valid" in plugin_source
    encoder_run = aoti_source.index(
        "pipeline.encoder->run(encoder_inputs, reinterpret_cast<void*>(stream))"
    )
    decoder_run = aoti_source.index(
        "pipeline.decoder->run(decoder_inputs, reinterpret_cast<void*>(stream))"
    )
    assert encoder_run < decoder_run
    assert "pipeline_for_device" in aoti_source
    assert "trtmc_sam3_tracker_step_register_pipeline" in aoti_source
    assert 'AotiLoader>(package_path, "model", false, kAotiRunnerCount' in aoti_source
    assert "constexpr std::size_t kAotiRunnerCount = 2" in aoti_source
    assert "run_mutex" not in aoti_source
    assert "TVMFFIFunctionSetGlobal(&name, function, 1)" in aoti_source
    assert "std::vector<std::unique_ptr<Entry>>& retained_entries()" in aoti_source
    assert "static auto* value = new std::vector<std::unique_ptr<Entry>>" in aoti_source
    assert "auto& entries = retained_entries()" in aoti_source
    assert "validate_tensor_contract" in aoti_source
    assert "source.get_device() != destination.device.device_id" in aoti_source
    assert "cudaMemcpyAsync" in aoti_source
    assert "cudaStreamSynchronize" not in plugin_source + aoti_source


def _split_artifacts(
    tmp_path: Path,
) -> tracker_step_aoti_exporter.Sam3TrackerSplitAotiArtifacts:
    package_specs = (
        ("encoder", 1, "sam3_tracker_encoder_b1_dynamic.pt2"),
        ("decoder", 1, "sam3_tracker_decoder_b1_static.pt2"),
        ("encoder", 2, "sam3_tracker_encoder_b2_dynamic.pt2"),
        ("decoder", 2, "sam3_tracker_decoder_b2_static.pt2"),
    )
    packages = []
    for stage, batch_size, section in package_specs:
        path = tmp_path / section
        payload = f"{stage}-b{batch_size}".encode()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        packages.append(
            tracker_step_aoti_exporter.TrackerAotiPackage(
                stage=stage,
                batch_size=batch_size,
                path=path,
                section=section,
                sha256=digest,
                package_global=(
                    f"trtmc.sam3.tracker_{stage}.b{batch_size}."
                    f"{'m1_10.p1_19' if stage == 'encoder' else 'static'}.{digest[:20]}"
                ),
            )
        )
    by_key = {(package.stage, package.batch_size): package for package in packages}
    pipeline_b1 = native_plugin_builder._pipeline_global(
        1, by_key[("encoder", 1)].sha256, by_key[("decoder", 1)].sha256
    )
    pipeline_b2 = native_plugin_builder._pipeline_global(
        2, by_key[("encoder", 2)].sha256, by_key[("decoder", 2)].sha256
    )
    exporter_manifest = b'{"schema_version":1}'
    bundle_sections = (
        (
            tracker_step_aoti_exporter.TRACKER_SPLIT_AOTI_MANIFEST_SECTION,
            exporter_manifest,
        ),
        *((package.section, package.path.read_bytes()) for package in packages),
    )
    return tracker_step_aoti_exporter.Sam3TrackerSplitAotiArtifacts(
        cache_directory=tmp_path,
        packages=tuple(packages),
        pipeline_global_b1=pipeline_b1,
        pipeline_global_b2=pipeline_b2,
        producer_abi=tracker_step_aoti_exporter.TrackerAotiProducerAbi(
            torch_version="2.9.0",
            transformers_version="5.2.0",
            cuda_version="12.8",
            compute_capability=(8, 9),
            host_architecture="x86_64",
            torch_cxx11_abi=False,
        ),
        carrier_abi=tracker_step_aoti_exporter.TRACKER_TEN_CARRIER_ABI,
        manifest_bytes=exporter_manifest,
        bundle_sections=bundle_sections,
    )


def _memory_artifacts(
    tmp_path: Path,
) -> tracker_memory_aoti_exporter.Sam3TrackerMemoryAotiArtifacts:
    specs = (("soft", 1, False), ("hard", 1, True), ("soft", 2, False), ("hard", 2, True))
    packages = []
    records = []
    for policy, batch_size, hard_mask in specs:
        section = f"sam3_tracker_memory_{policy}_b{batch_size}.pt2"
        path = tmp_path / section
        payload = f"memory-{policy}-b{batch_size}".encode()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        package_global = native_plugin_builder._memory_package_global(policy, batch_size, digest)
        packages.append(
            tracker_memory_aoti_exporter.MemoryAotiPackage(
                policy=policy,
                batch_size=batch_size,
                hard_mask=hard_mask,
                path=path,
                section=section,
                sha256=digest,
                package_global=package_global,
            )
        )
        records.append(
            {
                "policy": policy,
                "batch_size": batch_size,
                "fixed_shape": True,
                "inputs": [
                    {
                        "name": "tracker_feature_2",
                        "dtype": "float32",
                        "shape": [1, 256, 72, 72],
                    },
                    {
                        "name": "owned_tracker_mask" if hard_mask else "final_mask",
                        "dtype": "float32",
                        "shape": [
                            batch_size,
                            1,
                            1008 if hard_mask else 288,
                            1008 if hard_mask else 288,
                        ],
                    },
                    {
                        "name": "object_score_logits",
                        "dtype": "float32",
                        "shape": [batch_size, 1],
                    },
                    {
                        "name": "suppress_area_shrinkage",
                        "dtype": "int32",
                        "shape": [batch_size, 1],
                    },
                ],
                "outputs": [
                    {
                        "name": "packed_memory_and_position",
                        "dtype": "float32",
                        "shape": ([2, 5184, 1, 64] if batch_size == 1 else [2, 2, 5184, 64]),
                    }
                ],
                "hard_mask": hard_mask,
                "filename": f"{policy}-b{batch_size}-{digest}.pt2",
                "section": section,
                "sha256": digest,
                "package_global": package_global,
            }
        )
    producer = tracker_memory_aoti_exporter.MemoryAotiProducerAbi(
        torch_version="2.9.0",
        transformers_version="5.2.0",
        cuda_version="12.8",
        compute_capability=(8, 9),
        host_architecture="x86_64",
        torch_cxx11_abi=False,
        torch_aoti_abi_version=7,
    )
    manifest = {
        "schema_version": 2,
        "scope": "fixed_memory_encoder_soft_hard_b1_b2",
        "artifact_format": "torch.aot_inductor.package.pt2",
        "implementation": {
            "library": "transformers",
            "model_class": "Sam3TrackerVideoModel",
            "module": "Sam3TrackerVideoMemoryEncoder",
            "license": "Apache-2.0",
            "source_import_policy": "transformers-only",
        },
        "producer": {
            "torch_version": "2.9.0",
            "transformers_version": "5.2.0",
            "cuda_version": "12.8",
            "compute_capability": [8, 9],
            "host_architecture": "x86_64",
            "torch_cxx11_abi": False,
            "torch_aoti_abi_version": 7,
        },
        "input_abi": [
            {
                "policy": policy_abi.policy,
                "tensors": [
                    {"name": tensor.name, "dtype": tensor.dtype, "shape": list(tensor.shape)}
                    for tensor in policy_abi.tensors
                ],
            }
            for policy_abi in tracker_memory_aoti_exporter.TRACKER_MEMORY_INPUT_ABI
        ],
        "packages": records,
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    bundle_sections = (
        (
            tracker_memory_aoti_exporter.TRACKER_MEMORY_AOTI_MANIFEST_SECTION,
            manifest_bytes,
        ),
        *((package.section, package.path.read_bytes()) for package in packages),
    )
    return tracker_memory_aoti_exporter.Sam3TrackerMemoryAotiArtifacts(
        cache_directory=tmp_path,
        packages=tuple(packages),
        producer_abi=producer,
        input_abi=tracker_memory_aoti_exporter.TRACKER_MEMORY_INPUT_ABI,
        manifest_bytes=manifest_bytes,
        bundle_sections=bundle_sections,
    )


def _resize_artifacts(tmp_path: Path) -> hard_mask_resize_aoti_exporter.HardMaskResizeAotiArtifacts:
    packages = []
    records = []
    for batch_size in (1, 2):
        section = f"sam3_hard_mask_resize_b{batch_size}.pt2"
        path = tmp_path / section
        payload = f"resize-b{batch_size}".encode()
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        packages.append(
            hard_mask_resize_aoti_exporter.HardMaskResizeAotiPackage(
                batch_size=batch_size,
                path=path,
                section=section,
                sha256=digest,
                package_global=(
                    f"trtmc.sam3.tracker_memory.resize.b{batch_size}.fixed.{digest[:20]}"
                ),
            )
        )
        records.append(
            {
                "batch_size": batch_size,
                "filename": f"sam3_hard_mask_resize_b{batch_size}_{digest}.pt2",
                "section": section,
                "sha256": digest,
                "package_global": (
                    f"trtmc.sam3.tracker_memory.resize.b{batch_size}.fixed.{digest[:20]}"
                ),
            }
        )
    producer = tracker_memory_aoti_exporter.MemoryAotiProducerAbi(
        torch_version="2.9.0",
        transformers_version="5.2.0",
        cuda_version="12.8",
        compute_capability=(8, 9),
        host_architecture="x86_64",
        torch_cxx11_abi=False,
        torch_aoti_abi_version=7,
    )
    manifest = json.dumps(
        {
            "schema_version": 1,
            "scope": "torch_bilinear_288_to_1008_b1_b2",
            "artifact_format": "torch.aot_inductor.package.pt2",
            "implementation": {
                "library": "torch",
                "operator": "torch.nn.functional.interpolate",
                "mode": "bilinear",
                "align_corners": False,
                "source_size": 288,
                "target_size": 1008,
            },
            "producer": asdict(producer),
            "host_architecture": producer.host_architecture,
            "exporter_sha256": "a" * 64,
            "input_abi": [
                {"name": "tracker_mask", "dtype": "float32", "shape": ["B", 1, 288, 288]}
            ],
            "output_abi": [
                {
                    "name": "resized_tracker_mask",
                    "dtype": "float32",
                    "shape": ["B", 1, 1008, 1008],
                }
            ],
            "packages": records,
            "package_validation": {
                "reference": "same torch.interpolate eager execution",
                "maximum_absolute_error": 2.0e-5,
                "cases": [
                    {"batch_size": 1, "maximum_absolute_error": 1.0e-6, "passed": True},
                    {"batch_size": 2, "maximum_absolute_error": 1.0e-6, "passed": True},
                ],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hard_mask_resize_aoti_exporter.HardMaskResizeAotiArtifacts(
        cache_directory=tmp_path,
        packages=tuple(packages),
        producer_abi=producer,
        manifest_bytes=manifest,
        bundle_sections=(
            (hard_mask_resize_aoti_exporter.HARD_MASK_RESIZE_AOTI_MANIFEST_SECTION, manifest),
            *((package.section, package.path.read_bytes()) for package in packages),
        ),
    )


def test_runtime_manifest_binds_plugin_and_both_split_packages_per_batch(
    tmp_path: Path,
) -> None:
    split = _split_artifacts(tmp_path)
    plugin = tmp_path / native_plugin_builder._PLUGIN_NAME
    plugin.write_bytes(b"native-plugin")

    runtime = native_plugin_builder._assemble_runtime_artifacts(
        split,
        _memory_artifacts(tmp_path),
        _resize_artifacts(tmp_path),
        plugin,
        _inputs(),
        aoti_abi_version=7,
    )
    manifest = json.loads(runtime.runtime_manifest)

    assert manifest["step_scope"] == native_plugin_builder.TRACKER_STEP_RUNTIME_SCOPE
    assert manifest["producer"]["transformers_version"] == "5.2.0"
    assert manifest["producer"]["aoti_abi_version"] == 7
    assert len(manifest["packages"]) == 4
    assert [pipeline["batch_size"] for pipeline in manifest["pipelines"]] == [1, 2]
    assert manifest["pipelines"][0]["global_name"] == split.pipeline_global_b1
    assert manifest["pipelines"][1]["global_name"] == split.pipeline_global_b2
    assert manifest["plugin"]["sha256"] == hashlib.sha256(b"native-plugin").hexdigest()
    section_names = [name for name, _ in runtime.bundle_sections]
    assert tracker_memory_aoti_exporter.TRACKER_MEMORY_AOTI_MANIFEST_SECTION in section_names
    assert section_names[-2:] == [
        native_plugin_builder.TRACKER_STEP_NATIVE_PLUGIN_SECTION,
        native_plugin_builder.TRACKER_STEP_RUNTIME_MANIFEST_SECTION,
    ]


def test_runtime_manifest_rejects_tampered_package_or_pipeline(tmp_path: Path) -> None:
    split = _split_artifacts(tmp_path)
    plugin = tmp_path / native_plugin_builder._PLUGIN_NAME
    plugin.write_bytes(b"native-plugin")

    split.packages[0].path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="package hash mismatch"):
        native_plugin_builder._assemble_runtime_artifacts(
            split,
            _memory_artifacts(tmp_path),
            _resize_artifacts(tmp_path),
            plugin,
            _inputs(),
            aoti_abi_version=7,
        )

    split = _split_artifacts(tmp_path)
    mismatched = replace(split, pipeline_global_b1=split.pipeline_global_b2)
    with pytest.raises(RuntimeError, match="does not bind both packages"):
        native_plugin_builder._assemble_runtime_artifacts(
            mismatched,
            _memory_artifacts(tmp_path),
            _resize_artifacts(tmp_path),
            plugin,
            _inputs(),
            aoti_abi_version=7,
        )


def test_build_time_registration_binds_each_pipeline_to_both_package_paths(
    tmp_path: Path,
) -> None:
    split = _split_artifacts(tmp_path)

    class _Register:
        def __init__(self) -> None:
            self.calls = []
            self.argtypes = None
            self.restype = None

        def __call__(
            self,
            global_name,
            encoder_path,
            decoder_path,
            encoder_sha256,
            decoder_sha256,
            batch_size,
        ):
            self.calls.append(
                (
                    global_name,
                    encoder_path,
                    decoder_path,
                    encoder_sha256,
                    decoder_sha256,
                    batch_size,
                )
            )
            return 0

    register = _Register()
    library = type(
        "Library",
        (),
        {"trtmc_sam3_tracker_step_register_pipeline": register},
    )()

    native_plugin_builder._register_split_pipelines(library, split)

    assert register.calls == [
        (
            split.pipeline_global_b1.encode(),
            str(split.packages[0].path).encode(),
            str(split.packages[1].path).encode(),
            split.packages[0].sha256.encode(),
            split.packages[1].sha256.encode(),
            1,
        ),
        (
            split.pipeline_global_b2.encode(),
            str(split.packages[2].path).encode(),
            str(split.packages[3].path).encode(),
            split.packages[2].sha256.encode(),
            split.packages[3].sha256.encode(),
            2,
        ),
    ]


def test_build_time_registration_binds_all_four_memory_packages(tmp_path: Path) -> None:
    memory = _memory_artifacts(tmp_path)

    class _Register:
        def __init__(self) -> None:
            self.calls = []
            self.argtypes = None
            self.restype = None

        def __call__(self, global_name, path, digest, policy, batch_size):
            self.calls.append((global_name, path, digest, policy, batch_size))
            return 0

    register = _Register()
    library = type(
        "Library",
        (),
        {"trtmc_sam3_tracker_memory_register_package": register},
    )()

    native_plugin_builder._register_memory_packages(library, memory)

    assert register.calls == [
        (
            package.package_global.encode(),
            str(package.path).encode(),
            package.sha256.encode(),
            package.policy.encode(),
            package.batch_size,
        )
        for package in memory.packages
    ]


def test_build_time_registration_binds_b1_b2_resize_packages(tmp_path: Path) -> None:
    resize = _resize_artifacts(tmp_path)

    class _Register:
        def __init__(self) -> None:
            self.calls = []
            self.argtypes = None
            self.restype = None

        def __call__(self, global_name, path, digest, policy, batch_size):
            self.calls.append((global_name, path, digest, policy, batch_size))
            return 0

    register = _Register()
    library = type("Library", (), {"trtmc_sam3_tracker_memory_register_package": register})()
    native_plugin_builder._register_resize_packages(library, resize)
    assert register.calls == [
        (
            package.package_global.encode(),
            str(package.path).encode(),
            package.sha256.encode(),
            b"resize",
            package.batch_size,
        )
        for package in resize.packages
    ]


def test_resize_artifacts_reject_non_finite_validation_error(tmp_path: Path) -> None:
    split = _split_artifacts(tmp_path)
    resize = _resize_artifacts(tmp_path)
    plugin = tmp_path / native_plugin_builder._PLUGIN_NAME
    plugin.write_bytes(b"native-plugin")
    manifest = json.loads(resize.manifest_bytes)
    manifest["package_validation"]["cases"][0]["maximum_absolute_error"] = float("nan")
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    resize = replace(
        resize,
        manifest_bytes=manifest_bytes,
        bundle_sections=(
            (resize.bundle_sections[0][0], manifest_bytes),
            *resize.bundle_sections[1:],
        ),
    )
    with pytest.raises(RuntimeError, match="package validation mismatch"):
        native_plugin_builder._assemble_runtime_artifacts(
            split,
            _memory_artifacts(tmp_path),
            resize,
            plugin,
            _inputs(),
            aoti_abi_version=7,
        )


def test_memory_artifacts_fail_closed_on_hash_and_step_abi_mismatch(tmp_path: Path) -> None:
    split = _split_artifacts(tmp_path)
    memory = _memory_artifacts(tmp_path)
    plugin = tmp_path / native_plugin_builder._PLUGIN_NAME
    plugin.write_bytes(b"native-plugin")

    memory.packages[0].path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="package hash mismatch"):
        native_plugin_builder._assemble_runtime_artifacts(
            split,
            memory,
            _resize_artifacts(tmp_path),
            plugin,
            _inputs(),
            aoti_abi_version=7,
        )

    memory = _memory_artifacts(tmp_path)
    mismatched_memory = replace(
        memory,
        producer_abi=replace(memory.producer_abi, compute_capability=(12, 0)),
    )
    with pytest.raises(RuntimeError, match="producer ABI mismatch"):
        native_plugin_builder._assemble_runtime_artifacts(
            split,
            mismatched_memory,
            _resize_artifacts(tmp_path),
            plugin,
            _inputs(),
            aoti_abi_version=7,
        )
