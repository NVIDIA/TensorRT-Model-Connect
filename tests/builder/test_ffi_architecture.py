# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FFI kernel architecture: kernel setup and .so bundling.

Intent:
    Validates the refactored FFI kernel architecture:
    1. Kernel .so files round-trip through the bundle format
    2. FlashInfer kernel setup returns a valid (name, so_path) pair

Preconditions:
    - Bundle tests: No special deps (pure Python)
    - FlashInfer tests: FlashInfer + TVM-FFI + CUDA available

Postconditions:
    - Bundle kernel manifest + .so sections survive write/read round-trip
    - FlashInfer setup produces a registered kernel + valid .so path

Trace IDs: ARCH-TVM-FFI-002, UD-FFI-ARCH-001, UT-FFI-ARCH-001
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import pytest

try:
    import tensorrt_model_connect  # noqa: F401
except ImportError:
    pytest.skip("tensorrt_model_connect not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Test 1: Bundle kernel manifest round-trip (no GPU needed)
# ---------------------------------------------------------------------------


class TestBundleKernelManifest:
    """Verify kernel .so sections and manifest survive bundle write/read."""

    def test_kernel_manifest_round_trip(self):
        """Write a bundle with kernel_manifest.json + fake .so, read it back."""
        from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle

        info = BundleInfo(model_id="test-model", model_type="test")

        fake_so_data = b"\x7fELF" + b"\x00" * 100  # fake ELF header
        manifest = {
            "kernels": [
                {
                    "global_name": "test.kernel_f16_d64",
                    "func_name": "run",
                    "section": "kernel_test_kernel_f16_d64.so",
                }
            ]
        }
        manifest_json = json.dumps(manifest).encode("utf-8")

        sections = [
            BundleSection("config.json", b"{}"),
            BundleSection("kernel_test_kernel_f16_d64.so", fake_so_data),
            BundleSection("kernel_manifest.json", manifest_json),
        ]

        with tempfile.NamedTemporaryFile(suffix=".trtfb", delete=False) as f:
            bundle_path = f.name

        try:
            write_bundle(bundle_path, info, sections)

            # Read back and verify
            data = Path(bundle_path).read_bytes()
            assert data[:5] == b"TRTFB", "Bundle magic bytes missing"

            # Parse header
            header_len = struct.unpack("<Q", data[8:16])[0]
            header = json.loads(data[16 : 16 + header_len])

            assert "kernel_manifest.json" in header["sections"]
            assert "kernel_test_kernel_f16_d64.so" in header["sections"]

            # Extract manifest section
            m_info = header["sections"]["kernel_manifest.json"]
            body_start = 16 + header_len
            m_data = data[body_start + m_info["offset"] : body_start + m_info["offset"] + m_info["size"]]
            parsed = json.loads(m_data)
            assert len(parsed["kernels"]) == 1
            assert parsed["kernels"][0]["global_name"] == "test.kernel_f16_d64"

            # Extract .so section
            so_info = header["sections"]["kernel_test_kernel_f16_d64.so"]
            so_data = data[body_start + so_info["offset"] : body_start + so_info["offset"] + so_info["size"]]
            assert so_data == fake_so_data
        finally:
            Path(bundle_path).unlink(missing_ok=True)

    def test_bundle_without_kernel_manifest(self):
        """Bundles without kernel_manifest.json still work (backward compat)."""
        from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle

        info = BundleInfo(model_id="test-model", model_type="test")
        sections = [BundleSection("config.json", b"{}")]

        with tempfile.NamedTemporaryFile(suffix=".trtfb", delete=False) as f:
            bundle_path = f.name

        try:
            write_bundle(bundle_path, info, sections)
            data = Path(bundle_path).read_bytes()
            header_len = struct.unpack("<Q", data[8:16])[0]
            header = json.loads(data[16 : 16 + header_len])
            assert "kernel_manifest.json" not in header["sections"]
        finally:
            Path(bundle_path).unlink(missing_ok=True)

    def test_multiple_kernels_in_manifest(self):
        """Bundle can contain multiple kernel .so sections."""
        from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle

        info = BundleInfo(model_id="test-model", model_type="test")
        fake_so_1 = b"\x7fELF_kernel_1" + b"\x00" * 50
        fake_so_2 = b"\x7fELF_kernel_2" + b"\x00" * 75

        manifest = {
            "kernels": [
                {"global_name": "flashinfer.decode_f16_d64", "func_name": "run",
                 "section": "kernel_flashinfer_decode_f16_d64.so"},
                {"global_name": "cute.fused_swiglu", "func_name": "run",
                 "section": "kernel_cute_fused_swiglu.so"},
            ]
        }

        sections = [
            BundleSection("config.json", b"{}"),
            BundleSection("kernel_flashinfer_decode_f16_d64.so", fake_so_1),
            BundleSection("kernel_cute_fused_swiglu.so", fake_so_2),
            BundleSection("kernel_manifest.json", json.dumps(manifest).encode()),
        ]

        with tempfile.NamedTemporaryFile(suffix=".trtfb", delete=False) as f:
            bundle_path = f.name

        try:
            write_bundle(bundle_path, info, sections)
            data = Path(bundle_path).read_bytes()
            header_len = struct.unpack("<Q", data[8:16])[0]
            header = json.loads(data[16 : 16 + header_len])

            assert len(header["sections"]) == 4  # config + 2 .so + manifest
            assert "kernel_flashinfer_decode_f16_d64.so" in header["sections"]
            assert "kernel_cute_fused_swiglu.so" in header["sections"]
        finally:
            Path(bundle_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Test 2: FlashInfer kernel setup (FlashInfer + TVM-FFI required)
# ---------------------------------------------------------------------------


def _flashinfer_available():
    try:
        import flashinfer  # noqa: F401
        import tvm_ffi  # noqa: F401
        return True
    except ImportError:
        return False


requires_flashinfer = pytest.mark.skipif(
    not _flashinfer_available(),
    reason="FlashInfer + TVM-FFI not available",
)


class TestFlashInferKernelSetup:
    """Verify kernels/flashinfer_decode.setup() works correctly."""

    @requires_flashinfer
    def test_setup_returns_valid_kernel_name_and_so_path(self):
        """setup() returns (kernel_name, so_path) with valid .so file."""
        from tensorrt_model_connect.kernels import flashinfer_decode

        name, so_path = flashinfer_decode.setup(head_dim=64)

        assert name == "flashinfer.decode_f16_d64"
        assert Path(so_path).exists()
        assert Path(so_path).stat().st_size > 0

    @requires_flashinfer
    def test_setup_registers_tvm_ffi_global(self):
        """setup() registers the kernel as a TVM-FFI global function."""
        import tvm_ffi
        from tensorrt_model_connect.kernels import flashinfer_decode

        name, _ = flashinfer_decode.setup(head_dim=64)
        func = tvm_ffi.get_global_func(name)
        assert func is not None

    @requires_flashinfer
    def test_setup_different_head_dims(self):
        """setup() works for different head dimensions."""
        from tensorrt_model_connect.kernels import flashinfer_decode

        for hd in (64, 128):
            name, so_path = flashinfer_decode.setup(head_dim=hd)
            assert name == f"flashinfer.decode_f16_d{hd}"
            assert Path(so_path).exists()


# ---------------------------------------------------------------------------
# Test 3: Engine builder kernel_artifacts parameter
# ---------------------------------------------------------------------------


class TestEngineBuilderKernelArtifacts:
    """Verify build_bundle accepts kernel_artifacts and packages them."""

    def test_build_bundle_signature_has_kernel_artifacts(self):
        """build_bundle() accepts kernel_artifacts keyword argument."""
        import inspect
        from tensorrt_model_connect.engine_builder import build_bundle

        sig = inspect.signature(build_bundle)
        assert "kernel_artifacts" in sig.parameters
        param = sig.parameters["kernel_artifacts"]
        assert param.default is None
