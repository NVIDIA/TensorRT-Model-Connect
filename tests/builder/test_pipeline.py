# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pipeline.py — Python wrapper around the C++ trtmc CLI.

Pure Python tests with mocked subprocess calls. No GPU or TRT needed.

Trace: ARCH-FAC-001, UD-FAC-PIPELINE
Intent: Validate Pipeline subprocess wrapper init, binary detection, and CLI argument construction
Preconditions: subprocess calls are mocked; no real C++ binary or GPU required
Postconditions: Pipeline correctly stores paths, auto-detects binary, and constructs valid subprocess commands
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import tensorrt_model_connect.pipeline as pipeline_module
    from tensorrt_model_connect.pipeline import Pipeline
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


# ---------------------------------------------------------------------------
# Pipeline.__init__
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_explicit_binary(self):
        """Explicit binary path is stored directly, no auto-detection."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtmc")
        assert pipe.binary == "/usr/bin/trtmc"
        assert pipe.bundle_path == "/tmp/model.trtfb"
        assert pipe.hf_python is None

    def test_explicit_binary_and_hf_python(self):
        """Both binary and hf_python are stored when provided."""
        pipe = Pipeline(
            "/tmp/model.trtfb",
            binary="/usr/bin/trtmc",
            hf_python="/opt/venv/bin/python",
        )
        assert pipe.binary == "/usr/bin/trtmc"
        assert pipe.hf_python == "/opt/venv/bin/python"

    def test_auto_detect_calls_find_binary(self):
        """When binary is None, _find_binary is called."""
        with patch.object(Pipeline, "_find_binary", return_value="/auto/trtmc"):
            pipe = Pipeline("/tmp/model.trtfb")
            assert pipe.binary == "/auto/trtmc"

    def test_binary_not_found_raises(self):
        """_find_binary raises FileNotFoundError when nothing is found."""
        with patch.object(
            Pipeline, "_find_binary",
            side_effect=FileNotFoundError("trtmc binary not found"),
        ):
            with pytest.raises(FileNotFoundError, match="trtmc binary not found"):
                Pipeline("/tmp/model.trtfb")

    def test_bundle_path_converted_to_str(self):
        """Path objects are converted to str."""
        pipe = Pipeline(Path("/tmp/model.trtfb"), binary="/usr/bin/trtmc")
        assert isinstance(pipe.bundle_path, str)
        assert pipe.bundle_path == "/tmp/model.trtfb"

    def test_repr(self):
        """__repr__ includes the bundle path."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtmc")
        assert repr(pipe) == "Pipeline('/tmp/model.trtfb')"


# ---------------------------------------------------------------------------
# Pipeline.__call__
# ---------------------------------------------------------------------------


class TestPipelineCall:
    def _make_pipeline(self, hf_python=None):
        return Pipeline(
            "/tmp/model.trtfb", binary="/usr/bin/trtmc", hf_python=hf_python)

    def test_basic_prompt(self):
        """Basic text prompt constructs correct command."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello world!\n"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            output = pipe("Say hello", max_new_tokens=5)

            mock_run.assert_called_once_with(
                [
                    "/usr/bin/trtmc", "run", "/tmp/model.trtfb",
                    "--prompt", "Say hello",
                    "--max-new-tokens", "5",
                ],
                capture_output=True,
                text=True,
                timeout=120.0,
            )
            assert output == "Hello world!"

    def test_prompt_with_image(self):
        """Image argument is appended to the command."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "A cat sitting on a couch\n"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            output = pipe(
                "Describe this image",
                image="/tmp/photo.jpg",
                max_new_tokens=30,
            )

            cmd = mock_run.call_args[0][0]
            assert "--image" in cmd
            idx = cmd.index("--image")
            assert cmd[idx + 1] == "/tmp/photo.jpg"
            assert output == "A cat sitting on a couch"

    def test_prompt_with_lora_adapter(self):
        """A PEFT adapter directory and runtime ID are forwarded to the CLI."""
        pipe = self._make_pipeline()
        mock_result = MagicMock(returncode=0, stdout="answer\n")

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                   return_value=mock_result) as mock_run:
            output = pipe(
                "test prompt",
                lora_adapter="/tmp/test-adapter",
                lora_adapter_id="adapter-1",
            )

            cmd = mock_run.call_args[0][0]
            assert cmd[cmd.index("--lora-adapter") + 1] == "/tmp/test-adapter"
            assert cmd[cmd.index("--lora-adapter-id") + 1] == "adapter-1"
            assert output == "answer"

    def test_empty_lora_adapter_id_rejected(self):
        pipe = self._make_pipeline()
        with pytest.raises(ValueError, match="lora_adapter_id must not be empty"):
            pipe("test prompt", lora_adapter="/tmp/test-adapter", lora_adapter_id="")

    def test_prompt_without_image(self):
        """When image is None, --image is not in the command."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            assert "--image" not in cmd

    def test_hf_python_appended(self):
        """When hf_python is set, --hf-python is in the command."""
        pipe = self._make_pipeline(hf_python="/opt/venv/bin/python")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            assert "--hf-python" in cmd
            idx = cmd.index("--hf-python")
            assert cmd[idx + 1] == "/opt/venv/bin/python"

    def test_hf_python_not_appended_when_none(self):
        """When hf_python is None, --hf-python is not in the command."""
        pipe = self._make_pipeline(hf_python=None)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "output"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            assert "--hf-python" not in cmd

    def test_nonzero_returncode_raises(self):
        """Non-zero exit code raises RuntimeError with stderr."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CUDA out of memory"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(RuntimeError, match="trtmc run failed"):
                pipe("Hello")

    def test_nonzero_returncode_includes_exit_code(self):
        """RuntimeError message includes the exit code."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 137
        mock_result.stderr = "Killed"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(RuntimeError, match="exit=137"):
                pipe("Hello")

    def test_subprocess_timeout(self):
        """subprocess.TimeoutExpired is propagated."""
        pipe = self._make_pipeline()

        with patch(
            "tensorrt_model_connect.pipeline.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="trtmc", timeout=5.0),
        ):
            with pytest.raises(subprocess.TimeoutExpired):
                pipe("Hello", timeout=5.0)

    def test_custom_timeout(self):
        """Custom timeout is forwarded to subprocess.run."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "out"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello", timeout=60.0)
            assert mock_run.call_args[1]["timeout"] == 60.0

    def test_default_max_new_tokens(self):
        """Default max_new_tokens is 20."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "out"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            pipe("Hello")
            cmd = mock_run.call_args[0][0]
            idx = cmd.index("--max-new-tokens")
            assert cmd[idx + 1] == "20"

    def test_stdout_stripped(self):
        """Leading/trailing whitespace in stdout is stripped."""
        pipe = self._make_pipeline()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n  Hello world  \n\n"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result):
            assert pipe("x") == "Hello world"


# ---------------------------------------------------------------------------
# Pipeline.inspect
# ---------------------------------------------------------------------------


class TestPipelineInspect:
    def test_inspect_success(self):
        """inspect() returns stripped stdout."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtmc")
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Model ID:  example-model\nLayers: 28\n"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result) as mock_run:
            out = pipe.inspect()
            mock_run.assert_called_once_with(
                ["/usr/bin/trtmc", "inspect", "/tmp/model.trtfb"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert out == "Model ID:  example-model\nLayers: 28"

    def test_inspect_failure_raises(self):
        """Non-zero exit code in inspect raises RuntimeError."""
        pipe = Pipeline("/tmp/model.trtfb", binary="/usr/bin/trtmc")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not a valid .trtfb bundle"

        with patch("tensorrt_model_connect.pipeline.subprocess.run",
                    return_value=mock_result):
            with pytest.raises(RuntimeError, match="trtmc inspect failed"):
                pipe.inspect()


# ---------------------------------------------------------------------------
# Pipeline._find_binary
# ---------------------------------------------------------------------------


class TestFindBinary:
    def test_build_trtmc_exists(self, tmp_path):
        """Finds ./build/trtmc when it exists."""
        trtmc = tmp_path / "build" / "trtmc"
        trtmc.parent.mkdir()
        trtmc.write_text("#!/bin/sh\n", encoding="utf-8")
        trtmc.chmod(0o755)

        with patch(
            "tensorrt_model_connect.pipeline._native_binary_candidates",
            return_value=[trtmc],
        ):
            result = Pipeline._find_binary()
            assert result == str(trtmc)

    def test_found_on_path(self):
        """Falls back to shutil.which when local candidates missing."""
        with patch(
            "tensorrt_model_connect.pipeline._native_binary_candidates",
            return_value=[],
        ):
            with patch("tensorrt_model_connect.pipeline.shutil.which",
                        return_value="/usr/local/bin/trtmc"):
                result = Pipeline._find_binary()
                assert result == "/usr/local/bin/trtmc"

    def test_not_found_anywhere(self):
        """Raises FileNotFoundError when trtmc is not found anywhere."""
        with patch(
            "tensorrt_model_connect.pipeline._native_binary_candidates",
            return_value=[],
        ):
            with patch("tensorrt_model_connect.pipeline.shutil.which",
                        return_value=None):
                with pytest.raises(FileNotFoundError, match="trtmc binary not found"):
                    Pipeline._find_binary()

    def test_existing_executable_returns_first_executable(self, tmp_path):
        missing = tmp_path / "missing"
        non_executable = tmp_path / "non_executable"
        executable = tmp_path / "trtmc"
        non_executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

        assert pipeline_module._existing_executable(
            [missing, non_executable, executable]
        ) == executable

    def test_native_binary_candidates_include_package_and_source_paths(self):
        candidates = pipeline_module._native_binary_candidates()

        assert any(candidate.name == "trtmc" for candidate in candidates)
        assert any(candidate.parent.name == "build" for candidate in candidates)
