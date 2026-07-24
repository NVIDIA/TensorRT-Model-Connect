# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Python wrapper around the C++ trtmc CLI.

C++ is the single source of truth for inference. This wrapper calls the
trtmc binary as a subprocess, providing a Pythonic API.

Usage:
    import tensorrt_model_connect
    pipe = tensorrt_model_connect.Pipeline("model.trtfb")
    print(pipe("Hello world", max_new_tokens=10))
    print(pipe("Describe this image", image="photo.jpg", max_new_tokens=50))
"""

from __future__ import annotations

import os
import shutil
import subprocess
import json
from importlib import resources
from pathlib import Path
from typing import Iterable


def _normalize_kv_cache_memory(value: str | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("kv_cache_memory must be 'auto', a percentage, a size, or bytes")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("kv_cache_memory bytes must be positive")
        return f"{value}B"
    text = str(value).strip()
    if not text:
        raise ValueError("kv_cache_memory must not be empty")
    return text


def _normalize_max_sequence_length(value: str | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("max_sequence_length must be 'auto' or a positive token count")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("max_sequence_length must be positive")
        return str(value)
    text = str(value).strip()
    if not text:
        raise ValueError("max_sequence_length must not be empty")
    return text


def _memory_receipt_from_stderr(stderr: str) -> dict | None:
    if not isinstance(stderr, str):
        return None
    prefix = "[trtmc.memory] "
    for line in reversed(stderr.splitlines()):
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        if not payload.startswith("{"):
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _existing_executable(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _native_binary_candidates() -> list[Path]:
    candidates: list[Path] = []
    try:
        candidates.append(
            Path(resources.files("tensorrt_model_connect").joinpath("bin", "trtmc"))
        )
    except (FileNotFoundError, ModuleNotFoundError):
        pass

    package_file = Path(__file__).resolve()
    candidates.append(package_file.parents[2] / "build" / "trtmc")
    return candidates


class Pipeline:
    """Python wrapper that calls the C++ trtmc CLI as a subprocess.

    Args:
        bundle_path: Path to a .trtfb bundle file.
        binary: Path to the trtmc binary. Auto-detected if not specified.
        hf_python: Path to Python interpreter for tokenizer. Auto-detected.
    """

    def __init__(
        self,
        bundle_path: str,
        binary: str | None = None,
        hf_python: str | None = None,
        kv_cache_memory: str | int | None = None,
        max_sequence_length: str | int | None = None,
    ):
        self.bundle_path = str(bundle_path)

        if binary is not None:
            self.binary = binary
        else:
            self.binary = self._find_binary()

        self.hf_python = hf_python
        self.kv_cache_memory = _normalize_kv_cache_memory(kv_cache_memory)
        self.max_sequence_length = _normalize_max_sequence_length(
            max_sequence_length
        )
        self.last_memory_receipt: dict | None = None

    def __call__(
        self,
        prompt: str,
        *,
        image: str | None = None,
        lora_adapter: str | None = None,
        lora_adapter_id: str = "default",
        max_new_tokens: int = 20,
        timeout: float = 120.0,
    ) -> str:
        """Generate text (and optionally process an image).

        Args:
            prompt: Input text prompt.
            image: Optional path to an image file (for VL models).
            lora_adapter: Optional path to a PEFT LoRA adapter directory.
            lora_adapter_id: Runtime ID assigned to ``lora_adapter``.
            max_new_tokens: Maximum tokens to generate.
            timeout: Subprocess timeout in seconds.

        Returns:
            Generated text string.
        """
        cmd = [
            self.binary, "run", self.bundle_path,
            "--prompt", prompt,
            "--max-new-tokens", str(max_new_tokens),
        ]

        if image is not None:
            cmd.extend(["--image", str(image)])

        if lora_adapter is not None:
            if not lora_adapter_id:
                raise ValueError("lora_adapter_id must not be empty")
            cmd.extend([
                "--lora-adapter", str(lora_adapter),
                "--lora-adapter-id", lora_adapter_id,
            ])

        if self.hf_python is not None:
            cmd.extend(["--hf-python", self.hf_python])
        if self.kv_cache_memory is not None:
            cmd.extend(["--kv-cache-memory", self.kv_cache_memory])
        if self.max_sequence_length is not None:
            cmd.extend(["--max-sequence-length", self.max_sequence_length])

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        self.last_memory_receipt = _memory_receipt_from_stderr(result.stderr)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"trtmc run failed (exit={result.returncode}): {stderr}")

        return result.stdout.strip()

    @staticmethod
    def _find_binary() -> str:
        """Auto-detect the trtmc binary location."""
        native = _existing_executable(_native_binary_candidates())
        if native is not None:
            return str(native)

        # Check PATH
        found = shutil.which("trtmc")
        if found is not None:
            return found

        raise FileNotFoundError(
            "trtmc binary not found. Build it with: "
            "cmake -S . -B build -G Ninja && cmake --build build")

    def inspect(self) -> str:
        """Run trtmc inspect on the bundle."""
        cmd = [self.binary, "inspect", self.bundle_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"trtmc inspect failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def __repr__(self) -> str:
        return f"Pipeline({self.bundle_path!r})"
