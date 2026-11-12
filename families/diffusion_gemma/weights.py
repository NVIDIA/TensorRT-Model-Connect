# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Memory-map the checkpoint's tied text weights one engine at a time."""

from __future__ import annotations

import json
from pathlib import Path
import re
import struct

import ml_dtypes
import numpy as np

from .config import Config


_DTYPES = {"BF16": np.dtype(ml_dtypes.bfloat16), "F16": np.dtype("<f2"), "F32": np.dtype("<f4")}


def _is_finite(value: np.ndarray) -> bool:
    # Inspect BF16's exponent directly: its NumPy dtype otherwise converts
    # each element to FP32 to implement isfinite on this 52 GB checkpoint.
    if value.dtype == np.dtype(ml_dtypes.bfloat16):
        bits = value.view(np.uint16)
        return not bool(np.any((bits & 0x7F80) == 0x7F80))
    return bool(np.isfinite(value).all())


class Checkpoint:
    def __init__(self, directory: str | Path, config: Config):
        self.directory = Path(directory).resolve()
        self.config = config
        index = self.directory / "model.safetensors.index.json"
        self._headers: dict[str, tuple[Path, int, dict]] = {}
        self._cache_files: dict[Path, Path] | None = None
        if index.is_file():
            document = json.loads(index.read_text())
            self._locations = document.get("weight_map")
            if not isinstance(self._locations, dict) or not self._locations:
                raise ValueError("diffusion_gemma checkpoint has no safetensors weight map")
        else:
            _, _, header = self._read_header("model.safetensors")
            self._locations = {name: "model.safetensors" for name in header if name != "__metadata__"}

    def _shard_path(self, filename: str) -> Path:
        if not isinstance(filename, str) or not filename or Path(filename).name != filename or filename in {".", ".."}:
            raise ValueError("diffusion_gemma weight shard must be inside the checkpoint directory")
        lexical = self.directory / filename
        path = lexical.resolve()
        if path.parent == self.directory:
            return path
        repository = self.directory.parent.parent
        blobs = repository / "blobs"
        if (not lexical.is_symlink() or self.directory.parent.name != "snapshots"
                or not re.fullmatch(r"[0-9a-f]{40}", self.directory.name)
                or blobs.is_symlink() or path.parent != blobs
                or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", path.name)):
            raise ValueError("diffusion_gemma weight shard must be inside the checkpoint directory or its verified Hub cache")
        if self._cache_files is None:
            from huggingface_hub import scan_cache_dir

            # The Hub inventory establishes repository/revision/file provenance;
            # no family-owned content fingerprint is introduced.
            self._cache_files = {
                cached.file_path: cached.blob_path
                for repo in scan_cache_dir(repository.parent).repos
                if repo.repo_type == "model" and repo.repo_path == repository
                for revision in repo.revisions
                if revision.snapshot_path == self.directory and revision.commit_hash == self.directory.name
                for cached in revision.files
            }
        if self._cache_files.get(lexical) != path:
            raise ValueError("diffusion_gemma weight shard is not verified by the Hub cache inventory")
        return path

    def _read_header(self, filename: str) -> tuple[Path, int, dict]:
        if filename in self._headers:
            return self._headers[filename]
        path = self._shard_path(filename)
        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"diffusion_gemma truncated safetensors shard: {path.name}")
            header_size = struct.unpack("<Q", length_bytes)[0]
            if not 2 <= header_size <= 100 * 1024 * 1024:
                raise ValueError(f"diffusion_gemma invalid safetensors header: {path.name}")
            header_bytes = stream.read(header_size)
            if len(header_bytes) != header_size:
                raise ValueError(f"diffusion_gemma truncated safetensors header: {path.name}")
            header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError("diffusion_gemma safetensors header must be an object")
        result = path, 8 + header_size, header
        self._headers[filename] = result
        return result

    def tensor(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        filename = self._locations.get(name)
        if not isinstance(filename, str):
            raise ValueError(f"diffusion_gemma missing checkpoint tensor: {name}")
        path, data_offset, header = self._read_header(filename)
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"diffusion_gemma missing tensor in indexed shard: {name}")
        if tuple(entry.get("shape", ())) != shape:
            raise ValueError(f"diffusion_gemma {name} has shape {entry.get('shape')}, expected {shape}")
        dtype = _DTYPES.get(entry.get("dtype"))
        if dtype is None:
            raise ValueError(f"diffusion_gemma unsupported checkpoint dtype for {name}")
        offsets = entry.get("data_offsets")
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if (
            not isinstance(offsets, list) or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != size
            or data_offset + offsets[1] > path.stat().st_size
        ):
            raise ValueError(f"diffusion_gemma invalid tensor byte range: {name}")
        return np.memmap(path, mode="r", dtype=dtype, offset=data_offset + offsets[0], shape=shape)

    def layer(self, index: int) -> dict[str, np.ndarray]:
        prefix = f"model.decoder.layers.{index}."
        result = {
            name: self.tensor(prefix + name, shape)
            for name, shape in self.config.layer_weight_shapes(index).items()
        }
        encoder_scalar = self.tensor(f"model.encoder.language_model.layers.{index}.layer_scalar", (1,))
        if not np.array_equal(encoder_scalar, result["layer_scalar"]):
            raise ValueError("diffusion_gemma encoder and decoder layer scalars must match")
        return result

    def embedding(self) -> dict[str, np.ndarray]:
        return {
            name: self.tensor("model.decoder." + name, shape)
            for name, shape in self.config.embedding_weight_shapes().items()
        }

    def output(self) -> dict[str, np.ndarray]:
        return {
            name: self.tensor("model.decoder." + name, shape)
            for name, shape in self.config.output_weight_shapes().items()
        }

    def validate(self) -> None:
        """Fail before compilation for missing, malformed, or nonfinite text weights."""
        for group in (self.embedding(), self.output()):
            for name, value in group.items():
                if not _is_finite(value):
                    raise ValueError(f"diffusion_gemma nonfinite checkpoint tensor: {name}")
        for index in range(self.config.text_config["num_hidden_layers"]):
            print(f"diffusion_gemma: validating checkpoint layer {index + 1}/{self.config.text_config['num_hidden_layers']}", flush=True)
            for name, value in self.layer(index).items():
                if not _is_finite(value):
                    raise ValueError(f"diffusion_gemma nonfinite checkpoint tensor: layer {index} {name}")
