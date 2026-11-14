# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit tensor-only PyTorch checkpoint reader for original InternVL AWQ."""

import json
from pathlib import Path

import numpy as np

from .checkpoint_mapper import _ReaderCollection


class _TorchShardReader:
    """Read published AWQ shards using the restricted weights-only loader."""

    def __init__(self, path: Path):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("InternVL PyTorch checkpoint loading requires torch>=2.10.0") from error

        self.tensors = torch.load(path, map_location="cpu", weights_only=True, mmap=True)

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, name: str) -> np.ndarray:
        import torch

        tensor = self.tensors[name].detach().cpu()
        # NumPy cannot expose torch.bfloat16 directly. Keep packed INT32 intact.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()


def _open_torch_checkpoint(model_dir: Path) -> _ReaderCollection:
    """Open only the published PyTorch layout; never retry another framework."""
    torch_index = model_dir / "pytorch_model.bin.index.json"
    if torch_index.is_file():
        weight_map = json.loads(torch_index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("pytorch_model.bin.index.json has no weight_map")
        if any(not isinstance(shard, str) or Path(shard).name != shard for shard in weight_map.values()):
            raise ValueError("PyTorch shard names must be checkpoint-local filenames")
        shard_files = sorted(set(weight_map.values()))
        readers_by_file = {shard: _TorchShardReader(model_dir / shard) for shard in shard_files}
        return _ReaderCollection(
            list(readers_by_file.values()),
            tensor_map={name: readers_by_file[shard] for name, shard in weight_map.items()},
        )
    torch_single = model_dir / "pytorch_model.bin"
    if torch_single.is_file():
        return _ReaderCollection([_TorchShardReader(torch_single)])
    raise FileNotFoundError(f"No supported PyTorch checkpoint in {model_dir}")
