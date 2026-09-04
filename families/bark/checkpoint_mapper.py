# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""1:1 port of standard_checkpoint_mapper.cpp + tensor_math.cpp to Python.

Loads the family's exact Hugging Face pytorch_model.bin and maps keys to the flat weight dict expected by
standard_decoder_builder.py. All projections are transposed from HF
[out, in] layout to [in, out] for TRT matmul.
"""

from __future__ import annotations

from pathlib import Path


class WeightDict(dict):
    """A dict mapping logical weight names to flat float32 arrays.

    Keys follow the convention used by standard_decoder_builder.py:
      - embedding: [vocab, hidden]
      - layer.{i}.input_norm: [hidden]
      - layer.{i}.w_q: [hidden, attention_size]
      - layer.{i}.w_k: [hidden, kv_attention_size]
      - layer.{i}.w_v: [hidden, kv_attention_size]
      - layer.{i}.q_bias: [attention_size]       (optional)
      - layer.{i}.k_bias: [kv_attention_size]    (optional)
      - layer.{i}.v_bias: [kv_attention_size]    (optional)
      - layer.{i}.q_norm: [attention_size]        (optional)
      - layer.{i}.k_norm: [kv_attention_size]     (optional)
      - layer.{i}.w_o: [attention_size, hidden]
      - layer.{i}.post_attn_norm: [hidden]
      - layer.{i}.w_gate: [hidden, mlp_size]
      - layer.{i}.w_up: [hidden, mlp_size]
      - layer.{i}.w_down: [mlp_size, hidden]
      - final_norm: [hidden]
      - w_out: [hidden, vocab]
    """


# ---------------------------------------------------------------------------
# PyTorch checkpoint I/O helpers
# ---------------------------------------------------------------------------
class _TorchCheckpointReader:
    """One exact PyTorch state-dict reader."""

    def __init__(self, state: dict):
        self._state = state

    def keys(self) -> list[str]:
        return list(self._state)

    def get_tensor(self, name: str):
        return self._state[name]


class _ReaderCollection(list):
    """Checkpoint readers with one exact tensor-to-reader index."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {name: reader for reader in readers for name in reader.keys()}
        self.tensor_map = tensor_map


def _open_torch_checkpoint(model_dir: Path) -> _ReaderCollection:
    """Open the family's required pytorch_model.bin checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "This family requires torch in the build environment"
        ) from error

    path = model_dir / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required PyTorch checkpoint is missing: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return _ReaderCollection([_TorchCheckpointReader(state)])
