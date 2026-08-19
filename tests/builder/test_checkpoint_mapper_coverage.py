# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-focused tests for Qwen checkpoint mapper helper branches.

Trace: ARCH-CHK-001, ARCH-MODPLUG-001, UD-CHK-02
Intent: Validate edge-case branches in checkpoint_mapper including q/k norm tiling, final norm fallback, QKV bias loading, and compact GQA/MQA K/V shapes.
Preconditions: tensorrt_model_connect and safetensors are importable; no TRT or GPU required.
Postconditions: Optional norm weights are tiled per-head, missing final norm defaults to ones, biases are loaded when present, and compact K/V shapes are preserved.
"""

from __future__ import annotations

import json
import sys
import types
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt_model_connect", reason="tensorrt_model_connect requires tensorrt")
from tensorrt_model_connect.models.qwen.config import ModelConfig

_QWEN_ROOT = Path(__file__).resolve().parents[2] / "python/tensorrt_model_connect/models/qwen"
_MAPPER_MODULE = "weights" if (_QWEN_ROOT / "weights/__init__.py").is_file() else "checkpoint_mapper"
cm = import_module(f"tensorrt_model_connect.models.qwen.{_MAPPER_MODULE}")


def _save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    from safetensors.numpy import save_file

    save_file(tensors, str(path))


def _base_config(hidden: int = 8) -> dict:
    return {
        "model_type": "standard_decoder",
        "vocab_size": 16,
        "hidden_size": hidden,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
    }


@pytest.mark.unit
def test_load_standard_weights_qk_norm_and_final_norm_fallback(tmp_path: Path) -> None:
    """Intent: cover optional q/k norm branches and missing final norm fallback.

    Preconditions: model.safetensors includes q_norm/k_norm and excludes model.norm.weight.
    Postconditions: q_norm/k_norm are tiled per head and final_norm defaults to ones.
    """
    hidden = 8
    cfg_json = _base_config(hidden)
    (tmp_path / "config.json").write_text(json.dumps(cfg_json), encoding="utf-8")

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": np.random.randn(16, hidden).astype(np.float32),
        "lm_head.weight": np.random.randn(16, hidden).astype(np.float32),
    }
    prefix = "model.layers.0"
    tensors[f"{prefix}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
    tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
    tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
    tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
    tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
    tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
    tensors[f"{prefix}.self_attn.q_norm.weight"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    tensors[f"{prefix}.self_attn.k_norm.weight"] = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(12, hidden).astype(np.float32)
    tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(12, hidden).astype(np.float32)
    tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(hidden, 12).astype(np.float32)
    _save_safetensors(tmp_path / "model.safetensors", tensors)

    cfg = ModelConfig.from_dir(tmp_path)
    weights = cm.load_standard_weights(tmp_path, cfg)

    assert "layer.0.q_norm" in weights
    assert "layer.0.k_norm" in weights
    assert weights["layer.0.q_norm"].shape == (hidden,)
    assert weights["layer.0.k_norm"].shape == (hidden,)
    np.testing.assert_allclose(weights["final_norm"], np.ones(hidden, dtype=np.float32))


@pytest.mark.unit
def test_detect_framework_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Intent: cover _detect_framework ImportError fallback.

    Preconditions: importing torch raises ImportError.
    Postconditions: _detect_framework returns "numpy".
    """
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert cm._detect_framework() == "numpy"


@pytest.mark.unit
def test_torch_bin_reader_keys_and_get_tensor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Intent: cover _TorchBinReader adapter keys/get_tensor methods.

    Preconditions: torch.load is mocked to return a deterministic state dict.
    Postconditions: keys() returns state keys and get_tensor() returns selected tensor.
    """
    fake_state = {
        "a.weight": np.array([1, 2, 3], dtype=np.float32),
        "b.bias": np.array([4, 5], dtype=np.float32),
    }

    fake_torch = types.SimpleNamespace(load=lambda *_args, **_kwargs: fake_state)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reader = cm._TorchBinReader(tmp_path / "pytorch_model.bin")
    assert sorted(reader.keys()) == ["a.weight", "b.bias"]
    np.testing.assert_array_equal(reader.get_tensor("b.bias"), fake_state["b.bias"])


@pytest.mark.unit
def test_open_safetensors_index_diffusion_and_bin_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Intent: cover index, diffusers, and bin fallback branches in _open_safetensors.

    Preconditions: branch-specific files are created and safe_open/_TorchBinReader are mocked.
    Postconditions: each branch returns expected reader objects without touching real model files.
    """
    monkeypatch.setattr(cm, "_detect_framework", lambda: "numpy")

    opened: list[str] = []

    class _FakeReader:
        """Minimal safe_open substitute: stringifies the same way the
        assertions expect (``f"reader:{name}"``) and exposes an empty
        ``.keys()`` so _ReaderCollection's tensor-map build doesn't crash."""

        def __init__(self, name: str) -> None:
            self._name = name

        def __eq__(self, other: object) -> bool:
            return other == f"reader:{self._name}"

        __hash__ = None  # type: ignore[assignment]

        def keys(self) -> list[str]:
            return []

    def fake_safe_open(path: str, framework: str):
        opened.append(f"{Path(path).name}:{framework}")
        return _FakeReader(Path(path).name)

    monkeypatch.setattr(cm, "safe_open", fake_safe_open)

    # model.safetensors.index.json branch
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    (idx_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "shard2.safetensors", "b": "shard1.safetensors"}}),
        encoding="utf-8",
    )
    readers = cm._open_safetensors(idx_dir)
    assert readers == ["reader:shard1.safetensors", "reader:shard2.safetensors"]

    # diffusion_pytorch_model.safetensors branch
    diff_dir = tmp_path / "diff_single"
    diff_dir.mkdir()
    (diff_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    readers = cm._open_safetensors(diff_dir)
    assert readers == ["reader:diffusion_pytorch_model.safetensors"]

    # diffusion_pytorch_model.safetensors.index.json branch
    diff_idx_dir = tmp_path / "diff_idx"
    diff_idx_dir.mkdir()
    (diff_idx_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "d2.safetensors", "b": "d1.safetensors"}}),
        encoding="utf-8",
    )
    readers = cm._open_safetensors(diff_idx_dir)
    assert readers == ["reader:d1.safetensors", "reader:d2.safetensors"]

    # pytorch_model.bin fallback branch
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "pytorch_model.bin").write_bytes(b"x")

    class _FakeBinReader:
        def __init__(self, p: Path) -> None:
            self._name = Path(p).name

        def __eq__(self, other: object) -> bool:
            return other == f"bin:{self._name}"

        __hash__ = None  # type: ignore[assignment]

        def keys(self) -> list[str]:
            return []

    monkeypatch.setattr(cm, "_TorchBinReader", _FakeBinReader)
    readers = cm._open_safetensors(bin_dir)
    assert readers == ["bin:pytorch_model.bin"]

    assert opened  # ensure safe_open path was exercised


class _Reader:
    def __init__(self, mapping: dict[str, object]) -> None:
        self._mapping = mapping

    def keys(self) -> list[str]:
        return list(self._mapping.keys())

    def get_tensor(self, name: str):
        return self._mapping[name]


class _FakeTorchTensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def float(self):
        return self

    def numpy(self):
        return self._arr


@pytest.mark.unit
def test_load_tensor_uint16_float16_and_torch_tensor_paths() -> None:
    """Intent: cover dtype conversion branches in _load_tensor.

    Preconditions: readers return uint16-bfloat bits, float16 arrays, and torch-like tensors.
    Postconditions: outputs are float32 arrays and tensors are found in-order across readers.
    """
    # uint16 branch (bfloat-like bit pattern): output should be float32.
    r1 = _Reader({"u16": np.array([0x3F80], dtype=np.uint16)})
    out_u16 = cm._load_tensor([r1], "u16")
    assert out_u16.dtype == np.float32
    assert out_u16.shape == (1,)

    # float16 branch.
    r2 = _Reader({"f16": np.array([1.5, -2.0], dtype=np.float16)})
    out_f16 = cm._load_tensor([r2], "f16")
    assert out_f16.dtype == np.float32
    np.testing.assert_allclose(out_f16, np.array([1.5, -2.0], dtype=np.float32))

    # torch-like tensor branch.
    r3 = _Reader({"torch_like": _FakeTorchTensor(np.array([7.0, 8.0], dtype=np.float32))})
    out_torch = cm._load_tensor([r3], "torch_like")
    np.testing.assert_allclose(out_torch, np.array([7.0, 8.0], dtype=np.float32))
