# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint readers and tensor normalization for Parakeet TDT."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Mapping

import numpy as np


class WeightDict(dict):
    """Normalized family-owned build tensors."""


def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _transpose(value) -> np.ndarray:
    array = _array(value)
    if array.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got {array.shape}")
    return np.ascontiguousarray(array.T)


def _transpose_2d(value, name: str, precision: str = "fp32") -> np.ndarray:
    """Family graph-builder adapter with the repository's mapper signature."""
    del name, precision
    return _transpose(value)


def map_transducer_weights(
    state: Mapping[str, object],
    *,
    vocab_size: int,
    duration_count: int,
    decoder_layers: int,
    decoder_hidden_size: int,
    encoder_hidden_size: int,
) -> WeightDict:
    """Normalize the predictor/projector/joint tensors shared by HF and NeMo layouts."""
    out = WeightDict()
    out["pred_embedding"] = _array(state["decoder.embedding.weight"])
    for layer in range(decoder_layers):
        prefix = "decoder.lstm"
        w_ih = _array(state[f"{prefix}.weight_ih_l{layer}"])
        w_hh = _array(state[f"{prefix}.weight_hh_l{layer}"])
        b_ih = _array(state[f"{prefix}.bias_ih_l{layer}"])
        b_hh = _array(state[f"{prefix}.bias_hh_l{layer}"])
        expected = (4 * decoder_hidden_size, decoder_hidden_size)
        if w_ih.shape != expected or w_hh.shape != expected:
            raise ValueError(
                f"predictor layer {layer} has shapes {w_ih.shape}/{w_hh.shape}, expected {expected}"
            )
        out[f"pred.{layer}.w_ih_t"] = np.ascontiguousarray(w_ih.T)
        out[f"pred.{layer}.w_hh_t"] = np.ascontiguousarray(w_hh.T)
        out[f"pred.{layer}.bias"] = np.ascontiguousarray(b_ih + b_hh)

    out["decoder_projector_w"] = _transpose(state["decoder.decoder_projector.weight"])
    out["decoder_projector_b"] = _array(state["decoder.decoder_projector.bias"])
    out["encoder_projector_w"] = _transpose(state["encoder_projector.weight"])
    out["encoder_projector_b"] = _array(state["encoder_projector.bias"])

    joint_w = _array(state["joint.head.weight"])
    joint_b = _array(state["joint.head.bias"])
    expected_outputs = vocab_size + duration_count
    if joint_w.shape != (expected_outputs, decoder_hidden_size):
        raise ValueError(
            f"joint.head.weight has shape {joint_w.shape}, expected "
            f"{(expected_outputs, decoder_hidden_size)}"
        )
    if joint_b.shape != (expected_outputs,):
        raise ValueError(
            f"joint.head.bias has shape {joint_b.shape}, expected {(expected_outputs,)}"
        )
    out["joint_token_w"] = np.ascontiguousarray(joint_w[:vocab_size].T)
    out["joint_token_b"] = np.ascontiguousarray(joint_b[:vocab_size])
    out["joint_duration_w"] = np.ascontiguousarray(joint_w[vocab_size:].T)
    out["joint_duration_b"] = np.ascontiguousarray(joint_b[vocab_size:])
    out["_encoder_hidden"] = encoder_hidden_size
    out["_pred_hidden"] = decoder_hidden_size
    out["_pred_layers"] = decoder_layers
    out["_vocab"] = vocab_size
    out["_duration_count"] = duration_count
    return out


def load_hf_safetensors(model_dir: str | Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Parakeet HF checkpoint is missing {path.name}")
    return dict(load_file(path))


def load_nemo_archive(model_dir: str | Path) -> tuple[dict, dict]:
    import torch
    import yaml

    root = Path(model_dir)
    archive = root if root.suffix == ".nemo" else next(iter(sorted(root.glob("*.nemo"))), None)
    if archive is None:
        raise FileNotFoundError(f"No .nemo file found in {root}")
    state = config = None
    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name == "model_config.yaml":
                extracted = tar.extractfile(member)
                if extracted is not None:
                    config = yaml.safe_load(extracted.read())
            elif name == "model_weights.ckpt":
                extracted = tar.extractfile(member)
                if extracted is not None:
                    state = torch.load(
                        io.BytesIO(extracted.read()), map_location="cpu", weights_only=True
                    )
    if not isinstance(config, dict):
        raise FileNotFoundError(f"model_config.yaml not found in {archive}")
    if not isinstance(state, dict):
        raise FileNotFoundError(f"model_weights.ckpt not found in {archive}")
    if isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    return state, config
