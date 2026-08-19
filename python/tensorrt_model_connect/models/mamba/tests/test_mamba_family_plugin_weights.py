# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    _rand,
    _write_config,
    _write_safetensors,
)


class TestMambaPlugin:
    VOCAB, HIDDEN, LAYERS = 32, 16, 2
    D_INNER = 32
    STATE_SIZE = 8
    CONV_KERNEL = 4
    DT_RANK = 6

    def _make_tensors(self):
        t = {}
        t["backbone.embeddings.weight"] = _rand(self.VOCAB, self.HIDDEN)
        for i in range(self.LAYERS):
            p = f"backbone.layers.{i}"
            t[f"{p}.norm.weight"] = _rand(self.HIDDEN)
            # in_proj: [2*d_inner, hidden]
            t[f"{p}.mixer.in_proj.weight"] = _rand(2 * self.D_INNER, self.HIDDEN)
            # conv1d: [d_inner, 1, conv_kernel]
            t[f"{p}.mixer.conv1d.weight"] = _rand(
                self.D_INNER, 1, self.CONV_KERNEL)
            t[f"{p}.mixer.conv1d.bias"] = _rand(self.D_INNER)
            # x_proj: [dt_rank + 2*state_size, d_inner]
            t[f"{p}.mixer.x_proj.weight"] = _rand(
                self.DT_RANK + 2 * self.STATE_SIZE, self.D_INNER)
            # dt_proj: [d_inner, dt_rank]
            t[f"{p}.mixer.dt_proj.weight"] = _rand(self.D_INNER, self.DT_RANK)
            t[f"{p}.mixer.dt_proj.bias"] = _rand(self.D_INNER)
            # A_log: [d_inner, state_size]
            t[f"{p}.mixer.A_log"] = _rand(self.D_INNER, self.STATE_SIZE)
            # D: [d_inner]
            t[f"{p}.mixer.D"] = _rand(self.D_INNER)
            # out_proj: [hidden, d_inner]
            t[f"{p}.mixer.out_proj.weight"] = _rand(self.HIDDEN, self.D_INNER)
        t["backbone.norm_f.weight"] = _rand(self.HIDDEN)
        t["lm_head.weight"] = _rand(self.VOCAB, self.HIDDEN)
        return t

    def test_a_log_transform(self, tmp_path):
        """A_log should be transformed to A = -exp(A_log)."""
        import tensorrt_model_connect.models.mamba.model as plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            A_log = tensors[f"backbone.layers.{i}.mixer.A_log"]
            expected_A = -np.exp(A_log.astype(np.float32))
            np.testing.assert_allclose(
                weights[f"layer.{i}.A"], expected_A, atol=1e-5)

    def test_in_proj_split(self, tmp_path):
        """in_proj should be split into w_in_x and w_in_z."""
        import tensorrt_model_connect.models.mamba.model as plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        for i in range(self.LAYERS):
            in_proj = tensors[f"backbone.layers.{i}.mixer.in_proj.weight"]
            x_raw = in_proj[:self.D_INNER, :]
            z_raw = in_proj[self.D_INNER:, :]
            # Transposed: [hidden, d_inner]
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_in_x"],
                x_raw.T.astype(np.float32), atol=1e-6)
            np.testing.assert_allclose(
                weights[f"layer.{i}.w_in_z"],
                z_raw.T.astype(np.float32), atol=1e-6)

    def test_x_proj_split(self, tmp_path):
        """x_proj should be split into dt, B, C projections."""
        import tensorrt_model_connect.models.mamba.model as plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        x_proj = tensors["backbone.layers.0.mixer.x_proj.weight"]
        dt_raw = x_proj[:self.DT_RANK, :]
        B_raw = x_proj[self.DT_RANK:self.DT_RANK + self.STATE_SIZE, :]
        C_raw = x_proj[self.DT_RANK + self.STATE_SIZE:, :]

        # All transposed
        np.testing.assert_allclose(
            weights["layer.0.w_dt_in"],
            dt_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_B"],
            B_raw.T.astype(np.float32), atol=1e-6)
        np.testing.assert_allclose(
            weights["layer.0.w_C"],
            C_raw.T.astype(np.float32), atol=1e-6)

    def test_conv1d_reshaped(self, tmp_path):
        """conv1d weight [d_inner, 1, conv_kernel] should be reshaped to [d_inner, conv_kernel]."""
        import tensorrt_model_connect.models.mamba.model as plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": 1,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["layer.0.conv1d_weight"].shape == (
            self.D_INNER, self.CONV_KERNEL)

    def test_metadata_keys(self, tmp_path):
        """Mamba-specific dimension metadata should be stored."""
        import tensorrt_model_connect.models.mamba.model as plugin

        config = {
            "model_type": "mamba",
            "vocab_size": self.VOCAB,
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.LAYERS,
            "intermediate_size": self.D_INNER,
            "state_size": self.STATE_SIZE,
            "conv_kernel": self.CONV_KERNEL,
            "time_step_rank": self.DT_RANK,
        }
        tensors = self._make_tensors()
        _write_config(tmp_path, config)
        _write_safetensors(tmp_path, tensors)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["_d_inner"] == self.D_INNER
        assert weights["_state_size"] == self.STATE_SIZE
        assert weights["_conv_kernel"] == self.CONV_KERNEL
        assert weights["_dt_rank"] == self.DT_RANK
