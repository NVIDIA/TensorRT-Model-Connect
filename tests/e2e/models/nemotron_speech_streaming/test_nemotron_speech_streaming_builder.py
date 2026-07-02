# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic load tests for the Nemotron Speech Streaming RNNT family."""

from __future__ import annotations

import json

import pytest

pytest.importorskip(
    "tensorrt", reason="Nemotron Speech Streaming builder tests require TensorRT"
)

try:
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tests.e2e.models.nemotron_speech_streaming.synthetic_nemo import (
    make_nemo_archive,
    make_nemo_state_dict,
)


class TestNemotronSpeechStreamingPlugin:
    VOCAB, HIDDEN, ENC_LAYERS = 32, 16, 2
    HEADS, HEAD_DIM, FFN = 2, 8, 32
    MEL_BINS, CONV_KERNEL, SUB_CH = 8, 3, 4
    PRED_HIDDEN, PRED_LAYERS, JOINT_HIDDEN = 12, 2, 10

    @staticmethod
    def _write_config(model_dir, config):
        (model_dir / "config.json").write_text(json.dumps(config))

    def test_load_weights_keys(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch required for synthetic NeMo archive test")

        from tensorrt_model_connect.families.nemotron_speech_streaming import plugin

        sd = make_nemo_state_dict(
            self.VOCAB,
            self.HIDDEN,
            self.ENC_LAYERS,
            1,
            self.HEADS,
            self.HEAD_DIM,
            self.FFN,
            self.MEL_BINS,
            self.CONV_KERNEL,
            self.SUB_CH,
        )
        for i in range(self.ENC_LAYERS):
            sd.pop(f"encoder.layers.{i}.conv.batch_norm.running_mean", None)
            sd.pop(f"encoder.layers.{i}.conv.batch_norm.running_var", None)

        # RNNT predictor: embedding includes the blank row at VOCAB.
        sd["decoder.prediction.embed.weight"] = torch.randn(self.VOCAB + 1, self.PRED_HIDDEN)
        for i in range(self.PRED_LAYERS):
            sd[f"decoder.prediction.dec_rnn.weight_ih_l{i}"] = torch.randn(
                4 * self.PRED_HIDDEN, self.PRED_HIDDEN)
            sd[f"decoder.prediction.dec_rnn.weight_hh_l{i}"] = torch.randn(
                4 * self.PRED_HIDDEN, self.PRED_HIDDEN)
            sd[f"decoder.prediction.dec_rnn.bias_ih_l{i}"] = torch.randn(4 * self.PRED_HIDDEN)
            sd[f"decoder.prediction.dec_rnn.bias_hh_l{i}"] = torch.randn(4 * self.PRED_HIDDEN)

        sd["joint.enc.weight"] = torch.randn(self.JOINT_HIDDEN, self.HIDDEN)
        sd["joint.enc.bias"] = torch.randn(self.JOINT_HIDDEN)
        sd["joint.pred.weight"] = torch.randn(self.JOINT_HIDDEN, self.PRED_HIDDEN)
        sd["joint.pred.bias"] = torch.randn(self.JOINT_HIDDEN)
        sd["joint.joint_net.1.weight"] = torch.randn(self.VOCAB + 1, self.JOINT_HIDDEN)
        sd["joint.joint_net.1.bias"] = torch.randn(self.VOCAB + 1)

        nemo_cfg = {
            "target": "nemo.collections.asr.models.EncDecRNNTBPEModel",
            "model_defaults": {
                "enc_hidden": self.HIDDEN,
                "pred_hidden": self.PRED_HIDDEN,
                "joint_hidden": self.JOINT_HIDDEN,
            },
            "encoder": {
                "d_model": self.HIDDEN,
                "n_layers": self.ENC_LAYERS,
                "n_heads": self.HEADS,
                "ff_expansion_factor": self.FFN // self.HIDDEN,
                "conv_kernel_size": self.CONV_KERNEL,
                "conv_norm_type": "layer_norm",
                "conv_context_size": "causal",
                "causal_downsampling": True,
                "feat_in": self.MEL_BINS,
                "subsampling_conv_channels": self.SUB_CH,
            },
            "decoder": {
                "blank_idx": self.VOCAB,
                "prednet": {
                    "pred_hidden": self.PRED_HIDDEN,
                    "pred_rnn_layers": self.PRED_LAYERS,
                },
            },
            "joint": {
                "jointnet": {
                    "encoder_hidden": self.HIDDEN,
                    "pred_hidden": self.PRED_HIDDEN,
                    "joint_hidden": self.JOINT_HIDDEN,
                    "activation": "relu",
                }
            },
            "preprocessor": {"features": self.MEL_BINS},
        }

        make_nemo_archive(tmp_path, sd, nemo_cfg)
        self._write_config(tmp_path, {
            "model_type": "nemotron_speech_streaming",
            "hidden_size": self.PRED_HIDDEN,
            "num_hidden_layers": self.PRED_LAYERS,
            "num_attention_heads": 1,
            "vocab_size": self.VOCAB + 1,
            "mel_length": 80,
        })

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["_enc_layers"] == self.ENC_LAYERS
        assert weights["_pred_layers"] == self.PRED_LAYERS
        assert weights["_pred_hidden"] == self.PRED_HIDDEN
        assert weights["_joint_hidden"] == self.JOINT_HIDDEN
        assert weights["_blank_id"] == self.VOCAB
        assert weights["pred_embedding"].shape == (self.VOCAB + 1, self.PRED_HIDDEN)
        for i in range(self.PRED_LAYERS):
            assert weights[f"pred.{i}.w_ih_t"].shape == (self.PRED_HIDDEN, 4 * self.PRED_HIDDEN)
            assert weights[f"pred.{i}.w_hh_t"].shape == (self.PRED_HIDDEN, 4 * self.PRED_HIDDEN)
        assert weights["joint_enc_w"].shape == (self.HIDDEN, self.JOINT_HIDDEN)
        assert weights["joint_pred_w"].shape == (self.PRED_HIDDEN, self.JOINT_HIDDEN)
        assert weights["joint_out_w"].shape == (self.JOINT_HIDDEN, self.VOCAB + 1)

        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "nemotron_speech_streaming_speech_to_text_rnnt"
        assert overrides["rnnt_blank_id"] == self.VOCAB
        assert overrides["rnnt_causal_downsampling"] is True
