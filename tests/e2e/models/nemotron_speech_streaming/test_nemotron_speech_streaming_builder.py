# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic load tests for the Nemotron Speech Streaming RNNT family.

Covers both checkpoints the family supports today:
- ``nvidia/nemotron-speech-streaming-en-0.6b`` (monolingual).
- ``nvidia/nemotron-3.5-asr-streaming-0.6b`` (multilingual, prompt_kernel MLP).
"""

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

    # Multilingual-only constants.
    NUM_PROMPTS, PK_HIDDEN = 8, 32
    STREAMING_RIGHT_CONTEXTS_SORTED_DESC = [13, 6, 3, 0]
    STREAMING_CACHE_LEFT = 56
    PROMPT_DICTIONARY = {"en-US": 0, "es-ES": 1, "de-DE": 2, "fr-FR": 3}

    @staticmethod
    def _write_config(model_dir, config):
        (model_dir / "config.json").write_text(json.dumps(config))

    def _build_state_dict(self, with_prompt_kernel: bool = False):
        """Build a synthetic state dict for encoder + predictor + joint (+ prompt_kernel)."""
        import torch

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

        if with_prompt_kernel:
            pk_input_dim = self.HIDDEN + self.NUM_PROMPTS
            sd["prompt_kernel.0.weight"] = torch.randn(self.PK_HIDDEN, pk_input_dim)
            sd["prompt_kernel.0.bias"] = torch.randn(self.PK_HIDDEN)
            sd["prompt_kernel.2.weight"] = torch.randn(self.HIDDEN, self.PK_HIDDEN)
            sd["prompt_kernel.2.bias"] = torch.randn(self.HIDDEN)
        return sd

    def _build_nemo_cfg(self, att_context_size=None, prompt_dictionary=None):
        cfg = {
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
        if att_context_size is not None:
            cfg["encoder"]["att_context_size"] = att_context_size
        if prompt_dictionary is not None:
            cfg["train_ds"] = {"prompt_dictionary": dict(prompt_dictionary)}
        return cfg

    def _write_top_level_config(self, tmp_path, model_type: str):
        self._write_config(tmp_path, {
            "model_type": model_type,
            "hidden_size": self.PRED_HIDDEN,
            "num_hidden_layers": self.PRED_LAYERS,
            "num_attention_heads": 1,
            "vocab_size": self.VOCAB + 1,
            "mel_length": 80,
        })

    def test_load_weights_keys(self, tmp_path):
        """Monolingual en-0.6b: encoder + predictor + joint, no prompt_kernel."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch required for synthetic NeMo archive test")

        from tensorrt_model_connect.families.nemotron_speech_streaming import model as plugin

        sd = self._build_state_dict(with_prompt_kernel=False)
        nemo_cfg = self._build_nemo_cfg()
        make_nemo_archive(tmp_path, sd, nemo_cfg)
        self._write_top_level_config(tmp_path, "nemotron_speech_streaming")

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
        assert weights["_has_prompt_kernel"] is False

        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "nemotron_speech_streaming_speech_to_text_rnnt"
        assert overrides["rnnt_blank_id"] == self.VOCAB
        assert overrides["rnnt_causal_downsampling"] is True
        assert overrides["rnnt_has_prompt_kernel"] is False

    def test_load_weights_multilingual_artifacts(self, tmp_path):
        """Multilingual 3.5: prompt_kernel + 4 right-contexts + cache_left=56."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch required for synthetic NeMo archive test")

        from tensorrt_model_connect.families.nemotron_speech_streaming import model as plugin

        sd = self._build_state_dict(with_prompt_kernel=True)
        nemo_cfg = self._build_nemo_cfg(
            att_context_size=[[self.STREAMING_CACHE_LEFT, r]
                              for r in self.STREAMING_RIGHT_CONTEXTS_SORTED_DESC],
            prompt_dictionary=self.PROMPT_DICTIONARY,
        )
        make_nemo_archive(tmp_path, sd, nemo_cfg)
        self._write_top_level_config(tmp_path, "nemotron_3_5_asr_streaming")

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        assert weights["_streaming_right_contexts"] == self.STREAMING_RIGHT_CONTEXTS_SORTED_DESC
        assert 1 not in weights["_streaming_right_contexts"]
        assert weights["_streaming_cache_left"] == self.STREAMING_CACHE_LEFT

        pk_input_dim = self.HIDDEN + self.NUM_PROMPTS
        assert weights["_has_prompt_kernel"] is True
        assert weights["_num_prompts"] == self.NUM_PROMPTS
        assert weights["_pk_input_dim"] == pk_input_dim
        assert weights["_pk_output_dim"] == self.HIDDEN
        assert weights["pk_w0"].shape == (pk_input_dim, self.PK_HIDDEN)
        assert weights["pk_w2"].shape == (self.PK_HIDDEN, self.HIDDEN)
        assert weights["_prompt_dictionary"] == self.PROMPT_DICTIONARY

        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["rnnt_streaming_cache_left"] == self.STREAMING_CACHE_LEFT
        assert overrides["rnnt_streaming_right_contexts"] == \
            self.STREAMING_RIGHT_CONTEXTS_SORTED_DESC
        assert overrides["rnnt_has_prompt_kernel"] is True
        assert overrides["rnnt_num_prompts"] == self.NUM_PROMPTS
        assert overrides["rnnt_prompt_dictionary"] == self.PROMPT_DICTIONARY

    def test_missing_prompt_dictionary_raises(self, tmp_path):
        """prompt_kernel without train_ds.prompt_dictionary must fail loudly."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch required for synthetic NeMo archive test")

        from tensorrt_model_connect.families.nemotron_speech_streaming import model as plugin

        sd = self._build_state_dict(with_prompt_kernel=True)
        nemo_cfg = self._build_nemo_cfg(
            att_context_size=[[self.STREAMING_CACHE_LEFT, 13]],
            prompt_dictionary=None,
        )
        make_nemo_archive(tmp_path, sd, nemo_cfg)
        self._write_top_level_config(tmp_path, "nemotron_3_5_asr_streaming")

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(ValueError, match="prompt_dictionary"):
            plugin.load_weights(str(tmp_path), cfg)
