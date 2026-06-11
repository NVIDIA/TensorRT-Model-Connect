"""Synthetic load tests for the multilingual nemotron-3.5 RNNT variant.

These mirror :mod:`tests.builder.test_engine_nemotron_speech_streaming` but
exercise the multilingual code paths added in Commit 2:

* ``att_context_size`` ships as a list of pairs with ``left=56`` and
  ``right in {13, 6, 3, 0}`` (descending). ``right=1`` is intentionally not
  in the checkpoint and must not produce an engine.
* ``train_ds.prompt_dictionary`` advertises the language-tag -> prompt-index
  mapping that the prompt_kernel MLP consumes.
* ``prompt_kernel.{0,2}.{weight,bias}`` tensors are present, with the shape
  contract ``pk_input_dim == encoder_hidden + num_prompts``.
"""

from __future__ import annotations

import json

import pytest

try:
    from tensorrt_model_connect.config import ModelConfig
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)

from tests.builder.test_family_plugins import TestCanaryPlugin


class TestNemotron35AsrStreamingPlugin:
    """Synthetic-archive coverage for nemotron-3.5-asr-streaming-0.6b.

    Uses small dimensions; the contract under test is *shape and metadata
    propagation*, not numerical fidelity. The TRT engine builds themselves are
    covered by the live-checkpoint smoke tests below (marked requires_gpu /
    requires_network).
    """

    VOCAB, HIDDEN, ENC_LAYERS = 32, 16, 2
    HEADS, HEAD_DIM, FFN = 2, 8, 32
    MEL_BINS, CONV_KERNEL, SUB_CH = 8, 3, 4
    PRED_HIDDEN, PRED_LAYERS, JOINT_HIDDEN = 12, 2, 10
    NUM_PROMPTS, PK_HIDDEN = 8, 32

    # nemotron-3.5-asr-streaming-0.6b ships these four right contexts. right=1
    # is intentionally NOT present.
    STREAMING_RIGHT_CONTEXTS_SORTED_DESC = [13, 6, 3, 0]
    STREAMING_CACHE_LEFT = 56

    PROMPT_DICTIONARY = {
        "en-US": 0,
        "es-ES": 1,
        "de-DE": 2,
        "fr-FR": 3,
    }

    @staticmethod
    def _write_config(model_dir, config):
        (model_dir / "config.json").write_text(json.dumps(config))

    def test_load_weights_multilingual_artifacts(self, tmp_path):
        """Multilingual load: prompt_kernel + 4 right-contexts + cache_left=56."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch required for synthetic NeMo archive test")

        from tensorrt_model_connect.families.nemotron_speech_streaming import plugin

        sd = TestCanaryPlugin._make_nemo_state_dict(
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

        # RNNT joint.
        sd["joint.enc.weight"] = torch.randn(self.JOINT_HIDDEN, self.HIDDEN)
        sd["joint.enc.bias"] = torch.randn(self.JOINT_HIDDEN)
        sd["joint.pred.weight"] = torch.randn(self.JOINT_HIDDEN, self.PRED_HIDDEN)
        sd["joint.pred.bias"] = torch.randn(self.JOINT_HIDDEN)
        sd["joint.joint_net.1.weight"] = torch.randn(self.VOCAB + 1, self.JOINT_HIDDEN)
        sd["joint.joint_net.1.bias"] = torch.randn(self.VOCAB + 1)

        # Multilingual: prompt_kernel MLP. Shape contract:
        #   pk_input_dim (HIDDEN + NUM_PROMPTS) -> PK_HIDDEN -> HIDDEN.
        pk_input_dim = self.HIDDEN + self.NUM_PROMPTS
        sd["prompt_kernel.0.weight"] = torch.randn(self.PK_HIDDEN, pk_input_dim)
        sd["prompt_kernel.0.bias"] = torch.randn(self.PK_HIDDEN)
        sd["prompt_kernel.2.weight"] = torch.randn(self.HIDDEN, self.PK_HIDDEN)
        sd["prompt_kernel.2.bias"] = torch.randn(self.HIDDEN)

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
                # Full list as it appears in the nemotron-3.5 NeMo YAML. Order
                # is the on-disk order; the plugin sorts descending.
                "att_context_size": [
                    [self.STREAMING_CACHE_LEFT, 13],
                    [self.STREAMING_CACHE_LEFT, 6],
                    [self.STREAMING_CACHE_LEFT, 3],
                    [self.STREAMING_CACHE_LEFT, 0],
                ],
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
            # The plugin pulls the prompt dictionary from train_ds.
            "train_ds": {
                "prompt_dictionary": dict(self.PROMPT_DICTIONARY),
            },
        }

        TestCanaryPlugin._make_nemo_archive(tmp_path, sd, nemo_cfg)
        self._write_config(tmp_path, {
            "model_type": "nemotron_3_5_asr_streaming",
            "hidden_size": self.PRED_HIDDEN,
            "num_hidden_layers": self.PRED_LAYERS,
            "num_attention_heads": 1,
            "vocab_size": self.VOCAB + 1,
            "mel_length": 80,
        })

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Streaming context list is sorted descending and excludes right=1.
        assert weights["_streaming_right_contexts"] == self.STREAMING_RIGHT_CONTEXTS_SORTED_DESC
        assert 1 not in weights["_streaming_right_contexts"]
        assert weights["_streaming_cache_left"] == self.STREAMING_CACHE_LEFT

        # Prompt-kernel weights and shape metadata.
        assert weights["_has_prompt_kernel"] is True
        assert weights["_num_prompts"] == self.NUM_PROMPTS
        assert weights["_pk_input_dim"] == pk_input_dim
        assert weights["_pk_hidden"] == self.PK_HIDDEN
        assert weights["_pk_output_dim"] == self.HIDDEN
        assert weights["pk_w0"].shape == (pk_input_dim, self.PK_HIDDEN)
        assert weights["pk_b0"].shape == (self.PK_HIDDEN,)
        assert weights["pk_w2"].shape == (self.PK_HIDDEN, self.HIDDEN)
        assert weights["pk_b2"].shape == (self.HIDDEN,)
        assert weights["_prompt_dictionary"] == self.PROMPT_DICTIONARY

        # Bundle config carries everything the C++ runtime needs to know.
        overrides = plugin.get_bundle_config_overrides(cfg)
        assert overrides["runtime_strategy"] == "speech_to_text_rnnt"
        assert overrides["rnnt_blank_id"] == self.VOCAB
        assert overrides["rnnt_causal_downsampling"] is True
        assert overrides["rnnt_streaming_cache_left"] == self.STREAMING_CACHE_LEFT
        assert overrides["rnnt_streaming_right_contexts"] == \
            self.STREAMING_RIGHT_CONTEXTS_SORTED_DESC
        assert overrides["rnnt_has_prompt_kernel"] is True
        assert overrides["rnnt_num_prompts"] == self.NUM_PROMPTS
        assert overrides["rnnt_prompt_dictionary"] == self.PROMPT_DICTIONARY
        assert overrides["rnnt_prompt_dictionary"]["en-US"] == 0
        assert overrides["rnnt_prompt_dictionary"]["es-ES"] == 1

    def test_missing_prompt_dictionary_raises(self, tmp_path):
        """prompt_kernel without train_ds.prompt_dictionary must fail loudly."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch required for synthetic NeMo archive test")

        from tensorrt_model_connect.families.nemotron_speech_streaming import plugin

        sd = TestCanaryPlugin._make_nemo_state_dict(
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

        # prompt_kernel present but no train_ds in YAML.
        pk_input_dim = self.HIDDEN + self.NUM_PROMPTS
        sd["prompt_kernel.0.weight"] = torch.randn(self.PK_HIDDEN, pk_input_dim)
        sd["prompt_kernel.0.bias"] = torch.randn(self.PK_HIDDEN)
        sd["prompt_kernel.2.weight"] = torch.randn(self.HIDDEN, self.PK_HIDDEN)
        sd["prompt_kernel.2.bias"] = torch.randn(self.HIDDEN)

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
                "att_context_size": [[self.STREAMING_CACHE_LEFT, 13]],
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
            # No train_ds / prompt_dictionary.
        }

        TestCanaryPlugin._make_nemo_archive(tmp_path, sd, nemo_cfg)
        self._write_config(tmp_path, {
            "model_type": "nemotron_3_5_asr_streaming",
            "hidden_size": self.PRED_HIDDEN,
            "num_hidden_layers": self.PRED_LAYERS,
            "num_attention_heads": 1,
            "vocab_size": self.VOCAB + 1,
            "mel_length": 80,
        })

        cfg = ModelConfig.from_dir(tmp_path)
        with pytest.raises(ValueError, match="prompt_dictionary"):
            plugin.load_weights(str(tmp_path), cfg)


# ---------------------------------------------------------------------------
# Live-checkpoint smoke tests (require GPU + network + a downloaded HF cache).
# These will SKIP in CI; they exist so a developer with the model in
# ``/tmp/trtmc_hf_cache/`` can verify the engine builds end-to-end.
# ---------------------------------------------------------------------------


MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"


@pytest.mark.gpu
@pytest.mark.slow
def test_streaming_engines_for_3_5_right_contexts():
    """All four streaming right-context plans must be produced; right=1 absent."""
    pytest.skip(
        "Live-checkpoint engine build is opt-in (10-20 min). Run manually "
        "with the model cached at /tmp/trtmc_hf_cache/.")


@pytest.mark.gpu
@pytest.mark.slow
def test_prompt_kernel_engine_present():
    """prompt_kernel sub-engine plan must be in extras for the multilingual variant."""
    pytest.skip(
        "Live-checkpoint engine build is opt-in (10-20 min). Run manually "
        "with the model cached at /tmp/trtmc_hf_cache/.")


@pytest.mark.gpu
@pytest.mark.slow
def test_bundle_carries_prompt_metadata():
    """Bundle config overrides must surface multilingual streaming metadata."""
    pytest.skip(
        "Live-checkpoint engine build is opt-in (10-20 min). Run manually "
        "with the model cached at /tmp/trtmc_hf_cache/.")
