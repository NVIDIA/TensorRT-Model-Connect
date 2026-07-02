# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned plugin weight tests.

Concrete load_weights behavior belongs beside the model family it validates.
Shared test code is limited to filesystem and serialization helpers.
"""

from __future__ import annotations


import numpy as np
import pytest

from tests.builder.family_plugin_test_support import (
    ModelConfig,
    ParallelConfig,
    WeightDict,
    _write_config,
)


class TestCanaryPlugin:
    """Canary encoder-decoder ASR plugin loads from synthetic .nemo archive."""

    VOCAB, HIDDEN, ENC_LAYERS, DEC_LAYERS = 64, 16, 2, 2
    HEADS, HEAD_DIM, FFN = 2, 8, 32
    MEL_BINS, CONV_KERNEL, SUB_CH = 8, 3, 4

    @staticmethod
    def _make_tp_weights(
        *,
        hidden: int = HIDDEN,
        dec_layers: int = DEC_LAYERS,
        dec_heads: int = HEADS,
        ffn: int = FFN,
    ) -> WeightDict:
        rng = np.random.RandomState(123)

        def rand(*shape: int) -> np.ndarray:
            return rng.randn(*shape).astype(np.float32)

        weights = WeightDict({
            "_dec_layers": dec_layers,
            "_dec_heads": dec_heads,
            "_dec_ffn": ffn,
            "_enc_seq": 16,
            "dec_emb": rand(TestCanaryPlugin.VOCAB, hidden),
            "dec_pos": rand(128, hidden),
            "emb_ln": rand(hidden),
            "emb_ln_b": rand(hidden),
            "final_norm": rand(hidden),
            "final_norm_b": rand(hidden),
            "w_out": rand(hidden, TestCanaryPlugin.VOCAB),
            "out_bias": rand(TestCanaryPlugin.VOCAB),
        })
        for i in range(dec_layers):
            pfx = f"layer.{i}"
            for key in ("w_q", "w_k", "w_v", "xw_q", "xw_k", "xw_v"):
                weights[f"{pfx}.{key}"] = rand(hidden, hidden)
            for key in ("q_bias", "k_bias", "v_bias", "xb_q", "xb_k", "xb_v"):
                weights[f"{pfx}.{key}"] = rand(hidden)
            for key in ("w_o", "xw_o"):
                weights[f"{pfx}.{key}"] = rand(hidden, hidden)
            weights[f"{pfx}.o_bias"] = rand(hidden)
            weights[f"{pfx}.xb_o"] = rand(hidden)
            weights[f"{pfx}.w_fc1"] = rand(hidden, ffn)
            weights[f"{pfx}.fc1_bias"] = rand(ffn)
            weights[f"{pfx}.w_fc2"] = rand(ffn, hidden)
            weights[f"{pfx}.fc2_bias"] = rand(hidden)
            for key in (
                "input_norm", "input_norm_b",
                "xattn_norm", "xattn_norm_b",
                "ffn_norm", "ffn_norm_b",
            ):
                weights[f"{pfx}.{key}"] = rand(hidden)
        return weights

    @staticmethod
    def _tp_builder_module():
        return pytest.importorskip(
            "tensorrt_model_connect.families.canary.decoder_tp_builder",
            reason="TensorRT is required for Canary TP builder tests",
        )

    @staticmethod
    def _make_nemo_state_dict(vocab, hidden, enc_layers, dec_layers,
                              heads, head_dim, ffn, mel_bins, conv_kernel,
                              sub_ch):
        """Create synthetic NeMo state dict matching canary-1b-v2."""
        import torch

        sd = {}
        # Subsampling
        sd["encoder.pre_encode.conv.0.weight"] = torch.randn(sub_ch, 1, 3, 3)
        sd["encoder.pre_encode.conv.0.bias"] = torch.randn(sub_ch)
        for dw, pw in [(2, 3), (5, 6)]:
            sd[f"encoder.pre_encode.conv.{dw}.weight"] = torch.randn(sub_ch, 1, 3, 3)
            sd[f"encoder.pre_encode.conv.{dw}.bias"] = torch.randn(sub_ch)
            sd[f"encoder.pre_encode.conv.{pw}.weight"] = torch.randn(sub_ch, sub_ch, 1, 1)
            sd[f"encoder.pre_encode.conv.{pw}.bias"] = torch.randn(sub_ch)
        feat_after = mel_bins
        for _ in range(3):
            feat_after = (feat_after + 2 - 3) // 2 + 1
        sd["encoder.pre_encode.out.weight"] = torch.randn(hidden, sub_ch * feat_after)
        sd["encoder.pre_encode.out.bias"] = torch.randn(hidden)

        # Encoder layers (with biases)
        for i in range(enc_layers):
            p = f"encoder.layers.{i}"
            for proj in ("linear_q", "linear_k", "linear_v", "linear_out"):
                sd[f"{p}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden)
                sd[f"{p}.self_attn.{proj}.bias"] = torch.randn(hidden)
            sd[f"{p}.self_attn.linear_pos.weight"] = torch.randn(hidden, hidden)
            sd[f"{p}.self_attn.pos_bias_u"] = torch.randn(heads, head_dim)
            sd[f"{p}.self_attn.pos_bias_v"] = torch.randn(heads, head_dim)
            for norm in ("norm_self_att", "norm_feed_forward1",
                         "norm_feed_forward2", "norm_conv", "norm_out"):
                sd[f"{p}.{norm}.weight"] = torch.randn(hidden)
                sd[f"{p}.{norm}.bias"] = torch.randn(hidden)
            for fn in ("feed_forward1", "feed_forward2"):
                sd[f"{p}.{fn}.linear1.weight"] = torch.randn(ffn, hidden)
                sd[f"{p}.{fn}.linear1.bias"] = torch.randn(ffn)
                sd[f"{p}.{fn}.linear2.weight"] = torch.randn(hidden, ffn)
                sd[f"{p}.{fn}.linear2.bias"] = torch.randn(hidden)
            sd[f"{p}.conv.pointwise_conv1.weight"] = torch.randn(2*hidden, hidden, 1)
            sd[f"{p}.conv.pointwise_conv1.bias"] = torch.randn(2*hidden)
            sd[f"{p}.conv.depthwise_conv.weight"] = torch.randn(hidden, 1, conv_kernel)
            sd[f"{p}.conv.depthwise_conv.bias"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.weight"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.bias"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.running_mean"] = torch.randn(hidden)
            sd[f"{p}.conv.batch_norm.running_var"] = torch.abs(torch.randn(hidden))
            sd[f"{p}.conv.pointwise_conv2.weight"] = torch.randn(hidden, hidden, 1)
            sd[f"{p}.conv.pointwise_conv2.bias"] = torch.randn(hidden)

        # Decoder (note underscore prefixes _embedding, _decoder)
        sd["transf_decoder._embedding.token_embedding.weight"] = torch.randn(vocab, hidden)
        sd["transf_decoder._embedding.position_embedding.pos_enc"] = torch.randn(128, hidden)
        sd["transf_decoder._embedding.layer_norm.weight"] = torch.randn(hidden)
        sd["transf_decoder._embedding.layer_norm.bias"] = torch.randn(hidden)
        sd["transf_decoder._decoder.final_layer_norm.weight"] = torch.randn(hidden)
        sd["transf_decoder._decoder.final_layer_norm.bias"] = torch.randn(hidden)
        for i in range(dec_layers):
            p = f"transf_decoder._decoder.layers.{i}"
            for sub in ("first_sub_layer", "second_sub_layer"):
                for pn in ("query_net", "key_net", "value_net", "out_projection"):
                    sd[f"{p}.{sub}.{pn}.weight"] = torch.randn(hidden, hidden)
                    sd[f"{p}.{sub}.{pn}.bias"] = torch.randn(hidden)
            sd[f"{p}.third_sub_layer.dense_in.weight"] = torch.randn(ffn, hidden)
            sd[f"{p}.third_sub_layer.dense_in.bias"] = torch.randn(ffn)
            sd[f"{p}.third_sub_layer.dense_out.weight"] = torch.randn(hidden, ffn)
            sd[f"{p}.third_sub_layer.dense_out.bias"] = torch.randn(hidden)
            for ln in ("layer_norm_1", "layer_norm_2", "layer_norm_3"):
                sd[f"{p}.{ln}.weight"] = torch.randn(hidden)
                sd[f"{p}.{ln}.bias"] = torch.randn(hidden)

        sd["log_softmax.mlp.layer0.weight"] = torch.randn(vocab, hidden)
        sd["log_softmax.mlp.layer0.bias"] = torch.randn(vocab)
        return sd

    @staticmethod
    def _make_nemo_archive(tmp_path, state_dict, nemo_cfg):
        """Create a synthetic .nemo tar archive."""
        import io
        import tarfile
        import torch
        import yaml

        nemo_path = tmp_path / "canary.nemo"
        with tarfile.open(str(nemo_path), "w") as tar:
            # Write model_config.yaml
            cfg_bytes = yaml.dump(nemo_cfg).encode("utf-8")
            cfg_info = tarfile.TarInfo(name="model_config.yaml")
            cfg_info.size = len(cfg_bytes)
            tar.addfile(cfg_info, io.BytesIO(cfg_bytes))

            # Write model_weights.ckpt
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            buf.seek(0)
            ckpt_info = tarfile.TarInfo(name="model_weights.ckpt")
            ckpt_info.size = len(buf.getvalue())
            tar.addfile(ckpt_info, buf)

        return nemo_path

    def test_load_weights_keys(self, tmp_path):
        """Canary load_weights extracts correct keys from .nemo archive."""
        pytest.importorskip("torch", reason="torch required for canary test")
        pytest.importorskip("yaml", reason="yaml required for canary test")

        from tensorrt_model_connect.families.canary import plugin

        sd = self._make_nemo_state_dict(
            self.VOCAB, self.HIDDEN, self.ENC_LAYERS, self.DEC_LAYERS,
            self.HEADS, self.HEAD_DIM, self.FFN, self.MEL_BINS,
            self.CONV_KERNEL, self.SUB_CH)

        nemo_cfg = {
            "target": "EncDecMultiTaskModel",
            "encoder": {
                "d_model": self.HIDDEN,
                "n_layers": self.ENC_LAYERS,
                "n_heads": self.HEADS,
                "ff_expansion_factor": self.FFN // self.HIDDEN,
                "conv_kernel_size": self.CONV_KERNEL,
                "feat_in": self.MEL_BINS,
                "subsampling_conv_channels": self.SUB_CH,
            },
            "transf_decoder": {
                "config_dict": {
                    "num_layers": self.DEC_LAYERS,
                    "num_attention_heads": self.HEADS,
                    "inner_size": self.FFN,
                },
            },
            "preprocessor": {"features": self.MEL_BINS},
        }

        self._make_nemo_archive(tmp_path, sd, nemo_cfg)

        config = {
            "model_type": "canary",
            "hidden_size": self.HIDDEN,
            "num_hidden_layers": self.DEC_LAYERS,
            "num_attention_heads": self.HEADS,
            "intermediate_size": self.FFN,
            "vocab_size": self.VOCAB,
            "rms_norm_eps": 1e-5,
        }
        _write_config(tmp_path, config)

        cfg = ModelConfig.from_dir(tmp_path)
        weights = plugin.load_weights(str(tmp_path), cfg)

        # Encoder subsampling
        assert "enc_sub_conv0_w" in weights
        assert "enc_sub_dw0_w" in weights
        assert "enc_sub_dw1_w" in weights

        # Encoder layers
        for i in range(self.ENC_LAYERS):
            pfx = f"el.{i}"
            assert f"{pfx}.w_q" in weights
            assert f"{pfx}.pos_bias_u" in weights
            assert f"{pfx}.rpe_proj" in weights
            assert f"{pfx}.cpw1_w" in weights
            assert f"{pfx}.bn_w" in weights
            assert f"{pfx}.ff1.w1" in weights
            assert f"{pfx}.ff2.w1" in weights
            assert f"{pfx}.norm_out" in weights

        # Decoder
        assert "dec_emb" in weights
        assert weights["dec_emb"].shape == (self.VOCAB, self.HIDDEN)
        assert "emb_ln" in weights
        for i in range(self.DEC_LAYERS):
            pfx = f"layer.{i}"
            assert f"{pfx}.w_q" in weights
            assert f"{pfx}.xw_q" in weights
            assert f"{pfx}.w_fc1" in weights
            assert f"{pfx}.input_norm" in weights

        # Head
        assert "w_out" in weights
        assert "out_bias" in weights
        assert "final_norm" in weights

        assert weights["_enc_layers"] == self.ENC_LAYERS
        assert weights["_dec_layers"] == self.DEC_LAYERS
        assert weights["_hidden"] == self.HIDDEN
        assert weights["_vocab"] == self.VOCAB

    def test_tp_build_rejects_single_device_mode(self):
        decoder_tp_builder = self._tp_builder_module()

        with pytest.raises(ValueError, match="requires tensor_parallel mode"):
            decoder_tp_builder.build_canary_tp_decoder_engine(
                object(),
                WeightDict(),
                max_cache_length=4,
                parallel_config=ParallelConfig(),
            )

    def test_tp_validation_ignores_single_device_mode(self):
        decoder_tp_builder = self._tp_builder_module()

        decoder_tp_builder._validate_canary_tp(
            WeightDict(),
            hidden=self.HIDDEN,
            num_heads=self.HEADS,
            ffn_dim=self.FFN,
            parallel=ParallelConfig(),
        )

    @pytest.mark.parametrize(
        ("parallel", "overrides", "message"),
        [
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=-1),
                {},
                "concrete rank",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"hidden": HIDDEN + 1},
                "hidden size divisible",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"num_heads": HEADS + 1},
                "decoder_attention_heads divisible",
            ),
            (
                ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
                {"ffn_dim": FFN + 1},
                "decoder_ffn_dim divisible",
            ),
        ],
    )
    def test_tp_validation_rejects_bad_config_dimensions(
        self,
        parallel,
        overrides,
        message,
    ):
        decoder_tp_builder = self._tp_builder_module()
        kwargs = {
            "hidden": self.HIDDEN,
            "num_heads": self.HEADS,
            "ffn_dim": self.FFN,
        }
        kwargs.update(overrides)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_canary_tp(
                self._make_tp_weights(),
                parallel=parallel,
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("key", "shape", "message"),
        [
            ("layer.0.w_q", (HIDDEN, HIDDEN - 1), "output dim"),
            ("layer.0.w_o", (HIDDEN - 1, HIDDEN), "input dim"),
            ("layer.0.w_fc1", (HIDDEN, FFN - 1), "w_fc1 output dim"),
        ],
    )
    def test_tp_validation_rejects_unshardable_weight_shapes(
        self,
        key,
        shape,
        message,
    ):
        decoder_tp_builder = self._tp_builder_module()
        weights = self._make_tp_weights()
        weights[key] = np.zeros(shape, dtype=np.float32)

        with pytest.raises(ValueError, match=message):
            decoder_tp_builder._validate_canary_tp(
                weights,
                hidden=self.HIDDEN,
                num_heads=self.HEADS,
                ffn_dim=self.FFN,
                parallel=ParallelConfig(
                    mode="tensor_parallel",
                    tp_size=2,
                    rank=0,
                ),
            )

    def test_tp_sharding_returns_original_for_single_device_mode(self):
        decoder_tp_builder = self._tp_builder_module()
        weights = self._make_tp_weights()
        assert decoder_tp_builder.shard_canary_decoder_weights(
            weights,
            parallel=ParallelConfig(),
        ) is weights

    def test_tp_shards_rank_local_decoder_weights(self):
        decoder_tp_builder = self._tp_builder_module()
        weights = self._make_tp_weights()
        shard = decoder_tp_builder.shard_canary_decoder_weights(
            weights,
            parallel=ParallelConfig(
                mode="tensor_parallel",
                tp_size=2,
                rank=1,
            ),
        )

        assert isinstance(shard, WeightDict)
        assert shard["_tensor_parallel_size"] == 2
        assert shard["_tensor_parallel_rank"] == 1
        assert shard["_dec_layers"] == self.DEC_LAYERS

        np.testing.assert_array_equal(
            shard["layer.0.w_q"],
            weights["layer.0.w_q"][:, self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.xw_v"],
            weights["layer.0.xw_v"][:, self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.q_bias"],
            weights["layer.0.q_bias"][self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.xb_k"],
            weights["layer.0.xb_k"][self.HIDDEN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_o"],
            weights["layer.0.w_o"][self.HIDDEN // 2:, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.xw_o"],
            weights["layer.0.xw_o"][self.HIDDEN // 2:, :],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc1"],
            weights["layer.0.w_fc1"][:, self.FFN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.fc1_bias"],
            weights["layer.0.fc1_bias"][self.FFN // 2:],
        )
        np.testing.assert_array_equal(
            shard["layer.0.w_fc2"],
            weights["layer.0.w_fc2"][self.FFN // 2:, :],
        )
        assert shard["final_norm"] is weights["final_norm"]
