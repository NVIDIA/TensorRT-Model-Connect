"""Canary-owned synthetic NeMo archive helpers for model-family tests."""

from __future__ import annotations

from pathlib import Path


def make_nemo_state_dict(
    vocab: int,
    hidden: int,
    enc_layers: int,
    dec_layers: int,
    heads: int,
    head_dim: int,
    ffn: int,
    mel_bins: int,
    conv_kernel: int,
    sub_ch: int,
):
    """Create a synthetic NeMo state dict matching the Canary ASR layout."""
    import torch

    sd = {}
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

    for i in range(enc_layers):
        p = f"encoder.layers.{i}"
        for proj in ("linear_q", "linear_k", "linear_v", "linear_out"):
            sd[f"{p}.self_attn.{proj}.weight"] = torch.randn(hidden, hidden)
            sd[f"{p}.self_attn.{proj}.bias"] = torch.randn(hidden)
        sd[f"{p}.self_attn.linear_pos.weight"] = torch.randn(hidden, hidden)
        sd[f"{p}.self_attn.pos_bias_u"] = torch.randn(heads, head_dim)
        sd[f"{p}.self_attn.pos_bias_v"] = torch.randn(heads, head_dim)
        for norm in (
            "norm_self_att",
            "norm_feed_forward1",
            "norm_feed_forward2",
            "norm_conv",
            "norm_out",
        ):
            sd[f"{p}.{norm}.weight"] = torch.randn(hidden)
            sd[f"{p}.{norm}.bias"] = torch.randn(hidden)
        for fn in ("feed_forward1", "feed_forward2"):
            sd[f"{p}.{fn}.linear1.weight"] = torch.randn(ffn, hidden)
            sd[f"{p}.{fn}.linear1.bias"] = torch.randn(ffn)
            sd[f"{p}.{fn}.linear2.weight"] = torch.randn(hidden, ffn)
            sd[f"{p}.{fn}.linear2.bias"] = torch.randn(hidden)
        sd[f"{p}.conv.pointwise_conv1.weight"] = torch.randn(2 * hidden, hidden, 1)
        sd[f"{p}.conv.pointwise_conv1.bias"] = torch.randn(2 * hidden)
        sd[f"{p}.conv.depthwise_conv.weight"] = torch.randn(hidden, 1, conv_kernel)
        sd[f"{p}.conv.depthwise_conv.bias"] = torch.randn(hidden)
        sd[f"{p}.conv.batch_norm.weight"] = torch.randn(hidden)
        sd[f"{p}.conv.batch_norm.bias"] = torch.randn(hidden)
        sd[f"{p}.conv.batch_norm.running_mean"] = torch.randn(hidden)
        sd[f"{p}.conv.batch_norm.running_var"] = torch.abs(torch.randn(hidden))
        sd[f"{p}.conv.pointwise_conv2.weight"] = torch.randn(hidden, hidden, 1)
        sd[f"{p}.conv.pointwise_conv2.bias"] = torch.randn(hidden)

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


def make_nemo_archive(tmp_path: Path, state_dict: dict, nemo_cfg: dict) -> Path:
    """Create a synthetic .nemo tar archive."""
    import io
    import tarfile

    import torch
    import yaml

    nemo_path = tmp_path / "canary.nemo"
    with tarfile.open(str(nemo_path), "w") as tar:
        cfg_bytes = yaml.dump(nemo_cfg).encode("utf-8")
        cfg_info = tarfile.TarInfo(name="model_config.yaml")
        cfg_info.size = len(cfg_bytes)
        tar.addfile(cfg_info, io.BytesIO(cfg_bytes))

        buf = io.BytesIO()
        torch.save(state_dict, buf)
        buf.seek(0)
        ckpt_info = tarfile.TarInfo(name="model_weights.ckpt")
        ckpt_info.size = len(buf.getvalue())
        tar.addfile(ckpt_info, buf)

    return nemo_path
