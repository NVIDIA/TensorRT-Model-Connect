# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the official FP8-weight/BF16-compute components into TensorRT plans.

FP8 weights are widened and multiplied by BF16-rounded row scales exactly as
the publisher's Fp8Linear does. This is weight-only quantization, not FP8
activation inference. TensorRT retains the reference graph's explicit casts.
"""

import json
from pathlib import Path

import torch
from torch import nn

TAPS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


def materialize_fp8(module):
    from ideogram4.quantized_loading import Fp8Linear

    for name, child in list(module.named_children()):
        if isinstance(child, Fp8Linear):
            dtype = child.compute_dtype
            layer = nn.Linear(
                child.in_features, child.out_features, bias=child.bias is not None, dtype=dtype
            )
            with torch.no_grad():
                layer.weight.copy_(
                    child.weight.to(dtype) * child.weight_scale.to(dtype).unsqueeze(1)
                )
                if child.bias is not None:
                    layer.bias.copy_(child.bias.to(dtype))
            setattr(module, name, layer)
        else:
            materialize_fp8(child)
    return module


def text_encoder(directory: Path):
    from transformers import AutoConfig, AutoModel
    from safetensors.torch import load_file
    from ideogram4.quantized_loading import swap_linears_to_fp8, load_fp8_state_dict

    config = AutoConfig.from_pretrained(directory / "text_encoder", local_files_only=True)
    model = AutoModel.from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
    weights = load_file(directory / "text_encoder/model.safetensors")
    swap_linears_to_fp8(model, weights, compute_dtype=torch.bfloat16)
    load_fp8_state_dict(
        model, weights, device=torch.device("cpu"), dtype=torch.bfloat16, assign=True
    )
    return TextEncoder(materialize_fp8(model.language_model).eval())


class TextEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids, token_mask, positions):
        x = self.model.embed_tokens(token_ids.to(torch.int64))
        length = token_ids.shape[1]
        row = torch.arange(length, device=token_ids.device)
        allowed = (row[None, :] <= row[:, None])[None, None, :, :] & token_mask[
            :, None, None, :
        ].bool()
        mask = torch.where(allowed, 0.0, torch.finfo(x.dtype).min).to(x.dtype)
        rotary = self.model.rotary_emb(x, positions.unsqueeze(0).expand(3, -1, -1))
        captured = []
        for index, layer in enumerate(self.model.layers):
            x = layer(
                x,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                position_embeddings=rotary,
            )
            if index in TAPS:
                captured.append(x)
        stacked = torch.stack(captured, dim=-1).flatten(-2)
        return (stacked * token_mask.to(stacked.dtype).unsqueeze(-1)).float()


class Denoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, latents, time, positions, segments, roles):
        return self.model(
            llm_features=features,
            x=latents,
            t=time,
            position_ids=positions,
            segment_ids=segments,
            indicator=roles,
        )


class RMSNorm(nn.Module):
    """Keep the affine multiply inside PyTorch RMSNorm's FP32 accumulation."""

    def __init__(self, module):
        super().__init__()
        self.weight = module.weight
        self.eps = module.eps

    def forward(self, values):
        wide = values.float()
        normalized = wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(values.dtype)


def preserve_normalization(module):
    from ideogram4.modeling_ideogram4 import Ideogram4RMSNorm

    for name, child in list(module.named_children()):
        if isinstance(child, Ideogram4RMSNorm):
            # Exporting F.rms_norm directly rounds the normalized BF16 value
            # before its affine multiply. The publisher's fused PyTorch op
            # rounds only after that multiply, so spell out its accumulation.
            setattr(module, name, RMSNorm(child))
        else:
            preserve_normalization(child)
    return module


def denoiser(directory: Path, component: str):
    from ideogram4.modeling_ideogram4 import Ideogram4Config
    from ideogram4.pipeline_ideogram4 import _build_transformer
    from safetensors.torch import load_file

    config = json.loads((directory / component / "config.json").read_text())
    params = Ideogram4Config(
        emb_dim=config["num_attention_heads"] * config["attention_head_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        adanln_dim=config["adaln_dim"],
        in_channels=config["in_channels"],
        llm_features_dim=config["llm_features_dim"],
        rope_theta=config["rope_theta"],
        mrope_section=tuple(config["mrope_section"]),
        norm_eps=config["norm_eps"],
    )
    model = _build_transformer(
        params,
        load_file(directory / component / "diffusion_pytorch_model.safetensors"),
        torch.device("cpu"),
        torch.bfloat16,
    )
    return Denoiser(preserve_normalization(materialize_fp8(model)).eval())


class Decoder(nn.Module):
    def __init__(self, module, grid_h, grid_w):
        super().__init__()
        from ideogram4.latent_norm import get_latent_norm

        shift, scale = get_latent_norm()
        self.register_buffer("shift", shift)
        self.register_buffer("scale", scale)
        self.module = module
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, latents):
        z = latents * self.scale + self.shift
        z = z.reshape(1, self.grid_h, self.grid_w, 2, 2, 32)
        z = z.permute(0, 5, 1, 3, 2, 4).reshape(1, 32, self.grid_h * 2, self.grid_w * 2)
        return self.module(z.to(torch.bfloat16)).float()


def decoder(directory: Path, grid_h, grid_w):
    from ideogram4.pipeline_ideogram4 import _load_autoencoder

    model = _load_autoencoder(
        str(directory / "vae/diffusion_pytorch_model.safetensors"),
        torch.device("cpu"),
        torch.bfloat16,
    )
    return Decoder(model.decoder, grid_h, grid_w).eval()


def compile_engine(module, inputs, input_names, output_name, directory: Path, verbose=False):
    import onnxscript.optimizer
    import tensorrt as trt

    path = directory / "component.onnx"
    with torch.inference_mode():
        program = torch.onnx.export(
            module.eval(),
            inputs,
            input_names=input_names,
            output_names=[output_name],
            opset_version=23,
            dynamo=True,
            optimize=False,
        )
    # Fold constant axes without the exporter's algebraic pattern rewrites:
    # its RMSNormalization fusion moves a BF16 cast before the affine multiply.
    # TensorRT still owns fusion and lowering of the faithfully typed graph.
    program.model = onnxscript.optimizer.optimize(program.model)
    program.save(str(path), external_data=True)
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(path)):
        raise RuntimeError(
            "Ideogram4 ONNX import failed: "
            + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        )
    # The VAE has a 512-wide attention head. Permit TensorRT to lower attention
    # to its primitive operations when that shape has no dedicated fused kernel.
    for layer in network:
        if layer.type == trt.LayerType.ATTENTION_INPUT:
            layer.__class__ = trt.IAttentionInputLayer
            layer.attention.decomposable = True
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    settings.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Ideogram4 TensorRT engine build failed")
    return bytes(plan)
