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
    return TextEncoder(preserve_text_attention(materialize_fp8(model.language_model)).eval())


def rotary_positions(query, key, cos, sin):
    """Round both BF16 products before the rotary addition, as in PyTorch."""
    cos, sin = cos.unsqueeze(1).float(), sin.unsqueeze(1).float()

    def rotate(values):
        half = values.shape[-1] // 2
        shifted = torch.cat((-values[..., half:], values[..., :half]), dim=-1)
        left = (values.float() * cos).to(values.dtype).float()
        right = (shifted.float() * sin).to(values.dtype).float()
        return (left + right).to(values.dtype)

    return rotate(query), rotate(key)


class TextAttention(nn.Module):
    """Preserve Qwen's BF16 score rounding and FP32 probability normalization."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

        module = self.module
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
        value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = rotary_positions(query, key, *position_embeddings)
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)).to(query.dtype)
        scores = (scores.float() * module.scaling).to(query.dtype)
        if attention_mask is not None:
            scores = (scores.float() + attention_mask.float()).to(query.dtype)
        # Keep eager Qwen's separate BF16 score and scale boundaries.
        # Express stable softmax in FP32 so fusion retains those boundaries.
        wide = scores.float()
        exponentials = torch.exp(wide - wide.amax(dim=-1, keepdim=True))
        probabilities = (exponentials / exponentials.sum(dim=-1, keepdim=True)).to(query.dtype)
        output = torch.matmul(probabilities.float(), value.float()).to(query.dtype)
        output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return module.o_proj(output), None


def preserve_text_attention(module):
    for layer in module.layers:
        layer.self_attn = TextAttention(layer.self_attn)
    return module


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


class Attention(nn.Module):
    """Keep score and probability arithmetic in FP32 like the fused reference."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, values, segment_ids, cos, sin):
        batch, length, _ = values.shape
        module = self.module
        q, k, v = (
            module.qkv(values)
            .reshape(batch, length, 3, module.num_heads, module.head_dim)
            .unbind(dim=2)
        )
        q = module.norm_q(q).transpose(1, 2)
        k = module.norm_k(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = rotary_positions(q, k, cos, sin)
        mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
        scores = (q.float() @ k.float().transpose(-1, -2)) * (module.head_dim**-0.5)
        probabilities = torch.softmax(torch.where(mask, scores, float("-inf")), dim=-1)
        output = probabilities @ v.float()
        output = output.to(values.dtype).transpose(1, 2).reshape(batch, length, module.hidden_size)
        return module.o(output)


def preserve_attention(module):
    from ideogram4.modeling_ideogram4 import Ideogram4Attention

    for name, child in list(module.named_children()):
        if isinstance(child, Ideogram4Attention):
            setattr(module, name, Attention(child))
        else:
            preserve_attention(child)
    return module


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
    return Denoiser(preserve_attention(preserve_normalization(materialize_fp8(model))).eval())


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


def export_silu(values):
    """Export PyTorch SiLU with one BF16 rounding after FP32 activation math."""
    from onnxscript import opset23 as op

    wide = op.Cast(values, to=1)
    return op.CastLike(wide * op.Sigmoid(wide), values)


def compile_engine(
    module, inputs, input_names, output_name, directory: Path, verbose=False, sequence_bounds=None
):
    import onnxscript.optimizer
    import tensorrt as trt

    path = directory / "component.onnx"
    dynamic_shapes = None
    if sequence_bounds is not None and sequence_bounds[0] < sequence_bounds[1]:
        sequence = torch.export.Dim("sequence", min=sequence_bounds[0], max=sequence_bounds[1])
        dynamic_shapes = tuple({1: sequence} if value.ndim > 1 else None for value in inputs)
    with torch.inference_mode():
        program = torch.onnx.export(
            module.eval(),
            inputs,
            input_names=input_names,
            output_names=[output_name],
            opset_version=23,
            dynamo=True,
            optimize=False,
            dynamic_shapes=dynamic_shapes,
            custom_translation_table={torch.ops.aten.silu.default: export_silu},
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
    if dynamic_shapes is not None:
        profile = builder.create_optimization_profile()
        lower, upper = sequence_bounds
        for index in range(network.num_inputs):
            value = network.get_input(index)
            shape = tuple(value.shape)
            if -1 not in shape:
                continue
            minimum = tuple(lower if axis == 1 else size for axis, size in enumerate(shape))
            maximum = tuple(upper if axis == 1 else size for axis, size in enumerate(shape))
            profile.set_shape(value.name, minimum, maximum, maximum)
        settings.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError("Ideogram4 TensorRT engine build failed")
    return bytes(plan)
