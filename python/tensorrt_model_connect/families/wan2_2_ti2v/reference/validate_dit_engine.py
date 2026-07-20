#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare a fixed-shape TensorRT Wan2.2 DiT plan with upstream source."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import types
from pathlib import Path

from tensorrt_model_connect.trt_compat import trt
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--context-tokens", type=int, default=37)
    parser.add_argument("--timestep", type=float, default=500.0)
    parser.add_argument("--native-plugin", type=Path, action="append", default=[])
    parser.add_argument("--input-tensors", type=Path)
    parser.add_argument("--native-first", action="store_true")
    parser.add_argument("--benchmark-warmup", type=int, default=0)
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    parser.add_argument(
        "--save-native-debug-dir",
        type=Path,
        help="Optionally persist captured upstream debug tensors for standalone qualification.",
    )
    parser.add_argument(
        "--save-engine-debug-dir",
        type=Path,
        help="Optionally persist selected TensorRT debug outputs for standalone qualification.",
    )
    parser.add_argument(
        "--save-debug-names",
        default="",
        help="Comma-separated debug names to persist; empty saves every available debug tensor.",
    )
    args = parser.parse_args()
    saved_debug_names = {name.strip() for name in args.save_debug_names.split(",") if name.strip()}
    for native_plugin in args.native_plugin:
        ctypes.CDLL(str(native_plugin.resolve()), mode=ctypes.RTLD_GLOBAL)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    official_root = args.official_source.resolve()
    sys.path.insert(0, str(official_root))
    # Import the official model module without executing wan/__init__.py,
    # which eagerly imports unrelated task configs and optional dependencies.
    # The package stubs preserve normal relative imports inside model.py.
    wan_package = types.ModuleType("wan")
    wan_package.__path__ = [str(official_root / "wan")]
    modules_package = types.ModuleType("wan.modules")
    modules_package.__path__ = [str(official_root / "wan" / "modules")]
    sys.modules["wan"] = wan_package
    sys.modules["wan.modules"] = modules_package
    from wan.modules.model import (  # pylint: disable=import-outside-toplevel
        WanModel,
        rope_apply,
        sinusoidal_embedding_1d,
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    native = None
    if args.native_first:
        native = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
        native.blocks = torch.nn.ModuleList(list(native.blocks[: args.num_layers]))
        native.to(device)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError("Could not deserialize TensorRT DiT engine")
    context = engine.create_execution_context()
    latent_shape = tuple(engine.get_tensor_shape("latents"))
    output_shape = tuple(engine.get_tensor_shape("noise_prediction"))
    generator = torch.Generator(device=device).manual_seed(args.seed)
    captured = (
        torch.load(args.input_tensors, map_location="cpu", weights_only=True)
        if args.input_tensors is not None
        else None
    )
    if captured is None:
        latent = torch.randn(latent_shape, generator=generator, device=device)
        text_short = torch.randn(
            args.context_tokens,
            4096,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        timestep = torch.tensor([args.timestep], device=device, dtype=torch.float32)
    else:
        latent = captured["latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
        text_short = captured["context"].to(device=device)
        timestep = captured["timestep"].to(device=device, dtype=torch.float32)
        if tuple(latent.shape) != latent_shape:
            raise ValueError(f"Captured latent shape {tuple(latent.shape)} != {latent_shape}")
        if int(captured["seq_len"]) != 27280:
            raise ValueError(f"Captured seq_len is {captured['seq_len']}")
    text_padded = torch.zeros(1, 512, 4096, device=device, dtype=torch.float32)
    text_padded[0, : text_short.shape[0]] = text_short.float()
    scalar_timestep = timestep.reshape(-1)[:1]
    time_features = sinusoidal_embedding_1d(256, scalar_timestep).float()
    output = torch.empty(output_shape, device=device, dtype=torch.float32)

    debug_outputs = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if not name.startswith(("block_", "final_")) and name not in {
            "patch_hidden",
            "time_linear1",
            "time_silu",
            "time_embed",
            "time_proj",
            "text_hidden",
        }:
            continue
        debug_outputs[name] = torch.empty(
            tuple(engine.get_tensor_shape(name)), device=device, dtype=torch.float32
        )

    for name, tensor in (
        ("latents", latent),
        ("time_features", time_features),
        ("encoder_hidden_states", text_padded),
        ("noise_prediction", output),
    ):
        context.set_tensor_address(name, tensor.data_ptr())
    for name, tensor in debug_outputs.items():
        context.set_tensor_address(name, tensor.data_ptr())

    def execute_engine() -> None:
        stream = torch.cuda.current_stream(device).cuda_stream
        if not context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT DiT execution failed")
        torch.cuda.synchronize(device)

    def benchmark_engine() -> dict | None:
        if args.benchmark_iterations <= 0:
            return None
        for _ in range(args.benchmark_warmup):
            execute_engine()
        samples = []
        stream = torch.cuda.current_stream(device)
        for _ in range(args.benchmark_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT DiT benchmark execution failed")
            end.record(stream)
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return {
            "samples_ms": samples,
            "min_ms": min(samples),
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
        }

    if not args.native_first:
        execute_engine()
    engine_latency = benchmark_engine()

    if native is None:
        native = WanModel.from_pretrained(str(args.checkpoint)).eval().requires_grad_(False)
        native.blocks = torch.nn.ModuleList(list(native.blocks[: args.num_layers]))
        native.to(device)
    native_debug = {}
    hooks = []
    self_attention_context = {}
    self_attention_context_layers = set()

    def ensure_self_attention_context(layer_index: int) -> None:
        if layer_index in self_attention_context_layers:
            return
        module = native.blocks[layer_index].self_attn

        def capture_context(_module, inputs, *, key=layer_index):
            self_attention_context[key] = (inputs[2], inputs[3])

        hooks.append(module.register_forward_pre_hook(capture_context))
        self_attention_context_layers.add(layer_index)

    for name in debug_outputs:
        if name in {
            "patch_hidden",
            "time_linear1",
            "time_silu",
            "time_embed",
            "time_proj",
            "text_hidden",
        }:
            continue
        if name == "final_norm":
            hooks.append(
                native.head.norm.register_forward_hook(
                    lambda _module, _inputs, value, key=name: native_debug.__setitem__(
                        key, value.detach().float().cpu()
                    )
                )
            )
            continue
        if name == "final_input":
            hooks.append(
                native.head.head.register_forward_pre_hook(
                    lambda _module, inputs, key=name: native_debug.__setitem__(
                        key, inputs[0].detach().float().cpu()
                    )
                )
            )
            continue
        if name == "final_rows":
            hooks.append(
                native.head.head.register_forward_hook(
                    lambda _module, _inputs, value, key=name: native_debug.__setitem__(
                        key, value.detach().float().cpu()
                    )
                )
            )
            continue
        layer_index = int(name.split("_")[1])

        if name.endswith("_cross_k_weight"):
            native_debug[name] = (
                native.blocks[layer_index].cross_attn.norm_k.weight.detach().float().cpu()
            )
            continue

        if name.endswith("_input") and not name.endswith(
            ("_self_input", "_cross_input", "_ffn_input")
        ):
            module = native.blocks[layer_index]

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        if name.endswith("_modulation"):
            module = native.blocks[layer_index]

            def capture_modulation(module, inputs, kwargs, *, key=name):
                projected_time = kwargs["e"] if "e" in kwargs else inputs[1]
                combined = module.modulation.unsqueeze(0) + projected_time
                native_debug[key] = combined[:, 0].reshape(1, -1).detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_modulation, with_kwargs=True))
            continue
        if name.endswith("_self_input"):
            module = native.blocks[layer_index].self_attn.q

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        if name.endswith("_self_norm"):
            module = native.blocks[layer_index].norm1
        elif name.endswith("_self_q_linear"):
            module = native.blocks[layer_index].self_attn.q
        elif name.endswith("_self_k_linear"):
            module = native.blocks[layer_index].self_attn.k
        elif name.endswith("_self_v_linear"):
            module = native.blocks[layer_index].self_attn.v
        elif name.endswith("_self_q_norm"):
            module = native.blocks[layer_index].self_attn.norm_q
        elif name.endswith("_self_k_norm"):
            module = native.blocks[layer_index].self_attn.norm_k
        elif name.endswith(("_self_q_rotated", "_self_k_rotated")):
            self_attention = native.blocks[layer_index].self_attn
            module = (
                self_attention.norm_q if name.endswith("_self_q_rotated") else self_attention.norm_k
            )
            ensure_self_attention_context(layer_index)

            def capture_rotated(
                _module,
                _inputs,
                value,
                *,
                key=name,
                context_key=layer_index,
                heads=self_attention.num_heads,
                head_dim=self_attention.head_dim,
            ):
                grid_sizes, freqs = self_attention_context[context_key]
                batch, sequence = value.shape[:2]
                rotated = rope_apply(
                    value.view(batch, sequence, heads, head_dim), grid_sizes, freqs
                ).to(torch.bfloat16)
                native_debug[key] = rotated.flatten(2).detach().float().cpu()

            hooks.append(module.register_forward_hook(capture_rotated))
            continue
        elif name.endswith("_self_attention"):
            module = native.blocks[layer_index].self_attn.o

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        elif name.endswith("_self_projection"):
            module = native.blocks[layer_index].self_attn
        elif name.endswith("_post_self"):
            module = native.blocks[layer_index].norm3

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        elif name.endswith("_cross_norm"):
            module = native.blocks[layer_index].norm3
        elif name.endswith("_cross_input"):
            module = native.blocks[layer_index].cross_attn.q

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        elif name.endswith("_cross_q_linear"):
            module = native.blocks[layer_index].cross_attn.q
        elif name.endswith("_cross_k_linear"):
            module = native.blocks[layer_index].cross_attn.k
        elif name.endswith("_cross_v_linear"):
            module = native.blocks[layer_index].cross_attn.v
        elif name.endswith("_cross_q_norm"):
            module = native.blocks[layer_index].cross_attn.norm_q
        elif name.endswith("_cross_k_norm"):
            module = native.blocks[layer_index].cross_attn.norm_k
        elif name.endswith("_cross_attention"):
            module = native.blocks[layer_index].cross_attn.o

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        elif name.endswith("_cross_projection"):
            module = native.blocks[layer_index].cross_attn
        elif name.endswith("_post_cross"):
            module = native.blocks[layer_index].norm2

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        elif name.endswith("_ffn_norm"):
            module = native.blocks[layer_index].norm2
        elif name.endswith("_ffn_input"):
            module = native.blocks[layer_index].ffn[0]

            def capture_pre(_module, inputs, *, key=name):
                native_debug[key] = inputs[0].detach().float().cpu()

            hooks.append(module.register_forward_pre_hook(capture_pre))
            continue
        elif name.endswith("_ffn_linear1"):
            module = native.blocks[layer_index].ffn[0]
        elif name.endswith("_ffn_gelu"):
            module = native.blocks[layer_index].ffn[1]
        elif name.endswith("_ffn_projection"):
            module = native.blocks[layer_index].ffn
        elif name.endswith("_post_ffn"):
            module = native.blocks[layer_index]
        elif name.endswith("_hidden"):
            module = native.blocks[layer_index]
        elif name.endswith("_self_update"):
            module = native.blocks[layer_index].self_attn
        elif name.endswith("_cross_update"):
            module = native.blocks[layer_index].cross_attn
        elif name.endswith("_ffn_update"):
            module = native.blocks[layer_index].ffn
        else:
            continue

        def capture(_module, _inputs, value, *, key=name):
            native_debug[key] = value.detach().float().cpu()

        hooks.append(module.register_forward_hook(capture))
    num_patches = latent_shape[2] * (latent_shape[3] // 2) * (latent_shape[4] // 2)
    expanded_timestep = (
        timestep.reshape(1, num_patches)
        if timestep.numel() == num_patches
        else scalar_timestep.expand(1, num_patches)
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if "patch_hidden" in debug_outputs:
            native_debug["patch_hidden"] = (
                native.patch_embedding(latent).flatten(2).transpose(1, 2).detach().float().cpu()
            )
        if any(
            name in debug_outputs
            for name in ("time_linear1", "time_silu", "time_embed", "time_proj")
        ):
            with torch.amp.autocast("cuda", dtype=torch.float32):
                expanded_time = timestep.expand(1, num_patches)
                source_time_features = (
                    sinusoidal_embedding_1d(256, expanded_time.flatten())
                    .unflatten(0, (1, num_patches))
                    .float()
                )
                source_time_linear1 = native.time_embedding[0](source_time_features)
                source_time_silu = native.time_embedding[1](source_time_linear1)
                source_time = native.time_embedding[2](source_time_silu)
                source_time_proj = native.time_projection(source_time)
            if "time_linear1" in debug_outputs:
                native_debug["time_linear1"] = source_time_linear1.detach().float().cpu()
            if "time_silu" in debug_outputs:
                native_debug["time_silu"] = source_time_silu.detach().float().cpu()
            if "time_embed" in debug_outputs:
                native_debug["time_embed"] = source_time.detach().float().cpu()
            if "time_proj" in debug_outputs:
                native_debug["time_proj"] = source_time_proj.detach().float().cpu()
        if "text_hidden" in debug_outputs:
            padded_text = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)
            padded_text[0, : text_short.shape[0]] = text_short
            native_debug["text_hidden"] = native.text_embedding(padded_text).detach().float().cpu()
        reference = native(
            [latent[0]],
            expanded_timestep,
            [text_short],
            seq_len=num_patches,
        )[0].unsqueeze(0)
    torch.cuda.synchronize(device)
    if args.save_native_debug_dir is not None:
        args.save_native_debug_dir.mkdir(parents=True, exist_ok=True)
        saved_debug = {}
        for name, value in native_debug.items():
            if saved_debug_names and name not in saved_debug_names:
                continue
            path = args.save_native_debug_dir / f"{name}.pt"
            torch.save(value, path)
            saved_debug[name] = {
                "path": str(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        (args.save_native_debug_dir / "manifest.json").write_text(
            json.dumps(saved_debug, indent=2) + "\n"
        )
    if args.native_first:
        execute_engine()
    if args.save_engine_debug_dir is not None:
        args.save_engine_debug_dir.mkdir(parents=True, exist_ok=True)
        saved_debug = {}
        for name, value in debug_outputs.items():
            if saved_debug_names and name not in saved_debug_names:
                continue
            path = args.save_engine_debug_dir / f"{name}.pt"
            host_value = value.detach().float().cpu()
            torch.save(host_value, path)
            saved_debug[name] = {
                "path": str(path),
                "shape": list(host_value.shape),
                "dtype": str(host_value.dtype),
            }
        (args.save_engine_debug_dir / "manifest.json").write_text(
            json.dumps(saved_debug, indent=2) + "\n"
        )
    ref = reference.float().cpu()
    got = output.float().cpu()
    delta = got - ref
    exact_elements = int(
        torch.count_nonzero(
            got.contiguous().view(torch.int32) == ref.contiguous().view(torch.int32)
        )
    )
    report = {
        "kind": "wan2_2_ti2v_dit_tensorrt_parity",
        "device": torch.cuda.get_device_name(device),
        "num_layers": args.num_layers,
        "input_shape": list(latent_shape),
        "output_shape": list(output_shape),
        "num_patches": num_patches,
        "engine_latency": engine_latency,
        "metrics": {
            "bitwise_exact": exact_elements == ref.numel(),
            "exact_elements": exact_elements,
            "total_elements": ref.numel(),
            "max_abs_error": float(delta.abs().max()),
            "mean_abs_error": float(delta.abs().mean()),
            "rmse": float(delta.square().mean().sqrt()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    ref.flatten().double(), got.flatten().double(), dim=0
                )
            ),
        },
        "block_metrics": {},
    }
    for name, tensor in debug_outputs.items():
        ref_hidden = native_debug[name]
        if name.endswith("_modulation") and ref_hidden.numel() != tensor.numel():
            # The captured timestep is scalar, so every broadcast modulation row is
            # identical.  Compare one row rather than copying a multi-GiB debug tensor.
            got_hidden = tensor[:1].float().cpu()
        else:
            got_hidden = tensor.float().cpu()
        if ref_hidden.numel() != got_hidden.numel():
            ref_hidden = ref_hidden.reshape(-1, ref_hidden.shape[-1])[: got_hidden.shape[0]]
        ref_hidden = ref_hidden.reshape(got_hidden.shape).float()
        hidden_delta = got_hidden - ref_hidden
        exact_elements = int(
            torch.count_nonzero(
                got_hidden.contiguous().view(torch.int32)
                == ref_hidden.contiguous().view(torch.int32)
            )
        )
        report["block_metrics"][name] = {
            "bitwise_exact": exact_elements == ref_hidden.numel(),
            "exact_elements": exact_elements,
            "total_elements": ref_hidden.numel(),
            "max_abs_error": float(hidden_delta.abs().max()),
            "mean_abs_error": float(hidden_delta.abs().mean()),
            "rmse": float(hidden_delta.square().mean().sqrt()),
            "cosine_similarity": float(
                torch.nn.functional.cosine_similarity(
                    ref_hidden.flatten().double(), got_hidden.flatten().double(), dim=0
                )
            ),
        }
    if "final_rows" in debug_outputs:
        rows = (
            debug_outputs["final_rows"]
            .float()
            .cpu()
            .reshape(
                latent_shape[2],
                latent_shape[3] // 2,
                latent_shape[4] // 2,
                1,
                2,
                2,
                output_shape[1],
            )
        )
        source_layout = rows.permute(6, 0, 3, 1, 4, 2, 5).reshape(output_shape)
        native_rows = native_debug["final_rows"].reshape(rows.shape)
        native_source_layout = native_rows.permute(6, 0, 3, 1, 4, 2, 5).reshape(output_shape)
        flat_layout = rows.reshape(output_shape)
        report["unpatchify_diagnostics"] = {}
        for name, candidate in {
            "source_layout": source_layout,
            "bf16_source_layout": source_layout.to(torch.bfloat16).float(),
            "native_source_layout": native_source_layout,
            "flat_layout": flat_layout,
        }.items():
            layout_delta = got - candidate
            report["unpatchify_diagnostics"][name] = {
                "max_abs_error": float(layout_delta.abs().max()),
                "mean_abs_error": float(layout_delta.abs().mean()),
                "rmse": float(layout_delta.square().mean().sqrt()),
            }
        native_layout_delta = ref - native_source_layout
        report["unpatchify_diagnostics"]["native_layout_vs_reference"] = {
            "max_abs_error": float(native_layout_delta.abs().max()),
            "mean_abs_error": float(native_layout_delta.abs().mean()),
            "rmse": float(native_layout_delta.square().mean().sqrt()),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
