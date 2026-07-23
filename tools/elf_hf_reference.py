#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the official PyTorch ELF implementation with replayable sampling inputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shared-inputs-dir", required=True)
    parser.add_argument(
        "--generation-mode", choices=("conditional", "unconditional"), required=True
    )
    parser.add_argument("--sampling-method", choices=("ode", "sde"), required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--self-cond-cfg-scale", type=float, required=True)
    parser.add_argument("--sde-gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("fp16", "fp32", "bf16"), default="bf16")
    parser.add_argument("--initial-latents", type=Path)
    parser.add_argument("--sampling-steps", type=Path)
    parser.add_argument("--condition-latents", type=Path)
    parser.add_argument("--condition-mask", type=Path)
    parser.add_argument("--sde-noises", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sampling_steps(torch_mod, *, count: int, mean: float, std: float, generator):
    inner = torch_mod.randn((count - 1,), generator=generator, dtype=torch_mod.float32)
    inner = torch_mod.sigmoid(inner * std + mean).sort().values
    return torch_mod.cat((torch_mod.zeros(1), inner, torch_mod.ones(1)))


def _trim_terminal(token_ids: list[int], terminal_ids: set[int]) -> list[int]:
    end = len(token_ids)
    while end and token_ids[end - 1] in terminal_ids:
        end -= 1
    return token_ids[:end]


def _read_float32(path: Path, *, expected: int | None = None) -> torch.Tensor:
    values = np.fromfile(path, dtype=np.float32)
    if expected is not None and values.size != expected:
        raise ValueError(f"{path} contains {values.size} float32 values; expected {expected}")
    return torch.from_numpy(values.copy())


def _validate_replay_arguments(args: argparse.Namespace) -> bool:
    sampling_paths = (args.initial_latents, args.sampling_steps)
    if any(sampling_paths) and not all(sampling_paths):
        raise ValueError("--initial-latents and --sampling-steps must be provided together")
    condition_paths = (args.condition_latents, args.condition_mask)
    if any(condition_paths) and not all(condition_paths):
        raise ValueError("--condition-latents and --condition-mask must be provided together")
    if args.generation_mode == "conditional" and args.initial_latents and not all(condition_paths):
        raise ValueError("conditional replay requires condition latents and condition mask")
    return bool(args.initial_latents)


def main() -> int:
    args = _parse_args()
    replay_inputs = _validate_replay_arguments(args)
    reference_src = Path(args.reference_repo).resolve() / "src"
    sys.path.insert(0, str(reference_src))

    from configs.config import load_config_from_yaml
    from generation import _build_eval_model
    from modules.model import ELF_models
    from modules.t5_encoder import get_encoder
    from utils.checkpoint_utils import load_checkpoint
    from utils.data_utils import (
        get_dataloader,
        get_pad_token_id,
        load_jsonl_dataset,
    )
    from utils.encoder_utils import encode_text
    from utils.generation_utils import (
        _dlm_decode_batch,
        mask_after_eos,
        shift_left,
    )
    from utils.sampling_utils import _ode_step, _forward_sample, restore_cond
    from utils.train_utils import TrainState, get_optimizer

    if args.local_files_only:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = load_config_from_yaml(args.config)
    config.use_bf16 = args.precision == "bf16"
    config.use_compile = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # cuDNN SDPA can request an impractically large workspace for ELF's
        # 1,088-token XSum sequence on P2021. Prefer PyTorch's bounded-memory
        # SDPA implementations for a reference run.
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.encoder_model_name,
        local_files_only=args.local_files_only,
    )
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    terminal_ids = {int(pad_token_id), int(tokenizer.eos_token_id)}

    encoder = None
    if replay_inputs:
        replay_initial = _read_float32(args.initial_latents)
        if replay_initial.numel() % int(config.max_length) != 0:
            raise ValueError("initial replay latents do not divide evenly by config.max_length")
        text_encoder_dim = replay_initial.numel() // int(config.max_length)
    elif args.generation_mode == "conditional":
        encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
        encoder = encoder.to(device).eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        text_encoder_dim = encoder_config.d_model
    else:
        text_encoder_dim = 512

    model = ELF_models[config.model](
        text_encoder_dim=text_encoder_dim,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=tokenizer.vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    ).to(device)
    optimizer = get_optimizer(model, config, lr=1e-4)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        ema_params1=TrainState.init_ema(model),
        step=0,
        epoch=0,
        dropout_generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    state, _ = load_checkpoint(args.checkpoint, state)
    model = _build_eval_model(state, use_compile=False).to(device).eval()
    dtype = next(model.parameters()).dtype
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.precision]

    rows = _load_rows(Path(args.dataset))
    conditional_batches = None
    if args.generation_mode == "conditional" and not replay_inputs:
        dataset = load_jsonl_dataset(
            args.dataset, tokenizer, input_key="input", output_key="output"
        )
        conditional_batches = iter(
            get_dataloader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                max_seq_length=config.max_length,
                pad_token_id=pad_token_id,
                max_input_seq_length=config.max_input_length,
                distributed=False,
            )
        )

    shared_root = Path(args.shared_inputs_dir)
    shared_root.mkdir(parents=True, exist_ok=True)
    responses: list[dict] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", f"elf_{index:06d}"))
        sample_dir = shared_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        if replay_inputs:
            latent_count = int(config.max_length) * int(text_encoder_dim)
            z = replay_initial.reshape(1, config.max_length, text_encoder_dim).clone()
            t_steps = _read_float32(args.sampling_steps, expected=args.num_steps + 1)
            sde_noises = []
            if args.sde_noises:
                noise = _read_float32(args.sde_noises)
                sde_noises = list(
                    noise.reshape(-1, config.max_length, text_encoder_dim).unbind(0)
                )
            if args.generation_mode == "conditional":
                cond_seq = _read_float32(
                    args.condition_latents, expected=latent_count
                ).reshape(1, config.max_length, text_encoder_dim)
                cond_mask = _read_float32(
                    args.condition_mask, expected=int(config.max_length)
                ).reshape(1, config.max_length)
            else:
                cond_seq = torch.zeros_like(z)
                cond_mask = torch.zeros((1, config.max_length), dtype=torch.float32)
        else:
            generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
            t_steps = _sampling_steps(
                torch,
                count=args.num_steps,
                mean=float(config.denoiser_p_mean),
                std=float(config.denoiser_p_std),
                generator=generator,
            )
            z = torch.randn(
                (1, config.max_length, text_encoder_dim),
                generator=generator,
                dtype=torch.float32,
            ) * float(config.denoiser_noise_scale)
            sde_noises = []
            if args.sampling_method == "sde":
                for _ in range(args.num_steps - 1):
                    sde_noises.append(
                        torch.randn(z.shape, generator=generator, dtype=torch.float32)
                        * float(config.denoiser_noise_scale)
                    )
            z.numpy().tofile(sample_dir / "initial_latents.f32")
            t_steps.numpy().tofile(sample_dir / "sampling_steps.f32")
            if sde_noises:
                torch.stack(sde_noises).numpy().tofile(sample_dir / "sde_noises.f32")

        if args.generation_mode == "conditional" and not replay_inputs:
            batch = next(conditional_batches)
            input_ids = torch.from_numpy(np.asarray(batch["input_ids"])).to(device).long()
            encoder_mask = (
                torch.from_numpy(np.asarray(batch["encoder_attention_mask"])).to(device).float()
            )
            cond_mask = torch.from_numpy(np.asarray(batch["cond_seq_mask"])).to(device).float()
            cond_seq = encode_text(
                input_ids=input_ids,
                attention_mask=encoder_mask,
                encoder=encoder,
                latent_mean=config.latent_mean,
                latent_std=config.latent_std,
            ).to(dtype)
        elif not replay_inputs:
            cond_seq = torch.zeros_like(z, dtype=dtype, device=device)
            cond_mask = torch.zeros((1, config.max_length), dtype=dtype, device=device)

        z = z.to(device=device, dtype=dtype)
        t_steps = t_steps.to(device=device, dtype=dtype)
        cond_seq = cond_seq.to(device=device, dtype=dtype)
        cond_mask = cond_mask.to(device=device, dtype=dtype)
        z = restore_cond(z, cond_seq, cond_mask)
        x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_mask)
        step_kwargs = dict(
            model=model,
            config=config,
            cfg_scale=args.cfg_scale,
            self_cond_cfg_scale=args.self_cond_cfg_scale,
            cond_seq=cond_seq,
            cond_seq_mask=cond_mask,
        )
        started = time.perf_counter()
        with (
            torch.no_grad(),
            torch.amp.autocast(
                "cuda",
                dtype=autocast_dtype,
                enabled=device.type == "cuda" and args.precision != "fp32",
            ),
        ):
            for step_index in range(len(t_steps) - 2):
                t = float(t_steps[step_index].item())
                t_next = float(t_steps[step_index + 1].item())
                if args.sampling_method == "ode":
                    z, x_pred = _ode_step(
                        z=z, t=t, t_next=t_next, x_pred_prev=x_pred, **step_kwargs
                    )
                else:
                    h = t_next - t
                    alpha = max(0.0, min(1.0, 1.0 - args.sde_gamma * h))
                    t_back = alpha * t
                    eps = sde_noises[step_index].to(device=device, dtype=dtype)
                    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_mask)
                    t_batch = torch.full((1,), t_back, dtype=dtype, device=device)
                    v_pred, x_pred = _forward_sample(
                        z=z_back, t_batch=t_batch, x_pred_prev=x_pred, **step_kwargs
                    )
                    z = z_back + (t_next - t_back) * v_pred
            z, x_pred = _ode_step(
                z=z,
                t=float(t_steps[-2].item()),
                t_next=float(t_steps[-1].item()),
                x_pred_prev=x_pred,
                **step_kwargs,
            )
        if args.precision == "fp16" and device.type == "cuda":
            batch_size = z.shape[0]
            t_final = torch.full(
                (batch_size,), float(t_steps[-1].item()), dtype=z.dtype, device=device
            )
            scale = (
                torch.full(
                    (batch_size,), args.self_cond_cfg_scale, dtype=z.dtype, device=device
                )
                if config.num_self_cond_cfg_tokens > 0
                else None
            )
            z_input = torch.cat([z, torch.zeros_like(z)], dim=-1)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, decoder_logits = model(
                    z_input,
                    t_final,
                    deterministic=True,
                    self_cond_cfg_scale=scale,
                    decoder_step_active=True,
                )
            predicted = decoder_logits.argmax(dim=-1)
        else:
            predicted = _dlm_decode_batch(
                z, model, float(t_steps[-1].item()), config, args.self_cond_cfg_scale
            )
        if args.generation_mode == "conditional":
            cond_len = cond_mask.to(torch.int32).sum(dim=1)
            predicted = shift_left(predicted, cond_len, 0)[
                :, : config.max_length - config.max_input_length
            ]
        predicted = mask_after_eos(
            predicted, eos_token_id=tokenizer.eos_token_id, pad_token_id=pad_token_id
        )
        token_ids = _trim_terminal(
            [int(token_id) for token_id in predicted[0].detach().cpu().tolist()],
            terminal_ids,
        )
        responses.append(
            {
                "sample_id": sample_id,
                "output_text": tokenizer.decode(token_ids, skip_special_tokens=True).strip(),
                "generated_token_ids": token_ids,
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "source": "hf_elf_torch",
                "shared_inputs_dir": str(sample_dir),
                "shared_sampling_inputs": {
                    "initial_latents": str(
                        args.initial_latents or sample_dir / "initial_latents.f32"
                    ),
                    "sampling_steps": str(
                        args.sampling_steps or sample_dir / "sampling_steps.f32"
                    ),
                    **({"sde_noises": str(sample_dir / "sde_noises.f32")} if sde_noises else {}),
                },
            }
        )

    Path(args.output).write_text(
        json.dumps({"responses": responses}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
