"""Opt-in VoxCPM2 reference tensor dump hooks.

The normal VoxCPM reference path must stay identical to the model-card path.
This module only patches an already-loaded model when a caller explicitly sets
``TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR``. The emitted manifest mirrors the TRT
LocDiT tensor dump format so a developer can compare full raw tensors by step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def install_voxcpm2_tensor_dump(model: Any) -> bool:
    """Install an opt-in LocDiT tensor dump hook on an upstream VoxCPM2 model.

    Returns ``True`` when the hook was installed and ``False`` when no dump
    directory was requested. The hook also accepts
    ``TRTMC_VOXCPM2_HF_NOISE_RAW`` as a float32 ``[steps, patch, feat]`` raw
    noise source so HF can consume the same LocDiT noise patches as TRT.
    """

    dump_root = os.environ.get("TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR", "")
    if not dump_root:
        return False

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by preflight.
        raise RuntimeError(
            "VoxCPM2 HF tensor dump requires numpy and torch"
        ) from exc

    tts_model = getattr(model, "tts_model", model)
    feat_decoder = getattr(tts_model, "feat_decoder", None)
    lm_to_dit_proj = getattr(tts_model, "lm_to_dit_proj", None)
    res_to_dit_proj = getattr(tts_model, "res_to_dit_proj", None)
    if feat_decoder is None or lm_to_dit_proj is None or res_to_dit_proj is None:
        raise RuntimeError(
            "VoxCPM2 HF tensor dump expected tts_model.feat_decoder, "
            "lm_to_dit_proj, and res_to_dit_proj"
        )

    dump_dir = Path(dump_root)
    dump_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dump_dir / "manifest.jsonl"
    state: dict[str, Any] = {"step": 0}

    patch_size = int(getattr(tts_model, "patch_size", 0))
    feat_dim = int(getattr(tts_model, "feat_dim", getattr(feat_decoder, "in_channels", 0)))
    if patch_size <= 0 or feat_dim <= 0:
        raise RuntimeError("VoxCPM2 HF tensor dump could not resolve patch_size/feat_dim")

    noise_steps = None
    noise_raw_path = os.environ.get("TRTMC_VOXCPM2_HF_NOISE_RAW", "")
    if noise_raw_path:
        raw = np.fromfile(noise_raw_path, dtype=np.float32)
        patch_values = patch_size * feat_dim
        if raw.size % patch_values != 0:
            raise RuntimeError(
                "TRTMC_VOXCPM2_HF_NOISE_RAW element count is not divisible by "
                f"one LocDiT patch ({patch_values} values)"
            )
        noise_steps = raw.reshape((-1, patch_size, feat_dim))

    orig_lm_forward = lm_to_dit_proj.forward
    orig_res_forward = res_to_dit_proj.forward

    def lm_forward(*args: Any, **kwargs: Any) -> Any:
        if args:
            state["lm_hidden"] = args[0].detach()
        return orig_lm_forward(*args, **kwargs)

    def res_forward(*args: Any, **kwargs: Any) -> Any:
        if args:
            state["residual_hidden"] = args[0].detach()
        return orig_res_forward(*args, **kwargs)

    def dtype_name(tensor: Any) -> str:
        if tensor.dtype == torch.float32:
            return "float32"
        if tensor.dtype == torch.float16:
            return "float16"
        if tensor.dtype == torch.bfloat16:
            return "bfloat16"
        if tensor.dtype == torch.int32:
            return "int32"
        if tensor.dtype == torch.int64:
            return "int64"
        return str(tensor.dtype).removeprefix("torch.")

    def raw_bytes(tensor: Any) -> bytes:
        contiguous = tensor.detach().cpu().contiguous()
        if contiguous.dtype == torch.bfloat16:
            return contiguous.view(torch.uint8).numpy().tobytes()
        return contiguous.numpy().tobytes()

    def tensor_first_and_mean64(tensor: Any) -> dict[str, float | int]:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return {}
        if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            sample = flat[:64].to(dtype=torch.float32)
            return {
                "first": float(sample[0].item()),
                "mean64": float(sample.abs().mean().item()),
            }
        if tensor.dtype in (torch.int32, torch.int64):
            return {"first_int": int(flat[0].item())}
        return {}

    def dump_tensor(step: int, direction: str, name: str, tensor: Any) -> None:
        tensor = tensor.detach().cpu().contiguous()
        filename = f"locdit_{step:06d}_{direction}_{name}.raw"
        raw_path = dump_dir / filename
        raw_path.write_bytes(raw_bytes(tensor))

        record = {
            "stage": "locdit",
            "engine_section": "hf_reference",
            "phase": "locdit",
            "step": step,
            "direction": direction,
            "name": name,
            "dtype": dtype_name(tensor),
            "shape": [int(dim) for dim in tensor.shape],
            "nbytes": raw_path.stat().st_size,
            "path": str(raw_path),
        }
        record.update(tensor_first_and_mean64(tensor))
        with manifest_path.open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(record, sort_keys=False) + "\n")

    def next_noise(step: int, mu: Any, batch: int, temperature: float) -> Any:
        if noise_steps is not None:
            if step >= len(noise_steps):
                raise RuntimeError(
                    "TRTMC_VOXCPM2_HF_NOISE_RAW has "
                    f"{len(noise_steps)} patch(es), but generation needs step {step}"
                )
            patch = torch.from_numpy(noise_steps[step]).to(device=mu.device, dtype=mu.dtype)
            noise = patch.transpose(0, 1).contiguous().unsqueeze(0)
            if batch != 1:
                noise = noise.repeat(batch, 1, 1)
            return noise
        return torch.randn(
            (batch, feat_decoder.in_channels, patch_size),
            device=mu.device,
            dtype=mu.dtype,
        ) * temperature

    def decoder_forward(
        *,
        mu: Any,
        n_timesteps: int,
        patch_size: int,
        cond: Any,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> Any:
        step = int(state["step"])
        batch, _ = mu.shape
        noise = next_noise(step, mu, batch, float(temperature))
        noise = noise.to(dtype=mu.dtype)

        lm_hidden = state.get("lm_hidden")
        residual_hidden = state.get("residual_hidden")
        if lm_hidden is None or residual_hidden is None:
            raise RuntimeError(
                "VoxCPM2 HF tensor dump did not observe lm/residual projection inputs"
            )

        dump_tensor(
            step,
            "input",
            "inference_timesteps",
            torch.tensor([int(n_timesteps)], dtype=torch.int32),
        )
        dump_tensor(
            step,
            "input",
            "cfg_value",
            torch.tensor([float(cfg_value)], dtype=torch.float32),
        )
        dump_tensor(step, "input", "lm_hidden", lm_hidden)
        dump_tensor(step, "input", "residual_hidden", residual_hidden)
        dump_tensor(
            step,
            "input",
            "feat_cond",
            cond.transpose(1, 2).contiguous().reshape(-1, feat_dim),
        )
        dump_tensor(
            step,
            "input",
            "locdit_noise",
            noise.transpose(1, 2).contiguous().reshape(-1, feat_dim),
        )

        t_span = torch.linspace(
            1,
            0,
            int(n_timesteps) + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        t_span = t_span + float(sway_sampling_coef) * (
            torch.cos(torch.pi / 2 * t_span) - 1 + t_span
        )
        out = feat_decoder.solve_euler(
            x=noise,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )
        dump_tensor(
            step,
            "output",
            "audio_vae_latents",
            out.transpose(1, 2).contiguous().reshape(-1, feat_dim).to(dtype=torch.float32),
        )
        state["step"] = step + 1
        return out

    lm_to_dit_proj.forward = lm_forward
    res_to_dit_proj.forward = res_forward
    feat_decoder.forward = decoder_forward
    return True
