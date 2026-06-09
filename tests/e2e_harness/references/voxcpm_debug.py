"""Opt-in VoxCPM2 reference tensor dump hooks.

The normal VoxCPM reference path must stay identical to the model-card path.
This module only patches an already-loaded model when a caller explicitly sets
``TRTMC_VOXCPM2_HF_TENSOR_DUMP_DIR``. The emitted manifest mirrors the TRT
TSLM/RALM/LocDiT tensor dump format so a developer can compare full raw tensors
by step.
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

    def squeeze_batch(tensor: Any) -> Any:
        if getattr(tensor, "ndim", 0) > 0 and int(tensor.shape[0]) == 1:
            return tensor.squeeze(0)
        return tensor

    def model_float_dtype() -> Any:
        dtype_fn = getattr(tts_model, "_dtype", None)
        if callable(dtype_fn):
            return dtype_fn()
        return torch.bfloat16

    def position_ids(count: int, device: Any) -> Any:
        return torch.arange(count, device=device, dtype=torch.int32)

    orig_inference = getattr(tts_model, "_inference", None)
    orig_enc_to_lm_forward = getattr(getattr(tts_model, "enc_to_lm_proj", None), "forward", None)
    orig_base_forward = getattr(getattr(tts_model, "base_lm", None), "forward", None)
    orig_base_forward_step = getattr(getattr(tts_model, "base_lm", None), "forward_step", None)
    orig_fsq_forward = getattr(getattr(tts_model, "fsq_layer", None), "forward", None)
    orig_fusion_forward = getattr(getattr(tts_model, "fusion_concat_proj", None), "forward", None)
    orig_residual_forward = getattr(getattr(tts_model, "residual_lm", None), "forward", None)
    orig_residual_forward_step = getattr(
        getattr(tts_model, "residual_lm", None), "forward_step", None
    )
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

    def dump_tensor(
        phase: str,
        step: int,
        direction: str,
        name: str,
        tensor: Any,
        *,
        stage: str | None = None,
        engine_section: str = "hf_reference",
    ) -> None:
        tensor = tensor.detach().cpu().contiguous()
        filename = f"{phase}_{step:06d}_{direction}_{name}.raw"
        raw_path = dump_dir / filename
        raw_path.write_bytes(raw_bytes(tensor))

        record = {
            "stage": stage or phase,
            "engine_section": engine_section,
            "phase": phase,
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

    def dump_prefill_inputs(local_text_features: Any) -> None:
        text_tokens = state.get("text_tokens")
        text_mask = state.get("text_mask")
        audio_mask = state.get("audio_mask")
        if text_tokens is None or text_mask is None or audio_mask is None:
            return
        if state.get("prefill_inputs_dumped"):
            return
        state["prefill_inputs_dumped"] = True
        squeezed_tokens = squeeze_batch(text_tokens).to(dtype=torch.int32)
        squeezed_text_mask = squeeze_batch(text_mask).to(dtype=model_float_dtype())
        squeezed_audio_mask = squeeze_batch(audio_mask).to(dtype=model_float_dtype())
        squeezed_features = squeeze_batch(local_text_features).to(dtype=model_float_dtype())
        token_count = int(squeezed_tokens.shape[0])
        ids = position_ids(token_count, squeezed_tokens.device)
        for pos in range(token_count):
            dump_tensor(
                "tslm_prefill",
                pos,
                "input",
                "position_id",
                ids[pos : pos + 1],
                stage="tslm",
            )
            dump_tensor(
                "tslm_prefill",
                pos,
                "input",
                "text_tokens",
                squeezed_tokens[pos : pos + 1],
                stage="tslm",
            )
            dump_tensor(
                "tslm_prefill",
                pos,
                "input",
                "text_mask",
                squeezed_text_mask[pos : pos + 1],
                stage="tslm",
            )
            dump_tensor(
                "tslm_prefill",
                pos,
                "input",
                "audio_mask",
                squeezed_audio_mask[pos : pos + 1],
                stage="tslm",
            )
            dump_tensor(
                "tslm_prefill",
                pos,
                "input",
                "local_text_features",
                squeezed_features[pos : pos + 1],
                stage="tslm",
            )
        state["prefill_local_text_features"] = local_text_features.detach()

    def dump_tslm_prefill_outputs(raw_hidden: Any, semantic_lm_states: Any) -> None:
        if state.get("prefill_outputs_dumped"):
            return
        state["prefill_outputs_dumped"] = True
        semantic = squeeze_batch(semantic_lm_states).to(dtype=model_float_dtype())
        latest = semantic_lm_states[:, -1, :].to(dtype=model_float_dtype())
        for pos in range(int(semantic.shape[0])):
            row = semantic[pos : pos + 1]
            dump_tensor(
                "tslm_prefill",
                pos,
                "output",
                "semantic_lm_states",
                row,
                stage="tslm",
            )
            dump_tensor("tslm_prefill", pos, "output", "lm_hidden", row, stage="tslm")
        state["semantic_lm_states"] = semantic_lm_states.detach()
        state["lm_hidden"] = latest.detach()
        state["base_lm_raw_hidden"] = raw_hidden.detach()

    def dump_ralm_prefill_inputs() -> None:
        if state.get("ralm_prefill_inputs_dumped"):
            return
        semantic = state.get("semantic_lm_states")
        local_features = state.get("prefill_local_text_features")
        audio_mask = state.get("audio_mask")
        if semantic is None or local_features is None or audio_mask is None:
            return
        state["ralm_prefill_inputs_dumped"] = True
        squeezed_semantic = squeeze_batch(semantic).to(dtype=model_float_dtype())
        squeezed_audio_mask = squeeze_batch(audio_mask).to(dtype=model_float_dtype())
        squeezed_features = squeeze_batch(local_features).to(dtype=model_float_dtype())
        ids = position_ids(int(squeezed_semantic.shape[0]), squeezed_semantic.device)
        for pos in range(int(squeezed_semantic.shape[0])):
            dump_tensor(
                "ralm_prefill",
                pos,
                "input",
                "position_id",
                ids[pos : pos + 1],
                stage="ralm",
            )
            dump_tensor(
                "ralm_prefill",
                pos,
                "input",
                "semantic_lm_states",
                squeezed_semantic[pos : pos + 1],
                stage="ralm",
            )
            dump_tensor(
                "ralm_prefill",
                pos,
                "input",
                "audio_mask",
                squeezed_audio_mask[pos : pos + 1],
                stage="ralm",
            )
            dump_tensor(
                "ralm_prefill",
                pos,
                "input",
                "local_text_features",
                squeezed_features[pos : pos + 1],
                stage="ralm",
            )

    def dump_ralm_prefill_outputs(residual_hidden: Any) -> None:
        if state.get("ralm_prefill_outputs_dumped"):
            return
        state["ralm_prefill_outputs_dumped"] = True
        squeezed = squeeze_batch(residual_hidden).to(dtype=model_float_dtype())
        latest = residual_hidden[:, -1, :].to(dtype=model_float_dtype())
        for pos in range(int(squeezed.shape[0])):
            dump_tensor(
                "ralm_prefill",
                pos,
                "output",
                "residual_hidden",
                squeezed[pos : pos + 1],
                stage="ralm",
            )
        state["residual_hidden"] = latest.detach()

    def dump_refresh_tslm_input(local_text_features: Any, step: int, position_id: Any | None) -> None:
        if position_id is None:
            return
        one = torch.ones((1,), device=local_text_features.device, dtype=model_float_dtype())
        zero = torch.zeros((1,), device=local_text_features.device, dtype=model_float_dtype())
        text_tokens = torch.zeros((1,), device=local_text_features.device, dtype=torch.int32)
        dump_tensor(
            "tslm_refresh",
            step,
            "input",
            "position_id",
            position_id.detach().to(dtype=torch.int32),
            stage="tslm",
        )
        dump_tensor(
            "tslm_refresh", step, "input", "text_tokens", text_tokens, stage="tslm"
        )
        dump_tensor("tslm_refresh", step, "input", "text_mask", zero, stage="tslm")
        dump_tensor("tslm_refresh", step, "input", "audio_mask", one, stage="tslm")
        dump_tensor(
            "tslm_refresh",
            step,
            "input",
            "local_text_features",
            squeeze_batch(local_text_features).to(dtype=model_float_dtype()),
            stage="tslm",
        )

    def dump_refresh_tslm_output(semantic_lm_state: Any, step: int) -> None:
        dump_tensor(
            "tslm_refresh",
            step,
            "output",
            "semantic_lm_states",
            squeeze_batch(semantic_lm_state).to(dtype=model_float_dtype()),
            stage="tslm",
        )
        dump_tensor(
            "tslm_refresh",
            step,
            "output",
            "lm_hidden",
            semantic_lm_state.to(dtype=model_float_dtype()),
            stage="tslm",
        )
        state["lm_hidden"] = semantic_lm_state.detach()

    def dump_refresh_ralm_input(residual_input: Any, step: int, position_id: Any | None) -> None:
        local_features = state.get("refresh_local_text_features")
        semantic = state.get("lm_hidden")
        if position_id is None or local_features is None or semantic is None:
            return
        one = torch.ones((1,), device=residual_input.device, dtype=model_float_dtype())
        dump_tensor(
            "ralm_refresh",
            step,
            "input",
            "position_id",
            position_id.detach().to(dtype=torch.int32),
            stage="ralm",
        )
        dump_tensor(
            "ralm_refresh",
            step,
            "input",
            "semantic_lm_states",
            semantic.to(dtype=model_float_dtype()),
            stage="ralm",
        )
        dump_tensor("ralm_refresh", step, "input", "audio_mask", one, stage="ralm")
        dump_tensor(
            "ralm_refresh",
            step,
            "input",
            "local_text_features",
            squeeze_batch(local_features).to(dtype=model_float_dtype()),
            stage="ralm",
        )

    def dump_refresh_ralm_output(residual_hidden: Any, step: int) -> None:
        dump_tensor(
            "ralm_refresh",
            step,
            "output",
            "residual_hidden",
            residual_hidden.to(dtype=model_float_dtype()),
            stage="ralm",
        )
        state["residual_hidden"] = residual_hidden.detach()

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
            "locdit",
            step,
            "input",
            "inference_timesteps",
            torch.tensor([int(n_timesteps)], dtype=torch.int32),
            stage="locdit",
        )
        dump_tensor(
            "locdit",
            step,
            "input",
            "cfg_value",
            torch.tensor([float(cfg_value)], dtype=torch.float32),
            stage="locdit",
        )
        dump_tensor("locdit", step, "input", "lm_hidden", lm_hidden, stage="locdit")
        dump_tensor(
            "locdit",
            step,
            "input",
            "residual_hidden",
            residual_hidden,
            stage="locdit",
        )
        dump_tensor(
            "locdit",
            step,
            "input",
            "feat_cond",
            cond.transpose(1, 2).contiguous().reshape(-1, feat_dim),
            stage="locdit",
        )
        dump_tensor(
            "locdit",
            step,
            "input",
            "locdit_noise",
            noise.transpose(1, 2).contiguous().reshape(-1, feat_dim),
            stage="locdit",
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
            "locdit",
            step,
            "output",
            "audio_vae_latents",
            out.transpose(1, 2).contiguous().reshape(-1, feat_dim).to(dtype=torch.float32),
            stage="locdit",
        )
        state["step"] = step + 1
        return out

    if callable(orig_inference):

        def inference_wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) >= 4:
                text_tokens, text_mask, _feat, audio_mask = args[:4]
            else:
                text_tokens = kwargs.get("text")
                text_mask = kwargs.get("text_mask")
                audio_mask = kwargs.get("feat_mask")
            state["text_tokens"] = text_tokens.detach() if text_tokens is not None else None
            state["text_mask"] = text_mask.detach() if text_mask is not None else None
            state["audio_mask"] = audio_mask.detach() if audio_mask is not None else None
            state["prefill_inputs_dumped"] = False
            state["prefill_outputs_dumped"] = False
            state["ralm_prefill_inputs_dumped"] = False
            state["ralm_prefill_outputs_dumped"] = False
            yield from orig_inference(*args, **kwargs)

        tts_model._inference = inference_wrapper

    if callable(orig_enc_to_lm_forward):

        def enc_to_lm_forward(*args: Any, **kwargs: Any) -> Any:
            out = orig_enc_to_lm_forward(*args, **kwargs)
            if getattr(out, "ndim", 0) == 3 and state.get("text_tokens") is not None:
                token_count = int(state["text_tokens"].shape[1])
                if int(out.shape[1]) == token_count and not state.get("prefill_inputs_dumped"):
                    dump_prefill_inputs(out)
                elif int(out.shape[1]) == 1:
                    state["refresh_local_text_features"] = out.detach()
            return out

        tts_model.enc_to_lm_proj.forward = enc_to_lm_forward

    if callable(orig_base_forward):

        def base_forward(*args: Any, **kwargs: Any) -> Any:
            out = orig_base_forward(*args, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
            state["base_lm_raw_hidden"] = hidden.detach()
            return out

        tts_model.base_lm.forward = base_forward

    if callable(orig_base_forward_step):

        def base_forward_step(*args: Any, **kwargs: Any) -> Any:
            position_id = args[1] if len(args) > 1 else kwargs.get("position_id")
            state["refresh_position_id"] = position_id.detach() if position_id is not None else None
            out = orig_base_forward_step(*args, **kwargs)
            state["base_lm_raw_hidden"] = out.detach()
            return out

        tts_model.base_lm.forward_step = base_forward_step

    if callable(orig_fsq_forward):

        def fsq_forward(*args: Any, **kwargs: Any) -> Any:
            hidden = args[0] if args else kwargs.get("hidden")
            out = orig_fsq_forward(*args, **kwargs)
            if hidden is None:
                return out
            if getattr(hidden, "ndim", 0) == 3:
                text_mask = state.get("text_mask")
                audio_mask = state.get("audio_mask")
                if text_mask is not None and audio_mask is not None:
                    semantic = out * audio_mask.unsqueeze(-1).to(dtype=out.dtype)
                    semantic = semantic + hidden * text_mask.unsqueeze(-1).to(dtype=hidden.dtype)
                    dump_tslm_prefill_outputs(hidden, semantic)
            elif getattr(hidden, "ndim", 0) == 2:
                step = max(int(state.get("step", 0)) - 1, 0)
                position_id = state.get("refresh_position_id")
                local_features = state.get("refresh_local_text_features")
                if local_features is not None:
                    dump_refresh_tslm_input(local_features, step, position_id)
                dump_refresh_tslm_output(out, step)
            return out

        tts_model.fsq_layer.forward = fsq_forward

    if callable(orig_fusion_forward):

        def fusion_forward(*args: Any, **kwargs: Any) -> Any:
            out = orig_fusion_forward(*args, **kwargs)
            if getattr(out, "ndim", 0) == 3:
                dump_ralm_prefill_inputs()
            elif getattr(out, "ndim", 0) == 2:
                dump_refresh_ralm_input(
                    out,
                    max(int(state.get("step", 0)) - 1, 0),
                    state.get("refresh_position_id"),
                )
            return out

        tts_model.fusion_concat_proj.forward = fusion_forward

    if callable(orig_residual_forward):

        def residual_forward(*args: Any, **kwargs: Any) -> Any:
            out = orig_residual_forward(*args, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
            dump_ralm_prefill_outputs(hidden)
            return out

        tts_model.residual_lm.forward = residual_forward

    if callable(orig_residual_forward_step):

        def residual_forward_step(*args: Any, **kwargs: Any) -> Any:
            out = orig_residual_forward_step(*args, **kwargs)
            dump_refresh_ralm_output(out, max(int(state.get("step", 0)) - 1, 0))
            return out

        tts_model.residual_lm.forward_step = residual_forward_step

    lm_to_dit_proj.forward = lm_forward
    res_to_dit_proj.forward = res_forward
    feat_decoder.forward = decoder_forward
    return True
