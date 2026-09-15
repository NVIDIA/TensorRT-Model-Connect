# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline cosine-scheduled Euler integration with classifier-free guidance.

Inputs are already prepared acoustic conditions, NOT text or speech tokens.
The caller supplies noise explicitly to make reference comparisons reproducible.
"""

from __future__ import annotations

import math


def solve_euler(estimator, mu, mask, spks, cond, noise, *, steps=10, guidance=0.7):
    import torch

    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ValueError("steps must be an integer in [1, 1000]")
    if not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if mu.ndim != 3 or mu.shape[0] != 1 or mu.shape[-1] < 1:
        raise ValueError("Offline flow integration supports one utterance: mu=[1, channels, frames]")
    if cond.shape != mu.shape or noise.shape != mu.shape or mask.shape != (1, 1, mu.shape[-1]):
        raise ValueError("cond/noise must match mu; mask must have shape [1, 1, frames]")
    if spks.ndim != 2 or spks.shape[0] != 1:
        raise ValueError("spks must have shape [1, speaker_channels]")
    for value in (mu, mask, spks, cond, noise):
        if value.dtype != torch.float32 or value.device != mu.device or not torch.isfinite(value).all().item():
            raise ValueError("All integration inputs must be finite FP32 tensors on one device")
    if not ((mask == 0) | (mask == 1)).all().item() or not mask.any().item():
        raise ValueError("mask must be binary with at least one valid frame")
    with torch.inference_mode():
        times = 1 - torch.cos(torch.linspace(0, 1, steps + 1, device=mu.device, dtype=mu.dtype) * (torch.pi / 2))
        x = noise.clone()
        # Index 0 is conditioned. Index 1 shares x/time/mask, but has zero
        # mu/speaker/prompt. In particular, mask is NOT dropped for CFG.
        mu_in = torch.cat((mu, torch.zeros_like(mu))).contiguous()
        spks_in = torch.cat((spks, torch.zeros_like(spks))).contiguous()
        cond_in = torch.cat((cond, torch.zeros_like(cond))).contiguous()
        mask_in = mask.repeat(2, 1, 1).contiguous()
        # These buffers belong to this invocation, never to the engine or
        # caller. The synchronous estimator finishes before the next update.
        x_in = torch.empty_like(mu_in)
        t_in = torch.empty(2, device=mu.device, dtype=mu.dtype)
        for index in range(steps):
            x_in.copy_(x)
            t_in.copy_(times[index])
            velocity = estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False)
            if velocity.shape != (2, *mu.shape[1:]) or velocity.dtype != mu.dtype or velocity.device != mu.device:
                raise ValueError("Estimator output must be FP32 [2, channels, frames] on the input device")
            if not torch.isfinite(velocity).all().item():
                raise RuntimeError("Estimator produced nonfinite velocity")
            conditional, unconditional = velocity[:1], velocity[1:]
            guided = (1 + guidance) * conditional - guidance * unconditional
            x = x + (times[index + 1] - times[index]) * guided
        if not torch.isfinite(x).all().item():
            raise RuntimeError("Flow integration produced nonfinite mel features")
        return x


class OfflineFlow:
    """Compose native conditioning and DiT with explicit per-request noise.

    B=1, offline/finalized inputs only. Prompt mel frames must correspond to
    prompt tokens at exactly two frames/token. This does not accept text.
    """

    def __init__(self, conditioner, estimator):
        if conditioner.device != estimator.device:
            raise ValueError("Conditioner and estimator must use the same device")
        self.conditioner, self.estimator = conditioner, estimator
        self.device = estimator.device

    def prepare(self, tokens, prompt_tokens, prompt_features, speaker):
        import torch

        for name, value in (("tokens", tokens), ("prompt_tokens", prompt_tokens)):
            if value.ndim != 2 or value.shape[0] != 1 or value.dtype != torch.int32 or value.device != self.device:
                raise ValueError(f"{name} must be INT32 [1, N] on {self.device}")
        if tokens.shape[1] == 0:
            raise ValueError("At least one target speech token is required")
        prompt_frames = prompt_tokens.shape[1] * 2
        if (prompt_features.shape != (1, prompt_frames, 80) or prompt_features.dtype != torch.float32
                or prompt_features.device != self.device or not torch.isfinite(prompt_features).all().item()):
            raise ValueError("prompt_features must be finite FP32 [1, 2*prompt_tokens, 80] on the engine device")
        frames = (tokens.shape[1] + prompt_tokens.shape[1]) * 2
        self.estimator.profile.validate_frames(frames)
        prepared = self.conditioner(torch.cat((prompt_tokens, tokens), dim=1), speaker)
        cond = torch.zeros((1, 80, frames), dtype=torch.float32, device=self.device)
        cond[:, :, :prompt_frames] = prompt_features.transpose(1, 2)
        prepared.update(cond=cond, mask=torch.ones((1, 1, frames), dtype=torch.float32, device=self.device))
        return prepared

    def __call__(self, tokens, prompt_tokens, prompt_features, speaker, noise):
        conditions = self.prepare(tokens, prompt_tokens, prompt_features, speaker)
        mel = solve_euler(self.estimator, **conditions, noise=noise)
        return mel[:, :, prompt_tokens.shape[1] * 2:].contiguous()
