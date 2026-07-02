# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow Matching Euler Discrete Scheduler.

Implements the Euler method for flow matching
(continuous normalizing flows) as in the diffusers FlowMatchEulerDiscreteScheduler.

Pure numpy — no TRT or torch dependency.
"""

from __future__ import annotations

import numpy as np


class FlowMatchEulerScheduler:
    """Flow Matching Euler Discrete Scheduler.

    Flow matching interpolates between noise and data:
        z_t = (1 - t) * x + t * noise  (for t in [0, 1])

    The model predicts the velocity v = noise - x.
    The Euler step: z_{t-dt} = z_t - dt * v

    Includes an optional shift parameter for timestep adjustment.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self._timesteps = np.array([], dtype=np.float64)
        self._sigmas = np.array([], dtype=np.float64)

    @property
    def timesteps(self) -> np.ndarray:
        return self._timesteps

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute timestep schedule matching HF FlowMatchEulerDiscreteScheduler.

        The schedule is computed as:
        1. sigma_min = shift * (1/N) / (1 + (shift-1)/N)
        2. Linspace in t-space from 1000 to sigma_min*1000 (num_inference_steps points)
        3. Convert to sigma-space and apply shift
        4. Append sigma=0 as terminal value
        """
        N = float(self.num_train_timesteps)
        s = float(self.shift)

        # sigma_min is the shifted version of 1/N
        raw_sigma_min = 1.0 / N
        sigma_min = s * raw_sigma_min / (1.0 + (s - 1.0) * raw_sigma_min)

        t_max = N  # sigma_max * N
        t_min = sigma_min * N

        # Linspace in t-space: num_inference_steps points
        t_steps = np.linspace(t_max, t_min, num_inference_steps, dtype=np.float64)
        sigmas = t_steps / N

        # Apply shift
        if self.shift != 1.0:
            sigmas = s * sigmas / (1.0 + (s - 1.0) * sigmas)

        # Append terminal sigma=0
        sigmas = np.concatenate([sigmas, [0.0]])

        timesteps = sigmas[:-1] * self.num_train_timesteps

        self._sigmas = sigmas
        self._timesteps = timesteps.astype(np.float32)

    def step(
        self,
        model_output: np.ndarray,
        timestep: float,
        sample: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        """Euler step for flow matching.

        z_{t-dt} = z_t - (sigma_t - sigma_{t-1}) * model_output
        """
        sigma = self._sigmas[step_index]
        sigma_next = self._sigmas[step_index + 1]
        dt = sigma_next - sigma

        # Euler step: x = x + dt * velocity
        # In flow matching, model predicts velocity = dx/dt
        prev_sample = sample + dt * model_output.astype(np.float64)
        return prev_sample.astype(np.float32)

    def add_noise(
        self,
        original: np.ndarray,
        noise: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Add noise at the given sigma level.

        z_t = (1 - sigma) * original + sigma * noise
        """
        sigma = timestep / self.num_train_timesteps
        return ((1.0 - sigma) * original + sigma * noise).astype(np.float32)
