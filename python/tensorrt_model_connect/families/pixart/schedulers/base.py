# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base scheduler protocol for diffusion models."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Scheduler(Protocol):
    """Protocol for diffusion schedulers.

    All implementations use pure numpy — no TRT or torch dependency.
    """

    @property
    def timesteps(self) -> np.ndarray:
        """Current timestep schedule after set_timesteps()."""
        ...

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute the timestep schedule."""
        ...

    def step(
        self,
        model_output: np.ndarray,
        timestep: float,
        sample: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        """Perform one denoising step.

        Args:
            model_output: Denoiser output (noise prediction or velocity).
            timestep: Current timestep.
            sample: Current noisy sample.
            step_index: Index into the timestep schedule.

        Returns:
            Updated (denoised) sample.
        """
        ...

    def add_noise(
        self,
        original: np.ndarray,
        noise: np.ndarray,
        timestep: float,
    ) -> np.ndarray:
        """Add noise to a clean sample at the given timestep."""
        ...
