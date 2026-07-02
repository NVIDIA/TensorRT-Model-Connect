# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local calibration adapters for scale acquisition."""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING, Any, Iterable, Protocol

if TYPE_CHECKING:
    from ..config import ModelConfig


class CalibrationAdapter(Protocol):
    """Bridge between a family's reference model and scale acquisition."""

    def load_calibration_model(
        self,
        model_dir: str,
        config: "ModelConfig",
    ) -> tuple[Any, Any | None]:
        """Return ``(model, aux)`` where aux is tokenizer or family-specific state."""
        ...

    def iter_calibration_batches(
        self,
        model: Any,
        aux: Any | None,
        *,
        model_dir: str,
        config: "ModelConfig",
        num_samples: int,
        calibration_prompts: list[str] | None,
    ) -> Iterable[Any]:
        """Yield representative calibration batches."""
        ...

    def run_calibration_batch(self, model: Any, batch: Any) -> None:
        """Execute one batch through the model for calibration."""
        ...

    def map_layer_name(self, layer_name: str) -> str | None:
        """Map reference-model module names to builder quant seam IDs."""
        ...


class AutoCausalLMCalibrationAdapter:
    """Default adapter for standard decoder families backed by HF causal LM."""

    def load_calibration_model(
        self,
        model_dir: str,
        config: "ModelConfig",
    ) -> tuple[Any, Any | None]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=False,
        )
        model = model.eval().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=False,
        )
        return model, tokenizer

    def iter_calibration_batches(
        self,
        model: Any,
        aux: Any | None,
        *,
        model_dir: str,
        config: "ModelConfig",
        num_samples: int,
        calibration_prompts: list[str] | None,
    ) -> Iterable[Any]:
        tokenizer = aux
        prompts = calibration_prompts or self._default_prompts()
        if not prompts:
            return
        for prompt in itertools.islice(itertools.cycle(prompts), num_samples):
            yield tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).input_ids.to(model.device)

    def run_calibration_batch(self, model: Any, batch: Any) -> None:
        model(batch)

    def map_layer_name(self, layer_name: str) -> str | None:
        return layer_name

    @staticmethod
    def _default_prompts() -> list[str]:
        return [
            "The quick brown fox jumps over the lazy dog.",
            "In a recent study, researchers found that",
            "The capital of France is Paris, which is known for",
            "Machine learning models can be trained using",
            "Once upon a time in a land far away,",
        ] * 100


class StandardDecoderCalibrationAdapter(AutoCausalLMCalibrationAdapter):
    """Map HF decoder module names to standard-decoder builder seam IDs."""

    _DEFAULT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj$"), "w_q"),
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj$"), "w_k"),
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.v_proj$"), "w_v"),
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj$"), "w_o"),
        (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate_proj$"), "w_gate"),
        (re.compile(r"^model\.layers\.(\d+)\.mlp\.up_proj$"), "w_up"),
        (re.compile(r"^model\.layers\.(\d+)\.mlp\.down_proj$"), "w_down"),
    )

    def __init__(
        self,
        *,
        family: str | None = None,
        rules: tuple[tuple[re.Pattern[str], str], ...] | None = None,
    ) -> None:
        self.family = family
        self.rules = rules or self._DEFAULT_RULES

    def map_layer_name(self, layer_name: str) -> str | None:
        for pattern, suffix in self.rules:
            match = pattern.match(layer_name)
            if match is None:
                continue
            layer_idx = match.group(1)
            seam = f"layer.{layer_idx}.{suffix}"
            if self.family:
                return f"{self.family}/{seam}"
            return seam
        return None


def resolve_calibration_adapter(
    plugin: Any | None,
    format_name: str,
) -> CalibrationAdapter:
    """Return a family-supplied adapter or the default decoder adapter."""
    if plugin is not None:
        quant_adapter = getattr(plugin, "quant_adapter", None)
        if callable(quant_adapter):
            adapter = quant_adapter(format_name)
            if adapter is not None:
                return adapter
    return AutoCausalLMCalibrationAdapter()
