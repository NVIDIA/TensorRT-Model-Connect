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


class QwenVLCalibrationAdapter:
    """Calibration adapter for Qwen-VL (Qwen2.5-VL / Qwen3-VL).

    The default ``AutoCausalLMCalibrationAdapter`` loads the checkpoint as
    ``AutoModelForCausalLM``, which raises for vision-language configs
    (``Qwen3VLConfig``). This adapter loads the full VL model and feeds
    image+text calibration batches so the decoder's activation statistics
    reflect real multimodal inputs, then maps the VL decoder module names
    (``...language_model.layers.N.<proj>``) to the builder seam IDs
    (``layer.N.w_q`` etc.). Calibration runs in PyTorch, so it is unaffected
    by the TRT reduced-precision decoder issue.
    """

    _RULES: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"language_model\.layers\.(\d+)\.self_attn\.q_proj$"), "w_q"),
        (re.compile(r"language_model\.layers\.(\d+)\.self_attn\.k_proj$"), "w_k"),
        (re.compile(r"language_model\.layers\.(\d+)\.self_attn\.v_proj$"), "w_v"),
        (re.compile(r"language_model\.layers\.(\d+)\.self_attn\.o_proj$"), "w_o"),
        (re.compile(r"language_model\.layers\.(\d+)\.mlp\.gate_proj$"), "w_gate"),
        (re.compile(r"language_model\.layers\.(\d+)\.mlp\.up_proj$"), "w_up"),
        (re.compile(r"language_model\.layers\.(\d+)\.mlp\.down_proj$"), "w_down"),
        # Fallback for checkpoints without the language_model. prefix.
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.q_proj$"), "w_q"),
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.k_proj$"), "w_k"),
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.v_proj$"), "w_v"),
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.self_attn\.o_proj$"), "w_o"),
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.gate_proj$"), "w_gate"),
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.up_proj$"), "w_up"),
        (re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.down_proj$"), "w_down"),
    )

    def load_calibration_model(
        self, model_dir: str, config: "ModelConfig",
    ) -> tuple[Any, Any | None]:
        import torch
        import transformers
        from transformers import AutoProcessor

        model = None
        last_exc: Exception | None = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(
                    model_dir, torch_dtype=torch.float16,
                    device_map=None, trust_remote_code=False)
                break
            except Exception as exc:  # try the next class
                last_exc = exc
                model = None
        if model is None:
            from transformers import AutoModel
            try:
                model = AutoModel.from_pretrained(
                    model_dir, torch_dtype=torch.float16,
                    device_map=None, trust_remote_code=False)
            except Exception as exc:
                raise RuntimeError(
                    f"QwenVLCalibrationAdapter could not load the VL model: {exc}"
                ) from (last_exc or exc)
        model = model.eval().to("cuda")
        processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=False)
        return model, processor

    @staticmethod
    def _calibration_images():
        from PIL import Image, ImageDraw
        images = []
        for color in ("red", "green", "blue", "orange"):
            im = Image.new("RGB", (448, 448), "white")
            ImageDraw.Draw(im).ellipse([80, 80, 368, 368], fill=color)
            images.append(im)
        return images

    @staticmethod
    def _load_calibration_images():
        """Calibration image source. [assumption knob — see mc_quantization_understanding.md §10]
        TRTMC_CALIB_IMAGE_DIR (REAL, representative, DISJOINT-from-eval images) if set — recommended;
        else synthetic colored circles (_calibration_images) as an explicit PLACEHOLDER."""
        import os
        d = os.environ.get("TRTMC_CALIB_IMAGE_DIR")
        if d and os.path.isdir(d):
            from PIL import Image
            imgs = []
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    try:
                        imgs.append(Image.open(os.path.join(d, f)).convert("RGB"))
                    except Exception:
                        pass
                if len(imgs) >= 64:
                    break
            if imgs:
                print(f"[qwenvl-calib] using {len(imgs)} real calibration images from {d}", flush=True)
                return imgs
        print("[qwenvl-calib] TRTMC_CALIB_IMAGE_DIR unset -> synthetic PLACEHOLDER images "
              "(not representative; set the env for real accuracy)", flush=True)
        return QwenVLCalibrationAdapter._calibration_images()

    def iter_calibration_batches(
        self, model: Any, aux: Any | None, *, model_dir: str,
        config: "ModelConfig", num_samples: int,
        calibration_prompts: list[str] | None,
    ) -> Iterable[Any]:
        # IMAGE+TEXT calibration (2026-07-03; previously text-only). Feeding real image tokens makes
        # the decoder's activation statistics reflect true multimodal inference — the primary lever
        # for VL int8/nvfp4 accuracy (text-only calibration mis-set the activation scales for the
        # image+text workload). Reference (how, not gospel): TRT-LLM get_calib_dataloader multimodal
        # branch runs samples through the processor -> pixel_values+input_ids.
        # ASSUMPTION LOG — each line is a revisitable knob (see mc_quantization_understanding.md §10):
        #   [modality]   image+text via processor.apply_chat_template (handles the image-token
        #                expansion previously deferred as "version-fiddly"); per-batch text-only
        #                fallback if the processor image path raises (robust across transformers vers).
        #   [images]     TRTMC_CALIB_IMAGE_DIR if set = REAL representative images (recommended; MUST
        #                be DISJOINT from any eval set — no leakage). Else synthetic colored circles
        #                (_calibration_images) = PLACEHOLDER: exercises the image path but is NOT
        #                representative -> weak accuracy gain; replace with real images.
        #   [prompts]    calibration_prompts if provided, else generic describe/VQA prompts below.
        #   [chat tmpl]  loaded from the model's tokenizer_config.json (HF/model-specific provenance).
        #   [count]      num_samples (--quant-calibration-samples); image+text is slower than text.
        import itertools
        import json
        import os
        processor = aux
        tokenizer = getattr(processor, "tokenizer", processor)
        chat_template = None
        try:
            tc = os.path.join(model_dir, "tokenizer_config.json")
            if os.path.isfile(tc):
                chat_template = json.load(open(tc)).get("chat_template")
        except Exception:
            chat_template = None
        # Custom PAIRED calibration via a manifest (env-gated + additive; unset => unchanged behavior).
        # Each jsonl line = {"image": <path>, "prompt": <text>}; calibrates on (image_i, prompt_i) pairs —
        # the representative alternative to the synthetic/generic path when real eval-domain image+question
        # samples are available (activation ranges then reflect the true multimodal workload).
        manifest = os.environ.get("TRTMC_CALIB_MANIFEST")
        if manifest and os.path.isfile(manifest):
            from PIL import Image
            entries = [json.loads(line) for line in open(manifest) if line.strip()]
            print(f"[qwenvl-calib] paired manifest: {len(entries)} (image,prompt) pairs from {manifest}",
                  flush=True)
            used = 0
            for entry in itertools.islice(itertools.cycle(entries), num_samples):
                image = Image.open(entry["image"]).convert("RGB")
                prompt = entry["prompt"]
                try:
                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                    enc = processor.apply_chat_template(
                        msgs, chat_template=chat_template, add_generation_prompt=True,
                        tokenize=True, return_dict=True, return_tensors="pt")
                    used += 1
                except Exception:
                    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                yield {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
            print(f"[qwenvl-calib] paired manifest batches: {used}/{num_samples}", flush=True)
            return
        images = self._load_calibration_images()
        prompts = calibration_prompts or [
            "Describe this image in detail.",
            "What objects and text are present in the image?",
            "Answer the multiple-choice question about the image with the correct option letter.",
            "What is shown in this image and what can you infer from it?",
        ]
        pairs = [(im, pr) for im in images for pr in prompts]
        used_img = 0
        for im, pr in itertools.islice(itertools.cycle(pairs), num_samples):
            try:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": im}, {"type": "text", "text": pr}]}]
                enc = processor.apply_chat_template(
                    msgs, chat_template=chat_template, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt")
                used_img += 1
            except Exception:
                enc = tokenizer(pr, return_tensors="pt", truncation=True, max_length=256)
            yield {k: (v.to(model.device) if hasattr(v, "to") else v)
                   for k, v in enc.items()}
        print(f"[qwenvl-calib] image+text batches: {used_img}/{num_samples} "
              f"(rest = text-only fallback); images={len(images)}", flush=True)

    def run_calibration_batch(self, model: Any, batch: Any) -> None:
        import torch
        with torch.no_grad():
            model(**batch)

    def map_layer_name(self, layer_name: str) -> str | None:
        for pattern, suffix in self._RULES:
            match = pattern.search(layer_name)
            if match is not None:
                return f"layer.{match.group(1)}.{suffix}"
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
