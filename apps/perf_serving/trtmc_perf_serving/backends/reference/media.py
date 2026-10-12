# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision, audio, diffusion, and time-series references."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..base import BackendError, Invocation
from .common import (
    ReferenceSpec, invocation, load_audio, load_image, maybe_compile, reject_compile, required,
    tensor_observation, timed,
)


class Vision:
    """classify / detect / segment / segment_prompted / extract_features on ``image_path``.

    Timed boundary: the model forward after image preprocessing.
    """

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        if spec.operation == "classify" and spec.model.startswith("timm/"):
            import timm

            # timm resolves a pinned Hub checkpoint as "hf-hub:<repo>@<revision>".
            model = timm.create_model(f"hf-hub:{spec.model}" + (f"@{spec.revision}" if spec.revision else ""),
                                      pretrained=True)
            config = timm.data.resolve_data_config({}, model=model)
            self.transform = timm.data.create_transform(**config)
            self.processor = None
        else:
            loaders = {
                "classify": transformers.AutoModelForImageClassification,
                "detect": transformers.AutoModelForObjectDetection,
                "segment": transformers.AutoModelForSemanticSegmentation,
                "segment_prompted": transformers.SamModel,
                "extract_features": transformers.AutoModel,
            }
            processor_cls = (transformers.SamProcessor if spec.operation == "segment_prompted"
                             else transformers.AutoImageProcessor)
            # processor_kwargs (for example use_fast=False) must match the family's accepted reference.
            self.processor = processor_cls.from_pretrained(spec.model, **spec.pretrained_kwargs(),
                                                           **dict(spec.options.get("processor_kwargs", {})))
            model = loaders[spec.operation].from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = maybe_compile(model.to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        image = load_image(str(required(request, "image_path")))
        device, dtype = self.spec.device, self.spec.dtype
        if self.processor is None:
            pixels = self.transform(image).unsqueeze(0).to(device, dtype)
            output, model_ms = timed(lambda: self.model(pixels).float())
            # Same fields as the TRTMC ImageClassification observation (raw logits).
            observation = {**tensor_observation(output, artifact_base), "score_kind": "logit",
                           "scores": output[0].tolist()}
            return invocation(observation, model_ms)
        operation = self.spec.operation
        kwargs: dict[str, Any] = {}
        if operation == "segment_prompted":
            # Same convention as the TRTMC worker: normalized point -> floor(x * width) pixels.
            kwargs["input_points"] = [[[math.floor(float(request.get("point_x", 0.5)) * image.width),
                                        math.floor(float(request.get("point_y", 0.5)) * image.height)]]]
            kwargs["input_labels"] = [[1 if bool(request.get("is_foreground", True)) else 0]]
        batch = self.processor(images=image, return_tensors="pt", **kwargs)
        sizes = {key: batch.pop(key) for key in ("original_sizes", "reshaped_input_sizes") if key in batch}
        batch = {key: value.to(device, dtype) if value.is_floating_point() else value.to(device)
                 for key, value in batch.items()}
        outputs, model_ms = timed(lambda: self.model(**batch))
        # Post-processing is outside the timed boundary and yields TRTMC's observation fields.
        return invocation(self._observe(outputs, image, request, sizes, artifact_base), model_ms)

    def _observe(self, outputs: Any, image: Any, request: Mapping[str, Any], sizes: Mapping[str, Any],
                 artifact_base: Path) -> dict[str, Any]:
        operation = self.spec.operation
        if operation == "classify":
            logits = outputs.logits.float()
            return {**tensor_observation(logits, artifact_base), "score_kind": "logit", "scores": logits[0].tolist()}
        if operation == "detect":
            result = self.processor.post_process_object_detection(
                outputs, threshold=float(request.get("score_threshold", 0.5)),
                target_sizes=[(image.height, image.width)])[0]
            boxes = result["boxes"].float().cpu()
            return {"boxes": boxes.flatten().tolist(), "class_ids": result["labels"].cpu().tolist(),
                    "scores": result["scores"].float().cpu().tolist(), "detections": int(boxes.shape[0]),
                    "coordinates": "xyxy", "units": "pixels", "image_width": image.width,
                    "image_height": image.height}
        if operation == "segment":
            logits = torch.nn.functional.interpolate(outputs.logits.float(), size=(image.height, image.width),
                                                     mode="bilinear", align_corners=False)
            mask = logits.argmax(1)[0].cpu()
            return {"height": image.height, "width": image.width, "num_masks": 1, "mask_pixels": mask.numel(),
                    "mask": mask.flatten().tolist()}
        if operation == "segment_prompted":
            # Raw mask logits at the original resolution, like the TRTMC worker (mask_kind "logits").
            masks = self.processor.post_process_masks(outputs.pred_masks.float().cpu(), sizes["original_sizes"],
                                                      sizes["reshaped_input_sizes"], binarize=False)[0][0]
            return {"height": image.height, "width": image.width, "num_masks": int(masks.shape[0]),
                    "mask_pixels": int(masks.numel()), "mask_kind": "logits", "masks": masks.flatten().tolist(),
                    "iou_scores": outputs.iou_scores.float().flatten().cpu().tolist()}
        hidden = getattr(outputs, "pooler_output", None)
        hidden = hidden if hidden is not None else outputs.last_hidden_state
        return tensor_observation(hidden, artifact_base)


class SpeechRecognition:
    """``transcribe`` from ``audio_path`` with a Whisper-style speech-seq2seq model."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        self.processor = transformers.AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = maybe_compile(transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device).eval(), spec)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        audio = load_audio(str(required(request, "audio_path")), self.sample_rate)
        features = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_features
        features = features.to(self.spec.device, self.spec.dtype)
        kwargs: dict[str, Any] = {"max_new_tokens": int(request.get("max_new_tokens", 128))}
        # A stated language, else the decoder contract the model declares (``options``: the candidate's fixed
        # prompt): without one, generation detects the language in an extra pass the candidate does not make.
        language = request.get("language") or self.spec.options.get("language")
        if language:
            kwargs["language"] = language
        if self.spec.options.get("task"):
            kwargs["task"] = self.spec.options["task"]
        import transformers

        steps = DecodeSteps()
        kwargs["logits_processor"] = transformers.LogitsProcessorList([steps])
        ids, model_ms = timed(lambda: self.model.generate(features, **kwargs))
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        seconds = len(audio) / self.sample_rate
        return invocation({"text": text, "output_tokens": steps.count, "input_audio_seconds": seconds}, model_ms,
                          realtime_factor=seconds / (model_ms / 1000.0))


class DecodeSteps:
    """A logits processor counting decoding steps: generation calls its processors once per generated token (the
    end token and generated special tokens included, the forced prompt not), whatever the returned sequence keeps."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        self.count += 1
        return scores


class SpeechSynthesis:
    """``generate_audio`` from ``prompt`` (Bark-style text-to-speech)."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        reject_compile(spec, "text-to-speech")
        self.spec = spec
        self.processor = transformers.AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = transformers.BarkModel.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device).eval()

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        # Same generation as the accepted Bark reference: Bark's own (sampled) generation configs, a
        # fixed seed (42 by default), and a semantic token budget only when the request sets one.
        inputs = self.processor(str(required(request, "prompt"))).to(self.spec.device)
        torch.manual_seed(int(request.get("seed", 42)) if int(request.get("seed", 42)) >= 0 else 42)
        kwargs = {}
        if int(request.get("max_new_tokens", 0)) > 0:
            kwargs["semantic_max_new_tokens"] = int(request["max_new_tokens"])
        audio, model_ms = timed(lambda: self.model.generate(**inputs, **kwargs))
        rate = int(self.model.generation_config.sample_rate)
        observation = tensor_observation(audio, artifact_base)
        seconds = audio.shape[-1] / rate
        return invocation({**observation, "sample_rate": rate, "audio_seconds": seconds}, model_ms,
                          realtime_factor=seconds / (model_ms / 1000.0))


def retie_encoder_embeddings(pipe: Any) -> list[str]:
    """Tie T5-style text encoders' token embeddings back to ``shared`` when loading left them zero.

    Transformers 5.2 does not tie ``encoder.embed_tokens`` of ``UMT5EncoderModel`` (Wan) to
    ``shared``: the weight loads as missing and stays zero, so every prompt encodes to zeros and the
    pipeline renders the same output whatever the prompt. Returns the components it repaired.
    """
    repaired = []
    for name, component in getattr(pipe, "components", {}).items():
        shared = getattr(component, "shared", None)
        embed = getattr(getattr(component, "encoder", None), "embed_tokens", None)
        if shared is None or embed is None or embed.weight is shared.weight:
            continue
        if embed.weight.shape == shared.weight.shape and not bool(embed.weight.any()):
            embed.weight = shared.weight
            repaired.append(name)
    return repaired


class Diffusion:
    """``generate_image`` (image or video, or an edit of ``image_path``/``image_paths``) through a
    Diffusers pipeline; timed pipeline call."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import diffusers

        self.spec = spec
        pipe = diffusers.DiffusionPipeline.from_pretrained(spec.model, torch_dtype=spec.dtype, **spec.pretrained_kwargs())
        for name in retie_encoder_embeddings(pipe):
            print(f"trtmc-perf-serve: tied {name}.encoder.embed_tokens to {name}.shared (zero after loading)",
                  file=sys.stderr)
        if spec.options.get("cpu_offload"):  # weights larger than the GPU: components move in when they run
            pipe.enable_model_cpu_offload()
            self.pipe = pipe
        else:
            self.pipe = pipe.to(spec.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.parameters = set(inspect.signature(self.pipe.__call__).parameters)
        denoiser = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        if spec.mode == "compile":
            if denoiser is None:
                raise BackendError("compile mode requires a transformer or unet denoiser")
            maybe_compile(denoiser, spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        kwargs: dict[str, Any] = {"prompt": str(required(request, "prompt")), "output_type": "np"}
        for source, target in (("height", "height"), ("width", "width"), ("num_steps", "num_inference_steps"),
                               ("negative_prompt", "negative_prompt")):
            value = request.get(source)
            if value not in (None, "", 0, -1):
                kwargs[target] = value
        guidance = request.get("guidance_scale")
        if guidance not in (None, "") and float(guidance) != -1.0:  # -1 leaves the default; 0 turns guidance off
            kwargs["guidance_scale"] = float(guidance)
        # Qwen-Image pipelines take the classifier-free guidance scale as true_cfg_scale.
        if "true_cfg_scale" in self.parameters and float(request.get("cfg_scale") or -1) > 0:
            kwargs["true_cfg_scale"] = float(request["cfg_scale"])
        paths = request.get("image_paths") or ([request["image_path"]] if request.get("image_path") else [])
        if paths:
            if "image" not in self.parameters:
                raise BackendError(f"{type(self.pipe).__name__} takes no input image to edit")
            images = [load_image(path) for path in paths]
            kwargs["image"] = images if len(images) > 1 else images[0]
        video = request.get("media_type") == "video" or int(request.get("num_frames", 1)) > 1
        if video:
            kwargs["num_frames"] = int(request.get("num_frames", 1))
        kwargs["generator"] = torch.Generator(self.spec.device).manual_seed(max(int(request.get("seed", 0)), 0))
        if request.get("initial_latents_path"):
            kwargs["latents"] = self._replayed_latents(request)
        result, model_ms = timed(lambda: self.pipe(**kwargs))
        media = np.asarray(result.frames if video else result.images)
        return invocation(tensor_observation(media, artifact_base), model_ms)

    def _replayed_latents(self, request: Mapping[str, Any]) -> Any:
        """The replayed noise file (``latents.Replay``) in the form this pipeline takes as ``latents``."""
        from ...latents import LAYOUTS, canonical_shape

        name = type(self.pipe).__name__
        shape = canonical_shape(name, self.pipe.transformer.config, self.pipe.vae.config, request)
        if shape is None:
            raise BackendError(f"no latent replay layout for {name}")
        flat = np.fromfile(str(request["initial_latents_path"]), dtype=np.float32)
        if flat.size != math.prod(shape):
            raise BackendError(f"initial latents hold {flat.size} floats; {name} expects {shape}")
        latents = torch.from_numpy(flat.reshape(shape)).to(self.spec.device)
        layout = LAYOUTS[name]
        if layout.pack:  # [1, (1,) C, H, W] -> [1, H/2 * W/2, C * 4]
            latents = self.pipe._pack_latents(latents, 1, shape[-3], shape[-2], shape[-1])
        return latents.to(self.spec.dtype) if layout.cast_to_pipeline_dtype else latents


def _with_post_init(model_cls: type) -> type:
    """``model_cls`` finishing its construction with ``post_init``: Transformers 5.2's PatchTSMixer heads
    skip it, and loading then fails on the missing ``all_tied_weights_keys``."""
    class Initialized(model_cls):  # type: ignore[misc, valid-type]
        def __init__(self, config: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(config, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.post_init()

    Initialized.__name__, Initialized.__module__ = model_cls.__name__, model_cls.__module__
    return Initialized


class TimeSeries:
    """``solve`` (forecast) and ``regress`` for Transformers time-series models.

    ``past_values`` is the worker's flat row-major array with ``shape`` [time, channel].
    """

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        self.spec = spec
        config = transformers.AutoConfig.from_pretrained(spec.model, **spec.pretrained_kwargs())
        prefix = {"patchtst": "PatchTST", "patchtsmixer": "PatchTSMixer"}.get(config.model_type)
        if prefix is None:
            raise BackendError(f"no time-series reference for model type {config.model_type!r}")
        suffix = "ForRegression" if spec.operation == "regress" else "ForPrediction"
        model_cls = _with_post_init(getattr(transformers, prefix + suffix))
        self.model = maybe_compile(model_cls.from_pretrained(spec.model, **spec.pretrained_kwargs())
                                   .to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        values = np.asarray(required(request, "past_values"), dtype=np.float32)
        # Without an explicit shape, the flat values are [time, channel] over the model's input channels.
        channels = int(getattr(self.model.config, "num_input_channels", 1) or 1)
        rows, columns = request.get("shape", [values.size // channels, channels])
        past = torch.from_numpy(values.reshape(1, int(rows), int(columns))).to(self.spec.device, self.spec.dtype)
        output, model_ms = timed(lambda: self.model(past_values=past))
        field = "regression_outputs" if self.spec.operation == "regress" else "prediction_outputs"
        tensor = getattr(output, field)
        return invocation(tensor_observation(tensor[0] if isinstance(tensor, tuple) else tensor, artifact_base, inline=True),
                          model_ms)
