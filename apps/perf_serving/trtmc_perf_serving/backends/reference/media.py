# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vision, audio, diffusion, and time-series references."""

from __future__ import annotations

import math
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

            model = timm.create_model(f"hf-hub:{spec.model}", pretrained=True)
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
            spec.model, dtype=spec.dtype, **spec.pretrained_kwargs()).to(spec.device).eval(), spec)
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        audio = load_audio(str(required(request, "audio_path")), self.sample_rate)
        features = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_features
        features = features.to(self.spec.device, self.spec.dtype)
        kwargs: dict[str, Any] = {"max_new_tokens": int(request.get("max_new_tokens", 128))}
        if request.get("language"):
            kwargs["language"] = request["language"]
        ids, model_ms = timed(lambda: self.model.generate(features, **kwargs))
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        seconds = len(audio) / self.sample_rate
        return invocation({"text": text, "input_audio_seconds": seconds}, model_ms,
                          realtime_factor=seconds / (model_ms / 1000.0))


class SpeechSynthesis:
    """``generate_audio`` from ``prompt`` (Bark-style text-to-speech)."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import transformers

        reject_compile(spec, "text-to-speech")
        self.spec = spec
        self.processor = transformers.AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = transformers.BarkModel.from_pretrained(
            spec.model, dtype=spec.dtype, **spec.pretrained_kwargs()).to(spec.device).eval()

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


class Diffusion:
    """``generate_image`` (image or video) through a Diffusers pipeline; timed pipeline call."""

    def __init__(self, spec: ReferenceSpec) -> None:
        import diffusers

        self.spec = spec
        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            spec.model, torch_dtype=spec.dtype, **spec.pretrained_kwargs()).to(spec.device)
        self.pipe.set_progress_bar_config(disable=True)
        denoiser = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        if spec.mode == "compile":
            if denoiser is None:
                raise BackendError("compile mode requires a transformer or unet denoiser")
            maybe_compile(denoiser, spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        kwargs: dict[str, Any] = {"prompt": str(required(request, "prompt")), "output_type": "np"}
        for source, target in (("height", "height"), ("width", "width"), ("num_steps", "num_inference_steps"),
                               ("guidance_scale", "guidance_scale"), ("negative_prompt", "negative_prompt")):
            value = request.get(source)
            if value not in (None, "", 0, -1):
                kwargs[target] = value
        video = request.get("media_type") == "video" or int(request.get("num_frames", 1)) > 1
        if video:
            kwargs["num_frames"] = int(request.get("num_frames", 1))
        kwargs["generator"] = torch.Generator(self.spec.device).manual_seed(max(int(request.get("seed", 0)), 0))
        result, model_ms = timed(lambda: self.pipe(**kwargs))
        media = np.asarray(result.frames if video else result.images)
        return invocation(tensor_observation(media, artifact_base), model_ms)


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
        model_cls = getattr(transformers, prefix + suffix)
        self.model = maybe_compile(model_cls.from_pretrained(spec.model, **spec.pretrained_kwargs())
                                   .to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        values = np.asarray(required(request, "past_values"), dtype=np.float32)
        rows, columns = request.get("shape", [values.size, 1])
        past = torch.from_numpy(values.reshape(1, int(rows), int(columns))).to(self.spec.device, self.spec.dtype)
        output, model_ms = timed(lambda: self.model(past_values=past))
        field = "regression_outputs" if self.spec.operation == "regress" else "prediction_outputs"
        tensor = getattr(output, field)
        return invocation(tensor_observation(tensor[0] if isinstance(tensor, tuple) else tensor, artifact_base),
                          model_ms)
