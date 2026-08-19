# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sam3 model-owned E2E reference plugins."""

from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path

from . import _case_artifact_dir
from .contracts import E2ECase, RunContext, StageOutput, StageSpec
from .references.hf_transformers import (
    HfTransformersReference,
    _existing_path_reader,
    _json_output_reader,
    _reference_env,
    _resolve_pinned_sam3_model_ref,
    _torch_dtype_for_case,
    run_reference_subprocess,
)


class Sam3HfTransformersReference(HfTransformersReference):
    """Run HF Sam3 model-card image PCS reference with a text prompt."""

    def _run_prompted_segmentation_ref(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = (
            _case_artifact_dir(artifacts_dir, case.name)
            if ctx.artifacts_dir else artifacts_dir
        )
        output_path = str(Path(model_dir) / "hf_sam3.json")
        masks_path = str(Path(model_dir) / "hf_sam3_masks.npy")
        segmented_image_path = str(Path(model_dir) / "hf_sam3_segmented.png")

        image_path = self._resolve_image_path(case.inputs.get("image", ""))
        image_url = case.inputs.get("image_url", "")
        trust_remote_code = case.metadata.get("trust_remote_code", False)
        hf_id = case.hf_id
        model_revision = case.hf_revision
        model_ref = _resolve_pinned_sam3_model_ref(hf_id, model_revision)
        prompt = (
            case.inputs.get("text_prompt")
            or case.inputs.get("prompt")
            or case.metadata.get("text_prompt")
            or "ear"
        )
        torch_dtype_expr = _torch_dtype_for_case(case)

        script = textwrap.dedent(f"""\
            import json, os, torch, numpy as np
            from transformers import AutoTokenizer, Sam3ImageProcessorFast, Sam3Model, Sam3Processor
            from PIL import Image
            import requests

            hf_id = {hf_id!r}
            model_ref = {model_ref!r}
            model_revision = {model_revision!r}
            image_path = {image_path!r}
            image_url = {image_url!r}
            trust_remote_code = {trust_remote_code!r}
            output_path = {output_path!r}
            masks_path = {masks_path!r}
            segmented_image_path = {segmented_image_path!r}
            prompt = {prompt!r}

            load_kwargs = {{"trust_remote_code": trust_remote_code}}
            if model_revision and not os.path.isdir(model_ref):
                load_kwargs["revision"] = model_revision

            def _load_sam3_processor(model_ref):
                try:
                    return Sam3Processor.from_pretrained(
                        model_ref, **load_kwargs)
                except Exception as processor_error:
                    if not os.path.isdir(model_ref):
                        raise processor_error
                    processor_config_path = os.path.join(model_ref, "processor_config.json")
                    if os.path.isfile(processor_config_path):
                        with open(processor_config_path, encoding="utf-8") as f:
                            processor_config = json.load(f)
                    else:
                        processor_config = {{
                            "target_size": 1008,
                            "image_processor": {{
                                "do_convert_rgb": True,
                                "do_normalize": True,
                                "do_rescale": True,
                                "do_resize": True,
                                "image_mean": [0.5, 0.5, 0.5],
                                "image_std": [0.5, 0.5, 0.5],
                                "mask_size": {{"height": 288, "width": 288}},
                                "resample": 2,
                                "rescale_factor": 1.0 / 255.0,
                                "size": {{"height": 1008, "width": 1008}},
                            }},
                        }}
                    image_processor_kwargs = processor_config.get("image_processor")
                    if not isinstance(image_processor_kwargs, dict):
                        raise processor_error
                    image_processor_kwargs = dict(image_processor_kwargs)
                    image_processor_kwargs.pop("image_processor_type", None)
                    image_processor_kwargs.pop("processor_class", None)
                    image_processor = Sam3ImageProcessorFast(**image_processor_kwargs)
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_ref, **load_kwargs)
                    return Sam3Processor(
                        image_processor,
                        tokenizer,
                        target_size=processor_config.get("target_size"),
                    )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = _load_sam3_processor(model_ref)
            model = Sam3Model.from_pretrained(
                model_ref, torch_dtype={torch_dtype_expr},
                **load_kwargs).to(device)
            model.eval()

            if image_url:
                image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
            w, h = image.size
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            original_sizes = inputs.get("original_sizes")
            if hasattr(original_sizes, "cpu"):
                target_sizes = original_sizes.cpu().tolist()
            else:
                target_sizes = original_sizes
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=target_sizes,
            )[0]

            masks = results.get("masks")
            if masks is None:
                mask_np = np.zeros((0, h, w), dtype=np.uint8)
            else:
                mask_np = masks.cpu().numpy().astype(np.uint8)
                if mask_np.ndim == 2:
                    mask_np = mask_np[None, ...]
            np.save(masks_path, mask_np)

            scores = results.get("scores", [])
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy().astype(float).tolist()
            boxes = results.get("boxes", [])
            if hasattr(boxes, "detach"):
                boxes = boxes.detach().cpu().numpy().astype(float).tolist()

            if mask_np.shape[0] > 0:
                selected = int(np.argmax(scores)) if scores else 0
                selected = min(selected, mask_np.shape[0] - 1)
                overlay_mask = mask_np[selected].astype(bool)
                image_arr = np.asarray(image, dtype=np.float32)
                overlay = np.zeros_like(image_arr)
                overlay[..., 0] = 255.0
                overlay[..., 1] = 96.0
                alpha = 0.55
                image_arr[overlay_mask] = (
                    image_arr[overlay_mask] * (1.0 - alpha)
                    + overlay[overlay_mask] * alpha
                )
                Image.fromarray(np.clip(image_arr, 0, 255).astype(np.uint8)).save(
                    segmented_image_path)

            result = {{
                "text_prompt": prompt,
                "scores": scores,
                "mask_scores": scores,
                "boxes": boxes,
                "num_masks": int(mask_np.shape[0]),
                "mask_shape": list(mask_np.shape),
                "segmented_image_path": segmented_image_path,
                "reference_variant": "sam3_model_card_text_prompt",
            }}
            with open(output_path, "w") as f:
                json.dump(result, f)
            print(f"OK sam3 prompt={{prompt!r}} masks={{mask_np.shape[0]}}")
        """)

        python = ctx.reference_python_path() or sys.executable
        return run_reference_subprocess(
            command=[python, "-c", script],
            timeout_s=900,
            label="hf_sam3_prompted_segmentation",
            artifact_dir=ctx.artifacts_dir or "",
            case_name=case.name,
            stage_name=stage.name,
            env=_reference_env(ctx),
            output_readers=(
                _json_output_reader(output_path),
                _existing_path_reader(masks_path, "masks_path"),
                _existing_path_reader(segmented_image_path, "segmented_image_path"),
            ),
            failure_label="HF SAM3 prompted segmentation",
        )


reference = Sam3HfTransformersReference()
