# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy evidence for the exact five-frame SAM2 workload.

The checked-in state is deliberately unqualified.  An authoritative reference
must come from three bitwise-identical BF16 runs of the pinned compatible source
on L4, using the delivered config rather than the different same-named config
beside that source.  The one-object ABI is valid only when frame zero has one
post-NMS detection; this module never silently selects top-1.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .archive_contract import REFERENCE_CHECKPOINT_SHA256, REFERENCE_CONFIG_SHA256


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "sam2_exact_workload_golden_evidence"
COMPARISON_ARTIFACT_TYPE = "sam2_exact_workload_accuracy_comparison"
MASK_PAYLOAD_NAME = "masks.bitpack"
FRAME_INDICES = (0, 1, 2, 3, 4)
FRAME_COUNT = len(FRAME_INDICES)
SELECTED_OBJECT_ID = 0
ORIGINAL_IMAGE_SHAPE_HW = (1280, 1088)
MODEL_IMAGE_SHAPE_HW = (1024, 1024)
MASK_SHAPE = (FRAME_COUNT, 1, *ORIGINAL_IMAGE_SHAPE_HW)
BINARY_MASK_THRESHOLD = 0.0

# Exact checked-in capture runner.  Authority still requires an independently
# reviewed raw manifest pin below; the tool pin alone cannot qualify evidence.
AUTHORITATIVE_CAPTURE_TOOL_SHA256: str | None = (
    "5c1929cdf803cfc82012d1ad3529e417d8f798d6c4b99020be18bdb031216eeb"
)
AUTHORITATIVE_ARCHIVE_CONTRACT_SHA256 = (
    "f0d169032d21157e015eb7e6912b025c39db20c311c67d64df7567cabec8d07a"
)
AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256 = (
    "2cfae7b9c81708221ee5523c52c9dfb706b64a04d6da9bf25e3cae3879d8b689"
)
# Replace this pending all-zero slot only after independently reviewing the
# first three-run source artifact.  The raw pin binds provenance, capture
# identity, and determinism metadata; its fixed shape remains Ruff-stable.
AUTHORITATIVE_REFERENCE_MANIFEST_SHA256: str | None = (
    "c25251ee27da05afd75adc3c6869cbc2944b80c05c5d6e703b6ebbbba697a4f0"
)

PUBLIC_SAM2_BASE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
COMPATIBLE_SOURCE_COMMIT = "79ab25d6bd5535bcb748de9f3b90e16b1a24e58d"

# The compatible source is a clean public Meta SAM2 tree at the pinned base,
# followed by this exact seven-file bbox overlay.  In particular, the private
# fork's backbones/__init__.py is not part of the composition: it imports the
# unrelated HOI/vendor tree.
PUBLIC_SAM2_BASE_FILES_SHA256: Mapping[str, str] = {
    "sam2/__init__.py": "b87ca1e95cd54b81766e8c74acf0e937952639529e40d2b4088693286f49419e",
    "sam2/automatic_mask_generator.py": (
        "66df266dbe14412305ae3398f0ec1bb21b303a93216b102d767e6c4ee5d4c3d7"
    ),
    "sam2/benchmark.py": "9b7a3506b88842ec09031fb7cd0fc1d3fed82e137ce06dbca10495e6c180dda9",
    "sam2/build_sam.py": "856d64d71e44407401297b551884fb4cd1aba8082de0ff5fc60fac3dd42f094d",
    "sam2/modeling/__init__.py": (
        "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa"
    ),
    "sam2/modeling/backbones/__init__.py": (
        "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa"
    ),
    "sam2/modeling/backbones/hieradet.py": (
        "03785581ca304d0451ae0df7a08ee0bf1e1dbe66fad285066e6b9ffc0d88d64f"
    ),
    "sam2/modeling/backbones/image_encoder.py": (
        "16eaad232220386f510ca8dfab8655e63699977149a4daf9ce9c6f374e6777a9"
    ),
    "sam2/modeling/backbones/utils.py": (
        "c4a0657db2a92bda2a7ff116cd4017f6b1c59af6408093d8874eb08a0df476ad"
    ),
    "sam2/modeling/memory_attention.py": (
        "07358cb7f58ec3788e88ff4e6415f8a9d628a3963529d6e2c915494a5347c8e2"
    ),
    "sam2/modeling/memory_encoder.py": (
        "73f7089ae5fdacdfcaaf3deca1ca6d9f84d89e7deb9cf97589622a9ef17ba42f"
    ),
    "sam2/modeling/position_encoding.py": (
        "b51404718c0d38f381293c8e5e00a15d129651b7f09b1158002d8974a30967b5"
    ),
    "sam2/modeling/sam/__init__.py": (
        "34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa"
    ),
    "sam2/modeling/sam/mask_decoder.py": (
        "ca3523c58365574faddf1bfb54f374e4b4beba05127185ea2b17d22b916e099a"
    ),
    "sam2/modeling/sam/prompt_encoder.py": (
        "4965ccb4a4504aa4d7246b2ad66ffeacd2626415c86a261dc1c65ffd8ae1d40d"
    ),
    "sam2/modeling/sam/transformer.py": (
        "cda19052331e775190ce8b1159efcb828a4632d931a6d8bd4d47109f121782f1"
    ),
    "sam2/modeling/sam2_base.py": (
        "69a46b44e8625f509791352bd09beaafe589cd7d872721384e63d37f9fdc6e41"
    ),
    "sam2/modeling/sam2_utils.py": (
        "e35bbf13bc2e544a0272cd3f9539af38a752676ac3cd744be31cd8220afee804"
    ),
    "sam2/sam2_image_predictor.py": (
        "f13e5f9d94e5c8d9d2c3622dab20c8f334c089ef2ee5ea8e199da7d332b029ba"
    ),
    "sam2/sam2_video_predictor.py": (
        "912555920a77f72ded07839efa33fcceecc79f3523abbec783f0be46ee2c55c2"
    ),
    "sam2/sam2_video_predictor_legacy.py": (
        "e0c054112f21bfa63620f2026c8bb2e0f4e284b138d09426a08d39f2e882c3f2"
    ),
    "sam2/utils/__init__.py": ("34bd8069c54764e7b8d73a78905dbe6467140a2f73170875128f6ca4d8cdd0aa"),
    "sam2/utils/amg.py": "b7b33090e2af72e04dbb815c8f32aff41a4ed1abf9668f62b59f1bdd640ca5d8",
    "sam2/utils/misc.py": "01600c01c161cd079d7106fb1d4da845cf91aa31ab2bcecaf8cb151b6d6d20a2",
    "sam2/utils/transforms.py": (
        "ba3a64f4600c62f209206a6df3b40e3fcf133edae32fad658831bb0c2a6d1146"
    ),
}
COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256: Mapping[str, str] = {
    "sam2/modeling/backbones/bbox_head.py": (
        "6e61cb2b5196de883bc537675437f5a787c5a1f7f0b5be6a87b81e3549775672"
    ),
    "sam2/modeling/backbones/image_encoder.py": (
        "de721c53be342d03a8958e62d37867a1eb0e759cd1add9cf95b78db7595111e1"
    ),
    "sam2/modeling/backbones/utils.py": (
        "c8ed6228b28dd5a7b0b08111b71263dfbf1542d47a0a444cd0728ddb3d5287e5"
    ),
    "sam2/sam2_video_predictor.py": (
        "ed7d81cd3595125b206cb91a52c2243c141f1df4e7ce2d567e7eca8ede896be6"
    ),
    "sam2/utils/act_ckpt_utils.py": (
        "3ea2eef843dd933560475d6caadb6b658297e17aee1e846294646b250d552b70"
    ),
    "sam2/utils/csp_layer.py": ("6303552e820a41ea5ca5d228df5951eb88e7be11338434c0965f6a0e2b281f0d"),
    "sam2/utils/misc.py": ("2e2de67378bce862ec721916b56ffacee1caa502ad2c49ddc4e7287595a74612"),
}
COMPATIBLE_SOURCE_FILES_SHA256: Mapping[str, str] = {
    **PUBLIC_SAM2_BASE_FILES_SHA256,
    **COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256,
}
INPUT_IMAGES_SHA256: Mapping[str, str] = {
    "000000.jpg": "8a398f40747d5053cfc0d47d45090f2070a10afa4722e7d5b827a6ad0825a5aa",
    "000001.jpg": "2871555bca47da7473762ca87314b17bd55d100a0f982f78d6449080ff86856f",
    "000002.jpg": "5594181db7dd1c5da3ce05b945f74e66a5d8d098d71a7cb9e5e43834a393bbe2",
    "000003.jpg": "c3abc03371458939d09faf331749c2a87cc6fc91128eaab3901b179adb096a35",
    "000004.jpg": "3d8ea6042c82e7b340277c00666c4c2cefbae5de265ef06a71fe964905ed720b",
}
INPUT_IMAGES_DECODED_RGB_UINT8_SHA256: Mapping[str, str] = {
    "000000.jpg": "0bcadde0e5a6f8ba04f79c44f064c5b00d3cd1b250e2f2f3bbf10ef0630a9ce9",
    "000001.jpg": "0abfd57f9e3886a8c3068bf6bcc353b26d1e3a8a43819a80dfeb00f309b24ec3",
    "000002.jpg": "9166cc263c3edb262065fa3b98ee062cbf6d781dd656bae13def7f4141b7d025",
    "000003.jpg": "77525faadfc8a607e4e1556135887caaddd0b64d7cd677fcf47c38ecf9e25a4f",
    "000004.jpg": "cb0801b490ba13dfb6d36aeef06b049ff67ff11864ef62ccd858a0096d97c6af",
}
INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256: Mapping[str, str] = {
    "000000.jpg": "bfc4b87e211b8437ede1b2244fe4c4bd0565afa7b05514d42305c7a2c2b1c275",
    "000001.jpg": "a1ac93ad5fe41f2109245b25d7068097f7d09f26e0592a4375d12bbfa81e44bb",
    "000002.jpg": "4c3da2fcc6a9c154036fad6742cd8bef9dc89a6abf7dd37cd6d9026ebb3b2676",
    "000003.jpg": "663996af0514e10932c035aedf81191c4e2420c726be930bd0c9809241a3f238",
    "000004.jpg": "bc35d5119f83ac8e169c42f68fd00117ae3bdcbfcf3ed33c45da677b60aa8bfc",
}

MIN_BOX_IOU = 0.995
MAX_BOX_COORDINATE_ABS_ERROR = 0.5
MAX_SCORE_ABS_ERROR = 0.01
MIN_FRAME_MASK_IOU = 0.995
MIN_MACRO_MASK_IOU = 0.995
MIN_GLOBAL_MASK_IOU = 0.995

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
_REFERENCE_PRODUCER = "compatible_source_pytorch_bf16"
_CANDIDATE_PRODUCER = "candidate"
_Producer = Literal["compatible_source_pytorch_bf16", "candidate"]
_WORKLOAD = {
    "frame_indices": list(FRAME_INDICES),
    "post_nms_detection_count": 1,
    "selected_object_count": 1,
    "selected_object_ids": [SELECTED_OBJECT_ID],
    "original_image_shape_hw": list(ORIGINAL_IMAGE_SHAPE_HW),
    "model_image_shape_hw": list(MODEL_IMAGE_SHAPE_HW),
    "mask_materialization": {
        "source": "propagate_in_video.video_res_logits",
        "source_shape": [FRAME_COUNT, 1, *ORIGINAL_IMAGE_SHAPE_HW],
        "comparison": "greater_than",
        "threshold": BINARY_MASK_THRESHOLD,
        "upstream_resize_contract": {
            "source": "per_frame_pred_mask_logits",
            "source_shape": [1, 1, 256, 256],
            "resize_mode": "bilinear",
            "resize_size_hw": list(ORIGINAL_IMAGE_SHAPE_HW),
            "align_corners": False,
        },
    },
}


class Sam2GoldenEvidenceError(ValueError):
    """The evidence is malformed, inconsistent, or tampered with."""


@dataclass(frozen=True)
class FrameZeroBBox:
    original_xyxy: tuple[float, float, float, float]
    model_xyxy_1024: tuple[float, float, float, float]
    score: float
    label: int


@dataclass(frozen=True)
class WorkloadCapture:
    masks: np.ndarray
    frame_zero_bbox: FrameZeroBBox
    post_nms_detection_count: int = 1
    selected_object_id: int = SELECTED_OBJECT_ID


@dataclass(frozen=True)
class Provenance:
    source_commit: str
    source_overlay_declared_commit: str
    source_files_sha256: Mapping[str, str]
    checkpoint_sha256: str
    config_sha256: str
    image_files_sha256: Mapping[str, str]
    capture_tool_sha256: str
    environment: Mapping[str, Any]
    artifacts_sha256: Mapping[str, str]


@dataclass(frozen=True)
class LoadedEvidence:
    root: Path
    manifest: Mapping[str, Any]
    masks: np.ndarray
    manifest_sha256: str
    authoritative_reference: bool


def _is_reviewed_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None and value != "0" * 64


def repository_qualification_state() -> dict[str, object]:
    """Return the checked-in state without consulting local artifacts."""

    return {
        "status": "unqualified",
        "reason": "authoritative_compatible_source_golden_pending",
        "required_reference_producer": _REFERENCE_PRODUCER,
        "required_deterministic_runs": 3,
        "required_frames": list(FRAME_INDICES),
        "required_post_nms_detection_count": 1,
        "required_selected_object_count": 1,
        "capture_tool_sha256_pinned": _is_reviewed_sha256(AUTHORITATIVE_CAPTURE_TOOL_SHA256),
        "reference_manifest_sha256_pinned": _is_reviewed_sha256(
            AUTHORITATIVE_REFERENCE_MANIFEST_SHA256
        ),
    }


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: object) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _checked_hash(value: object, label: str, pattern: re.Pattern[str] = _SHA256) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise Sam2GoldenEvidenceError(f"{label} has an invalid hash")
    return value


def _digest_map(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise Sam2GoldenEvidenceError(f"{label} must be a non-empty digest mapping")
    result: dict[str, str] = {}
    for path, digest in value.items():
        if not isinstance(path, str) or not path or "\x00" in path:
            raise Sam2GoldenEvidenceError(f"{label} has an invalid path")
        result[path] = _checked_hash(digest, label)
    return dict(sorted(result.items()))


def _provenance_record(value: Provenance) -> dict[str, object]:
    if not isinstance(value.environment, Mapping) or not value.environment:
        raise Sam2GoldenEvidenceError("provenance environment must be a non-empty mapping")
    try:
        json.dumps(value.environment, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise Sam2GoldenEvidenceError("provenance environment must be finite JSON") from error
    return {
        "source_commit": _checked_hash(value.source_commit, "source commit", _GIT_SHA),
        "source_overlay_declared_commit": _checked_hash(
            value.source_overlay_declared_commit,
            "source overlay declared commit",
            _GIT_SHA,
        ),
        "source_files_sha256": _digest_map(value.source_files_sha256, "source files"),
        "checkpoint_sha256": _checked_hash(value.checkpoint_sha256, "checkpoint"),
        "config_sha256": _checked_hash(value.config_sha256, "config"),
        "image_files_sha256": _digest_map(value.image_files_sha256, "image files"),
        "capture_tool_sha256": _checked_hash(value.capture_tool_sha256, "capture tool"),
        "environment": dict(value.environment),
        "artifacts_sha256": _digest_map(value.artifacts_sha256, "artifacts"),
    }


def _bbox_record(value: FrameZeroBBox) -> dict[str, object]:
    def xyxy(raw: object, label: str) -> tuple[float, float, float, float]:
        try:
            result = tuple(float(item) for item in raw)  # type: ignore[union-attr]
        except (TypeError, ValueError) as error:
            raise Sam2GoldenEvidenceError(f"{label} must contain four coordinates") from error
        if len(result) != 4 or not all(math.isfinite(item) for item in result):
            raise Sam2GoldenEvidenceError(f"{label} must contain four finite coordinates")
        if result[0] >= result[2] or result[1] >= result[3]:
            raise Sam2GoldenEvidenceError(f"{label} must be a non-empty XYXY box")
        return result  # type: ignore[return-value]

    original = xyxy(value.original_xyxy, "original box")
    model = xyxy(value.model_xyxy_1024, "model box")
    expected = (
        model[0] * ORIGINAL_IMAGE_SHAPE_HW[1] / MODEL_IMAGE_SHAPE_HW[1],
        model[1] * ORIGINAL_IMAGE_SHAPE_HW[0] / MODEL_IMAGE_SHAPE_HW[0],
        model[2] * ORIGINAL_IMAGE_SHAPE_HW[1] / MODEL_IMAGE_SHAPE_HW[1],
        model[3] * ORIGINAL_IMAGE_SHAPE_HW[0] / MODEL_IMAGE_SHAPE_HW[0],
    )
    if any(abs(left - right) > 1e-4 for left, right in zip(original, expected)):
        raise Sam2GoldenEvidenceError("frame-zero boxes have inconsistent scaling")
    if (
        isinstance(value.label, bool)
        or not isinstance(value.label, int)
        or value.label not in {0, 1}
    ):
        raise Sam2GoldenEvidenceError("frame-zero label must be one of the two bbox classes")
    if isinstance(value.score, bool) or not isinstance(value.score, (int, float)):
        raise Sam2GoldenEvidenceError("frame-zero score must be numeric")
    score = float(value.score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise Sam2GoldenEvidenceError("frame-zero score must be finite and in [0, 1]")
    return {
        "frame_index": 0,
        "coordinate_format": "xyxy",
        "original_image_xyxy": list(original),
        "model_image_xyxy_1024": list(model),
        "score": score,
        "label": value.label,
    }


def _capture_record(value: WorkloadCapture) -> tuple[np.ndarray, dict[str, object]]:
    if (
        isinstance(value.post_nms_detection_count, bool)
        or not isinstance(value.post_nms_detection_count, int)
        or value.post_nms_detection_count != 1
    ):
        raise Sam2GoldenEvidenceError(
            "exact workload requires one post-NMS detection; top-1 selection is forbidden"
        )
    if (
        isinstance(value.selected_object_id, bool)
        or not isinstance(value.selected_object_id, int)
        or value.selected_object_id != 0
    ):
        raise Sam2GoldenEvidenceError("exact workload requires selected object id 0")
    masks = np.asarray(value.masks)
    if masks.shape != MASK_SHAPE:
        raise Sam2GoldenEvidenceError(
            f"binary masks must have exact shape {MASK_SHAPE}, got {masks.shape}"
        )
    if masks.dtype not in {np.dtype(np.bool_), np.dtype(np.uint8)}:
        raise Sam2GoldenEvidenceError("binary masks must have bool or uint8 dtype")
    if masks.dtype == np.uint8 and np.any((masks != 0) & (masks != 1)):
        raise Sam2GoldenEvidenceError("uint8 masks must contain only 0 and 1")
    masks = np.ascontiguousarray(masks, dtype=np.uint8)
    frames = [
        {
            "frame_index": index,
            "object_ids": [0],
            "foreground_pixels": int(frame.sum(dtype=np.int64)),
            "logical_uint8_sha256": _digest(frame.tobytes()),
        }
        for index, frame in zip(FRAME_INDICES, masks)
    ]
    return masks, {"frame_zero_bbox": _bbox_record(value.frame_zero_bbox), "frames": frames}


def compare_captures_exact(
    reference: WorkloadCapture, candidate: WorkloadCapture
) -> dict[str, object]:
    """Compare two repeat captures bit-for-bit."""

    reference_masks, reference_record = _capture_record(reference)
    candidate_masks, candidate_record = _capture_record(candidate)
    frame_exact = [
        bool(np.array_equal(left, right)) for left, right in zip(reference_masks, candidate_masks)
    ]
    bbox_exact = reference_record["frame_zero_bbox"] == candidate_record["frame_zero_bbox"]
    return {
        "exact": bbox_exact and all(frame_exact),
        "frame_zero_bbox_exact": bbox_exact,
        "frame_masks_exact": [
            {"frame_index": index, "exact": exact}
            for index, exact in zip(FRAME_INDICES, frame_exact)
        ],
    }


def _stored_capture(value: WorkloadCapture) -> tuple[np.ndarray, bytes, dict[str, object]]:
    masks, fields = _capture_record(value)
    packed = np.packbits(masks.reshape(-1), bitorder="little").tobytes()
    mask_record = {
        "encoding": "numpy_packbits_v1",
        "bitorder": "little",
        "path": MASK_PAYLOAD_NAME,
        "shape": list(MASK_SHAPE),
        "logical_dtype": "uint8_binary",
        "binary_threshold": BINARY_MASK_THRESHOLD,
        "logical_element_count": int(masks.size),
        "logical_uint8_sha256": _digest(masks.tobytes()),
        "packed_bytes": len(packed),
        "packed_sha256": _digest(packed),
    }
    identity = {"workload": _WORKLOAD, **fields, "masks": mask_record}
    return masks, packed, {**identity, "capture_sha256": _canonical_digest(identity)}


def _authority_errors(provenance: Mapping[str, Any]) -> list[str]:
    errors = []
    expected = {
        "source_commit": PUBLIC_SAM2_BASE_COMMIT,
        "source_overlay_declared_commit": COMPATIBLE_SOURCE_COMMIT,
        "source_files_sha256": dict(COMPATIBLE_SOURCE_FILES_SHA256),
        "checkpoint_sha256": REFERENCE_CHECKPOINT_SHA256,
        # This is the delivered package config.  The adjacent source YAML is different.
        "config_sha256": REFERENCE_CONFIG_SHA256,
        "image_files_sha256": dict(INPUT_IMAGES_SHA256),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            errors.append(f"{key.replace('_sha256', ' hash')} mismatch")
    artifacts = provenance.get("artifacts_sha256")
    required_capture_code = {
        "capture_code/tensorrt_model_connect.families.sam2.archive_contract": (
            AUTHORITATIVE_ARCHIVE_CONTRACT_SHA256
        ),
        "capture_code/tensorrt_model_connect.families.sam2.golden_evidence.normalized": (
            AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256
        ),
        "capture_code/tensorrt_model_connect.families.sam2.capture_golden": (
            AUTHORITATIVE_CAPTURE_TOOL_SHA256
        ),
    }
    if not isinstance(artifacts, Mapping):
        errors.append("capture code closure receipts missing")
    else:
        for key, expected_hash in required_capture_code.items():
            if artifacts.get(key) != expected_hash:
                errors.append(f"{key} mismatch")
    if not _is_reviewed_sha256(AUTHORITATIVE_CAPTURE_TOOL_SHA256):
        errors.append("authoritative capture-tool contract is not pinned")
    elif provenance.get("capture_tool_sha256") != AUTHORITATIVE_CAPTURE_TOOL_SHA256:
        errors.append("capture tool hash mismatch")
    environment = provenance.get("environment")
    required_environment = {
        "python": "3.12.3",
        "antlr4_python3_runtime": "4.9.3",
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "pillow_jpeg_codec": "6.2",
        "libjpeg_turbo": "3.1.4.1",
        "input_images_decoded_rgb_uint8_sha256": dict(INPUT_IMAGES_DECODED_RGB_UINT8_SHA256),
        "input_images_resized_1024_rgb_uint8_sha256": dict(
            INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256
        ),
        "hydra_core": "1.3.2",
        "iopath": "0.1.10",
        "omegaconf": "2.3.1",
        "portalocker": "4.1.0",
        "pyyaml": "6.0.3",
        "torch": "2.7.1+cu128",
        "torchvision": "0.22.1+cu128",
        "tqdm": "4.67.1",
        "torch_cuda": "12.8",
        "cuda_driver": "595.58.03",
        "cuda_capability": [8, 9],
        "autocast": "cuda bfloat16",
        "sam2_optional_extension_present": False,
        "tf32_matmul": True,
        "tf32_cudnn": True,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
        "cudnn_version": 90701,
        "deterministic_algorithms": False,
        "python_isolated": True,
        "python_safe_path": True,
        "python_no_user_site": True,
        "python_no_site": True,
        "controlled_site_packages": ("/workspace/ref-work/.venv/lib/python3.12/site-packages"),
        "venv_pyvenv_cfg_sha256": (
            "16529e11b2fe1e50d7bca13c16b18bdd5ff478ae2db7750e483aba6e3733d858"
        ),
        "capture_input_isolation": "private_read_only_verified_snapshot_v1",
        "capture_runs": 3,
        "async_loading_frames": True,
        "apply_postprocessing": True,
        "config_name": "configs/sam2.1/trtmc_delivery_bbox_59488bb78c7c.yaml",
    }
    if not isinstance(environment, Mapping):
        errors.append("environment mismatch")
    else:
        expected_environment_keys = {
            *required_environment,
            "gpu",
            "dependency_origins",
            "video_res_logits_dtypes",
        }
        if set(environment) != expected_environment_keys:
            errors.append("environment key set mismatch")
        for key, value in required_environment.items():
            if environment.get(key) != value:
                errors.append(f"environment {key} mismatch")
        if environment.get("gpu") != "NVIDIA L4":
            errors.append("environment gpu mismatch")
        dependency_origins = environment.get("dependency_origins")
        if not isinstance(dependency_origins, Mapping) or set(dependency_origins) != {
            "antlr4",
            "hydra",
            "iopath",
            "numpy",
            "omegaconf",
            "pillow",
            "portalocker",
            "pyyaml",
            "torch",
            "torchvision",
            "tqdm",
        }:
            errors.append("environment dependency origins mismatch")
        elif any(
            not isinstance(path, str)
            or not path.startswith("/workspace/ref-work/.venv/lib/python3.12/site-packages/")
            for path in dependency_origins.values()
        ):
            errors.append("environment dependency origin path mismatch")
        video_dtypes = environment.get("video_res_logits_dtypes")
        if (
            not isinstance(video_dtypes, list)
            or len(video_dtypes) != 3
            or any(
                not isinstance(run, list)
                or len(run) != FRAME_COUNT
                or any(dtype not in {"torch.bfloat16", "torch.float32"} for dtype in run)
                for run in video_dtypes
            )
        ):
            errors.append("environment video-res logits dtypes mismatch")
    return errors


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_evidence(
    destination: str | Path,
    *,
    capture: WorkloadCapture,
    provenance: Provenance,
    producer: _Producer,
    authoritative_source_run: bool = False,
    replay_captures: Sequence[WorkloadCapture] = (),
) -> dict[str, object]:
    """Write a lossless capture; authority needs two exact additional replays."""

    if producer not in {_REFERENCE_PRODUCER, _CANDIDATE_PRODUCER}:
        raise Sam2GoldenEvidenceError(f"unsupported evidence producer: {producer!r}")
    if authoritative_source_run and producer != _REFERENCE_PRODUCER:
        raise Sam2GoldenEvidenceError("only the compatible-source producer can be authoritative")

    _, packed, capture_fields = _stored_capture(capture)
    provenance_record = _provenance_record(provenance)
    replay_results = [compare_captures_exact(capture, item) for item in replay_captures]
    replay_hashes = [_stored_capture(item)[2]["capture_sha256"] for item in replay_captures]
    run_count = 1 + len(replay_captures)
    all_exact = all(result["exact"] for result in replay_results)
    if authoritative_source_run:
        errors = _authority_errors(provenance_record)
        if errors:
            raise Sam2GoldenEvidenceError(
                "authoritative source provenance failed: " + "; ".join(errors)
            )
        if run_count < 3:
            raise Sam2GoldenEvidenceError("authoritative reference requires at least three runs")
        if not all_exact:
            raise Sam2GoldenEvidenceError("authoritative captures are not bitwise deterministic")
        qualification = {
            "status": "authoritative_reference_candidate",
            "reason": "requires_checked_in_exact_manifest_sha256_pin",
        }
    else:
        qualification = {
            "status": "unqualified",
            "reason": (
                "authoritative_source_run_pending"
                if producer == _REFERENCE_PRODUCER
                else "authoritative_reference_comparison_pending"
            ),
        }

    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "qualification": qualification,
        "provenance": provenance_record,
        **capture_fields,
        "determinism": {
            "run_count": run_count,
            "all_exact": all_exact,
            "capture_sha256": [capture_fields["capture_sha256"], *replay_hashes],
            "replays": replay_results,
        },
    }
    root = Path(destination)
    if root.is_symlink():
        raise Sam2GoldenEvidenceError("evidence destination must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise Sam2GoldenEvidenceError("evidence destination must be empty")
    _atomic_write(root / MASK_PAYLOAD_NAME, packed)
    _atomic_write(
        root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def _read_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "manifest.json"
    if root.is_symlink() or path.is_symlink() or not path.is_file():
        raise Sam2GoldenEvidenceError("evidence manifest must be a regular file")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Sam2GoldenEvidenceError(f"unable to read evidence manifest: {error}") from error
    if not isinstance(value, dict):
        raise Sam2GoldenEvidenceError("evidence manifest must be a mapping")
    if value.get("artifact_type") != ARTIFACT_TYPE or value.get("schema_version") != SCHEMA_VERSION:
        raise Sam2GoldenEvidenceError("unsupported SAM2 evidence schema")
    return value, _digest(payload)


def _validated_manifest_provenance(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Sam2GoldenEvidenceError("provenance must be a mapping")
    try:
        provenance = Provenance(
            source_commit=value["source_commit"],
            source_overlay_declared_commit=value["source_overlay_declared_commit"],
            source_files_sha256=value["source_files_sha256"],
            checkpoint_sha256=value["checkpoint_sha256"],
            config_sha256=value["config_sha256"],
            image_files_sha256=value["image_files_sha256"],
            capture_tool_sha256=value["capture_tool_sha256"],
            environment=value["environment"],
            artifacts_sha256=value["artifacts_sha256"],
        )
    except KeyError as error:
        raise Sam2GoldenEvidenceError(f"provenance is missing {error.args[0]}") from error
    record = _provenance_record(provenance)
    if dict(value) != record:
        raise Sam2GoldenEvidenceError("provenance contains unexpected or noncanonical fields")
    return record


def load_evidence(directory: str | Path) -> LoadedEvidence:
    """Load an evidence directory and verify all hashes and exact-workload fields."""

    root = Path(directory)
    manifest, manifest_sha256 = _read_manifest(root)
    if manifest.get("workload") != _WORKLOAD:
        raise Sam2GoldenEvidenceError("evidence does not describe the exact SAM2 workload")
    provenance = _validated_manifest_provenance(manifest.get("provenance"))

    bbox = manifest.get("frame_zero_bbox")
    if not isinstance(bbox, Mapping):
        raise Sam2GoldenEvidenceError("frame-zero bbox is missing")
    try:
        parsed_bbox = FrameZeroBBox(
            tuple(bbox["original_image_xyxy"]),
            tuple(bbox["model_image_xyxy_1024"]),
            bbox["score"],
            bbox["label"],
        )
    except (KeyError, TypeError) as error:
        raise Sam2GoldenEvidenceError("frame-zero bbox is malformed") from error
    if dict(bbox) != _bbox_record(parsed_bbox):
        raise Sam2GoldenEvidenceError("frame-zero bbox contains unexpected fields")

    masks_record = manifest.get("masks")
    expected_mask_contract = {
        "encoding": "numpy_packbits_v1",
        "bitorder": "little",
        "path": MASK_PAYLOAD_NAME,
        "shape": list(MASK_SHAPE),
        "logical_dtype": "uint8_binary",
        "binary_threshold": BINARY_MASK_THRESHOLD,
        "logical_element_count": math.prod(MASK_SHAPE),
    }
    if not isinstance(masks_record, Mapping) or any(
        masks_record.get(key) != value for key, value in expected_mask_contract.items()
    ):
        raise Sam2GoldenEvidenceError("mask storage contract is invalid")
    logical_hash = _checked_hash(masks_record.get("logical_uint8_sha256"), "logical masks")
    packed_hash = _checked_hash(masks_record.get("packed_sha256"), "packed masks")
    packed_bytes = (math.prod(MASK_SHAPE) + 7) // 8
    if masks_record.get("packed_bytes") != packed_bytes:
        raise Sam2GoldenEvidenceError("packed mask byte count is invalid")
    payload_path = root / MASK_PAYLOAD_NAME
    if payload_path.is_symlink() or not payload_path.is_file():
        raise Sam2GoldenEvidenceError("packed mask payload must be a regular file")
    payload = payload_path.read_bytes()
    if len(payload) != packed_bytes or _digest(payload) != packed_hash:
        raise Sam2GoldenEvidenceError("packed mask payload does not match its receipt")
    masks = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="little",
        count=math.prod(MASK_SHAPE),
    ).reshape(MASK_SHAPE)
    masks = np.ascontiguousarray(masks, dtype=np.uint8)
    if _digest(masks.tobytes()) != logical_hash:
        raise Sam2GoldenEvidenceError("logical mask payload does not match its receipt")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != FRAME_COUNT:
        raise Sam2GoldenEvidenceError("evidence requires exactly five frame records")
    expected_frames = [
        {
            "frame_index": index,
            "object_ids": [0],
            "foreground_pixels": int(frame.sum(dtype=np.int64)),
            "logical_uint8_sha256": _digest(frame.tobytes()),
        }
        for index, frame in zip(FRAME_INDICES, masks)
    ]
    if frames != expected_frames:
        raise Sam2GoldenEvidenceError("frame mask hash or metadata does not match payload")

    identity = {
        "workload": manifest["workload"],
        "frame_zero_bbox": manifest["frame_zero_bbox"],
        "frames": frames,
        "masks": manifest["masks"],
    }
    capture_hash = _checked_hash(manifest.get("capture_sha256"), "capture")
    if _canonical_digest(identity) != capture_hash:
        raise Sam2GoldenEvidenceError("capture identity hash does not match manifest")

    determinism = manifest.get("determinism")
    qualification = manifest.get("qualification")
    producer = manifest.get("producer")
    if producer not in {_REFERENCE_PRODUCER, _CANDIDATE_PRODUCER}:
        raise Sam2GoldenEvidenceError("evidence producer is invalid")
    if not isinstance(determinism, Mapping) or not isinstance(qualification, Mapping):
        raise Sam2GoldenEvidenceError("qualification or determinism evidence is missing")
    run_count = determinism.get("run_count")
    hashes = determinism.get("capture_sha256")
    replays = determinism.get("replays")
    if (
        not isinstance(run_count, int)
        or isinstance(run_count, bool)
        or run_count < 1
        or not isinstance(hashes, list)
        or len(hashes) != run_count
        or hashes[0] != capture_hash
        or not isinstance(replays, list)
        or len(replays) != run_count - 1
    ):
        raise Sam2GoldenEvidenceError("determinism run accounting is invalid")
    for replay_hash in hashes:
        _checked_hash(replay_hash, "determinism capture")
    all_exact = all(isinstance(item, Mapping) and item.get("exact") is True for item in replays)
    if determinism.get("all_exact") is not all_exact:
        raise Sam2GoldenEvidenceError("determinism result is inconsistent")
    if all_exact and any(item != capture_hash for item in hashes):
        raise Sam2GoldenEvidenceError("deterministic replay hashes differ")
    if qualification.get("status") == "authoritative_reference_candidate":
        errors = _authority_errors(provenance)
        if producer != _REFERENCE_PRODUCER or run_count < 3 or not all_exact or errors:
            raise Sam2GoldenEvidenceError(
                "authoritative reference candidate evidence is incomplete"
            )
    elif qualification.get("status") != "unqualified":
        raise Sam2GoldenEvidenceError("qualification status is invalid")
    authoritative_reference = bool(
        qualification.get("status") == "authoritative_reference_candidate"
        and _is_reviewed_sha256(AUTHORITATIVE_REFERENCE_MANIFEST_SHA256)
        and manifest_sha256 == AUTHORITATIVE_REFERENCE_MANIFEST_SHA256
    )
    return LoadedEvidence(
        root,
        manifest,
        masks,
        manifest_sha256,
        authoritative_reference,
    )


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _mask_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, int, int]:
    intersection = int(np.logical_and(left, right).sum(dtype=np.int64))
    union = int(np.logical_or(left, right).sum(dtype=np.int64))
    return (1.0 if union == 0 else intersection / union), intersection, union


def compare_evidence(
    reference: str | Path,
    candidate: str | Path,
) -> dict[str, object]:
    """Reload and apply fixed gates; a pass does not qualify runtime or timing.

    Paths, rather than mutable ``LoadedEvidence`` objects, are required so the
    hashes and bit-packed payload are verified immediately before comparison.
    """

    ref = load_evidence(reference)
    cand = load_evidence(candidate)
    ref_bbox = ref.manifest["frame_zero_bbox"]
    cand_bbox = cand.manifest["frame_zero_bbox"]
    original_left, original_right = (
        ref_bbox["original_image_xyxy"],
        cand_bbox["original_image_xyxy"],
    )
    model_left, model_right = (
        ref_bbox["model_image_xyxy_1024"],
        cand_bbox["model_image_xyxy_1024"],
    )
    original_iou = _box_iou(original_left, original_right)
    model_iou = _box_iou(model_left, model_right)
    original_error = max(abs(a - b) for a, b in zip(original_left, original_right))
    model_error = max(abs(a - b) for a, b in zip(model_left, model_right))
    score_error = abs(ref_bbox["score"] - cand_bbox["score"])

    frames = []
    for index, (left, right) in enumerate(zip(ref.masks, cand.masks)):
        iou, intersection, union = _mask_metrics(left, right)
        frames.append(
            {
                "frame_index": index,
                "iou": iou,
                "intersection": intersection,
                "union": union,
                "reference_foreground_pixels": int(left.sum(dtype=np.int64)),
                "candidate_foreground_pixels": int(right.sum(dtype=np.int64)),
                "passed": iou >= MIN_FRAME_MASK_IOU,
            }
        )
    minimum_iou = min(frame["iou"] for frame in frames)
    macro_iou = sum(frame["iou"] for frame in frames) / FRAME_COUNT
    global_intersection = sum(frame["intersection"] for frame in frames)
    global_union = sum(frame["union"] for frame in frames)
    global_iou = 1.0 if global_union == 0 else global_intersection / global_union
    ref_provenance, cand_provenance = ref.manifest["provenance"], cand.manifest["provenance"]
    reference_authoritative = ref.authoritative_reference
    gates = {
        "reference_authoritative": reference_authoritative,
        "candidate_producer": cand.manifest["producer"] == _CANDIDATE_PRODUCER,
        "asset_identity_exact": all(
            ref_provenance[key] == cand_provenance[key]
            for key in ("checkpoint_sha256", "config_sha256", "image_files_sha256")
        ),
        "post_nms_detection_count_exact": True,  # enforced by both loaders
        "selected_object_identity_exact": True,  # enforced by both loaders
        "label_exact": ref_bbox["label"] == cand_bbox["label"],
        "original_box_iou": original_iou >= MIN_BOX_IOU,
        "model_box_iou": model_iou >= MIN_BOX_IOU,
        "original_box_coordinate_error": original_error <= MAX_BOX_COORDINATE_ABS_ERROR,
        "model_box_coordinate_error": model_error <= MAX_BOX_COORDINATE_ABS_ERROR,
        "score_error": score_error <= MAX_SCORE_ABS_ERROR,
        "minimum_frame_mask_iou": minimum_iou >= MIN_FRAME_MASK_IOU,
        "macro_mask_iou": macro_iou >= MIN_MACRO_MASK_IOU,
        "global_mask_iou": global_iou >= MIN_GLOBAL_MASK_IOU,
    }
    passed = all(gates.values())
    if not reference_authoritative:
        status, reason = "unqualified", "authoritative_compatible_source_reference_required"
    elif passed:
        status, reason = "accuracy_parity_passed", "all_exact_workload_accuracy_gates_passed"
    else:
        status, reason = (
            "accuracy_parity_rejected",
            "one_or_more_exact_workload_accuracy_gates_failed",
        )
    result = {
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "passed": passed,
        "runtime_qualified": False,
        "runtime_qualification_reason": "bundle_runtime_and_timing_qualification_are_out_of_scope",
        "reference_capture_sha256": ref.manifest["capture_sha256"],
        "candidate_capture_sha256": cand.manifest["capture_sha256"],
        "policy": {
            "minimum_box_iou": MIN_BOX_IOU,
            "maximum_box_coordinate_abs_error": MAX_BOX_COORDINATE_ABS_ERROR,
            "maximum_score_abs_error": MAX_SCORE_ABS_ERROR,
            "minimum_frame_mask_iou": MIN_FRAME_MASK_IOU,
            "minimum_macro_mask_iou": MIN_MACRO_MASK_IOU,
            "minimum_global_mask_iou": MIN_GLOBAL_MASK_IOU,
        },
        "gates": gates,
        "metrics": {
            "frame_zero_bbox": {
                "original_image_iou": original_iou,
                "model_image_iou": model_iou,
                "original_image_max_abs_error": original_error,
                "model_image_max_abs_error": model_error,
                "score_abs_error": score_error,
            },
            "masks": {
                "minimum_frame_iou": minimum_iou,
                "macro_iou": macro_iou,
                "global_iou": global_iou,
                "global_intersection": global_intersection,
                "global_union": global_union,
                "frames": frames,
            },
        },
    }
    result["comparison_sha256"] = _canonical_digest(result)
    return result


__all__ = [
    "ARTIFACT_TYPE",
    "AUTHORITATIVE_ARCHIVE_CONTRACT_SHA256",
    "AUTHORITATIVE_CAPTURE_TOOL_SHA256",
    "AUTHORITATIVE_GOLDEN_EVIDENCE_NORMALIZED_SHA256",
    "AUTHORITATIVE_REFERENCE_MANIFEST_SHA256",
    "BINARY_MASK_THRESHOLD",
    "COMPARISON_ARTIFACT_TYPE",
    "COMPATIBLE_SOURCE_COMMIT",
    "COMPATIBLE_SOURCE_FILES_SHA256",
    "COMPATIBLE_SOURCE_OVERLAY_FILES_SHA256",
    "FRAME_COUNT",
    "FRAME_INDICES",
    "FrameZeroBBox",
    "INPUT_IMAGES_SHA256",
    "INPUT_IMAGES_DECODED_RGB_UINT8_SHA256",
    "INPUT_IMAGES_RESIZED_1024_RGB_UINT8_SHA256",
    "LoadedEvidence",
    "MASK_SHAPE",
    "PUBLIC_SAM2_BASE_COMMIT",
    "PUBLIC_SAM2_BASE_FILES_SHA256",
    "Provenance",
    "Sam2GoldenEvidenceError",
    "WorkloadCapture",
    "compare_captures_exact",
    "compare_evidence",
    "load_evidence",
    "repository_qualification_state",
    "write_evidence",
]
