# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose InternVL's family-owned debug runner to shared development tools."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_FAMILY_MODULE = "tensorrt_model_connect.families.internvl.vl_debug_runner"
_FAMILY_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/tensorrt_model_connect/families/internvl/vl_debug_runner.py"
)
_SPEC = spec_from_file_location(_FAMILY_MODULE, _FAMILY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load InternVL debug runner from {_FAMILY_PATH}")
_RUNNER = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUNNER)

TrtRunner = _RUNNER.TrtRunner
VLTrtRunner = _RUNNER.VLTrtRunner
VisionTrtRunner = _RUNNER.VisionTrtRunner
load_config_from_bundle = _RUNNER.load_config_from_bundle
load_engine_from_bundle = _RUNNER.load_engine_from_bundle
load_preprocessor_config_from_bundle = _RUNNER.load_preprocessor_config_from_bundle
load_section_from_bundle = _RUNNER.load_section_from_bundle
load_vision_engine_from_bundle = _RUNNER.load_vision_engine_from_bundle
preprocess_image_inputs_for_trt = _RUNNER.preprocess_image_inputs_for_trt

__all__ = [
    "TrtRunner",
    "VLTrtRunner",
    "VisionTrtRunner",
    "load_config_from_bundle",
    "load_engine_from_bundle",
    "load_preprocessor_config_from_bundle",
    "load_section_from_bundle",
    "load_vision_engine_from_bundle",
    "preprocess_image_inputs_for_trt",
]
