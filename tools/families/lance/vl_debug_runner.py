# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance-owned VL debug entrypoints used by the shared diff dispatcher."""

from importlib import import_module

_family_runner = import_module(
    "tensorrt_model_connect.families.lance.vl_debug_runner"
)
TrtRunner = _family_runner.TrtRunner
VLTrtRunner = _family_runner.VLTrtRunner
VisionTrtRunner = _family_runner.VisionTrtRunner
load_config_from_bundle = _family_runner.load_config_from_bundle
load_engine_from_bundle = _family_runner.load_engine_from_bundle
load_preprocessor_config_from_bundle = (
    _family_runner.load_preprocessor_config_from_bundle
)
load_section_from_bundle = _family_runner.load_section_from_bundle
load_vision_engine_from_bundle = _family_runner.load_vision_engine_from_bundle
preprocess_image_inputs_for_trt = _family_runner.preprocess_image_inputs_for_trt

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
