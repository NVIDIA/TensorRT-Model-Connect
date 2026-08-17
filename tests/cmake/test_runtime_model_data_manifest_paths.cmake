# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED TRTMC_SOURCE_DIR OR TRTMC_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "TRTMC_SOURCE_DIR is required")
endif()

include("${TRTMC_SOURCE_DIR}/cmake/trtmc_model_data.cmake")

foreach(_path IN ITEMS
    "record.json"
    "authority-0001/record.json"
    "sam2-l4-trt11.1-contract5-0001.qualification-audit.json")
  _trtmc_model_data_relative_path_is_safe("${_path}" _is_safe)
  if(NOT _is_safe)
    message(FATAL_ERROR "safe runtime model data path was rejected: '${_path}'")
  endif()
endforeach()

foreach(_path IN ITEMS
    ""
    "/absolute.json"
    "../record.json"
    "authority/../record.json"
    "./record.json"
    ".hidden"
    "authority//record.json"
    "authority\\record.json"
    "authority;record.json")
  _trtmc_model_data_relative_path_is_safe("${_path}" _is_safe)
  if(_is_safe)
    message(FATAL_ERROR "unsafe runtime model data path was accepted: '${_path}'")
  endif()
endforeach()

set(_manifest [=[
runtime_optional_data_files = [
  "authority-0001/record.json",
  "authority-0001/audit.json",
]
]=])
_trtmc_model_manifest_safe_relative_path_list(
  "${_manifest}" "runtime_optional_data_files" "fixture/MODEL.toml" _parsed)
set(_expected
  "authority-0001/record.json"
  "authority-0001/audit.json")
if(NOT _parsed STREQUAL _expected)
  message(FATAL_ERROR
    "runtime model data manifest list parsed incorrectly: '${_parsed}' != '${_expected}'")
endif()
