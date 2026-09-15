# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read the complete TensorRT SDK version using the compiler, including aliased macros.
# include_dir: native SDK include directory; output: caller variable receiving x.y.z.build.
function(_edgellm_trt_version include_dir output)
  set(_version)
  set(_probe "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/edgellm-version.cpp")
  file(WRITE "${_probe}" "#include <NvInferVersion.h>\n")
  foreach(_part IN ITEMS MAJOR MINOR PATCH BUILD)
    file(APPEND "${_probe}" "TRTMC_EDGE_${_part}=NV_TENSORRT_${_part}\n")
  endforeach()
  execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -E -P -I "${include_dir}" "${_probe}"
    OUTPUT_VARIABLE _expanded COMMAND_ERROR_IS_FATAL ANY)
  foreach(_part IN ITEMS MAJOR MINOR PATCH BUILD)
    if(NOT _expanded MATCHES "TRTMC_EDGE_${_part}=[ \t]*([0-9]+)")
      message(FATAL_ERROR "Cannot determine TensorRT ${_part} from ${include_dir}")
    endif()
    list(APPEND _version "${CMAKE_MATCH_1}")
  endforeach()
  list(JOIN _version "." _version)
  set(${output} "${_version}" PARENT_SCOPE)
endfunction()

# Require the installed package GPU architecture to be present on this build host.
function(_edgellm_check_gpu architecture)
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _sms RESULT_VARIABLE _result OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "." "" _sms "${_sms}")
  string(REPLACE "\n" ";" _sms "${_sms}")
  if(NOT _result EQUAL 0 OR NOT architecture IN_LIST _sms)
    message(FATAL_ERROR "EdgeLLM requires a local GPU with architecture ${architecture}")
  endif()
endfunction()
