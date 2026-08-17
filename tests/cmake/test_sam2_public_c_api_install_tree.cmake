# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS TRTMC_BUILD_DIR TRTMC_SOURCE_DIR TRTMC_INSTALL_INCLUDEDIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

set(_install_prefix "${TRTMC_BUILD_DIR}/tests/sam2-public-c-api-install")
set(_object "${TRTMC_BUILD_DIR}/tests/sam2-public-c-api-external.o")
file(REMOVE_RECURSE "${_install_prefix}")
file(REMOVE "${_object}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${TRTMC_BUILD_DIR}"
          --prefix "${_install_prefix}" --component sam2_public_api_development
  RESULT_VARIABLE _install_status
  OUTPUT_VARIABLE _install_stdout
  ERROR_VARIABLE _install_stderr
)
if(NOT _install_status EQUAL 0)
  message(FATAL_ERROR
    "SAM2 public-header install failed (${_install_status})\n"
    "stdout:\n${_install_stdout}\nstderr:\n${_install_stderr}")
endif()

set(_include_dir "${_install_prefix}/${TRTMC_INSTALL_INCLUDEDIR}")
set(_public_header "${_include_dir}/trtmc/models/sam2_video.h")
if(NOT EXISTS "${_public_header}")
  message(FATAL_ERROR "installed SAM2 public C ABI header is missing: ${_public_header}")
endif()

find_program(_c_compiler NAMES cc gcc clang REQUIRED)
execute_process(
  COMMAND "${_c_compiler}" -std=c11 -Wall -Wextra -Wpedantic -Werror
          "-I${_include_dir}" -c
          "${TRTMC_SOURCE_DIR}/tests/cpp/models/sam2/test_sam2_public_c_api_external.c"
          -o "${_object}"
  RESULT_VARIABLE _compile_status
  OUTPUT_VARIABLE _compile_stdout
  ERROR_VARIABLE _compile_stderr
)
if(NOT _compile_status EQUAL 0)
  message(FATAL_ERROR
    "external SAM2 public C ABI consumer failed to compile (${_compile_status})\n"
    "stdout:\n${_compile_stdout}\nstderr:\n${_compile_stderr}")
endif()
