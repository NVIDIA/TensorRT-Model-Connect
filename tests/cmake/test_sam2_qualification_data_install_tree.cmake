# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS TRTMC_BUILD_DIR TRTMC_SOURCE_DIR TRTMC_INSTALL_DATADIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

set(_record "sam2-l4-trt11.1-contract5-0001.qualification-record.json")
set(_audit "sam2-l4-trt11.1-contract5-0001.qualification-audit.json")
set(_expected ${_record} ${_audit})
set(_source_dir "${TRTMC_SOURCE_DIR}/src/runtime/models/sam2")
set(_install_prefix "${TRTMC_BUILD_DIR}/tests/sam2-qualification-data-install")
set(_install_dir
  "${_install_prefix}/${TRTMC_INSTALL_DATADIR}/trtmc/model_data/sam2")
file(REMOVE_RECURSE "${_install_prefix}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${TRTMC_BUILD_DIR}"
          --prefix "${_install_prefix}" --component trtmc_model_data
  RESULT_VARIABLE _install_status
  OUTPUT_VARIABLE _install_stdout
  ERROR_VARIABLE _install_stderr
)
if(NOT _install_status EQUAL 0)
  message(FATAL_ERROR
    "optional SAM2 qualification-data install failed (${_install_status})\n"
    "stdout:\n${_install_stdout}\nstderr:\n${_install_stderr}")
endif()

foreach(_name IN LISTS _expected)
  set(_source "${_source_dir}/${_name}")
  set(_installed "${_install_dir}/${_name}")
  if(EXISTS "${_source}")
    if(NOT EXISTS "${_installed}")
      message(FATAL_ERROR "declared SAM2 qualification data was not installed: ${_name}")
    endif()
    file(SHA256 "${_source}" _source_sha256)
    file(SHA256 "${_installed}" _installed_sha256)
    if(NOT _source_sha256 STREQUAL _installed_sha256)
      message(FATAL_ERROR "installed SAM2 qualification data changed: ${_name}")
    endif()
  elseif(EXISTS "${_installed}")
    message(FATAL_ERROR "missing optional SAM2 qualification data was synthesized: ${_name}")
  endif()
endforeach()

file(GLOB_RECURSE _installed_files
  RELATIVE "${_install_dir}" "${_install_dir}/*")
foreach(_installed IN LISTS _installed_files)
  if(IS_DIRECTORY "${_install_dir}/${_installed}")
    continue()
  endif()
  list(FIND _expected "${_installed}" _expected_index)
  if(_expected_index EQUAL -1)
    message(FATAL_ERROR "undeclared SAM2 model data was installed: ${_installed}")
  endif()
endforeach()
