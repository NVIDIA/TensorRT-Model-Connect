# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(TRTMC_FAMILIES "" CACHE STRING
  "Semicolon-separated families to configure; empty configures all families")
if(TRTMC_FAMILIES STREQUAL "")
  file(GLOB _trtmc_family_runtime_cmake CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/families/*/runtime/CMakeLists.txt")
else()
  set(_trtmc_family_runtime_cmake "")
  foreach(_trtmc_family IN LISTS TRTMC_FAMILIES)
    set(_trtmc_runtime_cmake
      "${PROJECT_SOURCE_DIR}/families/${_trtmc_family}/runtime/CMakeLists.txt")
    if(NOT _trtmc_family MATCHES "^[a-z][a-z0-9_]*$"
       OR NOT EXISTS "${_trtmc_runtime_cmake}")
      message(FATAL_ERROR "Unknown build family: '${_trtmc_family}'")
    endif()
    list(APPEND _trtmc_family_runtime_cmake "${_trtmc_runtime_cmake}")
  endforeach()
  list(REMOVE_DUPLICATES _trtmc_family_runtime_cmake)
endif()

foreach(_trtmc_runtime_cmake IN LISTS _trtmc_family_runtime_cmake)
  get_filename_component(_trtmc_runtime_dir "${_trtmc_runtime_cmake}" DIRECTORY)
  get_filename_component(_trtmc_family_dir "${_trtmc_runtime_dir}" DIRECTORY)
  get_filename_component(_trtmc_family "${_trtmc_family_dir}" NAME)
  add_subdirectory(
    "${_trtmc_runtime_dir}"
    "${CMAKE_BINARY_DIR}/families/${_trtmc_family}")
endforeach()
