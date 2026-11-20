# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Optional native dependency provisioning. Model builds never acquire dependencies.
if(CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "Edge-LLM cross compilation is not supported")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../CheckNative.cmake")

# Model Connect and the offload must link the same native TensorRT installation.
function(_edgellm_check_trt_selection include_dir library)
  foreach(_kind IN ITEMS INCLUDE_DIR LIBRARY)
    get_filename_component(_parent "${TRTMC_TRT_${_kind}}" REALPATH)
    if(_kind STREQUAL "INCLUDE_DIR")
      get_filename_component(_selected "${include_dir}" REALPATH)
    else()
      get_filename_component(_selected "${library}" REALPATH)
    endif()
    if(NOT _parent STREQUAL _selected)
      message(FATAL_ERROR "Model Connect and EdgeLLM must use the same TensorRT ${_kind}: ${_parent} != ${_selected}. Set TRTMC_TRT_INCLUDE_DIR, TRTMC_TRT_LIBRARY and TRTMC_EDGELLM_TRT_ROOT to one SDK.")
    endif()
  endforeach()
  _edgellm_trt_version("${TRTMC_TRT_INCLUDE_DIR}" _parent_version)
  _edgellm_trt_version("${include_dir}" _selected_version)
  if(NOT _parent_version STREQUAL _selected_version)
    message(FATAL_ERROR "Model Connect and EdgeLLM TensorRT header versions differ")
  endif()
endfunction()
option(TRTMC_EDGELLM_ALL_KERNELS "Provision all upstream kernel groups supported by the local GPU" OFF)
option(TRTMC_EDGELLM_ONNX "Install upstream ONNX builder executables for an opted-in family" OFF)
set(TRTMC_EDGELLM_TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "Matching native TensorRT SDK")
set(TRTMC_EDGELLM_CUDA_ARCHITECTURE "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "One local GPU architecture")
set(TRTMC_EDGELLM_JOBS 2 CACHE STRING "Parallel jobs for the optional native SDK build")
set(TRTMC_EDGELLM_GIT_MIRROR "" CACHE PATH "Optional local mirror of the official pinned repository")
set(TRTMC_EDGELLM_SOURCE_DIR "" CACHE PATH "Optional populated, pinned checkout including its submodules")
set(_edge_version "0.12.0")
set(_edge_revision "513dcd18291e587c80928911a480761352f3c2d8")
if(NOT TRTMC_EDGELLM_CUDA_ARCHITECTURE MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Set TRTMC_EDGELLM_CUDA_ARCHITECTURE to one local GPU architecture")
endif()
_edgellm_check_gpu("${TRTMC_EDGELLM_CUDA_ARCHITECTURE}")

# The SDK's relocatable device code and its consumers must use the same
# compiler target. Keep the numeric GPU identity for checks and receipts.
set(EdgeLLM_CUDA_ARCHITECTURE "${TRTMC_EDGELLM_CUDA_ARCHITECTURE}")
# Match native detection in the pinned SDK's cmake/EdgeLLMPlatform.cmake.
if(EdgeLLM_CUDA_ARCHITECTURE MATCHES "^(100|101|103|110|121)$")
  string(APPEND EdgeLLM_CUDA_ARCHITECTURE "a")
endif()
_edgellm_trt_version("${TRTMC_TRT_INCLUDE_DIR}" _edge_trt_version)
_edgellm_check_trt_library("${TRTMC_TRT_LIBRARY}" "${_edge_trt_version}")
if(TRTMC_EDGELLM_TRT_ROOT AND NOT TENSORRT_ROOT)
  set(TENSORRT_ROOT "${TRTMC_EDGELLM_TRT_ROOT}")
endif()

# Never reuse a private prefix selected for an earlier ABI or kernel policy.
if(TRTMC_EDGELLM_PRIVATE_CONFIG AND TensorRTEdgeLLM_DIR STREQUAL TRTMC_EDGELLM_PRIVATE_CONFIG)
  unset(TensorRTEdgeLLM_DIR CACHE)
  unset(TensorRTEdgeLLM_DIR)
endif()
set(_edge_provisioned OFF)
if(NOT TRTMC_EDGELLM_SOURCE_DIR)
  find_package(TensorRTEdgeLLM ${_edge_version} EXACT CONFIG QUIET)
endif()
if(NOT TensorRTEdgeLLM_FOUND OR TRTMC_EDGELLM_SOURCE_DIR)
  if(NOT EXISTS "${TRTMC_EDGELLM_TRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "Provisioning Edge-LLM requires TRTMC_EDGELLM_TRT_ROOT pointing to a native TensorRT SDK")
  endif()
  set(_edge_groups "fmha;gdn")
  # Preserve existing SDK prefixes when the kernel selection is unchanged.
  set(_edge_group_key "all${TRTMC_EDGELLM_ALL_KERNELS}")
  if(TRTMC_EDGELLM_ALL_KERNELS)
    set(_edge_groups ALL)
  elseif(TRTMC_EDGELLM_CUDA_ARCHITECTURE MATCHES "^(100|101|103|110)$")
    # The upstream attention plugin requires CTM for single-rank decode on these targets.
    list(APPEND _edge_groups fmha_decode_ctm)
    string(REPLACE ";" "-" _edge_group_key "${_edge_groups}")
  endif()
  set(_edge_root "${CMAKE_CURRENT_BINARY_DIR}/_deps/edgellm/${_edge_version}-${CMAKE_SYSTEM_PROCESSOR}-sm${TRTMC_EDGELLM_CUDA_ARCHITECTURE}-cu${CUDAToolkit_VERSION}-trt${_edge_trt_version}-${_edge_group_key}")
  set(_edge_prefix "${_edge_root}/install")
  set(_edge_repository "https://github.com/NVIDIA/TensorRT-Edge-LLM.git")
  if(TRTMC_EDGELLM_GIT_MIRROR)
    set(_edge_repository "${TRTMC_EDGELLM_GIT_MIRROR}")
  endif()
  # Configure the SDK separately: upstream owns its top-level CUDA/C++ project,
  # submodules, prebuilt kernel selection, install, and transitive link targets.
  # A small bootstrap project gives find_package real, existing targets on
  # the first configure. ExternalProject stamps make subsequent runs no-ops.
  file(MAKE_DIRECTORY "${_edge_root}/driver")
  string(REPLACE ";" "|" _edge_groups_arg "${_edge_groups}")
  configure_file("${CMAKE_CURRENT_LIST_DIR}/Prepare.cmake.in" "${_edge_root}/driver/CMakeLists.txt" @ONLY)
  execute_process(COMMAND "${CMAKE_COMMAND}" -S "${_edge_root}/driver" -B "${_edge_root}/driver-build"
    -G "${CMAKE_GENERATOR}" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${CMAKE_COMMAND}" --build "${_edge_root}/driver-build" --config Release
    COMMAND_ERROR_IS_FATAL ANY)
  set(TensorRTEdgeLLM_DIR "${_edge_prefix}/lib/cmake/TensorRTEdgeLLM")
  find_package(TensorRTEdgeLLM ${_edge_version} EXACT CONFIG REQUIRED NO_DEFAULT_PATH)
  set(TRTMC_EDGELLM_PRIVATE_CONFIG "${TensorRTEdgeLLM_DIR}" CACHE INTERNAL "Managed Edge-LLM SDK config")
  set(_edge_provisioned ON)
  # The upstream bin directory includes benchmarks and model-specific examples.
  # No exported library target needs these; install only the requested tools below.
  foreach(_dir IN ITEMS lib include)
    install(DIRECTORY "${_edge_prefix}/${_dir}/" DESTINATION "${_dir}"
      USE_SOURCE_PERMISSIONS COMPONENT EdgeLLM)
  endforeach()
elseif(TRTMC_EDGELLM_ALL_KERNELS)
  message(FATAL_ERROR "The external SDK does not export its compiled kernel groups. Use TRTMC_EDGELLM_SOURCE_DIR to provision ALL from the pinned source rather than labeling an unverified package.")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/InstalledSDK.cmake")
