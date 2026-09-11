# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Optional native dependency provisioning. Model builds never acquire dependencies.
option(TRTMC_ENABLE_EDGELLM "Install the pinned native Edge-LLM SDK and builder" OFF)
if(NOT TRTMC_ENABLE_EDGELLM)
  return()
endif()
if(CMAKE_CROSSCOMPILING)
  message(FATAL_ERROR "Edge-LLM cross compilation is not supported")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/edgellm/CheckNative.cmake")
set(_edge_version "0.10.1")
set(_edge_revision "e8b29522938901f6df19ebeedd4b69bc8edbcd97")
set(_edge_root "${CMAKE_BINARY_DIR}/_deps/edgellm")
set(_edge_prefix "${_edge_root}/install")
find_package(EdgeLLM ${_edge_version} EXACT CONFIG QUIET)
if(EdgeLLM_FOUND AND NOT EdgeLLM_PREFIX STREQUAL _edge_prefix)
  if(NOT EdgeLLM_REVISION STREQUAL _edge_revision)
    message(FATAL_ERROR "EdgeLLM package does not match the pinned GitHub revision")
  endif()
  install(FILES "$<TARGET_FILE:EdgeLLM::Plugin>" DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT EdgeLLM)
  return()
endif()

include(ExternalProject)
include(CMakePackageConfigHelpers)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
find_package(Threads REQUIRED)
set(TRTMC_EDGELLM_TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "Native TensorRT SDK, including its Python wheel")
set(TRTMC_EDGELLM_CUDA_ARCHITECTURE "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "One local GPU architecture for Edge-LLM")
set(TRTMC_EDGELLM_JOBS 2 CACHE STRING "Parallel Edge-LLM native and AOT compilation jobs")
set(TRTMC_EDGELLM_WHEELHOUSE "" CACHE PATH "Optional complete offline Python wheelhouse")
set(TRTMC_EDGELLM_GIT_MIRROR "" CACHE PATH "Optional local mirror of the pinned upstream Git repository")
if(NOT TRTMC_EDGELLM_CUDA_ARCHITECTURE MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Set TRTMC_EDGELLM_CUDA_ARCHITECTURE to one local GPU architecture, e.g. 80")
endif()
if(NOT EXISTS "${TRTMC_EDGELLM_TRT_ROOT}/include/NvInfer.h")
  message(FATAL_ERROR "TRTMC_EDGELLM_TRT_ROOT must contain the native TensorRT SDK")
endif()
_edgellm_check_gpu("${TRTMC_EDGELLM_CUDA_ARCHITECTURE}")
_edgellm_trt_version("${TRTMC_EDGELLM_TRT_ROOT}/include" _edge_trt_version)
set(_edge_source "${_edge_root}/source")
set(_edge_build "${_edge_root}/build")
set(_edge_python "${_edge_prefix}/libexec/trtmc-edge-llm/bin/python")
set(_edge_repository "https://github.com/NVIDIA/TensorRT-Edge-LLM.git")
if(TRTMC_EDGELLM_GIT_MIRROR)
  set(_edge_repository "${TRTMC_EDGELLM_GIT_MIRROR}")
endif()
set(_edge_template_dir "${CMAKE_CURRENT_LIST_DIR}/edgellm")
file(MAKE_DIRECTORY "${_edge_prefix}/lib/cmake/EdgeLLM" "${_edge_prefix}/include/edgellm/cpp"
  "${_edge_prefix}/include/edgellm/3rdParty/nlohmannJson/include"
  "${_edge_prefix}/include/edgellm/3rdParty/stb" "${_edge_prefix}/include/edgellm/3rdParty/miniaudio")
configure_file("${_edge_template_dir}/CheckNative.cmake" "${_edge_prefix}/lib/cmake/EdgeLLM/CheckNative.cmake" COPYONLY)
foreach(_script IN ITEMS Prepare Install)
  configure_file("${_edge_template_dir}/${_script}.cmake.in" "${_edge_root}/${_script}.cmake" @ONLY)
endforeach()
configure_file("${_edge_template_dir}/EdgeLLMConfig.cmake.in"
  "${_edge_prefix}/lib/cmake/EdgeLLM/EdgeLLMConfig.cmake" @ONLY)
write_basic_package_version_file("${_edge_prefix}/lib/cmake/EdgeLLM/EdgeLLMConfigVersion.cmake"
  VERSION "${_edge_version}" COMPATIBILITY ExactVersion)
ExternalProject_Add(trtmc_edgellm_dependency
  PREFIX "${_edge_root}/ep" SOURCE_DIR "${_edge_source}" BINARY_DIR "${_edge_build}"
  GIT_REPOSITORY "${_edge_repository}" GIT_TAG "${_edge_revision}"
  GIT_SUBMODULES_RECURSE TRUE UPDATE_DISCONNECTED TRUE
  LIST_SEPARATOR |
  CMAKE_COMMAND "${_edge_prefix}/libexec/trtmc-edge-llm/bin/cmake"
  PATCH_COMMAND "${CMAKE_COMMAND}" -P "${_edge_root}/Prepare.cmake"
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
    "-DCMAKE_CUDA_ARCHITECTURES=${TRTMC_EDGELLM_CUDA_ARCHITECTURE}"
    "-DCUDA_DIR=${CUDAToolkit_LIBRARY_ROOT}" "-DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}"
    "-DCUDA_CTK_VERSION=${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}"
    "-DTRT_PACKAGE_DIR=${TRTMC_EDGELLM_TRT_ROOT}" "-DPython3_EXECUTABLE=${_edge_python}"
    -DEDGELLM_WHEEL_PAYLOAD_DIR=unused -DENABLE_CUTE_DSL=fmha|gdn
    "-DCUTE_DSL_ARTIFACT_TAG=sm_${TRTMC_EDGELLM_CUDA_ARCHITECTURE}"
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target edgellmCore NvInfer_edgellm_plugin
    --parallel "${TRTMC_EDGELLM_JOBS}"
  INSTALL_COMMAND "${CMAKE_COMMAND}" -P "${_edge_root}/Install.cmake"
  BUILD_BYPRODUCTS "${_edge_prefix}/lib/libedgellmCore.a"
    "${_edge_prefix}/lib/libNvInfer_edgellm_plugin.so"
    "${_edge_prefix}/lib/libcutedsl.a"
  LOG_DOWNLOAD ON LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
ExternalProject_Add_StepDependencies(trtmc_edgellm_dependency patch "${_edge_root}/Prepare.cmake")
ExternalProject_Add_StepDependencies(trtmc_edgellm_dependency install "${_edge_root}/Install.cmake")
# Generated package targets refer to declared future byproducts; their build dependency
# prevents consumers from compiling or linking until installation completes.
find_package(EdgeLLM ${_edge_version} EXACT CONFIG REQUIRED
  PATHS "${_edge_prefix}/lib/cmake/EdgeLLM" NO_DEFAULT_PATH)
add_dependencies(EdgeLLM::Core trtmc_edgellm_dependency)
add_dependencies(EdgeLLM::Plugin trtmc_edgellm_dependency)
install(DIRECTORY "${_edge_prefix}/" DESTINATION . USE_SOURCE_PERMISSIONS COMPONENT EdgeLLM)
# Family DSOs may use lib64; their dynamically loaded plugin must remain adjacent.
install(FILES "$<TARGET_FILE:EdgeLLM::Plugin>" DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT EdgeLLM)
