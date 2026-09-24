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
include("${CMAKE_CURRENT_LIST_DIR}/CheckNative.cmake")

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
option(TRTMC_EDGELLM_ALL_KERNELS "Build all pinned Edge operator groups supported by the native GPU" OFF)
option(TRTMC_EDGELLM_ONNX "Install the pinned ONNX exporter and native engine builder" OFF)
set(_edge_cute_groups "fmha|gdn")
set(_edge_cute_cli_groups "fmha,gdn")
if(TRTMC_EDGELLM_ALL_KERNELS)
  set(_edge_cute_groups ALL)
  set(_edge_cute_cli_groups ALL)
endif()
set(_edge_build_targets edgellmCore NvInfer_edgellm_plugin)
set(_edge_onnx_byproducts "")
if(TRTMC_EDGELLM_ONNX)
  list(APPEND _edge_build_targets llm_build)
  list(APPEND _edge_onnx_byproducts "${CMAKE_BINARY_DIR}/_deps/edgellm/install/bin/edgellm-onnx-build")
endif()
set(_edge_version "0.10.1")
set(_edge_revision "e8b29522938901f6df19ebeedd4b69bc8edbcd97")
set(_edge_root "${CMAKE_BINARY_DIR}/_deps/edgellm")
set(_edge_prefix "${_edge_root}/install")
set(TRTMC_EDGELLM_CUDA_ARCHITECTURE "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "One local GPU architecture for Edge-LLM")
if(NOT TRTMC_EDGELLM_CUDA_ARCHITECTURE MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Set TRTMC_EDGELLM_CUDA_ARCHITECTURE to one local GPU architecture, e.g. 80")
endif()
# Do not import this build tree's previous generated package before regenerating
# it: its baked SDK checks and imported targets may describe the old configure.
set(_edge_package_dir "${_edge_prefix}/lib/cmake/EdgeLLM")
get_filename_component(_edge_package_real "${_edge_package_dir}" REALPATH)
if(EdgeLLM_DIR)
  get_filename_component(_edge_cached_real "${EdgeLLM_DIR}" REALPATH)
  if(_edge_cached_real STREQUAL _edge_package_real)
    unset(EdgeLLM_DIR CACHE)
    unset(EdgeLLM_DIR)
  endif()
endif()
set(_edge_saved_ignore_path "${CMAKE_IGNORE_PATH}")
list(APPEND CMAKE_IGNORE_PATH "${_edge_package_dir}" "${_edge_package_real}")
find_package(EdgeLLM ${_edge_version} EXACT CONFIG QUIET)
set(CMAKE_IGNORE_PATH "${_edge_saved_ignore_path}")

function(_edgellm_install_plugin)
  # Preserve the complete SONAME chain when lib and lib64 differ. Install-time
  # expansion also honors cmake --install --prefix and DESTDIR.
  install(CODE "file(INSTALL
    DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\"
    TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN
    FILES \"$<TARGET_FILE:EdgeLLM::Plugin>\")" COMPONENT EdgeLLM)
endfunction()
function(_edgellm_check_external_artifacts)
  foreach(_tool IN ITEMS EdgeLLM_PYTHON_EXECUTABLE EdgeLLM_BUILDER_LAUNCHER)
    if(NOT EXISTS "${${_tool}}" OR IS_DIRECTORY "${${_tool}}")
      message(FATAL_ERROR "External EdgeLLM package is incomplete: ${_tool} missing at ${${_tool}}")
    endif()
  endforeach()
  foreach(_target IN ITEMS EdgeLLM::Core EdgeLLM::Plugin)
    get_target_property(_artifact ${_target} IMPORTED_LOCATION)
    if(NOT EXISTS "${_artifact}" OR IS_DIRECTORY "${_artifact}")
      message(FATAL_ERROR "External EdgeLLM package is incomplete: ${_target} missing at ${_artifact}")
    endif()
  endforeach()
  if(NOT EXISTS "${EdgeLLM_PREFIX}/lib/libcutedsl.a" OR IS_DIRECTORY "${EdgeLLM_PREFIX}/lib/libcutedsl.a")
    message(FATAL_ERROR "External EdgeLLM package is incomplete: missing libcutedsl.a")
  endif()
endfunction()

if(EdgeLLM_FOUND AND NOT EdgeLLM_PREFIX STREQUAL _edge_prefix)
  _edgellm_check_external_artifacts()
  _edgellm_check_trt_selection("${EdgeLLM_TRT_INCLUDE_DIR}" "${EdgeLLM_TRT_LIBRARY}")
  if(NOT EdgeLLM_CUDA_ARCHITECTURE STREQUAL TRTMC_EDGELLM_CUDA_ARCHITECTURE)
    message(FATAL_ERROR "EdgeLLM package architecture ${EdgeLLM_CUDA_ARCHITECTURE} differs from requested ${TRTMC_EDGELLM_CUDA_ARCHITECTURE}")
  endif()
  _edgellm_check_trt_library("${EdgeLLM_TRT_LIBRARY}" "${EdgeLLM_TENSORRT_VERSION}")
  _edgellm_json_include(_edge_json_include)
  _edgellm_check_json_headers("${_edge_json_include}" "${EdgeLLM_PREFIX}/include/edgellm/3rdParty/nlohmannJson")
  if(NOT EdgeLLM_REVISION STREQUAL _edge_revision)
    message(FATAL_ERROR "EdgeLLM package does not match the pinned GitHub revision")
  endif()
  if(TRTMC_EDGELLM_ALL_KERNELS AND NOT EdgeLLM_ALL_KERNELS)
    message(FATAL_ERROR "EdgeLLM package lacks requested full native operator coverage; rebuild with TRTMC_EDGELLM_ALL_KERNELS=ON")
  endif()
  if(TRTMC_EDGELLM_ONNX AND (NOT EdgeLLM_ONNX OR NOT EXISTS "${EdgeLLM_ONNX_BUILDER}"))
    message(FATAL_ERROR "EdgeLLM package lacks requested ONNX tools; rebuild with TRTMC_EDGELLM_ONNX=ON")
  endif()
  _edgellm_install_plugin()
  return()
endif()

include(ExternalProject)
include(CMakePackageConfigHelpers)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
if(CUDAToolkit_VERSION_MAJOR EQUAL 12 AND Python3_VERSION VERSION_GREATER_EQUAL "3.13")
  message(FATAL_ERROR "Pinned CUDA 12 CuPy kernels require Python 3.10-3.12; select Python3_EXECUTABLE accordingly")
endif()
if(Python3_VERSION VERSION_GREATER_EQUAL "3.14")
  message(FATAL_ERROR "Pinned EdgeLLM NumPy requires Python 3.10-3.13; select Python3_EXECUTABLE accordingly")
endif()
find_package(Threads REQUIRED)
set(TRTMC_EDGELLM_TRT_ROOT "$ENV{TRT_ROOT}" CACHE PATH "Native TensorRT SDK, including its Python wheel")
set(TRTMC_EDGELLM_JOBS 2 CACHE STRING "Parallel Edge-LLM native and AOT compilation jobs")
set(TRTMC_EDGELLM_WHEELHOUSE "" CACHE PATH "Optional complete offline Python wheelhouse")
set(TRTMC_EDGELLM_GIT_MIRROR "" CACHE PATH "Optional local mirror of the pinned upstream Git repository")
if(NOT EXISTS "${TRTMC_EDGELLM_TRT_ROOT}/include/NvInfer.h")
  message(FATAL_ERROR "TRTMC_EDGELLM_TRT_ROOT must contain the native TensorRT SDK")
endif()
unset(_edge_selected_trt_library CACHE)
unset(_edge_selected_trt_library)
find_library(_edge_selected_trt_library nvinfer
  PATHS "${TRTMC_EDGELLM_TRT_ROOT}/lib" "${TRTMC_EDGELLM_TRT_ROOT}/lib64"
  NO_DEFAULT_PATH REQUIRED)
_edgellm_check_trt_selection("${TRTMC_EDGELLM_TRT_ROOT}/include" "${_edge_selected_trt_library}")
_edgellm_check_gpu("${TRTMC_EDGELLM_CUDA_ARCHITECTURE}")
_edgellm_trt_version("${TRTMC_EDGELLM_TRT_ROOT}/include" _edge_trt_version)
set(_edge_source "${_edge_root}/source")
set(_edge_build "${_edge_root}/build")
set(_edge_python "${_edge_prefix}/libexec/trtmc-edge-llm/bin/python")
set(_edge_repository "https://github.com/NVIDIA/TensorRT-Edge-LLM.git")
if(TRTMC_EDGELLM_GIT_MIRROR)
  set(_edge_repository "${TRTMC_EDGELLM_GIT_MIRROR}")
endif()
set(_edge_template_dir "${CMAKE_CURRENT_LIST_DIR}")
_edgellm_json_include(_edge_json_include)
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
  # Preparation installs tools; it does not patch upstream sources. Keep it in
  # the configure step so template changes invalidate disconnected builds too.
  CONFIGURE_COMMAND "${CMAKE_COMMAND}" -P "${_edge_root}/Prepare.cmake"
    COMMAND "${_edge_prefix}/libexec/trtmc-edge-llm/bin/cmake"
    -S <SOURCE_DIR> -B <BINARY_DIR> -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
    "-DCMAKE_CUDA_ARCHITECTURES=${TRTMC_EDGELLM_CUDA_ARCHITECTURE}"
    "-DCUDA_DIR=${CUDAToolkit_LIBRARY_ROOT}" "-DCUDAToolkit_ROOT=${CUDAToolkit_LIBRARY_ROOT}"
    "-DCUDA_CTK_VERSION=${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}"
    "-DTRT_PACKAGE_DIR=${TRTMC_EDGELLM_TRT_ROOT}" "-DPython3_EXECUTABLE=${_edge_python}"
    -DEDGELLM_WHEEL_PAYLOAD_DIR=unused "-DENABLE_CUTE_DSL=${_edge_cute_groups}"
    "-DCUTE_DSL_ARTIFACT_TAG=sm_${TRTMC_EDGELLM_CUDA_ARCHITECTURE}"
  BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target ${_edge_build_targets}
    --parallel "${TRTMC_EDGELLM_JOBS}"
  INSTALL_COMMAND "${CMAKE_COMMAND}" -P "${_edge_root}/Install.cmake"
  BUILD_BYPRODUCTS "${_edge_prefix}/lib/libedgellmCore.a"
    "${_edge_prefix}/lib/libNvInfer_edgellm_plugin.so"
    "${_edge_prefix}/lib/libcutedsl.a" ${_edge_onnx_byproducts}
  LOG_DOWNLOAD ON LOG_CONFIGURE ON LOG_BUILD ON LOG_INSTALL ON LOG_OUTPUT_ON_FAILURE ON)
ExternalProject_Add_StepDependencies(trtmc_edgellm_dependency configure "${_edge_root}/Prepare.cmake")
ExternalProject_Add_StepDependencies(trtmc_edgellm_dependency install "${_edge_root}/Install.cmake")
# Generated package targets refer to declared future byproducts; their build dependency
# prevents consumers from compiling or linking until installation completes.
set(EdgeLLM_DIR "${_edge_package_dir}" CACHE PATH "Edge-LLM package directory" FORCE)
find_package(EdgeLLM ${_edge_version} EXACT CONFIG REQUIRED
  PATHS "${_edge_prefix}/lib/cmake/EdgeLLM" NO_DEFAULT_PATH)
add_dependencies(EdgeLLM::Core trtmc_edgellm_dependency)
add_dependencies(EdgeLLM::Plugin trtmc_edgellm_dependency)
# Runtime consumers use the interpreter/modules and the prefix-relative launcher.
# Build-only console scripts/activation files embed build-tree paths; keep them
# available for rebuilding the dependency, but do not publish them in the SDK.
install(DIRECTORY "${_edge_prefix}/" DESTINATION . USE_SOURCE_PERMISSIONS COMPONENT EdgeLLM
  PATTERN "libexec/trtmc-edge-llm/bin" EXCLUDE)
install(DIRECTORY "${_edge_prefix}/libexec/trtmc-edge-llm/bin/"
  DESTINATION libexec/trtmc-edge-llm/bin USE_SOURCE_PERMISSIONS COMPONENT EdgeLLM
  FILES_MATCHING REGEX "/python([0-9]+(\\.[0-9]+)?)?$")
# Family DSOs may use lib64; their dynamically loaded plugin must remain adjacent.
_edgellm_install_plugin()
