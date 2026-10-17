# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LFM2_CUDNN_RUNTIME_NAMES
  libcudnn.so.9 libcudnn_graph.so.9 libcudnn_heuristic.so.9
  libcudnn_engines_precompiled.so.9 libcudnn_engines_runtime_compiled.so.9
  libcudnn_ops.so.9 libcudnn_cnn.so.9 libcudnn_adv.so.9)

function(_lfm2_check_cudnn root license)
  get_filename_component(_root "${root}" REALPATH)
  foreach(_header IN ITEMS cudnn.h cudnn_version.h cudnn_graph.h)
    if(NOT EXISTS "${_root}/include/${_header}")
      message(FATAL_ERROR "Incomplete LFM2 cuDNN SDK: ${_root}/include/${_header}")
    endif()
    get_filename_component(_real "${_root}/include/${_header}" REALPATH)
    get_filename_component(_parent "${_real}" DIRECTORY)
    if(NOT _parent STREQUAL "${_root}/include")
      message(FATAL_ERROR "LFM2 cuDNN header escapes the selected SDK: ${_real}")
    endif()
  endforeach()
  set(_libdir "")
  foreach(_suffix IN ITEMS lib lib64)
    if(EXISTS "${_root}/${_suffix}/libcudnn.so.9")
      set(_libdir "${_root}/${_suffix}")
      break()
    endif()
  endforeach()
  foreach(_name IN LISTS LFM2_CUDNN_RUNTIME_NAMES)
    if(NOT _libdir OR NOT EXISTS "${_libdir}/${_name}")
      message(FATAL_ERROR "Incomplete LFM2 cuDNN SDK: missing ${_name} under ${_root}")
    endif()
    get_filename_component(_real "${_libdir}/${_name}" REALPATH)
    get_filename_component(_parent "${_real}" DIRECTORY)
    if(NOT _parent STREQUAL _libdir)
      message(FATAL_ERROR "LFM2 cuDNN library escapes the selected SDK: ${_real}")
    endif()
  endforeach()
  file(READ "${_root}/include/cudnn_version.h" _header)
  set(_header_version "")
  foreach(_part IN ITEMS MAJOR MINOR PATCHLEVEL)
    if(NOT _header MATCHES "#[ \t]*define[ \t]+CUDNN_${_part}[ \t]+([0-9]+)")
      message(FATAL_ERROR "LFM2 cuDNN header has no ${_part} version")
    endif()
    list(APPEND _header_version "${CMAKE_MATCH_1}")
  endforeach()
  if(NOT _header_version STREQUAL "9;20;0")
    message(FATAL_ERROR "LFM2 cuDNN requires 9.20.0 headers, found ${_header_version}")
  endif()
  if(NOT license)
    foreach(_name IN ITEMS LICENSE.txt LICENSE License.txt)
      if(EXISTS "${_root}/${_name}")
        set(license "${_root}/${_name}")
        break()
      endif()
    endforeach()
  endif()
  if(NOT license OR NOT EXISTS "${license}" OR IS_DIRECTORY "${license}")
    message(FATAL_ERROR "Set LFM2_CUDNN_LICENSE_FILE to the selected SDK's complete license")
  endif()
  get_filename_component(license "${license}" REALPATH)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import ctypes,sys; c=ctypes.CDLL(sys.argv[1]); c.cudnnGetVersion.restype=ctypes.c_size_t; c.cudnnGetCudartVersion.restype=ctypes.c_size_t; print(str(c.cudnnGetVersion())+':'+str(c.cudnnGetCudartVersion()//1000))"
    "${_libdir}/libcudnn.so.9"
    RESULT_VARIABLE _status OUTPUT_VARIABLE _version ERROR_VARIABLE _error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT "${_status}" STREQUAL "0" OR NOT _version STREQUAL "92000:13")
    message(FATAL_ERROR "LFM2 selected cuDNN must be 9.20.0 for CUDA 13: ${_libdir}; got '${_version}': ${_error}")
  endif()
  message(STATUS "LFM2 cuDNN SDK ${_root}: ${_version}")
  set(LFM2_CUDNN_ROOT "${_root}" PARENT_SCOPE)
  set(LFM2_CUDNN_LICENSE_FILE "${license}" PARENT_SCOPE)
  set(LFM2_CUDNN_INCLUDE_DIR "${_root}/include" PARENT_SCOPE)
  set(_lfm2_cudnn_libdir "${_libdir}" PARENT_SCOPE)
  set(_lfm2_cudnn_license "${license}" PARENT_SCOPE)
endfunction()

# Build-time staging also validates a downloaded SDK before compilation.
if(DEFINED LFM2_CUDNN_STAGE_ROOT)
  _lfm2_check_cudnn("${LFM2_CUDNN_ROOT}" "${LFM2_CUDNN_LICENSE_FILE}")
  file(MAKE_DIRECTORY "${LFM2_CUDNN_STAGE_ROOT}")
  foreach(_name IN LISTS LFM2_CUDNN_RUNTIME_NAMES)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${_lfm2_cudnn_libdir}/${_name}" "${LFM2_CUDNN_STAGE_ROOT}/${_name}"
      COMMAND_ERROR_IS_FATAL ANY)
  endforeach()
  execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_lfm2_cudnn_license}" "${LFM2_CUDNN_STAGE_ROOT}/lfm2-cudnn-LICENSE.txt"
    COMMAND_ERROR_IS_FATAL ANY)
  return()
endif()

if(NOT TARGET trtmc_byok_tvm_ffi)
  return()
endif()
set(_lfm2_sdpa_default OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CUDAToolkit_VERSION_MAJOR EQUAL 13)
  set(_lfm2_sdpa_default ON)
endif()
option(LFM2_ENABLE_CUDNN_SDPA "Build the LFM2 CUDA 13 cuDNN attention binding" ${_lfm2_sdpa_default})
if(NOT LFM2_ENABLE_CUDNN_SDPA)
  message(STATUS "LFM2 cuDNN SDPA disabled; legacy LFM2 BYOK targets remain available")
  return()
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CUDAToolkit_VERSION_MAJOR EQUAL 13)
  message(FATAL_ERROR "LFM2 cuDNN SDPA requires Linux and CUDA 13; disable LFM2_ENABLE_CUDNN_SDPA for legacy BYOK builds")
endif()

include(ExternalProject)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LFM2_CUDNN_ROOT "" CACHE PATH "Complete native cuDNN 9.20 CUDA 13 SDK root")
set(LFM2_CUDNN_LICENSE_FILE "" CACHE FILEPATH "License for the selected cuDNN SDK")
set(LFM2_CUDNN_ARCHIVE "" CACHE FILEPATH "Offline pinned native cuDNN archive")
set(LFM2_CUDNN_FRONTEND_ARCHIVE "" CACHE FILEPATH "Offline pinned cuDNN frontend archive")
set(_lfm2_deps "${CMAKE_CURRENT_BINARY_DIR}/_deps")
set(LFM2_CUDNN_FRONTEND_SOURCE_DIR "${_lfm2_deps}/cudnn_frontend")
set(LFM2_CUDNN_FRONTEND_INCLUDE_DIR "${LFM2_CUDNN_FRONTEND_SOURCE_DIR}/include")
file(READ "${CMAKE_CURRENT_LIST_DIR}/../ci/native-dependencies.json" _lfm2_dependencies)
string(JSON _frontend_url GET "${_lfm2_dependencies}" artifacts frontend url)
string(JSON _frontend_hash GET "${_lfm2_dependencies}" artifacts frontend sha256)
if(LFM2_CUDNN_FRONTEND_ARCHIVE)
  if(NOT EXISTS "${LFM2_CUDNN_FRONTEND_ARCHIVE}")
    message(FATAL_ERROR "Missing LFM2_CUDNN_FRONTEND_ARCHIVE: ${LFM2_CUDNN_FRONTEND_ARCHIVE}")
  endif()
  set(_frontend_url "${LFM2_CUDNN_FRONTEND_ARCHIVE}")
endif()
file(MAKE_DIRECTORY "${LFM2_CUDNN_FRONTEND_INCLUDE_DIR}")
ExternalProject_Add(lfm2_cudnn_frontend_source
  PREFIX "${_lfm2_deps}/frontend_ep" SOURCE_DIR "${LFM2_CUDNN_FRONTEND_SOURCE_DIR}"
  URL "${_frontend_url}" DOWNLOAD_NAME cudnn-frontend.tar.gz
  URL_HASH "SHA256=${_frontend_hash}"
  CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "" UPDATE_COMMAND "")

set(_lfm2_cudnn_source_target "")
if(LFM2_CUDNN_ROOT)
  _lfm2_check_cudnn("${LFM2_CUDNN_ROOT}" "${LFM2_CUDNN_LICENSE_FILE}")
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(_sdk_arch linux-x86_64)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_sdk_arch linux-sbsa)
  else()
    message(FATAL_ERROR "LFM2 cuDNN SDK provisioning supports x86_64 or aarch64 SBSA")
  endif()
  string(JSON _sdk_url GET "${_lfm2_dependencies}" artifacts "${_sdk_arch}" url)
  string(JSON _sdk_hash GET "${_lfm2_dependencies}" artifacts "${_sdk_arch}" sha256)
  if(LFM2_CUDNN_ARCHIVE)
    if(NOT EXISTS "${LFM2_CUDNN_ARCHIVE}")
      message(FATAL_ERROR "Missing LFM2_CUDNN_ARCHIVE: ${LFM2_CUDNN_ARCHIVE}")
    endif()
    set(_sdk_url "${LFM2_CUDNN_ARCHIVE}")
  endif()
  set(LFM2_CUDNN_ROOT "${_lfm2_deps}/cudnn")
  set(LFM2_CUDNN_INCLUDE_DIR "${LFM2_CUDNN_ROOT}/include")
  file(MAKE_DIRECTORY "${LFM2_CUDNN_INCLUDE_DIR}")
  ExternalProject_Add(lfm2_cudnn_source
    PREFIX "${_lfm2_deps}/cudnn_ep" SOURCE_DIR "${LFM2_CUDNN_ROOT}"
    URL "${_sdk_url}" URL_HASH "SHA256=${_sdk_hash}"
    CONFIGURE_COMMAND "" BUILD_COMMAND "" INSTALL_COMMAND "" UPDATE_COMMAND "")
  set(_lfm2_cudnn_source_target lfm2_cudnn_source)
endif()

set(LFM2_CUDNN_STAGED_LIBRARIES "")
foreach(_name IN LISTS LFM2_CUDNN_RUNTIME_NAMES)
  list(APPEND LFM2_CUDNN_STAGED_LIBRARIES "${CMAKE_BINARY_DIR}/${_name}")
endforeach()
set(LFM2_CUDNN_STAGED_LICENSE "${CMAKE_BINARY_DIR}/lfm2-cudnn-LICENSE.txt")
add_custom_target(lfm2_cudnn_stage
  COMMAND "${CMAKE_COMMAND}" "-DLFM2_CUDNN_STAGE_ROOT=${CMAKE_BINARY_DIR}"
    "-DLFM2_CUDNN_ROOT=${LFM2_CUDNN_ROOT}"
    "-DLFM2_CUDNN_LICENSE_FILE=${LFM2_CUDNN_LICENSE_FILE}"
    "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}" -P "${CMAKE_CURRENT_LIST_FILE}"
  DEPENDS ${_lfm2_cudnn_source_target}
  BYPRODUCTS ${LFM2_CUDNN_STAGED_LIBRARIES} "${LFM2_CUDNN_STAGED_LICENSE}"
  VERBATIM)
add_library(lfm2_cudnn SHARED IMPORTED)
set_target_properties(lfm2_cudnn PROPERTIES
  IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/libcudnn.so.9"
  INTERFACE_INCLUDE_DIRECTORIES "${LFM2_CUDNN_INCLUDE_DIR}")
add_dependencies(lfm2_cudnn lfm2_cudnn_stage)
