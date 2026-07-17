# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(trtmc_sam3_tracker_step_native_plugin LANGUAGES CXX)

foreach(SAM3_REQUIRED_ABI_FIELD
    SAM3_TORCH_VERSION
    SAM3_TVM_FFI_VERSION
    SAM3_TENSORRT_VERSION
    SAM3_TORCH_CXX11_ABI)
  if(NOT DEFINED ${SAM3_REQUIRED_ABI_FIELD} OR "${${SAM3_REQUIRED_ABI_FIELD}}" STREQUAL "")
    message(FATAL_ERROR "${SAM3_REQUIRED_ABI_FIELD} is required for a fail-closed SAM3 plugin")
  endif()
endforeach()
if(NOT SAM3_TORCH_CXX11_ABI MATCHES "^[01]$")
  message(FATAL_ERROR "SAM3_TORCH_CXX11_ABI must be 0 or 1")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

find_path(SAM3_TVM_FFI_INCLUDE_DIR tvm/ffi/c_api.h
  HINTS "${SAM3_TVM_FFI_ROOT}/include"
  REQUIRED
)
find_library(SAM3_TVM_FFI_LIBRARY
  NAMES tvm_ffi libtvm_ffi.so
  HINTS "${SAM3_TVM_FFI_ROOT}/lib"
  REQUIRED
)

find_path(SAM3_TRT_INCLUDE_DIR NvInferRuntime.h
  HINTS
    "${SAM3_TENSORRT_ROOT}/include"
    "${SAM3_TENSORRT_ROOT}/tensorrt/include"
    "${SAM3_TENSORRT_ROOT}/tensorrt_libs/include"
  PATHS
    /usr/include/aarch64-linux-gnu
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
  REQUIRED
)
find_library(SAM3_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so
  HINTS
    "${SAM3_TENSORRT_ROOT}/lib"
    "${SAM3_TENSORRT_ROOT}/lib64"
    "${SAM3_TENSORRT_ROOT}/tensorrt_libs"
  PATHS
    /opt/venv/lib/python3.12/site-packages/tensorrt_libs
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
  REQUIRED
)

add_library(trtmc_sam3_tracker_step_native_plugin SHARED
  sam3_tracker_memory_aoti_bridge.cpp
  sam3_tracker_memory_ffi_plugin.cpp
  sam3_tracker_step_aoti_bridge.cpp
  sam3_tracker_step_ffi_plugin.cpp
)

separate_arguments(SAM3_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

target_include_directories(trtmc_sam3_tracker_step_native_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SAM3_TRT_INCLUDE_DIR}
  ${SAM3_TVM_FFI_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(trtmc_sam3_tracker_step_native_plugin PRIVATE
  ${SAM3_TRT_LIBRARY}
  ${SAM3_TVM_FFI_LIBRARY}
  ${TORCH_LIBRARIES}
  CUDA::cudart
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_sam3_tracker_step_native_plugin PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
  ${SAM3_TORCH_CXX_FLAGS}
)
target_compile_definitions(trtmc_sam3_tracker_step_native_plugin PRIVATE
  SAM3_TORCH_VERSION="${SAM3_TORCH_VERSION}"
  SAM3_TVM_FFI_VERSION="${SAM3_TVM_FFI_VERSION}"
  SAM3_TENSORRT_VERSION="${SAM3_TENSORRT_VERSION}"
  SAM3_TORCH_CXX11_ABI=${SAM3_TORCH_CXX11_ABI}
)

get_filename_component(SAM3_TRT_LIBRARY_DIR "${SAM3_TRT_LIBRARY}" DIRECTORY)
get_filename_component(SAM3_TVM_FFI_LIBRARY_DIR "${SAM3_TVM_FFI_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_sam3_tracker_step_native_plugin PROPERTIES
  OUTPUT_NAME trtmc_sam3_tracker_step_native_plugin
  BUILD_RPATH "${SAM3_TRT_LIBRARY_DIR};${SAM3_TVM_FFI_LIBRARY_DIR};${TORCH_INSTALL_PREFIX}/lib"
)

option(SAM3_NATIVE_PLUGIN_BUILD_TESTS "Build SAM3 native plugin tests" OFF)
if(SAM3_NATIVE_PLUGIN_BUILD_TESTS)
  enable_testing()
  add_executable(test_sam3_tracker_memory_ffi_plugin
    test_sam3_tracker_memory_ffi_plugin.cpp
    sam3_tracker_memory_ffi_plugin.cpp
  )
  target_include_directories(test_sam3_tracker_memory_ffi_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SAM3_TRT_INCLUDE_DIR}
    ${SAM3_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(test_sam3_tracker_memory_ffi_plugin PRIVATE
    ${SAM3_TRT_LIBRARY}
    ${SAM3_TVM_FFI_LIBRARY}
    CUDA::cudart
  )
  target_compile_options(test_sam3_tracker_memory_ffi_plugin PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  set_target_properties(test_sam3_tracker_memory_ffi_plugin PROPERTIES
    BUILD_RPATH "${SAM3_TRT_LIBRARY_DIR};${SAM3_TVM_FFI_LIBRARY_DIR}"
  )
  add_test(NAME sam3_tracker_memory_ffi_shape_bridge
           COMMAND test_sam3_tracker_memory_ffi_plugin)

  add_executable(test_sam3_tracker_memory_aoti_registry
    test_sam3_tracker_memory_aoti_registry.cpp
  )
  target_include_directories(test_sam3_tracker_memory_aoti_registry PRIVATE
    ${SAM3_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(test_sam3_tracker_memory_aoti_registry PRIVATE
    trtmc_sam3_tracker_step_native_plugin
    ${SAM3_TVM_FFI_LIBRARY}
  )
  target_compile_options(test_sam3_tracker_memory_aoti_registry PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  set_target_properties(test_sam3_tracker_memory_aoti_registry PROPERTIES
    BUILD_RPATH "${SAM3_TRT_LIBRARY_DIR};${SAM3_TVM_FFI_LIBRARY_DIR};${TORCH_INSTALL_PREFIX}/lib"
  )
  add_test(NAME sam3_tracker_memory_aoti_registration_generations
           COMMAND test_sam3_tracker_memory_aoti_registry)

  add_executable(test_sam3_tracker_step_aoti_registry
    test_sam3_tracker_step_aoti_registry.cpp
  )
  target_include_directories(test_sam3_tracker_step_aoti_registry PRIVATE
    ${SAM3_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(test_sam3_tracker_step_aoti_registry PRIVATE
    trtmc_sam3_tracker_step_native_plugin
    ${SAM3_TVM_FFI_LIBRARY}
  )
  target_compile_options(test_sam3_tracker_step_aoti_registry PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  set_target_properties(test_sam3_tracker_step_aoti_registry PROPERTIES
    BUILD_RPATH "${SAM3_TRT_LIBRARY_DIR};${SAM3_TVM_FFI_LIBRARY_DIR};${TORCH_INSTALL_PREFIX}/lib"
  )
  add_test(NAME sam3_tracker_step_aoti_registration_generations
           COMMAND test_sam3_tracker_step_aoti_registry)

  add_executable(test_sam3_tracker_step_ffi_plugin
    test_sam3_tracker_step_ffi_plugin.cpp
    sam3_tracker_step_ffi_plugin.cpp
  )
  target_include_directories(test_sam3_tracker_step_ffi_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SAM3_TRT_INCLUDE_DIR}
    ${SAM3_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(test_sam3_tracker_step_ffi_plugin PRIVATE
    ${SAM3_TRT_LIBRARY}
    ${SAM3_TVM_FFI_LIBRARY}
    CUDA::cudart
  )
  target_compile_options(test_sam3_tracker_step_ffi_plugin PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  set_target_properties(test_sam3_tracker_step_ffi_plugin PROPERTIES
    BUILD_RPATH "${SAM3_TRT_LIBRARY_DIR};${SAM3_TVM_FFI_LIBRARY_DIR}"
  )
  add_test(NAME sam3_tracker_step_ffi_shape_bridge
           COMMAND test_sam3_tracker_step_ffi_plugin)
endif()
