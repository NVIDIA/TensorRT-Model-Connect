# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(tensorrt_model_connect VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

set(TRTMC_CXX_WARNINGS
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
)

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)
option(TRTMC_BUILD_BACKEND_TRT "Build the standard TensorRT backend DSO" ON)

set(_trtmc_dependency_roots
  /usr
  /usr/local
  /usr/local/cuda
  /opt/tensorrt
)
if(DEFINED ENV{CUDA_PATH})
  list(PREPEND _trtmc_dependency_roots "$ENV{CUDA_PATH}")
endif()
if(DEFINED ENV{TRT_ROOT})
  list(PREPEND _trtmc_dependency_roots "$ENV{TRT_ROOT}")
endif()

if(TRTMC_BUILD_BACKEND_TRT)
  find_path(TRTMC_TRT_INCLUDE_DIR
    NAMES NvInferRuntime.h
    HINTS ${_trtmc_dependency_roots}
    PATH_SUFFIXES include include/zapped_headers
    REQUIRED
  )
  find_library(TRTMC_TRT_LIBRARY
    NAMES nvinfer libnvinfer.so.11 nvinfer_11
    HINTS ${_trtmc_dependency_roots}
    PATH_SUFFIXES lib lib64 lib/x64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
    REQUIRED
  )
endif()

option(TRTMC_ENABLE_BYOK "Enable the optional TVM-FFI BYOK bridge" ON)
set(TRTMC_HAS_TVM_FFI OFF)
if(TRTMC_ENABLE_BYOK)
  if(NOT TRTMC_BUILD_BACKEND_TRT)
    message(FATAL_ERROR "TRTMC_ENABLE_BYOK=ON requires TRTMC_BUILD_BACKEND_TRT=ON")
  endif()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import importlib.util; s=importlib.util.find_spec('tvm_ffi'); print(s.submodule_search_locations[0] if s else '')"
      OUTPUT_VARIABLE TRTMC_TVM_FFI_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()
  find_path(TRTMC_TVM_FFI_INCLUDE_DIR
    NAMES tvm/ffi/c_api.h
    HINTS "${TRTMC_TVM_FFI_ROOT}"
    PATH_SUFFIXES include
  )
  find_library(TRTMC_TVM_FFI_LIBRARY
    NAMES tvm_ffi
    HINTS "${TRTMC_TVM_FFI_ROOT}"
    PATH_SUFFIXES lib
  )
  if(TRTMC_TVM_FFI_INCLUDE_DIR AND TRTMC_TVM_FFI_LIBRARY)
    set(TRTMC_HAS_TVM_FFI ON)
    get_filename_component(
      TRTMC_TVM_FFI_LIBRARY_DIR "${TRTMC_TVM_FFI_LIBRARY}" DIRECTORY
    )
  else()
    message(FATAL_ERROR
      "TRTMC_ENABLE_BYOK=ON requires apache-tvm-ffi==0.1.12; "
      "install the project dependencies or configure -DTRTMC_ENABLE_BYOK=OFF"
    )
  endif()
endif()

# Family CMake files use these two variables to link the shared Engine API.
set(TRTMC_CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
if(WIN32 AND CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "Static")
  set(TRTMC_CUDART_LIBRARY CUDA::cudart_static)
else()
  set(TRTMC_CUDART_LIBRARY CUDA::cudart)
endif()

add_library(trtmc_core SHARED
  core/runtime/bundle/bundle_format.cpp
  core/runtime/primitives/cuda_common.cpp
  core/runtime/primitives/device_tensor.cpp
  core/runtime/primitives/trt_common.cpp
)
target_include_directories(trtmc_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/runtime/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/core
)
target_include_directories(trtmc_core SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(trtmc_core
  PUBLIC
    ${TRTMC_CUDART_LIBRARY}
  PRIVATE
    nlohmann_json::nlohmann_json
)
target_compile_options(trtmc_core PRIVATE ${TRTMC_CXX_WARNINGS})
set_target_properties(trtmc_core PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

add_library(trtmc_runtime SHARED
  core/runtime/loader/family_loader.cpp
  core/runtime/platform/dynamic_library.cpp
)
target_include_directories(trtmc_runtime
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/runtime/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}/core
)
target_link_libraries(trtmc_runtime
  PRIVATE
    trtmc_core
    ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_runtime PRIVATE ${TRTMC_CXX_WARNINGS})
set_target_properties(trtmc_runtime PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

if(TRTMC_BUILD_BACKEND_TRT)
add_library(trtmc_backend_trt SHARED
  core/runtime/tensorrt/trt_backend.cpp
  core/runtime/tensorrt/trt_logger.cpp
  core/runtime/tensorrt/trt_module_impl.cpp
)
target_include_directories(trtmc_backend_trt
  PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/core
)
target_include_directories(trtmc_backend_trt SYSTEM PRIVATE
  ${TRTMC_TRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(trtmc_backend_trt
  PRIVATE
    trtmc_core
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUDART_LIBRARY}
    ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_backend_trt PRIVATE ${TRTMC_CXX_WARNINGS})
set_target_properties(trtmc_backend_trt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
endif()

option(TRTMC_BUILD_BACKEND_RTX "Build TensorRT-RTX backend DSO" OFF)
set(TRTMC_RTX_RUNTIME_DIR "" CACHE PATH "TensorRT-RTX runtime DLL directory on Windows")
if(TRTMC_BUILD_BACKEND_RTX)
  if(NOT IS_DIRECTORY "${TRTMC_RTX_INCLUDE_DIR}" OR
     NOT EXISTS "${TRTMC_RTX_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR
      "TRTMC_BUILD_BACKEND_RTX=ON requires TRTMC_RTX_INCLUDE_DIR with NvInfer.h"
    )
  endif()
  if(NOT IS_DIRECTORY "${TRTMC_RTX_LIBRARY_DIR}")
    message(FATAL_ERROR
      "TRTMC_BUILD_BACKEND_RTX=ON requires TRTMC_RTX_LIBRARY_DIR"
    )
  endif()
  find_library(TRTMC_RTX_LIBRARY
    NAMES tensorrt_rtx tensorrt_rtx_1_6
    PATHS "${TRTMC_RTX_LIBRARY_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT TRTMC_RTX_LIBRARY)
    message(FATAL_ERROR
      "TRTMC_RTX_LIBRARY_DIR does not contain libtensorrt_rtx"
    )
  endif()

  if(WIN32)
    find_file(TRTMC_RTX_RUNTIME_DLL
      NAMES tensorrt_rtx_1_6.dll tensorrt_rtx.dll
      PATHS "${TRTMC_RTX_RUNTIME_DIR}"
      NO_DEFAULT_PATH
    )
    if(NOT TRTMC_RTX_RUNTIME_DLL)
      message(FATAL_ERROR
        "TRTMC_BUILD_BACKEND_RTX=ON requires TRTMC_RTX_RUNTIME_DIR with the runtime DLL"
      )
    endif()
  endif()

  add_library(trtmc_backend_rtx SHARED
    core/runtime/tensorrt/rtx_backend.cpp
    core/runtime/tensorrt/runtime_cache_persistence.cpp
    core/runtime/tensorrt/trt_logger.cpp
    core/runtime/tensorrt/trt_module_impl.cpp
  )
  target_include_directories(trtmc_backend_rtx
    PRIVATE
      ${PROJECT_SOURCE_DIR}/core/runtime/include
      ${PROJECT_SOURCE_DIR}/core
  )
  target_include_directories(trtmc_backend_rtx SYSTEM PRIVATE
    ${TRTMC_RTX_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(trtmc_backend_rtx
    PRIVATE
      trtmc_core
      ${TRTMC_RTX_LIBRARY}
      ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(trtmc_backend_rtx PRIVATE ${TRTMC_CXX_WARNINGS})
  target_compile_definitions(trtmc_backend_rtx PRIVATE
    TRTMC_TRT_DELETE_OBJECTS=1
    TRTMC_TRT_HAS_TENSOR_ALIAS_API=1
    TRTMC_TRT_HAS_COMMUNICATOR_API=1
  )
  set_target_properties(trtmc_backend_rtx PROPERTIES
    OUTPUT_NAME trtmc_backend_trt_rtx
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
  )
  if(WIN32)
    add_custom_command(TARGET trtmc_backend_rtx POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TRTMC_RTX_RUNTIME_DLL}"
        "$<TARGET_FILE_DIR:trtmc_backend_rtx>"
      COMMENT "Copying TensorRT-RTX runtime DLL next to the backend"
    )
  endif()
endif()

if(TRTMC_HAS_TVM_FFI)
  add_library(trtmc_byok_tvm_ffi SHARED
    core/runtime/byok/byok.cpp
    core/runtime/byok/tvm_ffi_function.cpp
    core/runtime/byok/tvm_ffi_kernel_creator.cpp
    core/runtime/byok/tvm_ffi_kernel_plugin.cpp
  )
  target_include_directories(trtmc_byok_tvm_ffi
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/runtime/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${PROJECT_SOURCE_DIR}/core
  )
  target_include_directories(trtmc_byok_tvm_ffi SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_byok_tvm_ffi PRIVATE
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUDART_LIBRARY}
    ${TRTMC_TVM_FFI_LIBRARY}
    nlohmann_json::nlohmann_json
  )
  target_compile_definitions(trtmc_byok_tvm_ffi PRIVATE
    TRTMC_HAS_TRT=1
    TRTMC_HAS_TVM_FFI=1
  )
  target_compile_options(trtmc_byok_tvm_ffi PRIVATE ${TRTMC_CXX_WARNINGS})
  set_target_properties(trtmc_byok_tvm_ffi PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN;${TRTMC_TVM_FFI_LIBRARY_DIR}"
  )
endif()

option(TRTMC_BUILD_TESTS "Build tests" ON)
option(TRTMC_BUILD_EXAMPLES "Build examples" ON)
if(TRTMC_BUILD_TESTS)
  enable_testing()
endif()

# A family owns its target, sources, dependencies, warnings, and installation.
# The root knows only the directory convention; adding a family never changes
# a central source or target list.
set(TRTMC_RUNTIME_MODELS "" CACHE STRING
  "Semicolon-separated family runtimes to build; empty builds every family"
)
file(GLOB _trtmc_family_runtime_cmake CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/families/*/runtime/CMakeLists.txt"
)
foreach(_trtmc_runtime_cmake IN LISTS _trtmc_family_runtime_cmake)
  get_filename_component(_trtmc_runtime_dir "${_trtmc_runtime_cmake}" DIRECTORY)
  get_filename_component(_trtmc_family_dir "${_trtmc_runtime_dir}" DIRECTORY)
  get_filename_component(_trtmc_family "${_trtmc_family_dir}" NAME)
  if(TRTMC_RUNTIME_MODELS AND NOT _trtmc_family IN_LIST TRTMC_RUNTIME_MODELS)
    continue()
  endif()
  add_subdirectory(
    "${_trtmc_runtime_dir}"
    "${CMAKE_BINARY_DIR}/families/${_trtmc_family}"
  )
endforeach()

add_library(trtmc_cli STATIC
  apps/cli/cli.cpp
  apps/cli/io.cpp
  apps/cli/windows_media.cpp
)
target_include_directories(trtmc_cli
  PUBLIC ${PROJECT_SOURCE_DIR}/apps
  PRIVATE
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_cli
  PUBLIC trtmc_runtime
  PRIVATE
    trtmc_core
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)
if(WIN32)
  target_link_libraries(trtmc_cli PRIVATE mfplat mfreadwrite mfuuid ole32)
endif()
target_compile_definitions(trtmc_cli PRIVATE TRTMC_VERSION_STRING="${PROJECT_VERSION}")
target_compile_options(trtmc_cli PRIVATE ${TRTMC_CXX_WARNINGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(apps/cli/io.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic"
  )
endif()

add_executable(trtmc apps/cli/main.cpp)
if(WIN32)
  target_sources(trtmc PRIVATE
    apps/cli/windows_utf8_argv.cpp
    apps/cli/trtmc_windows_utf8.manifest
  )
endif()
target_include_directories(trtmc PRIVATE ${PROJECT_SOURCE_DIR}/apps)
target_link_libraries(trtmc PRIVATE trtmc_cli)
target_compile_options(trtmc PRIVATE ${TRTMC_CXX_WARNINGS})
set_target_properties(trtmc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

if(TRTMC_BUILD_EXAMPLES)
  add_executable(trtmc_benchmark_worker apps/benchmark/native/benchmark_worker.cpp)
  target_include_directories(trtmc_benchmark_worker PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
  target_link_libraries(trtmc_benchmark_worker PRIVATE
    trtmc_runtime
    nlohmann_json::nlohmann_json
  )
  target_compile_options(trtmc_benchmark_worker PRIVATE ${TRTMC_CXX_WARNINGS})
  set_target_properties(trtmc_benchmark_worker PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )

  add_executable(trtmc_dataset_benchmark apps/benchmark/native/dataset_benchmark.cpp)
  target_include_directories(trtmc_dataset_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/apps/benchmark/native
  )
  target_link_libraries(trtmc_dataset_benchmark PRIVATE
    trtmc_runtime
    nlohmann_json::nlohmann_json
  )
  target_compile_options(trtmc_dataset_benchmark PRIVATE ${TRTMC_CXX_WARNINGS})
  set_target_properties(trtmc_dataset_benchmark PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_bundle_format_v1 core/runtime/tests/test_bundle_format_v1.cpp)
  target_include_directories(test_bundle_format_v1 PRIVATE ${PROJECT_SOURCE_DIR}/core)
  target_link_libraries(test_bundle_format_v1 PRIVATE trtmc_core)
  target_compile_options(test_bundle_format_v1 PRIVATE ${TRTMC_CXX_WARNINGS})
  add_test(NAME bundle_format_v1 COMMAND test_bundle_format_v1)

  add_executable(test_task_api core/runtime/tests/test_task_api.cpp)
  target_include_directories(test_task_api PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_compile_options(test_task_api PRIVATE ${TRTMC_CXX_WARNINGS})
  add_test(NAME task_api COMMAND test_task_api)

  add_executable(test_cli apps/cli/tests/test_cli.cpp)
  target_include_directories(test_cli PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/apps
  )
  target_link_libraries(test_cli PRIVATE trtmc_cli)
  target_compile_options(test_cli PRIVATE ${TRTMC_CXX_WARNINGS})
  add_test(NAME cli COMMAND test_cli)

  set(_trtmc_test_runtime_root "${CMAKE_BINARY_DIR}/tests/runtime")
  add_library(trtmc_test_backend_fake SHARED core/runtime/tests/fake_backend.cpp)
  target_include_directories(trtmc_test_backend_fake PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(trtmc_test_backend_fake PRIVATE ${TRTMC_CUDART_LIBRARY})
  set_target_properties(trtmc_test_backend_fake PROPERTIES
    OUTPUT_NAME trtmc_backend_fake
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}"
    RUNTIME_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}/$<0:>"
  )

  add_library(trtmc_test_backend_fake_rtx SHARED core/runtime/tests/fake_backend.cpp)
  target_include_directories(trtmc_test_backend_fake_rtx PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_test_backend_fake_rtx PRIVATE ${TRTMC_CUDART_LIBRARY})
  target_compile_definitions(trtmc_test_backend_fake_rtx PRIVATE
    TRTMC_FAKE_BACKEND_NAME="trt_rtx"
  )
  set_target_properties(trtmc_test_backend_fake_rtx PROPERTIES
    OUTPUT_NAME trtmc_backend_trt_rtx
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}"
    RUNTIME_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}/$<0:>"
  )

  add_library(trtmc_test_family_fake SHARED core/runtime/tests/fake_family.cpp)
  target_include_directories(trtmc_test_family_fake PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(trtmc_test_family_fake PRIVATE trtmc_core)
  set_target_properties(trtmc_test_family_fake PROPERTIES
    OUTPUT_NAME trtmc_model_fake
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}"
    RUNTIME_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}/$<0:>"
    BUILD_RPATH "\$ORIGIN/../.."
  )

  add_executable(test_family_loader core/runtime/tests/test_family_loader.cpp)
  target_include_directories(test_family_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/core
  )
  target_link_libraries(test_family_loader PRIVATE trtmc_runtime)
  target_compile_options(test_family_loader PRIVATE ${TRTMC_CXX_WARNINGS})
  add_dependencies(test_family_loader
    trtmc_test_backend_fake
    trtmc_test_backend_fake_rtx
    trtmc_test_family_fake
  )
  add_test(NAME family_loader COMMAND test_family_loader "${_trtmc_test_runtime_root}")

  add_executable(test_dynamic_library core/runtime/tests/test_dynamic_library.cpp)
  target_include_directories(test_dynamic_library PRIVATE ${PROJECT_SOURCE_DIR}/core)
  target_link_libraries(test_dynamic_library PRIVATE trtmc_runtime)
  target_compile_definitions(test_dynamic_library PRIVATE
    TRTMC_TEST_LIBRARY="$<TARGET_FILE:trtmc_test_backend_fake>"
  )
  target_compile_options(test_dynamic_library PRIVATE ${TRTMC_CXX_WARNINGS})
  add_dependencies(test_dynamic_library trtmc_test_backend_fake)
  add_test(NAME dynamic_library COMMAND test_dynamic_library)

  add_executable(test_runtime_cache_persistence
    core/runtime/tests/test_runtime_cache_persistence.cpp
    core/runtime/tensorrt/runtime_cache_persistence.cpp
  )
  target_include_directories(test_runtime_cache_persistence PRIVATE ${PROJECT_SOURCE_DIR}/core)
  target_compile_options(test_runtime_cache_persistence PRIVATE ${TRTMC_CXX_WARNINGS})
  add_test(NAME runtime_cache_persistence COMMAND test_runtime_cache_persistence)

  if(WIN32)
    add_executable(test_windows_utf8_argv
      apps/cli/tests/test_windows_utf8_argv.cpp
      apps/cli/windows_utf8_argv.cpp
      apps/cli/trtmc_windows_utf8.manifest
    )
    target_include_directories(test_windows_utf8_argv PRIVATE ${PROJECT_SOURCE_DIR}/apps)
    add_test(NAME windows_utf8_argv COMMAND test_windows_utf8_argv)

    add_executable(test_windows_media apps/cli/tests/test_windows_media.cpp)
    target_include_directories(test_windows_media PRIVATE
      ${PROJECT_SOURCE_DIR}/apps
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(test_windows_media PRIVATE trtmc_cli)
    add_test(NAME windows_media COMMAND test_windows_media)
  endif()

  if(TRTMC_BUILD_BACKEND_TRT OR TRTMC_BUILD_BACKEND_RTX)
  if(TRTMC_BUILD_BACKEND_TRT)
    set(_trtmc_dynamic_test_backend trtmc_backend_trt)
    set(_trtmc_dynamic_test_include ${TRTMC_TRT_INCLUDE_DIR})
    set(_trtmc_dynamic_test_library ${TRTMC_TRT_LIBRARY})
  else()
    set(_trtmc_dynamic_test_backend trtmc_backend_rtx)
    set(_trtmc_dynamic_test_include ${TRTMC_RTX_INCLUDE_DIR})
    set(_trtmc_dynamic_test_library ${TRTMC_RTX_LIBRARY})
  endif()
  add_executable(test_trt_module_dynamic_input
    core/runtime/tests/test_trt_module_dynamic_input.cpp
  )
  target_include_directories(test_trt_module_dynamic_input PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/core
  )
  target_include_directories(test_trt_module_dynamic_input SYSTEM PRIVATE
    ${_trtmc_dynamic_test_include}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(test_trt_module_dynamic_input PRIVATE
    ${_trtmc_dynamic_test_backend}
    trtmc_core
    ${_trtmc_dynamic_test_library}
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_trt_module_dynamic_input PRIVATE ${TRTMC_CXX_WARNINGS})
  add_test(NAME trt_module_dynamic_input COMMAND test_trt_module_dynamic_input)
  set_tests_properties(trt_module_dynamic_input PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  endif()

  if(TRTMC_BUILD_EXAMPLES)
    add_executable(test_dataset_answer apps/benchmark/tests/native/test_dataset_answer.cpp)
    target_include_directories(test_dataset_answer PRIVATE
      ${PROJECT_SOURCE_DIR}/apps/benchmark/native
    )
    target_compile_options(test_dataset_answer PRIVATE ${TRTMC_CXX_WARNINGS})
    add_test(NAME dataset_answer COMMAND test_dataset_answer)

    add_executable(test_benchmark_worker_e2e
      apps/benchmark/tests/native/test_benchmark_worker_e2e.cpp
    )
    target_link_libraries(test_benchmark_worker_e2e PRIVATE nlohmann_json::nlohmann_json)
    target_compile_options(test_benchmark_worker_e2e PRIVATE ${TRTMC_CXX_WARNINGS})
    add_dependencies(test_benchmark_worker_e2e
      trtmc_benchmark_worker
      trtmc_test_backend_fake
      trtmc_test_family_fake
    )
    add_test(NAME benchmark_worker_e2e
      COMMAND test_benchmark_worker_e2e
        $<TARGET_FILE:trtmc_benchmark_worker>
        "${_trtmc_test_runtime_root}"
    )
    set_tests_properties(benchmark_worker_e2e PROPERTIES LABELS "cpu;e2e")
  endif()
endif()

if(TRTMC_BUILD_TESTS AND TRTMC_HAS_TVM_FFI)
  add_executable(test_byok_shape_spec core/runtime/tests/test_byok_shape_spec.cpp)
  target_include_directories(test_byok_shape_spec SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(test_byok_shape_spec PRIVATE
    trtmc_byok_tvm_ffi
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_TVM_FFI_LIBRARY}
  )
  target_compile_options(test_byok_shape_spec PRIVATE ${TRTMC_CXX_WARNINGS})
  add_test(NAME byok_shape_spec COMMAND test_byok_shape_spec)
endif()

if(TRTMC_BUILD_EXAMPLES AND TRTMC_HAS_TVM_FFI)
  add_library(trtmc_byok_identity_copy SHARED
    examples/byok/identity_copy_kernel.cpp
  )
  target_include_directories(trtmc_byok_identity_copy SYSTEM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_byok_identity_copy PRIVATE
    ${TRTMC_CUDART_LIBRARY}
    ${TRTMC_TVM_FFI_LIBRARY}
  )
  set_target_properties(trtmc_byok_identity_copy PROPERTIES
    PREFIX ""
    OUTPUT_NAME identity_copy_kernel
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/byok"
  )

  if(TRTMC_BUILD_TESTS)
    add_executable(test_byok_tvm_ffi examples/byok/test_tvm_ffi_plugin.cpp)
    target_include_directories(test_byok_tvm_ffi PRIVATE
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_byok_tvm_ffi SYSTEM PRIVATE
      ${TRTMC_TRT_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${TRTMC_TVM_FFI_INCLUDE_DIR}
    )
    target_link_libraries(test_byok_tvm_ffi PRIVATE
      trtmc_byok_tvm_ffi
      ${TRTMC_TRT_LIBRARY}
      ${TRTMC_CUDART_LIBRARY}
      ${TRTMC_TVM_FFI_LIBRARY}
    )
    target_compile_definitions(test_byok_tvm_ffi PRIVATE
      TRTMC_HAS_TRT=1
      TRTMC_HAS_TVM_FFI=1
    )
    target_compile_options(test_byok_tvm_ffi PRIVATE ${TRTMC_CXX_WARNINGS})
    add_dependencies(test_byok_tvm_ffi trtmc_byok_identity_copy)
    add_test(NAME byok_tvm_ffi
      COMMAND test_byok_tvm_ffi $<TARGET_FILE:trtmc_byok_identity_copy>
    )
    set_tests_properties(byok_tvm_ffi PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  endif()
endif()

include(CMakePackageConfigHelpers)

install(TARGETS trtmc_core trtmc_runtime
  EXPORT trtmcTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if(TARGET trtmc_backend_trt)
  install(TARGETS trtmc_backend_trt
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
if(TARGET trtmc_backend_rtx)
  install(TARGETS trtmc_backend_rtx
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  if(WIN32)
    install(FILES "${TRTMC_RTX_RUNTIME_DLL}" DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()
if(TRTMC_HAS_TVM_FFI)
  install(TARGETS trtmc_byok_tvm_ffi
    EXPORT trtmcTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()
install(TARGETS trtmc
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(TRTMC_BUILD_EXAMPLES)
  install(TARGETS trtmc_benchmark_worker trtmc_dataset_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
install(FILES
  core/runtime/include/trtmc/byok.h
  core/runtime/include/trtmc/bundle.h
  core/runtime/include/trtmc/task.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc
)
install(FILES
  core/runtime/include/trtmc/runtime/device_tensor.h
  core/runtime/include/trtmc/runtime/family_factory.h
  core/runtime/include/trtmc/runtime/family_loader.h
  core/runtime/include/trtmc/runtime/tensor.h
  core/runtime/include/trtmc/runtime/trt_backend.h
  core/runtime/include/trtmc/runtime/trt_module.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc/runtime
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/trtmcConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/trtmcConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtmc
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/trtmcConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ExactVersion
)
install(EXPORT trtmcTargets
  FILE trtmcTargets.cmake
  NAMESPACE trtmc::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtmc
)
install(FILES
  "${PROJECT_BINARY_DIR}/trtmcConfig.cmake"
  "${PROJECT_BINARY_DIR}/trtmcConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/trtmc
)
