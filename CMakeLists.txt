# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(tensorrt_model_connect VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

include(GNUInstallDirs)
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(Threads REQUIRED)

set(_trtmc_dependency_roots
  /usr
  /usr/local
  /usr/local/cuda
  /opt/tensorrt
)
if(DEFINED ENV{TRT_ROOT})
  list(PREPEND _trtmc_dependency_roots "$ENV{TRT_ROOT}")
endif()

find_path(TRTMC_TRT_INCLUDE_DIR
  NAMES NvInferRuntime.h
  HINTS ${_trtmc_dependency_roots}
  PATH_SUFFIXES include include/zapped_headers
  REQUIRED
)
find_library(TRTMC_TRT_LIBRARY
  NAMES nvinfer libnvinfer.so.11
  HINTS ${_trtmc_dependency_roots}
  PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu lib/x86_64-linux-gnu
  REQUIRED
)

option(TRTMC_ENABLE_BYOK "Enable the optional TVM-FFI BYOK bridge" ON)
set(TRTMC_HAS_TVM_FFI OFF)
if(TRTMC_ENABLE_BYOK)
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
set(TRTMC_CUDART_LIBRARY CUDA::cudart)

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
    CUDA::cudart
  PRIVATE
    nlohmann_json::nlohmann_json
)
target_compile_options(trtmc_core PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_core PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

add_library(trtmc_runtime SHARED
  core/runtime/loader/family_loader.cpp
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
target_compile_options(trtmc_runtime PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_runtime PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

add_library(trtmc_c SHARED
  core/api/runtime/api.cpp
  core/api/runtime/text_runtime.cpp
  core/api/runtime/text_tasks.cpp
  core/api/runtime/image_tasks.cpp
  core/api/runtime/control.cpp
  core/api/runtime/audio.cpp
  core/api/runtime/features.cpp
  core/api/runtime/numeric.cpp
  core/api/runtime/scores.cpp
  core/api/runtime/stream.cpp
  core/api/runtime/video.cpp
  core/api/runtime/perception.cpp
  core/api/runtime/language.cpp
  core/api/runtime/tracking.cpp
  core/api/runtime/speech.cpp
  core/api/runtime/action.cpp
  core/api/runtime/recurrent.cpp
)
add_library(trtmc::c ALIAS trtmc_c)
target_include_directories(trtmc_c
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/api/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/runtime/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(trtmc_c PRIVATE trtmc_runtime trtmc_core Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(trtmc_c PRIVATE
  TRTMC_C_BUILD
  TRTMC_VERSION_STRING="${PROJECT_VERSION}"
)
target_compile_options(trtmc_c PRIVATE -Wall -Wextra -Wpedantic)
target_link_options(trtmc_c PRIVATE
  "-Wl,--version-script=${PROJECT_SOURCE_DIR}/core/api/runtime/exports.map"
)
set_target_properties(trtmc_c PROPERTIES
  EXPORT_NAME c
  VERSION 1
  SOVERSION 1
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  LINK_DEPENDS "${PROJECT_SOURCE_DIR}/core/api/runtime/exports.map"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

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
    CUDA::cudart
    ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_backend_trt PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_backend_trt PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)

option(TRTMC_BUILD_BACKEND_RTX "Build TensorRT-RTX backend DSO" OFF)
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
    NAMES tensorrt_rtx
    PATHS "${TRTMC_RTX_LIBRARY_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT TRTMC_RTX_LIBRARY)
    message(FATAL_ERROR
      "TRTMC_RTX_LIBRARY_DIR does not contain libtensorrt_rtx"
    )
  endif()

  add_library(trtmc_backend_rtx SHARED
    core/runtime/tensorrt/rtx_backend.cpp
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
      CUDA::cudart
  )
  target_compile_options(trtmc_backend_rtx PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(trtmc_backend_rtx PROPERTIES
    OUTPUT_NAME trtmc_backend_trt_rtx
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
  )
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
    CUDA::cudart
    ${TRTMC_TVM_FFI_LIBRARY}
    nlohmann_json::nlohmann_json
  )
  target_compile_definitions(trtmc_byok_tvm_ffi PRIVATE
    TRTMC_HAS_TRT=1
    TRTMC_HAS_TVM_FFI=1
  )
  target_compile_options(trtmc_byok_tvm_ffi PRIVATE -Wall -Wextra -Wpedantic)
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
file(GLOB _trtmc_family_runtime_cmake CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/families/*/runtime/CMakeLists.txt"
)
foreach(_trtmc_runtime_cmake IN LISTS _trtmc_family_runtime_cmake)
  get_filename_component(_trtmc_runtime_dir "${_trtmc_runtime_cmake}" DIRECTORY)
  get_filename_component(_trtmc_family_dir "${_trtmc_runtime_dir}" DIRECTORY)
  get_filename_component(_trtmc_family "${_trtmc_family_dir}" NAME)
  add_subdirectory(
    "${_trtmc_runtime_dir}"
    "${CMAKE_BINARY_DIR}/families/${_trtmc_family}"
  )
endforeach()

add_library(trtmc_media_io STATIC apps/cli/io.cpp)
target_include_directories(trtmc_media_io PRIVATE
  ${PROJECT_SOURCE_DIR}/apps
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_media_io SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
target_compile_options(trtmc_media_io PRIVATE -Wall -Wextra -Wpedantic)

add_library(trtmc_cli STATIC
  apps/cli/cli.cpp
  apps/cli/sdk_features.cpp
  apps/cli/sdk_action.cpp
  apps/cli/sdk_replay.cpp
  apps/cli/sdk_audio.cpp
  apps/cli/sdk_image.cpp
  apps/cli/sdk_video.cpp
  apps/cli/sdk_numeric.cpp
  apps/cli/sdk_perception.cpp
  apps/cli/sdk_language.cpp
)
target_include_directories(trtmc_cli
  PUBLIC ${PROJECT_SOURCE_DIR}/apps
  PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_cli
  PUBLIC trtmc_runtime
  PRIVATE
    trtmc_media_io
    trtmc_c
    trtmc_core
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS}
)
target_compile_definitions(trtmc_cli PRIVATE TRTMC_VERSION_STRING="${PROJECT_VERSION}")
target_compile_options(trtmc_cli PRIVATE -Wall -Wextra -Wpedantic)
set_source_files_properties(apps/cli/io.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic"
)

add_executable(trtmc apps/cli/main.cpp)
target_include_directories(trtmc PRIVATE ${PROJECT_SOURCE_DIR}/apps)
target_link_libraries(trtmc PRIVATE trtmc_cli)
target_compile_options(trtmc PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc PROPERTIES
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

if(TRTMC_BUILD_EXAMPLES)
  add_executable(trtmc_benchmark_worker apps/benchmark/native/benchmark_worker.cpp)
  target_include_directories(trtmc_benchmark_worker PRIVATE
    ${PROJECT_SOURCE_DIR}/apps
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
  target_link_libraries(trtmc_benchmark_worker PRIVATE
    trtmc_media_io
    trtmc_c
    trtmc_runtime
    nlohmann_json::nlohmann_json
  )
  target_compile_options(trtmc_benchmark_worker PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(trtmc_benchmark_worker PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )

  add_executable(trtmc_dataset_benchmark apps/benchmark/native/dataset_benchmark.cpp)
  target_include_directories(trtmc_dataset_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR}/apps
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/apps/benchmark/native
  )
  target_link_libraries(trtmc_dataset_benchmark PRIVATE
    trtmc_c
    trtmc_runtime
    nlohmann_json::nlohmann_json
  )
  target_compile_options(trtmc_dataset_benchmark PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(trtmc_dataset_benchmark PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_bundle_format_v1 core/runtime/tests/test_bundle_format_v1.cpp)
  target_include_directories(test_bundle_format_v1 PRIVATE ${PROJECT_SOURCE_DIR}/core)
  target_link_libraries(test_bundle_format_v1 PRIVATE trtmc_core)
  target_compile_options(test_bundle_format_v1 PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME bundle_format_v1 COMMAND test_bundle_format_v1)

  add_executable(test_task_api core/runtime/tests/test_task_api.cpp)
  target_include_directories(test_task_api PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_compile_options(test_task_api PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME task_api COMMAND test_task_api)

  add_executable(test_internal_config core/runtime/tests/test_internal_config.cpp)
  target_include_directories(test_internal_config PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_internal_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME internal_config COMMAND test_internal_config)

  add_executable(test_cpp_config core/api/tests/test_cpp_config.cpp)
  target_link_libraries(test_cpp_config PRIVATE trtmc_c)
  target_compile_options(test_cpp_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME cpp_config COMMAND test_cpp_config)

  add_executable(test_header_helpers core/api/tests/test_header_helpers.cpp)
  target_include_directories(test_header_helpers PRIVATE
    ${PROJECT_SOURCE_DIR}/core/api/include
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_features(test_header_helpers PRIVATE cxx_std_17)
  target_compile_options(test_header_helpers PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME header_helpers COMMAND test_header_helpers)
  set_tests_properties(header_helpers PROPERTIES LABELS cpu)

  add_executable(test_cli apps/cli/tests/test_cli.cpp)
  target_include_directories(test_cli PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/apps
  )
  target_link_libraries(test_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME cli COMMAND test_cli)

  set(_trtmc_test_runtime_root "${CMAKE_BINARY_DIR}/tests/runtime")
  add_library(trtmc_test_backend_fake SHARED core/runtime/tests/fake_backend.cpp)
  target_include_directories(trtmc_test_backend_fake PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(trtmc_test_backend_fake PRIVATE CUDA::cudart)
  set_target_properties(trtmc_test_backend_fake PROPERTIES
    OUTPUT_NAME trtmc_backend_fake
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}"
  )

  add_library(trtmc_test_backend_fake_rtx SHARED core/runtime/tests/fake_backend.cpp)
  target_include_directories(trtmc_test_backend_fake_rtx PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_test_backend_fake_rtx PRIVATE CUDA::cudart)
  target_compile_definitions(trtmc_test_backend_fake_rtx PRIVATE
    TRTMC_FAKE_BACKEND_NAME="trt_rtx"
  )
  set_target_properties(trtmc_test_backend_fake_rtx PROPERTIES
    OUTPUT_NAME trtmc_backend_trt_rtx
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}"
  )

  add_library(trtmc_test_family_fake SHARED core/runtime/tests/fake_family.cpp)
  target_include_directories(trtmc_test_family_fake PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(trtmc_test_family_fake PRIVATE trtmc_core)
  set_target_properties(trtmc_test_family_fake PROPERTIES
    OUTPUT_NAME trtmc_model_fake
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_test_runtime_root}"
    BUILD_RPATH "\$ORIGIN/../.."
  )

  add_executable(test_family_loader core/runtime/tests/test_family_loader.cpp)
  target_include_directories(test_family_loader PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/core
  )
  target_link_libraries(test_family_loader PRIVATE trtmc_runtime)
  target_compile_options(test_family_loader PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(test_family_loader
    trtmc_test_backend_fake
    trtmc_test_backend_fake_rtx
    trtmc_test_family_fake
  )
  add_test(NAME family_loader COMMAND test_family_loader "${_trtmc_test_runtime_root}")

  set(_trtmc_api_test_runtime_root "${CMAKE_BINARY_DIR}/tests/api-runtime")
  add_library(trtmc_api_test_backend SHARED core/runtime/tests/fake_backend.cpp)
  target_include_directories(trtmc_api_test_backend PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_api_test_backend PRIVATE CUDA::cudart)
  set_target_properties(trtmc_api_test_backend PROPERTIES
    OUTPUT_NAME trtmc_backend_fake
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_api_test_runtime_root}"
  )

  add_library(trtmc_api_test_family SHARED core/api/tests/fake_family.cpp)
  target_include_directories(trtmc_api_test_family PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_api_test_family PRIVATE trtmc_core)
  set_target_properties(trtmc_api_test_family PROPERTIES
    OUTPUT_NAME trtmc_model_api_fixture
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_api_test_runtime_root}"
    BUILD_RPATH "\$ORIGIN/../.."
  )

  add_executable(test_c_api core/api/tests/test_c_api.c)
  target_link_libraries(test_c_api PRIVATE trtmc_c)
  target_compile_options(test_c_api PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_c_api trtmc_api_test_backend trtmc_api_test_family)
  add_test(NAME c_api_e2e COMMAND test_c_api "${_trtmc_api_test_runtime_root}")
  set_tests_properties(c_api_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_cpp_api core/api/tests/test_cpp_api.cpp)
  target_link_libraries(test_cpp_api PRIVATE trtmc_c)
  target_compile_options(test_cpp_api PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_cpp_api trtmc_api_test_backend trtmc_api_test_family)
  add_test(NAME cpp_api_e2e COMMAND test_cpp_api "${_trtmc_api_test_runtime_root}")
  set_tests_properties(cpp_api_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_cli apps/cli/tests/test_sdk_cli.cpp)
  target_link_libraries(test_sdk_cli PRIVATE trtmc_cli nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_cli trtmc_api_test_backend trtmc_api_test_family)
  add_test(NAME sdk_cli_e2e COMMAND test_sdk_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_cli_e2e PROPERTIES LABELS "cpu;e2e")

  foreach(_trtmc_fixture IN ITEMS text image audio features numeric stream video perception language tracking speech action recurrent)
    add_library(trtmc_api_test_${_trtmc_fixture}_family SHARED
      core/api/tests/${_trtmc_fixture}_family.cpp
    )
    target_include_directories(trtmc_api_test_${_trtmc_fixture}_family PRIVATE
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(trtmc_api_test_${_trtmc_fixture}_family PRIVATE trtmc_core Threads::Threads)
    set_target_properties(trtmc_api_test_${_trtmc_fixture}_family PROPERTIES
      OUTPUT_NAME trtmc_model_${_trtmc_fixture}_fixture
      LIBRARY_OUTPUT_DIRECTORY "${_trtmc_api_test_runtime_root}"
      BUILD_RPATH "\$ORIGIN/../.."
    )
  endforeach()

  add_executable(test_text_c_api core/api/tests/text_c_test.c)
  add_executable(test_text_cpp_api core/api/tests/text_test.cpp)
  foreach(_trtmc_test IN ITEMS test_text_c_api test_text_cpp_api)
    target_link_libraries(${_trtmc_test} PRIVATE trtmc_c)
    target_compile_options(${_trtmc_test} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_dependencies(${_trtmc_test} trtmc_api_test_backend trtmc_api_test_text_family)
    add_test(NAME ${_trtmc_test} COMMAND ${_trtmc_test} "${_trtmc_api_test_runtime_root}")
    set_tests_properties(${_trtmc_test} PROPERTIES LABELS "cpu;e2e")
  endforeach()

  add_executable(test_image_c_api core/api/tests/test_image_api.c)
  add_executable(test_image_cpp_api core/api/tests/test_image_cpp.cpp)
  foreach(_trtmc_test IN ITEMS test_image_c_api test_image_cpp_api)
    target_link_libraries(${_trtmc_test} PRIVATE trtmc_c)
    target_compile_options(${_trtmc_test} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_dependencies(${_trtmc_test} trtmc_api_test_backend trtmc_api_test_image_family)
    add_test(NAME ${_trtmc_test} COMMAND ${_trtmc_test} "${_trtmc_api_test_runtime_root}")
    set_tests_properties(${_trtmc_test} PROPERTIES LABELS "cpu;e2e")
  endforeach()

  add_dependencies(test_sdk_cli trtmc_api_test_text_family trtmc_api_test_stream_family)

  add_executable(test_sdk_features_cli apps/cli/tests/test_sdk_features_cli.cpp)
  target_link_libraries(test_sdk_features_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_features_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_features_cli trtmc_api_test_backend trtmc_api_test_features_family)
  add_test(NAME sdk_features_cli_e2e
    COMMAND test_sdk_features_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_features_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_action_cli apps/cli/tests/test_sdk_action_cli.cpp)
  target_link_libraries(test_sdk_action_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_action_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_action_cli trtmc_api_test_backend trtmc_api_test_action_family)
  add_test(NAME sdk_action_cli_e2e COMMAND test_sdk_action_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_action_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_replay_cli apps/cli/tests/test_sdk_replay_cli.cpp)
  target_link_libraries(test_sdk_replay_cli PRIVATE trtmc_cli nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_replay_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_replay_cli trtmc_api_test_backend trtmc_api_test_numeric_family)
  add_test(NAME sdk_replay_cli_e2e COMMAND test_sdk_replay_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_replay_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_audio_cli apps/cli/tests/test_sdk_audio_cli.cpp)
  target_link_libraries(test_sdk_audio_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_audio_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_audio_cli trtmc_api_test_backend trtmc_api_test_audio_family
    trtmc_api_test_speech_family)
  add_test(NAME sdk_audio_cli_e2e COMMAND test_sdk_audio_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_audio_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_image_cli apps/cli/tests/test_sdk_image_cli.cpp)
  target_link_libraries(test_sdk_image_cli PRIVATE trtmc_cli nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_image_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_image_cli trtmc_api_test_backend trtmc_api_test_image_family)
  add_test(NAME sdk_image_cli_e2e COMMAND test_sdk_image_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_image_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_video_cli apps/cli/tests/test_sdk_video_cli.cpp)
  target_link_libraries(test_sdk_video_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_video_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_video_cli trtmc_api_test_backend trtmc_api_test_video_family)
  add_test(NAME sdk_video_cli_e2e COMMAND test_sdk_video_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_video_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_numeric_cli apps/cli/tests/test_sdk_numeric_cli.cpp)
  target_link_libraries(test_sdk_numeric_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_numeric_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_numeric_cli trtmc_api_test_backend trtmc_api_test_numeric_family)
  add_test(NAME sdk_numeric_cli_e2e COMMAND test_sdk_numeric_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_numeric_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_perception_cli apps/cli/tests/test_sdk_perception_cli.cpp)
  target_link_libraries(test_sdk_perception_cli PRIVATE trtmc_cli trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_perception_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_perception_cli trtmc_api_test_backend trtmc_api_test_perception_family
    trtmc_api_test_tracking_family)
  add_test(NAME sdk_perception_cli_e2e COMMAND test_sdk_perception_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_perception_cli_e2e PROPERTIES LABELS "cpu;e2e")

  add_executable(test_sdk_language_cli apps/cli/tests/test_sdk_language_cli.cpp)
  target_link_libraries(test_sdk_language_cli PRIVATE trtmc_cli nlohmann_json::nlohmann_json)
  target_compile_options(test_sdk_language_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_dependencies(test_sdk_language_cli trtmc_api_test_backend trtmc_api_test_language_family)
  add_test(NAME sdk_language_cli_e2e COMMAND test_sdk_language_cli "${_trtmc_api_test_runtime_root}")
  set_tests_properties(sdk_language_cli_e2e PROPERTIES LABELS "cpu;e2e")

  foreach(_trtmc_group IN ITEMS audio features numeric stream video perception language tracking speech action recurrent)
    add_executable(test_${_trtmc_group}_c_api core/api/tests/${_trtmc_group}_c_test.c)
    add_executable(test_${_trtmc_group}_cpp_api core/api/tests/${_trtmc_group}_test.cpp)
    foreach(_trtmc_language IN ITEMS c cpp)
      set(_trtmc_test test_${_trtmc_group}_${_trtmc_language}_api)
      target_link_libraries(${_trtmc_test} PRIVATE trtmc_c Threads::Threads ${CMAKE_DL_LIBS})
      target_compile_options(${_trtmc_test} PRIVATE -Wall -Wextra -Wpedantic -Werror)
      add_dependencies(${_trtmc_test} trtmc_api_test_backend trtmc_api_test_${_trtmc_group}_family)
      add_test(NAME ${_trtmc_test} COMMAND ${_trtmc_test} "${_trtmc_api_test_runtime_root}")
      set_tests_properties(${_trtmc_test} PROPERTIES LABELS "cpu;e2e")
    endforeach()
  endforeach()

  set(_trtmc_control_test_root "${CMAKE_BINARY_DIR}/tests/control-runtime")
  foreach(_trtmc_backend IN ITEMS fake trt_rtx)
    add_library(trtmc_control_test_backend_${_trtmc_backend} SHARED
      core/runtime/tests/fake_backend.cpp
    )
    target_include_directories(trtmc_control_test_backend_${_trtmc_backend} PRIVATE
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_compile_definitions(trtmc_control_test_backend_${_trtmc_backend} PRIVATE
      TRTMC_FAKE_BACKEND_NAME="${_trtmc_backend}"
    )
    target_link_libraries(trtmc_control_test_backend_${_trtmc_backend} PRIVATE CUDA::cudart)
    set_target_properties(trtmc_control_test_backend_${_trtmc_backend} PROPERTIES
      OUTPUT_NAME trtmc_backend_${_trtmc_backend}
      LIBRARY_OUTPUT_DIRECTORY "${_trtmc_control_test_root}"
    )
  endforeach()

  add_library(trtmc_control_test_family SHARED core/api/tests/fake_control_family.cpp)
  target_include_directories(trtmc_control_test_family PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_control_test_family PRIVATE trtmc_core)
  set_target_properties(trtmc_control_test_family PROPERTIES
    OUTPUT_NAME trtmc_model_control_fixture
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_control_test_root}"
    BUILD_RPATH "\$ORIGIN/../.."
  )

  add_library(trtmc_control_test_byok SHARED core/api/tests/fake_byok.cpp)
  set_target_properties(trtmc_control_test_byok PROPERTIES
    OUTPUT_NAME trtmc_byok_tvm_ffi
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_control_test_root}"
  )
  add_library(trtmc_control_test_bad_byok SHARED core/runtime/tests/fake_backend.cpp)
  target_include_directories(trtmc_control_test_bad_byok PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_control_test_bad_byok PRIVATE CUDA::cudart)
  set_target_properties(trtmc_control_test_bad_byok PROPERTIES
    OUTPUT_NAME trtmc_byok_tvm_ffi
    LIBRARY_OUTPUT_DIRECTORY "${_trtmc_control_test_root}/missing-symbol"
  )

  add_executable(test_control_c_api core/api/tests/test_control_api.c)
  add_executable(test_control_cpp_api core/api/tests/test_control_cpp.cpp)
  foreach(_trtmc_test IN ITEMS test_control_c_api test_control_cpp_api)
    target_link_libraries(${_trtmc_test} PRIVATE trtmc_c ${CMAKE_DL_LIBS})
    target_compile_options(${_trtmc_test} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_dependencies(${_trtmc_test}
      trtmc_control_test_family trtmc_control_test_byok trtmc_control_test_bad_byok
      trtmc_control_test_backend_fake trtmc_control_test_backend_trt_rtx
    )
    add_test(NAME ${_trtmc_test} COMMAND ${_trtmc_test} "${_trtmc_control_test_root}")
    set_tests_properties(${_trtmc_test} PROPERTIES LABELS "cpu;e2e")
  endforeach()

  add_executable(test_trt_module_dynamic_input
    core/runtime/tests/test_trt_module_dynamic_input.cpp
  )
  target_include_directories(test_trt_module_dynamic_input PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/core
  )
  target_include_directories(test_trt_module_dynamic_input SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  target_link_libraries(test_trt_module_dynamic_input PRIVATE
    trtmc_backend_trt
    trtmc_core
    ${TRTMC_TRT_LIBRARY}
    CUDA::cudart
  )
  target_compile_options(test_trt_module_dynamic_input PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME trt_module_dynamic_input COMMAND test_trt_module_dynamic_input)
  set_tests_properties(trt_module_dynamic_input PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)

  if(TRTMC_BUILD_EXAMPLES)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(trtmc_example_audio_streaming examples/audio_streaming/main.cpp)
    target_include_directories(trtmc_example_audio_streaming PRIVATE ${PROJECT_SOURCE_DIR}/apps)
    target_link_libraries(trtmc_example_audio_streaming PRIVATE trtmc_c trtmc_runtime)
    target_compile_options(trtmc_example_audio_streaming PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(trtmc_example_audio_streaming trtmc_api_test_backend trtmc_api_test_speech_family)
    add_test(NAME example_audio_streaming_sdk COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/examples/audio_streaming/check_sdk_audio_streaming.py"
      --binary $<TARGET_FILE:trtmc_example_audio_streaming>
      --runtime-root "${_trtmc_api_test_runtime_root}"
      --sdk-library $<TARGET_FILE:trtmc_c>)
    set_tests_properties(example_audio_streaming_sdk PROPERTIES LABELS "cpu;e2e")

    add_executable(trtmc_example_recorded_control examples/models/lerobot_act/recorded_control/main.cpp)
    target_include_directories(trtmc_example_recorded_control PRIVATE ${PROJECT_SOURCE_DIR}/apps)
    target_include_directories(trtmc_example_recorded_control SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
    target_link_libraries(trtmc_example_recorded_control PRIVATE trtmc_c trtmc_runtime)
    target_compile_options(trtmc_example_recorded_control PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(trtmc_example_recorded_control trtmc_api_test_backend trtmc_api_test_action_family)
    add_test(NAME example_recorded_control_sdk COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/examples/models/lerobot_act/recorded_control/check_sdk_recorded_control.py"
      --binary $<TARGET_FILE:trtmc_example_recorded_control>
      --runtime-root "${_trtmc_api_test_runtime_root}")
    set_tests_properties(example_recorded_control_sdk PROPERTIES LABELS "cpu;e2e")

    add_executable(trtmc_example_preprocessed examples/models/foundationpose/preprocessed_refinement/main.cpp)
    target_include_directories(trtmc_example_preprocessed PRIVATE ${PROJECT_SOURCE_DIR}/apps)
    target_link_libraries(trtmc_example_preprocessed PRIVATE trtmc_c trtmc_runtime)
    target_compile_options(trtmc_example_preprocessed PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(trtmc_example_preprocessed trtmc_api_test_backend trtmc_api_test_perception_family)
    add_test(NAME example_preprocessed_sdk COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/examples/models/foundationpose/preprocessed_refinement/check_sdk_preprocessed.py"
      --binary $<TARGET_FILE:trtmc_example_preprocessed>
      --runtime-root "${_trtmc_api_test_runtime_root}"
      --sdk-library $<TARGET_FILE:trtmc_c>)
    set_tests_properties(example_preprocessed_sdk PROPERTIES LABELS "cpu;e2e")

    add_executable(test_voicechat_sdk_io examples/models/nemotron_voicechat/full_duplex/test_sdk_session_io.cpp)
    target_link_libraries(test_voicechat_sdk_io PRIVATE trtmc_c Threads::Threads)
    target_compile_options(test_voicechat_sdk_io PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(test_voicechat_sdk_io trtmc_api_test_backend trtmc_api_test_speech_family)
    add_test(NAME voicechat_sdk_io COMMAND test_voicechat_sdk_io "${_trtmc_api_test_runtime_root}")
    set_tests_properties(voicechat_sdk_io PROPERTIES LABELS "cpu;e2e")

    add_executable(test_dataset_answer apps/benchmark/tests/native/test_dataset_answer.cpp)
    target_include_directories(test_dataset_answer PRIVATE
      ${PROJECT_SOURCE_DIR}/apps/benchmark/native
    )
    target_compile_options(test_dataset_answer PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME dataset_answer COMMAND test_dataset_answer)

    add_library(trtmc_dataset_test_family SHARED apps/benchmark/tests/native/dataset_family.cpp)
    target_include_directories(trtmc_dataset_test_family PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
    target_link_libraries(trtmc_dataset_test_family PRIVATE trtmc_core nlohmann_json::nlohmann_json)
    set_target_properties(trtmc_dataset_test_family PROPERTIES
      OUTPUT_NAME trtmc_model_dataset_fixture
      LIBRARY_OUTPUT_DIRECTORY "${_trtmc_api_test_runtime_root}"
      BUILD_RPATH "\$ORIGIN/../.."
    )
    add_executable(test_dataset_benchmark_e2e apps/benchmark/tests/native/test_dataset_benchmark_e2e.cpp)
    target_link_libraries(test_dataset_benchmark_e2e PRIVATE nlohmann_json::nlohmann_json)
    target_compile_options(test_dataset_benchmark_e2e PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(test_dataset_benchmark_e2e trtmc_dataset_benchmark
      trtmc_api_test_backend trtmc_api_test_family trtmc_dataset_test_family)
    add_test(NAME dataset_benchmark_e2e COMMAND test_dataset_benchmark_e2e
      $<TARGET_FILE:trtmc_dataset_benchmark> "${_trtmc_api_test_runtime_root}")
    set_tests_properties(dataset_benchmark_e2e PROPERTIES LABELS "cpu;e2e")

    add_executable(test_benchmark_worker_e2e
      apps/benchmark/tests/native/test_benchmark_worker_e2e.cpp
    )
    target_link_libraries(test_benchmark_worker_e2e PRIVATE nlohmann_json::nlohmann_json)
    target_compile_options(test_benchmark_worker_e2e PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(test_benchmark_worker_e2e
      trtmc_benchmark_worker
      trtmc_test_backend_fake
      trtmc_test_family_fake
      trtmc_api_test_family
      trtmc_api_test_backend
      trtmc_api_test_text_family
      trtmc_api_test_language_family
      trtmc_api_test_numeric_family
    )
    add_test(NAME benchmark_worker_e2e
      COMMAND test_benchmark_worker_e2e
        $<TARGET_FILE:trtmc_benchmark_worker>
        "${_trtmc_test_runtime_root}"
        "${_trtmc_api_test_runtime_root}"
    )
    set_tests_properties(benchmark_worker_e2e PROPERTIES LABELS "cpu;e2e")

    add_executable(test_benchmark_audio_e2e apps/benchmark/tests/native/test_benchmark_audio_e2e.cpp)
    target_include_directories(test_benchmark_audio_e2e PRIVATE
      ${PROJECT_SOURCE_DIR}/apps ${PROJECT_SOURCE_DIR}/core/runtime/include)
    target_link_libraries(test_benchmark_audio_e2e PRIVATE trtmc_media_io nlohmann_json::nlohmann_json)
    target_compile_options(test_benchmark_audio_e2e PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(test_benchmark_audio_e2e trtmc_benchmark_worker trtmc_api_test_backend
      trtmc_api_test_audio_family trtmc_api_test_speech_family)
    add_test(NAME benchmark_audio_e2e COMMAND test_benchmark_audio_e2e
      $<TARGET_FILE:trtmc_benchmark_worker> "${_trtmc_api_test_runtime_root}")
    set_tests_properties(benchmark_audio_e2e PROPERTIES LABELS "cpu;e2e")

    add_executable(test_benchmark_remaining_e2e apps/benchmark/tests/native/test_benchmark_remaining_e2e.cpp)
    target_link_libraries(test_benchmark_remaining_e2e PRIVATE nlohmann_json::nlohmann_json)
    target_compile_options(test_benchmark_remaining_e2e PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(test_benchmark_remaining_e2e trtmc_benchmark_worker trtmc_api_test_backend
      trtmc_api_test_perception_family trtmc_api_test_features_family trtmc_api_test_image_family
      trtmc_api_test_video_family trtmc_api_test_action_family)
    add_test(NAME benchmark_remaining_e2e COMMAND test_benchmark_remaining_e2e
      $<TARGET_FILE:trtmc_benchmark_worker> "${_trtmc_api_test_runtime_root}")
    set_tests_properties(benchmark_remaining_e2e PROPERTIES LABELS "cpu;e2e")
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
  target_compile_options(test_byok_shape_spec PRIVATE -Wall -Wextra -Wpedantic)
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
    CUDA::cudart
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
      CUDA::cudart
      ${TRTMC_TVM_FFI_LIBRARY}
    )
    target_compile_definitions(test_byok_tvm_ffi PRIVATE
      TRTMC_HAS_TRT=1
      TRTMC_HAS_TVM_FFI=1
    )
    target_compile_options(test_byok_tvm_ffi PRIVATE -Wall -Wextra -Wpedantic)
    add_dependencies(test_byok_tvm_ffi trtmc_byok_identity_copy)
    add_test(NAME byok_tvm_ffi
      COMMAND test_byok_tvm_ffi $<TARGET_FILE:trtmc_byok_identity_copy>
    )
    set_tests_properties(byok_tvm_ffi PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
  endif()
endif()

include(CMakePackageConfigHelpers)

install(TARGETS trtmc_core trtmc_runtime
  EXPORT trtmcRuntimeTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(TARGETS trtmc_backend_trt
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
)
if(TARGET trtmc_backend_rtx)
  install(TARGETS trtmc_backend_rtx
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
  )
endif()
if(TRTMC_HAS_TVM_FFI)
  install(TARGETS trtmc_byok_tvm_ffi
    EXPORT trtmcRuntimeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
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
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc COMPONENT runtime
)
install(FILES
  core/runtime/include/trtmc/runtime/device_tensor.h
  core/runtime/include/trtmc/runtime/family_factory.h
  core/runtime/include/trtmc/runtime/family_loader.h
  core/runtime/include/trtmc/runtime/tensor.h
  core/runtime/include/trtmc/runtime/trt_backend.h
  core/runtime/include/trtmc/runtime/trt_module.h
  core/runtime/include/trtmc/runtime/span.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc/runtime COMPONENT runtime
)
install(DIRECTORY core/runtime/include/trtmc/internal/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc/internal COMPONENT runtime
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/trtmcConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/trtmcConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/trtmc
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/trtmcConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ExactVersion
)
install(EXPORT trtmcRuntimeTargets
  FILE trtmcRuntimeTargets.cmake
  NAMESPACE trtmc::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/trtmc
  COMPONENT runtime
)
install(TARGETS trtmc_c
  EXPORT trtmcTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sdk
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT sdk NAMELINK_COMPONENT sdk
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT sdk
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY core/api/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT sdk)
install(FILES core/runtime/include/trtmc/runtime/span.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/trtmc/runtime COMPONENT sdk
)
install(EXPORT trtmcTargets
  FILE trtmcTargets.cmake
  NAMESPACE trtmc::
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/trtmc
  COMPONENT sdk
)
install(FILES
  "${PROJECT_BINARY_DIR}/trtmcConfig.cmake"
  "${PROJECT_BINARY_DIR}/trtmcConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cmake/trtmc
  COMPONENT sdk
)
