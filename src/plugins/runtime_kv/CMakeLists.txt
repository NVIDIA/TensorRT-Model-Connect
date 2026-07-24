# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "Runtime KV TensorRT plugin: skipped (CUDA compiler unavailable)")
  return()
endif()

option(
  TRTMC_ENABLE_CUDNN_RUNTIME_KV_SDPA
  "Enable the qualified cuDNN 9.20 SDPA path in the runtime-KV plugin"
  ON
)
set(
  TRTMC_CUDNN_FRONTEND_INCLUDE_DIR
  ""
  CACHE PATH
  "Offline cuDNN Frontend v1.21.0 include directory"
)

set(_trtmc_runtime_kv_cudnn_sdpa OFF)
set(_trtmc_runtime_kv_search_paths ${_trtmc_default_search_paths})
if(TRTMC_ENABLE_CUDNN_RUNTIME_KV_SDPA AND _trtmc_python)
  execute_process(
    COMMAND "${_trtmc_python}" -c
      "import importlib.util,sys; s=importlib.util.find_spec('nvidia.cudnn'); sys.stdout.write(next(iter(s.submodule_search_locations), '') if s and s.submodule_search_locations else '')"
    OUTPUT_VARIABLE _trtmc_cudnn_python_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_trtmc_cudnn_python_root)
    list(APPEND _trtmc_runtime_kv_search_paths
      "${_trtmc_cudnn_python_root}"
      "${_trtmc_cudnn_python_root}/include"
      "${_trtmc_cudnn_python_root}/lib"
    )
  endif()
endif()

if(TRTMC_ENABLE_CUDNN_RUNTIME_KV_SDPA)
  find_path(TRTMC_CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    PATHS ${_trtmc_runtime_kv_search_paths}
    PATH_SUFFIXES "" include
  )
  find_library(TRTMC_CUDNN_LIBRARY
    NAMES cudnn libcudnn.so.9
    PATHS ${_trtmc_runtime_kv_search_paths}
    PATH_SUFFIXES "" lib lib64
  )
  # The core has already selected NVRTC from the canonical directory of its
  # CUDART. Reuse exactly that DSO so builder, core, and runtime plugin cannot
  # drift through same-SONAME search-order differences.
  set(TRTMC_NVRTC_LIBRARY "${TRTMC_COHERENT_NVRTC_LIBRARY}")

  set(_trtmc_cudnn_9_20_or_newer OFF)
  if(TRTMC_CUDNN_INCLUDE_DIR AND
     EXISTS "${TRTMC_CUDNN_INCLUDE_DIR}/cudnn_version.h")
    file(READ
      "${TRTMC_CUDNN_INCLUDE_DIR}/cudnn_version.h"
      _trtmc_cudnn_version_header)
    string(REGEX MATCH
      "#define[ \t]+CUDNN_MAJOR[ \t]+([0-9]+)"
      _trtmc_cudnn_major_match
      "${_trtmc_cudnn_version_header}")
    set(_trtmc_cudnn_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH
      "#define[ \t]+CUDNN_MINOR[ \t]+([0-9]+)"
      _trtmc_cudnn_minor_match
      "${_trtmc_cudnn_version_header}")
    set(_trtmc_cudnn_minor "${CMAKE_MATCH_1}")
    if(_trtmc_cudnn_major STREQUAL "9" AND
       _trtmc_cudnn_minor GREATER_EQUAL 20)
      set(_trtmc_cudnn_9_20_or_newer ON)
    endif()
  endif()

  if(_trtmc_cudnn_9_20_or_newer AND
     TRTMC_CUDNN_LIBRARY AND TRTMC_NVRTC_LIBRARY)
    if(TRTMC_CUDNN_FRONTEND_INCLUDE_DIR)
      if(NOT EXISTS
         "${TRTMC_CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
        message(FATAL_ERROR
          "TRTMC_CUDNN_FRONTEND_INCLUDE_DIR does not contain cudnn_frontend.h")
      endif()
      set(_trtmc_cudnn_frontend_include
        "${TRTMC_CUDNN_FRONTEND_INCLUDE_DIR}")
    else()
      # Header-only dependency. Populate the exact reviewed revision without
      # configuring upstream samples/tests (which also keeps CMake 3.20
      # compatibility). Offline builds set the cache path above.
      include(FetchContent)
      FetchContent_Declare(
        trtmc_cudnn_frontend
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG 7b9b711c22b6823e87150213ecd8449260db8610
        GIT_SHALLOW FALSE
      )
      FetchContent_GetProperties(trtmc_cudnn_frontend)
      if(NOT trtmc_cudnn_frontend_POPULATED)
        FetchContent_Populate(trtmc_cudnn_frontend)
      endif()
      set(_trtmc_cudnn_frontend_include
        "${trtmc_cudnn_frontend_SOURCE_DIR}/include")
    endif()
    set(_trtmc_runtime_kv_cudnn_sdpa ON)
  endif()
endif()

add_library(trtmc_trt_plugins SHARED
  cudnn_attention.cpp
  native_contiguous_attention_plugin.cu
  native_contiguous_attention_creator.cpp
  runtime_kv_plugin_api.cpp
)
target_include_directories(trtmc_trt_plugins
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(trtmc_trt_plugins SYSTEM PRIVATE
  ${TRTMC_TRT_INCLUDE_DIR}
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_trt_plugins PRIVATE
  ${TRTMC_TRT_LIBRARY}
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
if(_trtmc_runtime_kv_cudnn_sdpa)
  target_include_directories(trtmc_trt_plugins SYSTEM PRIVATE
    ${TRTMC_CUDNN_INCLUDE_DIR}
    ${_trtmc_cudnn_frontend_include}
  )
  target_link_libraries(trtmc_trt_plugins PRIVATE
    ${TRTMC_CUDNN_LIBRARY}
    ${TRTMC_NVRTC_LIBRARY}
    ${CMAKE_DL_LIBS}
  )
  target_compile_definitions(trtmc_trt_plugins PRIVATE
    TRTMC_RUNTIME_KV_CUDNN_SDPA=1
    TRTMC_CUDNN_FRONTEND_REVISION="7b9b711c22b6823e87150213ecd8449260db8610"
    CUDNN_FRONTEND_SKIP_JSON_LIB
  )
  get_filename_component(
    _trtmc_cudnn_library_dir
    "${TRTMC_CUDNN_LIBRARY}"
    DIRECTORY)
else()
  target_compile_definitions(trtmc_trt_plugins PRIVATE
    TRTMC_RUNTIME_KV_CUDNN_SDPA=0
    TRTMC_CUDNN_FRONTEND_REVISION="7b9b711c22b6823e87150213ecd8449260db8610"
  )
  set(_trtmc_cudnn_library_dir "")
endif()
target_compile_options(trtmc_trt_plugins
  PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_trt_plugins PROPERTIES
  OUTPUT_NAME trtmc_trt_plugins
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH
    "\$ORIGIN;${_trtmc_cuda_runtime_library_dir};${_trtmc_cudnn_library_dir}"
  INSTALL_RPATH
    "\$ORIGIN;${_trtmc_cuda_runtime_library_dir};\$ORIGIN/../../tensorrt_libs;\$ORIGIN/../../nvidia/cudnn/lib;\$ORIGIN/../../nvidia/cu13/lib"
  CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS trtmc_trt_plugins
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# The rejected aliased-I/O append prototype remains available only as a test
# fixture. It is deliberately excluded from the installed production DSO so
# its v1 creator cannot become part of the runtime deserialization surface.
add_library(trtmc_native_kv_append_fixture SHARED EXCLUDE_FROM_ALL
  native_kv_append_plugin.cu
  native_kv_append_creator.cpp
)
target_include_directories(trtmc_native_kv_append_fixture
  PRIVATE ${PROJECT_SOURCE_DIR}/src
)
target_include_directories(
  trtmc_native_kv_append_fixture SYSTEM PRIVATE
  ${TRTMC_TRT_INCLUDE_DIR}
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_native_kv_append_fixture PRIVATE
  ${TRTMC_TRT_LIBRARY}
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_native_kv_append_fixture
  PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_native_kv_append_fixture PROPERTIES
  OUTPUT_NAME trtmc_native_kv_append_fixture
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(_trtmc_runtime_kv_cudnn_sdpa)
  message(STATUS
    "Runtime KV TensorRT plugin: enabled with cuDNN Frontend v1.21.0 "
    "(7b9b711c22b6823e87150213ecd8449260db8610)")
else()
  message(WARNING
    "Runtime KV TensorRT plugin: cuDNN 9.20 SDPA unavailable; "
    "qualified NativeContiguousAttention creation will fail closed")
endif()
