# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

target_sources(trtmc_model_qwen PRIVATE adapter.cpp)

set(TRTMC_EDGE_LLM_SOURCE_DIR "" CACHE PATH
  "Installed TensorRT Edge-LLM source directory for the optional Qwen adapter"
)
set(TRTMC_EDGE_LLM_BUILD_DIR "" CACHE PATH
  "Installed TensorRT Edge-LLM build directory for the optional Qwen adapter"
)
if((TRTMC_EDGE_LLM_SOURCE_DIR AND NOT TRTMC_EDGE_LLM_BUILD_DIR) OR
   (TRTMC_EDGE_LLM_BUILD_DIR AND NOT TRTMC_EDGE_LLM_SOURCE_DIR))
  message(FATAL_ERROR
    "Qwen Edge-LLM requires both TRTMC_EDGE_LLM_SOURCE_DIR and TRTMC_EDGE_LLM_BUILD_DIR"
  )
endif()

set(_trtmc_qwen_edge_real_runtime OFF)
if(TRTMC_EDGE_LLM_SOURCE_DIR AND TRTMC_EDGE_LLM_BUILD_DIR)
  find_file(TRTMC_EDGE_LLM_CORE_LIBRARY
    NAMES libedgellmCore.a
    PATHS
      "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp"
      "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/Release"
      "${TRTMC_EDGE_LLM_BUILD_DIR}/cpp/RelWithDebInfo"
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_file(TRTMC_EDGE_LLM_PLUGIN_LIBRARY
    NAMES libNvInfer_edgellm_plugin.so.1.0 libNvInfer_edgellm_plugin.so
    PATHS
      "${TRTMC_EDGE_LLM_BUILD_DIR}"
      "${TRTMC_EDGE_LLM_BUILD_DIR}/Release"
      "${TRTMC_EDGE_LLM_BUILD_DIR}/RelWithDebInfo"
    NO_DEFAULT_PATH
    REQUIRED
  )
  if(NOT EXISTS "${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp/runtime/llmInferenceRuntime.h")
    message(FATAL_ERROR
      "TRTMC_EDGE_LLM_SOURCE_DIR does not contain the official runtime headers"
    )
  endif()

  add_library(trtmc_qwen_edge_llm_bridge SHARED EXCLUDE_FROM_ALL
    bridge.cpp
    device_link_stub.cu
  )
  target_include_directories(trtmc_qwen_edge_llm_bridge PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${TRTMC_EDGE_LLM_SOURCE_DIR}/cpp
    ${TRTMC_TRT_INCLUDE_DIR}
  )
  target_include_directories(trtmc_qwen_edge_llm_bridge SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_qwen_edge_llm_bridge PRIVATE
    -Wl,--whole-archive
    ${TRTMC_EDGE_LLM_CORE_LIBRARY}
    -Wl,--no-whole-archive
    ${TRTMC_TRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
  )
  target_compile_options(trtmc_qwen_edge_llm_bridge PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(trtmc_qwen_edge_llm_bridge PROPERTIES
    OUTPUT_NAME trtmc_qwen_edge_llm
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "\$ORIGIN"
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_custom_command(TARGET trtmc_qwen_edge_llm_bridge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${TRTMC_EDGE_LLM_PLUGIN_LIBRARY}"
      "${CMAKE_BINARY_DIR}/libNvInfer_edgellm_plugin.so"
  )
  add_custom_target(trtmc_qwen_edge_llm_runtime
    DEPENDS
      trtmc_model_qwen
      trtmc_qwen_edge_llm_bridge
  )
  set(_trtmc_qwen_edge_real_runtime ON)
endif()

if(TRTMC_BUILD_TESTS AND NOT _trtmc_qwen_edge_real_runtime)
  add_library(trtmc_qwen_edge_llm_fake_bridge SHARED EXCLUDE_FROM_ALL
    ../tests/cpp/fake_bridge.cpp
  )
  target_include_directories(trtmc_qwen_edge_llm_fake_bridge PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(trtmc_qwen_edge_llm_fake_bridge PRIVATE
    -Wall -Wextra -Wpedantic
  )
  set_target_properties(trtmc_qwen_edge_llm_fake_bridge PROPERTIES
    OUTPUT_NAME trtmc_qwen_edge_llm
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  add_executable(test_qwen_edge_llm_runtime ../tests/cpp/test_runtime.cpp)
  target_include_directories(test_qwen_edge_llm_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_qwen_edge_llm_runtime PRIVATE
    trtmc_runtime
    trtmc_core
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_qwen_edge_llm_runtime PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_dependencies(test_qwen_edge_llm_runtime
    trtmc_model_qwen
    trtmc_qwen_edge_llm_fake_bridge
  )
  add_test(NAME test_qwen_edge_llm_runtime
    COMMAND test_qwen_edge_llm_runtime "${CMAKE_BINARY_DIR}"
  )
endif()
