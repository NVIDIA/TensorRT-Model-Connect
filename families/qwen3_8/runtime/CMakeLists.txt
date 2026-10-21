# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_8 SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  hybrid_state.cpp
  kv_cache.cpp
  mtp_scheduler.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  prefill_runner.cpp
  recurrent_state.cpp
  sampler.cpp
)
target_include_directories(trtmc_model_qwen3_8 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_link_libraries(trtmc_model_qwen3_8 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_qwen3_8 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)
set_target_properties(trtmc_model_qwen3_8 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen3_8 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# TensorRT plugins for engines built with the cuTile GDN kernels. Such an engine carries its
# cubins and names the plugin; the plugin library only has to be loaded before the engine is
# deserialized, which plugin_helpers.cpp does from the directory of this library. It calls the
# CUDA driver API, so it is skipped where libcuda is not installed (for example CPU-only builds).
option(TRTMC_QWEN3_8_NATIVE_PLUGIN "Build the qwen3_8 cuTile GDN TensorRT plugin library" ON)
if(TRTMC_QWEN3_8_NATIVE_PLUGIN)
  find_library(TRTMC_QWEN3_8_CUDA_DRIVER_LIBRARY NAMES cuda
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64
          /usr/local/cuda/lib64/stubs)
  if(TRTMC_QWEN3_8_CUDA_DRIVER_LIBRARY AND TRTMC_TRT_INCLUDE_DIR AND TRTMC_CUDA_INCLUDE_DIR)
    add_library(trtmc_qwen3_8_native_plugin SHARED
      ${CMAKE_CURRENT_SOURCE_DIR}/../native_plugins/gdn_tilegym_plugin.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../native_plugins/gdn_prefill_plugin.cpp
    )
    target_include_directories(trtmc_qwen3_8_native_plugin SYSTEM PRIVATE
      ${TRTMC_TRT_INCLUDE_DIR}
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(trtmc_qwen3_8_native_plugin PRIVATE
      ${TRTMC_TRT_LIBRARY}
      ${TRTMC_QWEN3_8_CUDA_DRIVER_LIBRARY}
    )
    target_compile_options(trtmc_qwen3_8_native_plugin PRIVATE -O3 -Wall -Wextra -Wpedantic)
    set_target_properties(trtmc_qwen3_8_native_plugin PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      BUILD_RPATH "\$ORIGIN"
      INSTALL_RPATH "\$ORIGIN"
    )
    add_dependencies(trtmc_model_qwen3_8 trtmc_qwen3_8_native_plugin)
    install(TARGETS trtmc_qwen3_8_native_plugin LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  else()
    message(STATUS "qwen3_8: native GDN plugin library not built (needs libcuda, TensorRT and CUDA headers)")
  endif()
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_8_recurrent_output_initializers
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_recurrent_output_initializers.cpp
  )
  target_include_directories(test_qwen3_8_recurrent_output_initializers PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_qwen3_8_recurrent_output_initializers PRIVATE
    trtmc_model_qwen3_8
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME test_qwen3_8_recurrent_output_initializers
    COMMAND test_qwen3_8_recurrent_output_initializers
  )

  # Keep native mock fixtures in the existing test, but load the family DSO
  # through the public loader for real bundles. Prelinking the optional Edge
  # DSO exposes its private CUDA symbols globally, unlike the ordinary CLI.
  add_executable(test_qwen3_8_recurrent_pipeline
    ${PROJECT_SOURCE_DIR}/families/qwen3_8/tests/cpp/test_qwen3_8_recurrent_pipeline.cpp
    bpe_tokenizer.cpp
    chat_templates.cpp
    hybrid_state.cpp
    kv_cache.cpp
    mtp_scheduler.cpp
    pipeline.cpp
    prefill_runner.cpp
    recurrent_state.cpp
    sampler.cpp
  )
  target_include_directories(test_qwen3_8_recurrent_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_qwen3_8_recurrent_pipeline SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_qwen3_8_recurrent_pipeline PRIVATE
    trtmc_backend_trt
    trtmc_core
    trtmc::c
    nlohmann_json::nlohmann_json
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUDART_LIBRARY}
  )
  add_test(NAME test_qwen3_8_recurrent_pipeline COMMAND test_qwen3_8_recurrent_pipeline)
  set_tests_properties(test_qwen3_8_recurrent_pipeline PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )
endif()

include(edge_llm/Adapter.cmake)
