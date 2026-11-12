# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_qwen3_vl_moe SHARED
  bpe_tokenizer.cpp
  image_preprocessor.cpp
  kv_cache.cpp
  lora_peft_artifact.cpp
  lora_peft_loader.cpp
  lora_runtime.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
  task_sdk.cpp
)

set_source_files_properties(image_preprocessor.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)

target_include_directories(trtmc_model_qwen3_vl_moe
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen3_vl_moe SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_qwen3_vl_moe PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ICU::uc
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_qwen3_vl_moe PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen3_vl_moe PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen3_vl_moe
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_vl_moe_runtime ../tests/cpp/test_runtime.cpp)
  target_include_directories(test_qwen3_vl_moe_runtime PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_qwen3_vl_moe_runtime PRIVATE
    trtmc_model_qwen3_vl_moe trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
  add_test(NAME qwen3_vl_moe_runtime COMMAND test_qwen3_vl_moe_runtime)
  set_tests_properties(qwen3_vl_moe_runtime PROPERTIES LABELS cpu)
  add_test(NAME qwen3_vl_moe_kv_capacity COMMAND test_qwen3_vl_moe_runtime --gpu-cache)
  set_tests_properties(qwen3_vl_moe_kv_capacity PROPERTIES LABELS gpu)

  add_executable(test_qwen3_vl_moe_threads ../tests/cpp/test_preprocess_threads.cpp)
  target_include_directories(test_qwen3_vl_moe_threads PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_qwen3_vl_moe_threads SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
  target_link_libraries(test_qwen3_vl_moe_threads PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
  add_test(NAME qwen3_vl_moe_thread_unwind COMMAND test_qwen3_vl_moe_threads)
  set_tests_properties(qwen3_vl_moe_thread_unwind PROPERTIES LABELS cpu)

  add_executable(test_qwen3_vl_moe_tokenizer ../tests/cpp/tokenizer_probe.cpp)
  target_include_directories(test_qwen3_vl_moe_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_qwen3_vl_moe_tokenizer PRIVATE nlohmann_json::nlohmann_json ICU::uc)

  add_executable(test_qwen3_vl_moe_sdk ../tests/cpp/run_task_sdk.cpp)
  target_include_directories(test_qwen3_vl_moe_sdk PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  target_link_libraries(test_qwen3_vl_moe_sdk PRIVATE trtmc_c)
  add_dependencies(test_qwen3_vl_moe_sdk trtmc_model_qwen3_vl_moe trtmc_backend_trt)
  set_target_properties(test_qwen3_vl_moe_tokenizer test_qwen3_vl_moe_sdk
    test_qwen3_vl_moe_runtime test_qwen3_vl_moe_threads PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "\$ORIGIN"
  )
endif()
