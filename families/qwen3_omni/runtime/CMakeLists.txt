# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_omni SHARED
  bpe_tokenizer.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  residual_multinomial_kernel.cu
  sampler.cpp
)

set_source_files_properties(
  residual_multinomial_kernel.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_qwen3_omni
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen3_omni SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_qwen3_omni PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_qwen3_omni PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen3_omni PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen3_omni
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen3_omni_audio_plan
    ${PROJECT_SOURCE_DIR}/families/qwen3_omni/tests/cpp/test_qwen3_omni_audio_plan.cpp
  )
  target_include_directories(test_qwen3_omni_audio_plan PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_qwen3_omni_audio_plan PRIVATE trtmc_model_qwen3_omni)
  add_test(NAME test_qwen3_omni_audio_plan COMMAND test_qwen3_omni_audio_plan)

  add_executable(test_qwen3_omni_sampler
    ${PROJECT_SOURCE_DIR}/families/qwen3_omni/tests/cpp/test_qwen3_omni_sampler.cpp
    residual_multinomial_kernel.cu
    sampler.cpp
  )
  set_source_files_properties(
    residual_multinomial_kernel.cu
    PROPERTIES LANGUAGE CUDA
  )
  target_include_directories(test_qwen3_omni_sampler PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_qwen3_omni_sampler SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_qwen3_omni_sampler PRIVATE ${TRTMC_CUDART_LIBRARY})
  target_compile_options(test_qwen3_omni_sampler PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
  )
  add_test(NAME test_qwen3_omni_sampler COMMAND test_qwen3_omni_sampler)
  set_tests_properties(test_qwen3_omni_sampler PROPERTIES SKIP_RETURN_CODE 77)
endif()
