# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_asr SHARED
  frontend.cpp
  pipeline.cpp
  plugin.cpp
  runtime_config.cpp
  text.cpp
)

target_include_directories(trtmc_model_qwen3_asr
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen3_asr SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_qwen3_asr PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_qwen3_asr PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_qwen3_asr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen3_asr
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # Front end, text and prompt helpers: CPU-only, no engine or GPU.
  add_executable(test_qwen3_asr_runtime
    ${PROJECT_SOURCE_DIR}/families/qwen3_asr/tests/cpp/test_runtime.cpp
    ${PROJECT_SOURCE_DIR}/families/qwen3_asr/runtime/frontend.cpp
    ${PROJECT_SOURCE_DIR}/families/qwen3_asr/runtime/runtime_config.cpp
    ${PROJECT_SOURCE_DIR}/families/qwen3_asr/runtime/text.cpp
  )
  target_include_directories(test_qwen3_asr_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_qwen3_asr_runtime PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_qwen3_asr_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME qwen3_asr_runtime COMMAND test_qwen3_asr_runtime)
  set_tests_properties(qwen3_asr_runtime PROPERTIES LABELS "cpu")

  # Public SDK consumers for the family E2E. They link only the public C runtime.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_qwen3_asr_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/qwen3_asr/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_qwen3_asr_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_qwen3_asr_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_qwen3_asr_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_qwen3_asr_runtime
    test_qwen3_asr_sdk_c
    test_qwen3_asr_sdk_cpp
  )
endif()
