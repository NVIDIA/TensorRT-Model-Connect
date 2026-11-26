# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_openai_gpt SHARED
  engine_contract.cpp
  kv_cache.cpp
  model.cpp
  plugin.cpp
  sampler.cpp
  text_config.cpp
  tokenizer.cpp
)
target_include_directories(trtmc_model_openai_gpt PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_openai_gpt SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_openai_gpt PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_openai_gpt PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_openai_gpt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_openai_gpt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# The tokenizer reimplements Hugging Face Tokenizers procedures, and its normalization tables
# derive from the Unicode Character Database.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/openai_gpt)

if(TRTMC_BUILD_TESTS)
  # Host-only: tokenizer contract, encoding and decoding, sampling, Config and the engine
  # contract (against a fake module). Also the native side of
  # tests/test_tokenizer_parity.py, which finds it in the build directory.
  add_executable(test_openai_gpt_text_runtime
    ${PROJECT_SOURCE_DIR}/families/openai_gpt/tests/cpp/test_openai_gpt_text_runtime.cpp
    engine_contract.cpp
    sampler.cpp
    text_config.cpp
    tokenizer.cpp
  )
  target_include_directories(test_openai_gpt_text_runtime PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_openai_gpt_text_runtime SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_openai_gpt_text_runtime PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_openai_gpt_text_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_openai_gpt_text_runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_openai_gpt_text_runtime COMMAND test_openai_gpt_text_runtime)
  set_tests_properties(test_openai_gpt_text_runtime PROPERTIES LABELS "cpu")

  # Direct public C and C++ SDK callers, run by the family E2E.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_openai_gpt_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/openai_gpt/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_openai_gpt_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_openai_gpt_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_openai_gpt_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach()
  add_dependencies(test_openai_gpt_text_runtime test_openai_gpt_sdk_c test_openai_gpt_sdk_cpp)
endif()
