# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_gemma3n SHARED plugin.cpp prompt.cpp tokenizer.cpp image.cpp)
target_include_directories(trtmc_model_gemma3n PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_gemma3n SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_gemma3n PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_gemma3n PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_gemma3n PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_gemma3n LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  # Source-only checks of the family's host processing (no GPU, no checkpoint).
  add_executable(test_gemma3n_processing
    ../tests/cpp/test_processing.cpp prompt.cpp tokenizer.cpp image.cpp)
  target_include_directories(test_gemma3n_processing PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_gemma3n_processing PRIVATE nlohmann_json::nlohmann_json)
  target_compile_definitions(test_gemma3n_processing PRIVATE
    GEMMA3N_TEST_DATA="${PROJECT_SOURCE_DIR}/families/gemma3n/tests/data")
  # The family E2E also runs it on the checkpoint's tokenizer (--prompt-ids).
  set_target_properties(test_gemma3n_processing PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_gemma3n_processing COMMAND test_gemma3n_processing)

  # Direct public C and C++ SDK consumers, run by the family E2E.
  add_executable(test_gemma3n_sdk_c ${PROJECT_SOURCE_DIR}/families/gemma3n/tests/sdk_consumer.c)
  target_link_libraries(test_gemma3n_sdk_c PRIVATE trtmc_c)
  add_executable(test_gemma3n_sdk_cpp ${PROJECT_SOURCE_DIR}/families/gemma3n/tests/sdk_consumer.cpp)
  target_link_libraries(test_gemma3n_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  foreach(_language IN ITEMS c cpp)
    target_compile_options(test_gemma3n_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_gemma3n_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach()
  add_dependencies(trtmc_model_gemma3n test_gemma3n_sdk_c test_gemma3n_sdk_cpp)
endif()
