# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_paligemma SHARED
  image_preprocess.cpp
  paligemma_bpe.cpp
  plugin.cpp
  prompt.cpp
)
target_include_directories(trtmc_model_paligemma PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_paligemma SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_paligemma PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_paligemma PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_paligemma PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_paligemma LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  # Public C++ SDK consumer the family E2E drives through the images_text_to_text Task.
  add_executable(test_paligemma_sdk_cpp
    ${PROJECT_SOURCE_DIR}/families/paligemma/tests/sdk_consumer.cpp
  )
  target_link_libraries(test_paligemma_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_paligemma_sdk_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_paligemma_sdk_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  # Run, and built, by test_e2e.py against a built bundle, so it is a target rather than a CTest case.
  add_executable(paligemma_tokenizer_check
    ${PROJECT_SOURCE_DIR}/families/paligemma/tests/cpp/tokenizer_check.cpp
    ${PROJECT_SOURCE_DIR}/families/paligemma/runtime/paligemma_bpe.cpp
    ${PROJECT_SOURCE_DIR}/families/paligemma/runtime/prompt.cpp
  )
  target_include_directories(paligemma_tokenizer_check PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(paligemma_tokenizer_check PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(paligemma_tokenizer_check PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(paligemma_tokenizer_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  add_executable(test_paligemma_runtime
    ${PROJECT_SOURCE_DIR}/families/paligemma/tests/cpp/test_runtime.cpp
    ${PROJECT_SOURCE_DIR}/families/paligemma/runtime/image_preprocess.cpp
    ${PROJECT_SOURCE_DIR}/families/paligemma/runtime/paligemma_bpe.cpp
    ${PROJECT_SOURCE_DIR}/families/paligemma/runtime/prompt.cpp
  )
  target_include_directories(test_paligemma_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_paligemma_runtime PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_paligemma_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME paligemma_runtime COMMAND test_paligemma_runtime
    ${PROJECT_SOURCE_DIR}/families/paligemma/tests/data/pillow_bicubic.json)
  set_tests_properties(paligemma_runtime PROPERTIES LABELS "cpu")
endif()
