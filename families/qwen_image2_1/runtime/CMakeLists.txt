# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen_image2_1 SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  runtime_config.cpp
  sampling.cpp
)

target_include_directories(trtmc_model_qwen_image2_1
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen_image2_1 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_qwen_image2_1 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_qwen_image2_1 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen_image2_1 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen_image2_1
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen_image2_1_sampling
    ${PROJECT_SOURCE_DIR}/families/qwen_image2_1/tests/cpp/test_sampling.cpp
  )
  target_include_directories(test_qwen_image2_1_sampling PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_qwen_image2_1_sampling SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_qwen_image2_1_sampling PRIVATE
    trtmc_model_qwen_image2_1
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_qwen_image2_1_sampling PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME qwen_image2_1_sampling COMMAND test_qwen_image2_1_sampling)
  set_tests_properties(qwen_image2_1_sampling PROPERTIES LABELS "cpu")

  # Public SDK consumers for the family E2E. They link only the public C runtime.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_qwen_image2_1_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/qwen_image2_1/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_qwen_image2_1_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_qwen_image2_1_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_qwen_image2_1_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_qwen_image2_1_sampling
    test_qwen_image2_1_sdk_c
    test_qwen_image2_1_sdk_cpp
  )
endif()
