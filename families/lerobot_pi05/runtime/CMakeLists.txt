# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_lerobot_pi05 SHARED
  normalization.cpp
  paligemma_bpe.cpp
  pipeline.cpp
  plugin.cpp
  prompt.cpp
)

target_include_directories(trtmc_model_lerobot_pi05
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lerobot_pi05 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lerobot_pi05 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_lerobot_pi05 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_lerobot_pi05 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lerobot_pi05
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(trtmc_lerobot_pi05_qualification
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/tests/cpp/qualification.cpp
  )
  target_include_directories(trtmc_lerobot_pi05_qualification PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_lerobot_pi05_qualification PRIVATE trtmc_c)
  target_compile_options(trtmc_lerobot_pi05_qualification PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(trtmc_lerobot_pi05_qualification
    trtmc_model_lerobot_pi05 trtmc_backend_trt)

  # Run by test_e2e.py against a built bundle, so it is a target rather than a CTest case.
  add_executable(trtmc_lerobot_pi05_tokenizer_check
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/tests/cpp/tokenizer_check.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/runtime/paligemma_bpe.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/runtime/prompt.cpp
  )
  target_include_directories(trtmc_lerobot_pi05_tokenizer_check PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(trtmc_lerobot_pi05_tokenizer_check PRIVATE
    trtmc_core
    nlohmann_json::nlohmann_json
  )
  target_compile_options(trtmc_lerobot_pi05_tokenizer_check PRIVATE -Wall -Wextra -Wpedantic -Werror)

  add_executable(test_lerobot_pi05_prompt
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/tests/cpp/test_prompt.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/runtime/prompt.cpp
  )
  target_include_directories(test_lerobot_pi05_prompt PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_lerobot_pi05_prompt PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME lerobot_pi05_prompt COMMAND test_lerobot_pi05_prompt)
  set_tests_properties(lerobot_pi05_prompt PROPERTIES LABELS "cpu")

  add_executable(test_lerobot_pi05_normalization
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/tests/cpp/test_normalization.cpp
    ${PROJECT_SOURCE_DIR}/families/lerobot_pi05/runtime/normalization.cpp
  )
  target_include_directories(test_lerobot_pi05_normalization PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_lerobot_pi05_normalization PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME lerobot_pi05_normalization COMMAND test_lerobot_pi05_normalization)
  set_tests_properties(lerobot_pi05_normalization PROPERTIES LABELS "cpu")
endif()
