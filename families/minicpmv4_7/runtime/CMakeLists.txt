# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)

add_library(trtmc_model_minicpmv4_7 SHARED
  bpe_tokenizer.cpp
  image_preprocess.cpp
  plugin.cpp
  prompt.cpp
  sampler.cpp
  state.cpp
)
target_include_directories(trtmc_model_minicpmv4_7 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_minicpmv4_7 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_minicpmv4_7 PRIVATE
  trtmc_core
  ICU::uc
  ICU::i18n
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_minicpmv4_7 PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
set_target_properties(trtmc_model_minicpmv4_7 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minicpmv4_7 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# The library carries code adapted from BSD-3-Clause and Apache-2.0 sources; ship their notices with it.
install(FILES ${PROJECT_SOURCE_DIR}/families/minicpmv4_7/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/minicpmv4_7)

if(TRTMC_BUILD_TESTS)
  # Host-only preprocessing, prompt and canvas-position contract (no GPU, no bundle).
  add_executable(test_minicpmv4_7_prompt
    ${PROJECT_SOURCE_DIR}/families/minicpmv4_7/tests/cpp/test_prompt.cpp
    image_preprocess.cpp
    prompt.cpp
  )
  target_include_directories(test_minicpmv4_7_prompt PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_minicpmv4_7_prompt PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_minicpmv4_7_prompt PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME minicpmv4_7_prompt COMMAND test_minicpmv4_7_prompt)
  set_tests_properties(minicpmv4_7_prompt PROPERTIES LABELS "cpu")

  # Explicit E2E helper: the runtime's host preprocessing for one request, dumped for comparison
  # with the provider processor. Never a default CTest case.
  add_executable(minicpmv4_7_preprocess_dump
    ${PROJECT_SOURCE_DIR}/families/minicpmv4_7/tests/cpp/preprocess_dump.cpp
    bpe_tokenizer.cpp
    image_preprocess.cpp
    prompt.cpp
  )
  target_include_directories(minicpmv4_7_preprocess_dump PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(minicpmv4_7_preprocess_dump PRIVATE trtmc_core ICU::uc ICU::i18n
    nlohmann_json::nlohmann_json)
  target_compile_options(minicpmv4_7_preprocess_dump PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(minicpmv4_7_preprocess_dump PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_dependencies(test_minicpmv4_7_prompt minicpmv4_7_preprocess_dump)
endif()
