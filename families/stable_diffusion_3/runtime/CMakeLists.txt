# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_stable_diffusion_3 SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  unigram_tokenizer.cpp
)

target_include_directories(trtmc_model_stable_diffusion_3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_stable_diffusion_3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_stable_diffusion_3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_stable_diffusion_3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_stable_diffusion_3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_stable_diffusion_3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# The tokenizers' Unicode tables (unicode_data.h) are Unicode data.
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/stable_diffusion_3
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_stable_diffusion_3_sampling
    ${PROJECT_SOURCE_DIR}/families/stable_diffusion_3/tests/cpp/test_sampling.cpp
  )
  target_include_directories(test_stable_diffusion_3_sampling PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_stable_diffusion_3_sampling PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME stable_diffusion_3_sampling COMMAND test_stable_diffusion_3_sampling)

  add_executable(test_stable_diffusion_3_unicode
    ${PROJECT_SOURCE_DIR}/families/stable_diffusion_3/tests/cpp/test_unicode.cpp
  )
  target_include_directories(test_stable_diffusion_3_unicode PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_stable_diffusion_3_unicode PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME stable_diffusion_3_unicode COMMAND test_stable_diffusion_3_unicode)

  # Prompt ids through the runtime tokenizers; tests/test_e2e.py compares them
  # with the checkpoint's tokenizers.
  add_executable(test_stable_diffusion_3_tokenize
    ${PROJECT_SOURCE_DIR}/families/stable_diffusion_3/tests/cpp/tokenize_prompts.cpp
    bpe_tokenizer.cpp
    unigram_tokenizer.cpp
  )
  target_include_directories(test_stable_diffusion_3_tokenize PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_stable_diffusion_3_tokenize PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_stable_diffusion_3_tokenize PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(test_stable_diffusion_3_tokenize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )

  foreach(_language IN ITEMS c cpp)
    add_executable(test_stable_diffusion_3_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/stable_diffusion_3/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_stable_diffusion_3_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_stable_diffusion_3_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_stable_diffusion_3_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_stable_diffusion_3
    test_stable_diffusion_3_sdk_c test_stable_diffusion_3_sdk_cpp test_stable_diffusion_3_tokenize)
  # Premerge E2E loads a prebuilt family library and builds only the CTest
  # targets and SDK consumers by name.
  add_dependencies(test_stable_diffusion_3_unicode test_stable_diffusion_3_tokenize)
endif()
