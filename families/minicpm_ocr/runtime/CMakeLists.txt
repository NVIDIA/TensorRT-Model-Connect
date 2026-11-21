# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The tokenizer applies the checkpoint's NFC normalizer and Split pattern with ICU.
find_package(ICU REQUIRED COMPONENTS uc i18n)

add_library(trtmc_model_minicpm_ocr SHARED
  bpe_tokenizer.cpp
  decoder.cpp
  image_processor.cpp
  model.cpp
  plugin.cpp
  prompt.cpp
  vision_encoder.cpp
)

target_include_directories(trtmc_model_minicpm_ocr
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_minicpm_ocr SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_minicpm_ocr PRIVATE
  trtmc_core
  ICU::uc
  ICU::i18n
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_minicpm_ocr PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_minicpm_ocr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minicpm_ocr
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # Host-only checks of preprocessing and prompt expansion; no GPU or bundle.
  add_executable(test_minicpm_ocr_host
    ${PROJECT_SOURCE_DIR}/families/minicpm_ocr/tests/cpp/test_minicpm_ocr_host.cpp
    image_processor.cpp
    prompt.cpp
  )
  target_include_directories(test_minicpm_ocr_host PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_minicpm_ocr_host PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME minicpm_ocr_host COMMAND test_minicpm_ocr_host)
  set_tests_properties(minicpm_ocr_host PROPERTIES LABELS "cpu")

  # Host-only checks of the tokenizer's NFC normalizer and Split pattern.
  add_executable(test_minicpm_ocr_tokenizer
    ${PROJECT_SOURCE_DIR}/families/minicpm_ocr/tests/cpp/test_minicpm_ocr_tokenizer.cpp
    bpe_tokenizer.cpp
  )
  target_include_directories(test_minicpm_ocr_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_minicpm_ocr_tokenizer PRIVATE
    nlohmann_json::nlohmann_json
    ICU::uc
    ICU::i18n
  )
  target_compile_options(test_minicpm_ocr_tokenizer PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME minicpm_ocr_tokenizer COMMAND test_minicpm_ocr_tokenizer)
  set_tests_properties(minicpm_ocr_tokenizer PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_minicpm_ocr_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/minicpm_ocr/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_minicpm_ocr_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_minicpm_ocr_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_minicpm_ocr_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_minicpm_ocr_host test_minicpm_ocr_sdk_c test_minicpm_ocr_sdk_cpp)
  add_dependencies(test_minicpm_ocr_tokenizer test_minicpm_ocr_sdk_c test_minicpm_ocr_sdk_cpp)
endif()
