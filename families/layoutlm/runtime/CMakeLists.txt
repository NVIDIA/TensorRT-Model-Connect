# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Unicode general categories of the GPT-2 pre-tokenizer come from ICU. Tesseract
# is not a build dependency: the runtime loads the user-installed
# libtesseract.so.5 on first use.
find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_layoutlm SHARED
  document_qa.cpp
  pipeline.cpp
  plugin.cpp
  tesseract.cpp
  tokenizer.cpp
)
target_include_directories(trtmc_model_layoutlm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_layoutlm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_layoutlm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ICU::uc
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_layoutlm PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_layoutlm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_layoutlm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # Host-side document QA against reference goldens: no GPU, no Tesseract.
  add_executable(test_layoutlm_document_qa
    ${PROJECT_SOURCE_DIR}/families/layoutlm/tests/cpp/test_document_qa.cpp
    document_qa.cpp
    tesseract.cpp
    tokenizer.cpp
  )
  target_include_directories(test_layoutlm_document_qa PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_layoutlm_document_qa PRIVATE
    nlohmann_json::nlohmann_json
    ICU::uc
    ${CMAKE_DL_LIBS}
  )
  target_compile_options(test_layoutlm_document_qa PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_layoutlm_document_qa PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME layoutlm_document_qa
    COMMAND test_layoutlm_document_qa
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/families/layoutlm
  )
  set_tests_properties(layoutlm_document_qa PROPERTIES LABELS cpu)

  foreach(_language IN ITEMS c cpp)
    add_executable(test_layoutlm_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/layoutlm/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_layoutlm_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_layoutlm_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_layoutlm_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_layoutlm_document_qa test_layoutlm_sdk_c test_layoutlm_sdk_cpp)
endif()
