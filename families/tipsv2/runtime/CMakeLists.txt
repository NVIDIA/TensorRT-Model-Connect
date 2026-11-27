# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_tipsv2 SHARED
  image_preprocess.cpp
  tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)
target_include_directories(trtmc_model_tipsv2 PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_tipsv2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_tipsv2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_tipsv2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_tipsv2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_tipsv2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
# The library carries code adapted from Pillow (MIT-CMU); ship its notice with it.
install(FILES ${PROJECT_SOURCE_DIR}/families/tipsv2/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/tipsv2
)

if(TRTMC_BUILD_TESTS)
  # The runtime rules that need no engine: Pillow's bilinear resize, the SentencePiece BPE
  # tokenizer, and the asset reader's refusals.
  add_executable(test_tipsv2_runtime
    ${PROJECT_SOURCE_DIR}/families/tipsv2/tests/cpp/test_runtime.cpp
    ${PROJECT_SOURCE_DIR}/families/tipsv2/runtime/image_preprocess.cpp
    ${PROJECT_SOURCE_DIR}/families/tipsv2/runtime/tokenizer.cpp
  )
  target_include_directories(test_tipsv2_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_tipsv2_runtime PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_tipsv2_runtime PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_tipsv2_runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  add_test(NAME tipsv2_runtime COMMAND test_tipsv2_runtime
    ${PROJECT_SOURCE_DIR}/families/tipsv2/tests/data/pillow_bilinear.json)
  set_tests_properties(tipsv2_runtime PROPERTIES LABELS "cpu")

  # Run, and built, by test_e2e.py against a built bundle, so it is a target rather than a CTest
  # case: it reads the bundle's own tokenizer asset and diffs the native ids against the
  # reference tokenizer's.
  add_executable(tipsv2_tokenizer_check
    ${PROJECT_SOURCE_DIR}/families/tipsv2/tests/cpp/tokenizer_check.cpp
    ${PROJECT_SOURCE_DIR}/families/tipsv2/runtime/tokenizer.cpp
  )
  target_include_directories(tipsv2_tokenizer_check PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(tipsv2_tokenizer_check PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(tipsv2_tokenizer_check PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(tipsv2_tokenizer_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()
