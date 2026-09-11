# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_phi4_multimodal SHARED
  bpe_tokenizer.cpp
  image_preprocessor.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

set_source_files_properties(
  PROPERTIES LANGUAGE CUDA
)
set_source_files_properties(image_preprocessor.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)

target_include_directories(trtmc_model_phi4_multimodal
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_phi4_multimodal SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_phi4_multimodal PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_phi4_multimodal PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_phi4_multimodal PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_phi4_multimodal
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_phi4_multimodal_runtime_config_contract
    ${PROJECT_SOURCE_DIR}/families/phi4_multimodal/tests/cpp/test_phi4_multimodal_runtime_config_contract.cpp
    image_preprocessor.cpp
  )
  target_include_directories(test_phi4_multimodal_runtime_config_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_include_directories(test_phi4_multimodal_runtime_config_contract SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
  target_link_libraries(test_phi4_multimodal_runtime_config_contract PRIVATE
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_phi4_multimodal_runtime_config_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_phi4_multimodal_runtime_config_contract
    COMMAND test_phi4_multimodal_runtime_config_contract
  )
  set_tests_properties(test_phi4_multimodal_runtime_config_contract PROPERTIES LABELS cpu)
endif()
