# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen_vl SHARED
  bpe_tokenizer.cpp
  distributed_runtime.cpp
  image_preprocessor.cpp
  kv_cache.cpp
  lora_peft_artifact.cpp
  lora_peft_loader.cpp
  lora_runtime.cpp
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

target_include_directories(trtmc_model_qwen_vl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_qwen_vl SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_qwen_vl PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_qwen_vl PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_qwen_vl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_qwen_vl
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_qwen_vl_image_normalization
    ${PROJECT_SOURCE_DIR}/families/qwen_vl/tests/cpp/test_image_normalization.cpp
    image_preprocessor.cpp
  )
  target_include_directories(test_qwen_vl_image_normalization PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(test_qwen_vl_image_normalization SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
  target_link_libraries(test_qwen_vl_image_normalization PRIVATE
    nlohmann_json::nlohmann_json
    Threads::Threads
  )
  target_compile_options(test_qwen_vl_image_normalization PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME qwen_vl_image_normalization COMMAND test_qwen_vl_image_normalization)

  add_executable(test_qwen_vl_dynamic_vision_metadata
    ${PROJECT_SOURCE_DIR}/families/qwen_vl/tests/cpp/test_dynamic_vision_metadata.cpp
    image_preprocessor.cpp)
  target_include_directories(test_qwen_vl_dynamic_vision_metadata PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(test_qwen_vl_dynamic_vision_metadata SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb)
  target_link_libraries(test_qwen_vl_dynamic_vision_metadata PRIVATE
    nlohmann_json::nlohmann_json Threads::Threads)
  target_compile_options(test_qwen_vl_dynamic_vision_metadata PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME qwen_vl_dynamic_vision_metadata COMMAND test_qwen_vl_dynamic_vision_metadata
    ${PROJECT_SOURCE_DIR}/families/qwen_vl/tests/data/dynamic-vision/hf-position-metadata.json)

endif()
