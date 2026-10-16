# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_internvl SHARED
  bpe_tokenizer.cpp
  distributed_runtime.cpp
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

target_include_directories(trtmc_model_internvl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_internvl SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_internvl PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_internvl PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_internvl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_internvl
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_internvl_image_normalization
    ${PROJECT_SOURCE_DIR}/families/internvl/tests/cpp/test_image_normalization.cpp
    image_preprocessor.cpp
  )
  target_include_directories(test_internvl_image_normalization PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(test_internvl_image_normalization SYSTEM PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/stb
  )
  target_link_libraries(test_internvl_image_normalization PRIVATE
    nlohmann_json::nlohmann_json
    Threads::Threads
  )
  target_compile_options(test_internvl_image_normalization PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME internvl_image_normalization COMMAND test_internvl_image_normalization)
endif()

include(edge_llm/Adapter.cmake)
