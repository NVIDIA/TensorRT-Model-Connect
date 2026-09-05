# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_bark SHARED
  distributed_runtime.cpp
  wordpiece_tokenizer.cpp
  decode_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
  sparse_multinomial_kernel.cu
)

set_source_files_properties(
  sparse_multinomial_kernel.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_bark
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_bark SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_bark PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_bark PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_bark PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_bark
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_bark_pipeline
    ${PROJECT_SOURCE_DIR}/families/bark/tests/cpp/test_bark_pipeline.cpp
  )
  target_include_directories(test_bark_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_bark_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_bark_pipeline PRIVATE
    trtmc_model_bark
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_bark_pipeline PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME bark_pipeline COMMAND test_bark_pipeline)
  set_tests_properties(bark_pipeline PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)
endif()
