# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_wan_t2v SHARED
  unigram_tokenizer.cpp
  diffusion_helpers.cpp
  distributed_runtime.cpp
  gpu_matmul.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  runtime_config.cpp
)

set_source_files_properties(
  pipeline.cpp
  PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

target_include_directories(trtmc_model_wan_t2v
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_wan_t2v SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_wan_t2v PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
  CUDA::cublas
)
target_compile_options(trtmc_model_wan_t2v PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_wan_t2v PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_wan_t2v
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_wan_parallel_runtime_config
    ${PROJECT_SOURCE_DIR}/families/wan_t2v/tests/cpp/test_wan_parallel_runtime_config.cpp
  )
  target_include_directories(test_wan_parallel_runtime_config PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_wan_parallel_runtime_config SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_wan_parallel_runtime_config PRIVATE trtmc_model_wan_t2v)
  target_compile_options(test_wan_parallel_runtime_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME wan_parallel_runtime_config COMMAND test_wan_parallel_runtime_config)

  foreach(test_name IN ITEMS
      test_wan_generation_plan
      test_wan_generation_conditioning
      test_wan_denoising_step_seam
      test_wan_pipeline)
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/wan_t2v/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(${test_name} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_wan_t2v
    )
    target_compile_options(${test_name} PRIVATE
      -Wall -Wextra -Wpedantic
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
