# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron_labs_diffusion SHARED
  bpe_tokenizer.cpp
  chat_templates.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

target_include_directories(trtmc_model_nemotron_labs_diffusion
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_nemotron_labs_diffusion SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_nemotron_labs_diffusion PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_nemotron_labs_diffusion PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_nemotron_labs_diffusion PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_nemotron_labs_diffusion
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_nemotron_labs_diffusion_runtime_config
    ${PROJECT_SOURCE_DIR}/families/nemotron_labs_diffusion/tests/cpp/test_nemotron_labs_diffusion_runtime_config.cpp
  )
  target_include_directories(test_nemotron_labs_diffusion_runtime_config PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_nemotron_labs_diffusion_runtime_config PRIVATE trtmc_model_nemotron_labs_diffusion)
  target_compile_options(test_nemotron_labs_diffusion_runtime_config PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME nemotron_labs_diffusion_runtime_config COMMAND test_nemotron_labs_diffusion_runtime_config)
endif()
