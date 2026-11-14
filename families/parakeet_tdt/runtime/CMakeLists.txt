# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_parakeet_tdt SHARED
  audio_helpers.cpp bpe_tokenizer.cpp pipeline.cpp plugin.cpp resampler.cpp
)
target_include_directories(trtmc_model_parakeet_tdt PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_parakeet_tdt SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_parakeet_tdt PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_parakeet_tdt PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_parakeet_tdt PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_parakeet_tdt LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_parakeet_tdt_sdk_cpp ../tests/sdk_consumer.cpp)
  target_link_libraries(test_parakeet_tdt_sdk_cpp PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  target_compile_options(test_parakeet_tdt_sdk_cpp PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(test_parakeet_tdt_sdk_cpp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endif()
