# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_ltx_video SHARED
  unigram_tokenizer.cpp
  diffusion_helpers.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_ltx_video
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_ltx_video SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_ltx_video PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_ltx_video PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_ltx_video PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_ltx_video
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_ltx_video_scheduler
    ../tests/cpp/test_scheduler.cpp
  )
  target_include_directories(test_ltx_video_scheduler PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(test_ltx_video_scheduler PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME ltx_video_scheduler
    COMMAND test_ltx_video_scheduler
  )

  add_executable(test_ltx_video_runtime_contract
    ../tests/cpp/test_runtime_contract.cpp
  )
  target_include_directories(test_ltx_video_runtime_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_ltx_video_runtime_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_ltx_video_runtime_contract PRIVATE
    trtmc_model_ltx_video
  )
  target_compile_options(test_ltx_video_runtime_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME ltx_video_runtime_contract
    COMMAND test_ltx_video_runtime_contract
  )
endif()
