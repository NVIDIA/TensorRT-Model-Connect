# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_sam3 SHARED
  bpe_tokenizer.cpp
  plugin.cpp
  plugin_helpers.cpp
  sam3_pipeline.cpp
  sam3_video_processor.cpp
  sam3_video_session.cpp
  sam3_video_kernels.cu
)

set_source_files_properties(
  sam3_video_kernels.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_sam3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_sam3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_sam3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_sam3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_sam3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_sam3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_sam3_runtime_contract
    ../tests/cpp/test_runtime_contract.cpp
  )
  target_include_directories(test_sam3_runtime_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_sam3_runtime_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_sam3_runtime_contract PRIVATE
    trtmc_model_sam3
    trtmc_core
  )
  target_compile_options(test_sam3_runtime_contract PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME sam3_runtime_contract
    COMMAND test_sam3_runtime_contract
  )
endif()
