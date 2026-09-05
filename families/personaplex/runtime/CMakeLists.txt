# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_personaplex SHARED
  distributed_runtime.cpp
  resampler.cpp
  decode_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampling_kernels.cu
)

set_source_files_properties(
  sampling_kernels.cu
  PROPERTIES LANGUAGE CUDA
)

set_source_files_properties(
  pipeline.cpp
  PROPERTIES
    COMPILE_OPTIONS "$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-nonnull>;$<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-free-nonheap-object>"
)

target_include_directories(trtmc_model_personaplex
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_personaplex SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_personaplex PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_personaplex PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_personaplex PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_personaplex
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_personaplex_decode_runtime
    ${PROJECT_SOURCE_DIR}/families/personaplex/tests/cpp/test_decode_runtime.cpp
    ${PROJECT_SOURCE_DIR}/families/personaplex/runtime/decode_runtime.cpp
  )
  target_include_directories(test_personaplex_decode_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(test_personaplex_decode_runtime PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_personaplex_decode_runtime COMMAND test_personaplex_decode_runtime)

  foreach(test_name IN ITEMS
      test_personaplex_speech_decode_stop_policy
      test_personaplex_speech_depth_plan
      test_personaplex_speech_generation_helpers
      test_personaplex_speech_mimi_decode_plan
      test_personaplex_speech_temporal_embed_plan)
    string(REPLACE "test_personaplex_" "" test_source ${test_name})
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/personaplex/tests/cpp/test_${test_source}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
    )
    target_compile_options(${test_name} PRIVATE
      -Wall -Wextra -Wpedantic
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
