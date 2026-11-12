# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_nemotron3_diarization SHARED
  frontend.cpp
  pipeline.cpp
  plugin.cpp
  speaker_cache.cpp
)

target_include_directories(trtmc_model_nemotron3_diarization
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_nemotron3_diarization SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_nemotron3_diarization PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
# The front end and the speaker-cache scores reproduce the reference's float32
# arithmetic step by step; a fused multiply-add would round differently, and
# the cache ranks frames by those scores.
target_compile_options(trtmc_model_nemotron3_diarization PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
)
set_target_properties(trtmc_model_nemotron3_diarization PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_nemotron3_diarization
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_nemotron3_diarization_runtime
    ${PROJECT_SOURCE_DIR}/families/nemotron3_diarization/tests/cpp/test_runtime.cpp
  )
  target_include_directories(test_nemotron3_diarization_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_nemotron3_diarization_runtime SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_nemotron3_diarization_runtime PRIVATE
    trtmc_model_nemotron3_diarization
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_nemotron3_diarization_runtime PRIVATE
    -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
  )
  target_compile_definitions(test_nemotron3_diarization_runtime PRIVATE
    NEMOTRON3_DIARIZATION_TEST_DATA="${PROJECT_SOURCE_DIR}/families/nemotron3_diarization/tests/data"
  )
  add_test(NAME nemotron3_diarization_runtime COMMAND test_nemotron3_diarization_runtime)
  set_tests_properties(nemotron3_diarization_runtime PROPERTIES LABELS "cpu")

  # Public SDK consumers for the single-device E2E. They link only the public
  # C runtime, and building the CPU test builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_nemotron3_diarization_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/nemotron3_diarization/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_nemotron3_diarization_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_nemotron3_diarization_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_nemotron3_diarization_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_nemotron3_diarization_runtime
    test_nemotron3_diarization_sdk_c
    test_nemotron3_diarization_sdk_cpp
  )
endif()
