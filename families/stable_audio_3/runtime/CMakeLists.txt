# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_stable_audio_3 SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_stable_audio_3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_stable_audio_3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_stable_audio_3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_stable_audio_3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_stable_audio_3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_stable_audio_3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_stable_audio_3_sampling
    ${PROJECT_SOURCE_DIR}/families/stable_audio_3/tests/cpp/test_sampling.cpp
  )
  target_include_directories(test_stable_audio_3_sampling PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_stable_audio_3_sampling PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME stable_audio_3_sampling COMMAND test_stable_audio_3_sampling)

  foreach(_language IN ITEMS c cpp)
    add_executable(test_stable_audio_3_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/stable_audio_3/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_stable_audio_3_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_stable_audio_3_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_stable_audio_3_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_stable_audio_3 test_stable_audio_3_sdk_c test_stable_audio_3_sdk_cpp)
endif()
