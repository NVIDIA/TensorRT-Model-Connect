# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_stable_audio SHARED
  noise.cpp
  pipeline.cpp
  plugin.cpp
  request.cpp
  scheduler.cpp
  unigram_tokenizer.cpp
)

target_include_directories(trtmc_model_stable_audio
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_stable_audio SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_stable_audio PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
# The scheduler and guidance arithmetic mirror the reference's separately rounded
# float32 operations, so a multiply-add must not be contracted into an FMA.
target_compile_options(trtmc_model_stable_audio PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
)
set_target_properties(trtmc_model_stable_audio PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_stable_audio
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_stable_audio_sampling
    ${PROJECT_SOURCE_DIR}/families/stable_audio/tests/cpp/test_sampling.cpp
    ${PROJECT_SOURCE_DIR}/families/stable_audio/runtime/noise.cpp
    ${PROJECT_SOURCE_DIR}/families/stable_audio/runtime/scheduler.cpp
  )
  target_include_directories(test_stable_audio_sampling PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(test_stable_audio_sampling PRIVATE
    -Wall -Wextra -Wpedantic -ffp-contract=off
  )
  add_test(NAME stable_audio_sampling COMMAND test_stable_audio_sampling)

  add_executable(test_stable_audio_request
    ${PROJECT_SOURCE_DIR}/families/stable_audio/tests/cpp/test_request.cpp
    ${PROJECT_SOURCE_DIR}/families/stable_audio/runtime/request.cpp
  )
  target_include_directories(test_stable_audio_request PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_stable_audio_request PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME stable_audio_request COMMAND test_stable_audio_request)

  add_executable(test_stable_audio_tokenizer
    ${PROJECT_SOURCE_DIR}/families/stable_audio/tests/cpp/test_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/families/stable_audio/runtime/unigram_tokenizer.cpp
  )
  target_include_directories(test_stable_audio_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_link_libraries(test_stable_audio_tokenizer PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_stable_audio_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME stable_audio_tokenizer COMMAND test_stable_audio_tokenizer)

  foreach(_language IN ITEMS c cpp)
    add_executable(test_stable_audio_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/stable_audio/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_stable_audio_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_stable_audio_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_stable_audio_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_stable_audio test_stable_audio_sdk_c test_stable_audio_sdk_cpp)
endif()
