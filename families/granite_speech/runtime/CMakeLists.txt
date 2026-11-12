# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_granite_speech SHARED
  pipeline.cpp
  plugin.cpp
  text.cpp
)
target_include_directories(trtmc_model_granite_speech PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_granite_speech SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_granite_speech PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_granite_speech PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_granite_speech PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_granite_speech
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_granite_speech_runtime
    ${PROJECT_SOURCE_DIR}/families/granite_speech/tests/cpp/test_runtime.cpp
  )
  target_include_directories(test_granite_speech_runtime PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_granite_speech_runtime SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_granite_speech_runtime PRIVATE
    trtmc_model_granite_speech
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_granite_speech_runtime PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME granite_speech_runtime COMMAND test_granite_speech_runtime)

  foreach(_language IN ITEMS c cpp)
    add_executable(test_granite_speech_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/granite_speech/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_granite_speech_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_granite_speech_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_granite_speech_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
endif()
