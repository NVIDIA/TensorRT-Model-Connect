# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_voxtral SHARED
  frontend.cpp
  pipeline.cpp
  plugin.cpp
  text.cpp
)
target_include_directories(trtmc_model_voxtral PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_voxtral SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_voxtral PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_voxtral PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_voxtral PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_voxtral LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  # Host-only front end and text tests: no CUDA, no TensorRT.
  add_executable(test_voxtral_frontend
    ${PROJECT_SOURCE_DIR}/families/voxtral/tests/cpp/test_voxtral_frontend.cpp
    frontend.cpp
    text.cpp
  )
  target_include_directories(test_voxtral_frontend PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_voxtral_frontend PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME voxtral_frontend COMMAND test_voxtral_frontend)
  set_tests_properties(voxtral_frontend PROPERTIES LABELS "cpu")

  foreach(_language c cpp)
    add_executable(test_voxtral_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/voxtral/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_voxtral_sdk_${_language} PRIVATE trtmc_c)
    if(_language STREQUAL "cpp")
      target_link_libraries(test_voxtral_sdk_${_language} PRIVATE nlohmann_json::nlohmann_json)
    endif()
    target_compile_options(test_voxtral_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_voxtral_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
endif()
