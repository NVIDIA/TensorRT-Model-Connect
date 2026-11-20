# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_vibevoice SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  scheduler.cpp
  script.cpp
  voice.cpp
)

target_include_directories(trtmc_model_vibevoice
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_vibevoice SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_vibevoice PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
# The host-side sampler mirrors the reference's float32 arithmetic; contracting
# a*b+c into one fused multiply-add would round differently.
target_compile_options(trtmc_model_vibevoice PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
)
set_target_properties(trtmc_model_vibevoice PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_vibevoice
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_vibevoice_host
    ${PROJECT_SOURCE_DIR}/families/vibevoice/tests/cpp/test_host.cpp
    ${PROJECT_SOURCE_DIR}/families/vibevoice/runtime/bpe_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/families/vibevoice/runtime/scheduler.cpp
    ${PROJECT_SOURCE_DIR}/families/vibevoice/runtime/script.cpp
    ${PROJECT_SOURCE_DIR}/families/vibevoice/runtime/voice.cpp
  )
  target_include_directories(test_vibevoice_host PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_vibevoice_host PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_vibevoice_host PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
  target_compile_definitions(test_vibevoice_host PRIVATE
    VIBEVOICE_TEST_DATA="${PROJECT_SOURCE_DIR}/families/vibevoice/tests/data"
  )
  add_test(NAME vibevoice_host COMMAND test_vibevoice_host)

  foreach(_language IN ITEMS c cpp)
    add_executable(test_vibevoice_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/vibevoice/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_vibevoice_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_vibevoice_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_vibevoice_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_vibevoice test_vibevoice_sdk_c test_vibevoice_sdk_cpp)
endif()
