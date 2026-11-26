# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The tokenizer pins ICU's Unicode tables from Unicode 15.0 (ICU 72) on (bpe_tokenizer.cpp).
find_package(ICU 72 REQUIRED COMPONENTS uc i18n)

add_library(trtmc_model_minicpmv4_6 SHARED
  bpe_tokenizer.cpp
  decoder.cpp
  engine_io.cpp
  image.cpp
  metadata.cpp
  plugin.cpp
  prompt.cpp
  sampler.cpp
  state.cpp
  vision_encoder.cpp
)

target_include_directories(trtmc_model_minicpmv4_6
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_minicpmv4_6 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_minicpmv4_6 PRIVATE
  trtmc_core
  ICU::uc
  ICU::i18n
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_minicpmv4_6 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_minicpmv4_6 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minicpmv4_6
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# The library carries code adapted from Apache-2.0, BSD-3-Clause and MIT-CMU sources; ship their notices with it.
install(FILES ${PROJECT_SOURCE_DIR}/families/minicpmv4_6/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/minicpmv4_6
)

if(TRTMC_BUILD_TESTS)
  # Host-only checks: image slicing and resampling, tokenizer, prompt formatting, metadata,
  # sampling and the engine I/O contracts (against a fake module).
  add_executable(test_minicpmv4_6_host
    ${PROJECT_SOURCE_DIR}/families/minicpmv4_6/tests/cpp/test_minicpmv4_6_host.cpp
    bpe_tokenizer.cpp
    engine_io.cpp
    image.cpp
    metadata.cpp
    prompt.cpp
    sampler.cpp
  )
  target_include_directories(test_minicpmv4_6_host PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minicpmv4_6_host SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_minicpmv4_6_host PRIVATE
    ICU::uc
    ICU::i18n
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_minicpmv4_6_host PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_minicpmv4_6_host COMMAND test_minicpmv4_6_host)

  # Public SDK consumers run by the family E2E from the native build directory.
  foreach(kind IN ITEMS c cpp)
    add_executable(test_minicpmv4_6_sdk_${kind}
      ${PROJECT_SOURCE_DIR}/families/minicpmv4_6/tests/sdk_consumer.${kind}
    )
    target_link_libraries(test_minicpmv4_6_sdk_${kind} PRIVATE trtmc_c)
    target_compile_options(test_minicpmv4_6_sdk_${kind} PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(test_minicpmv4_6_sdk_${kind} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  target_link_libraries(test_minicpmv4_6_sdk_cpp PRIVATE nlohmann_json::nlohmann_json)
endif()
