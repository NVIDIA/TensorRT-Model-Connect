# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_trocr SHARED
  detokenizer.cpp
  image_preprocess.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_trocr
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_trocr SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_trocr PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_trocr PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_trocr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_trocr
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # CPU contract: preprocessing against Pillow, the byte-level decoder, the
  # request contract and the runtime parameter. No GPU, no checkpoint.
  add_executable(test_trocr_runtime_contract
    ${PROJECT_SOURCE_DIR}/families/trocr/tests/cpp/test_runtime_contract.cpp
  )
  target_include_directories(test_trocr_runtime_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_trocr_runtime_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_trocr_runtime_contract PRIVATE
    trtmc_model_trocr
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_trocr_runtime_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME trocr_runtime_contract COMMAND test_trocr_runtime_contract)
  set_tests_properties(trocr_runtime_contract PROPERTIES LABELS "cpu")

  # Public SDK consumers for the selected single-device E2E. They link only the
  # public C runtime, and building the CPU contract builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_trocr_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/trocr/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_trocr_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_trocr_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_trocr_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_trocr_runtime_contract
    test_trocr_sdk_c
    test_trocr_sdk_cpp
  )
endif()
