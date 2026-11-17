# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_flux3 SHARED
  bpe_tokenizer.cpp
  engine_contract.cpp
  pipeline.cpp
  plugin.cpp
  runtime_config.cpp
  sampling.cpp
)

target_include_directories(trtmc_model_flux3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_flux3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_flux3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
# The host sampling math reproduces separately rounded float32 operations;
# fused multiply-add contraction would change its results.
target_compile_options(trtmc_model_flux3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
)
set_target_properties(trtmc_model_flux3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_flux3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  # Host math only: no TensorRT, CUDA runtime or bundle.
  add_executable(test_flux3_sampling
    ${PROJECT_SOURCE_DIR}/families/flux3/tests/cpp/test_sampling.cpp
    sampling.cpp
  )
  target_include_directories(test_flux3_sampling PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_compile_options(test_flux3_sampling PRIVATE
    -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
  )
  add_test(NAME test_flux3_sampling COMMAND test_flux3_sampling)
  set_tests_properties(test_flux3_sampling PROPERTIES LABELS "cpu")

  # Public SDK consumers for the family E2E. They link only the public C
  # runtime; building the CPU test builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_flux3_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/flux3/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_flux3_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_flux3_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_flux3_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_flux3_sampling test_flux3_sdk_c test_flux3_sdk_cpp)
endif()
