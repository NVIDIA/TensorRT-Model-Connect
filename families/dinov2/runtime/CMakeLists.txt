# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_dinov2 SHARED
  image_preprocess.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_dinov2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_dinov2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_dinov2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_dinov2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_dinov2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_dinov2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(_test IN ITEMS task_contract image_preprocess)
    add_executable(test_dinov2_${_test}
      ${PROJECT_SOURCE_DIR}/families/dinov2/tests/cpp/test_${_test}.cpp
    )
    target_include_directories(test_dinov2_${_test} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_dinov2_${_test} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(test_dinov2_${_test} PRIVATE
      trtmc_model_dinov2 trtmc_core ${TRTMC_CUDART_LIBRARY}
    )
    target_compile_options(test_dinov2_${_test} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME dinov2_${_test} COMMAND test_dinov2_${_test})
    set_tests_properties(dinov2_${_test} PROPERTIES LABELS "cpu")
  endforeach()
endif()
