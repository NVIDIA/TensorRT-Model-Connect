# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_swinv2 SHARED
  image_preprocess.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_swinv2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_swinv2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_swinv2 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_swinv2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_swinv2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_swinv2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_swinv2_image_preprocess
    ${PROJECT_SOURCE_DIR}/families/swinv2/tests/cpp/test_image_preprocess.cpp
  )
  target_include_directories(test_swinv2_image_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_swinv2_image_preprocess PRIVATE
    trtmc_model_swinv2
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_swinv2_image_preprocess PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(
    NAME swinv2_image_preprocess
    COMMAND test_swinv2_image_preprocess
  )
  set_tests_properties(swinv2_image_preprocess PROPERTIES LABELS "cpu")
endif()
