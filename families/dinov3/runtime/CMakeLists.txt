# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_dinov3 SHARED
  image_preprocess.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_dinov3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_dinov3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_dinov3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_dinov3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_dinov3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_dinov3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_dinov3_image_preprocess
    ${PROJECT_SOURCE_DIR}/families/dinov3/tests/cpp/test_dinov3_image_preprocess.cpp
  )
  target_include_directories(test_dinov3_image_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_link_libraries(test_dinov3_image_preprocess PRIVATE
    trtmc_model_dinov3
  )
  target_compile_options(test_dinov3_image_preprocess PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME test_dinov3_image_preprocess
    COMMAND test_dinov3_image_preprocess
  )

  add_executable(test_dinov3_pipeline
    ${PROJECT_SOURCE_DIR}/families/dinov3/tests/cpp/test_dinov3_pipeline.cpp
  )
  target_include_directories(test_dinov3_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_dinov3_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_dinov3_pipeline PRIVATE trtmc_model_dinov3)
  target_compile_options(test_dinov3_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_dinov3_pipeline COMMAND test_dinov3_pipeline)
endif()

# The CLI adapter owns image I/O and uses the existing family runtime contract.
add_library(trtmc_cli_dinov3 SHARED cli.cpp)
target_include_directories(trtmc_cli_dinov3 PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_cli_dinov3 SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
target_link_libraries(trtmc_cli_dinov3 PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_dinov3 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_dinov3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_dinov3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_dinov3_cli ${PROJECT_SOURCE_DIR}/families/dinov3/tests/cpp/test_cli.cpp cli.cpp)
  target_include_directories(test_dinov3_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_dinov3_cli SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
  target_link_libraries(test_dinov3_cli PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  target_compile_options(test_dinov3_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME dinov3_cli COMMAND test_dinov3_cli)
  set_tests_properties(dinov3_cli PROPERTIES LABELS cpu)
endif()
