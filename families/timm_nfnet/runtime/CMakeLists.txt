# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_timm_nfnet SHARED
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)
set_source_files_properties(image_preprocess_seam.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)
target_include_directories(trtmc_model_timm_nfnet PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_timm_nfnet SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_timm_nfnet PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_timm_nfnet PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_timm_nfnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_timm_nfnet
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_timm_nfnet_image_preprocess
    ${PROJECT_SOURCE_DIR}/families/timm_nfnet/tests/cpp/test_image_preprocess_seam.cpp
  )
  target_include_directories(test_timm_nfnet_image_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_timm_nfnet_image_preprocess PRIVATE
    trtmc_model_timm_nfnet
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_timm_nfnet_image_preprocess PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME timm_nfnet_image_preprocess
    COMMAND test_timm_nfnet_image_preprocess
  )
endif()

# The family CLI is an application adapter; the model does not depend on its loader.
add_library(trtmc_cli_timm_nfnet SHARED cli.cpp)
target_include_directories(trtmc_cli_timm_nfnet PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_cli_timm_nfnet SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
target_link_libraries(trtmc_cli_timm_nfnet PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_timm_nfnet PRIVATE -Wall -Wextra -Wno-unused-function)
set_target_properties(trtmc_cli_timm_nfnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_timm_nfnet LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_library(timm_nfnet_cli_fixture SHARED ${PROJECT_SOURCE_DIR}/families/timm_nfnet/tests/cpp/test_cli.cpp)
  target_compile_definitions(timm_nfnet_cli_fixture PRIVATE TRTMC_FAMILY_CLI_FIXTURE)
  target_include_directories(timm_nfnet_cli_fixture PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(timm_nfnet_cli_fixture PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  set_target_properties(timm_nfnet_cli_fixture PROPERTIES
    OUTPUT_NAME trtmc_model_timm_nfnet
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/timm_nfnet-cli"
  )
  add_dependencies(timm_nfnet_cli_fixture trtmc_test_backend_fake)
  add_custom_command(TARGET timm_nfnet_cli_fixture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:trtmc_test_backend_fake>
      $<TARGET_FILE_DIR:timm_nfnet_cli_fixture>
  )
  add_executable(test_timm_nfnet_cli ${PROJECT_SOURCE_DIR}/families/timm_nfnet/tests/cpp/test_cli.cpp)
  target_include_directories(test_timm_nfnet_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_timm_nfnet_cli PRIVATE trtmc_cli_timm_nfnet nlohmann_json::nlohmann_json)
  target_compile_options(test_timm_nfnet_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(test_timm_nfnet_cli timm_nfnet_cli_fixture)
  add_test(NAME timm_nfnet_cli COMMAND test_timm_nfnet_cli $<TARGET_FILE_DIR:timm_nfnet_cli_fixture>)
  set_tests_properties(timm_nfnet_cli PROPERTIES LABELS cpu)
endif()
