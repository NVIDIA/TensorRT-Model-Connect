# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_timm_convnext SHARED
  image_preprocess_seam.cpp
  pipeline.cpp
  plugin.cpp
)
set_source_files_properties(image_preprocess_seam.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)
target_include_directories(trtmc_model_timm_convnext PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_timm_convnext SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_timm_convnext PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_timm_convnext PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_timm_convnext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_timm_convnext
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_timm_convnext_task_contract
    ${PROJECT_SOURCE_DIR}/families/timm_convnext/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_timm_convnext_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_timm_convnext_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_timm_convnext_task_contract PRIVATE
    trtmc_model_timm_convnext trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_timm_convnext_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME timm_convnext_task_contract COMMAND test_timm_convnext_task_contract)
  set_tests_properties(timm_convnext_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_timm_convnext_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/timm_convnext/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_timm_convnext_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_timm_convnext_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_timm_convnext_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_timm_convnext_task_contract
    test_timm_convnext_sdk_c test_timm_convnext_sdk_cpp
  )

  add_executable(test_timm_convnext_image_preprocess
    ${PROJECT_SOURCE_DIR}/families/timm_convnext/tests/cpp/test_image_preprocess_seam.cpp
  )
  target_include_directories(test_timm_convnext_image_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_timm_convnext_image_preprocess PRIVATE
    trtmc_model_timm_convnext
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_timm_convnext_image_preprocess PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME timm_convnext_image_preprocess
    COMMAND test_timm_convnext_image_preprocess
  )
endif()

# The family CLI is an application adapter; the model does not depend on its loader.
add_library(trtmc_cli_timm_convnext SHARED cli.cpp)
target_include_directories(trtmc_cli_timm_convnext PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_cli_timm_convnext SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/stb)
target_link_libraries(trtmc_cli_timm_convnext PRIVATE trtmc_c trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_timm_convnext PRIVATE -Wall -Wextra -Wno-unused-function)
set_target_properties(trtmc_cli_timm_convnext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_cli_timm_convnext LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_library(timm_convnext_cli_fixture SHARED ${PROJECT_SOURCE_DIR}/families/timm_convnext/tests/cpp/test_cli.cpp)
  target_compile_definitions(timm_convnext_cli_fixture PRIVATE TRTMC_FAMILY_CLI_FIXTURE)
  target_include_directories(timm_convnext_cli_fixture PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(timm_convnext_cli_fixture PRIVATE trtmc_core nlohmann_json::nlohmann_json)
  set_target_properties(timm_convnext_cli_fixture PROPERTIES
    OUTPUT_NAME trtmc_model_timm_convnext
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/timm_convnext-cli"
  )
  add_dependencies(timm_convnext_cli_fixture trtmc_test_backend_fake)
  add_custom_command(TARGET timm_convnext_cli_fixture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:trtmc_test_backend_fake>
      $<TARGET_FILE_DIR:timm_convnext_cli_fixture>
  )
  add_executable(test_timm_convnext_cli ${PROJECT_SOURCE_DIR}/families/timm_convnext/tests/cpp/test_cli.cpp)
  target_include_directories(test_timm_convnext_cli PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_timm_convnext_cli PRIVATE trtmc_cli_timm_convnext nlohmann_json::nlohmann_json)
  target_compile_options(test_timm_convnext_cli PRIVATE -Wall -Wextra -Wpedantic)
  add_dependencies(test_timm_convnext_cli timm_convnext_cli_fixture)
  add_test(NAME timm_convnext_cli COMMAND test_timm_convnext_cli $<TARGET_FILE_DIR:timm_convnext_cli_fixture>)
  set_tests_properties(timm_convnext_cli PROPERTIES LABELS cpu)
endif()
