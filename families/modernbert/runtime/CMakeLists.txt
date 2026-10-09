# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_modernbert SHARED
  distributed_runtime.cpp
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_modernbert
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_modernbert SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_modernbert PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_modernbert PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_modernbert PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_modernbert
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_modernbert_task_contract
    ${PROJECT_SOURCE_DIR}/families/modernbert/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_modernbert_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_modernbert_task_contract SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_modernbert_task_contract PRIVATE trtmc_model_modernbert)
  target_compile_options(test_modernbert_task_contract PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME modernbert_task_contract COMMAND test_modernbert_task_contract)
  set_tests_properties(modernbert_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_modernbert_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/modernbert/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_modernbert_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_modernbert_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_modernbert_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_modernbert test_modernbert_sdk_c test_modernbert_sdk_cpp)
endif()
