# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_fnet SHARED
  distributed_runtime.cpp
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
)

target_include_directories(trtmc_model_fnet
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_fnet SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_fnet PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_fnet PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_fnet PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_fnet
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_fnet_tokenizer
    ${PROJECT_SOURCE_DIR}/families/fnet/tests/cpp/test_fnet_tokenizer.cpp)
  target_include_directories(test_fnet_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_fnet_tokenizer PRIVATE trtmc_model_fnet)
  add_test(NAME fnet_tokenizer COMMAND test_fnet_tokenizer)
  set_tests_properties(fnet_tokenizer PROPERTIES LABELS "cpu")
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_fnet_task_contract
    ${PROJECT_SOURCE_DIR}/families/fnet/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_fnet_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_fnet_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_fnet_task_contract PRIVATE
    trtmc_model_fnet trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_fnet_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME fnet_task_contract COMMAND test_fnet_task_contract)
  set_tests_properties(fnet_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_fnet_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/fnet/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_fnet_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_fnet_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_fnet_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_fnet_task_contract
    test_fnet_sdk_c test_fnet_sdk_cpp
  )
endif()
