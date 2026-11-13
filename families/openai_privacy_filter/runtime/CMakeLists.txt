# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_openai_privacy_filter SHARED
  o200k_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_openai_privacy_filter
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_openai_privacy_filter SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_openai_privacy_filter PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_openai_privacy_filter PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_openai_privacy_filter PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_openai_privacy_filter
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/openai_privacy_filter
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_openai_privacy_filter_task_contract
    ${PROJECT_SOURCE_DIR}/families/openai_privacy_filter/tests/cpp/test_openai_privacy_filter_task_contract.cpp
  )
  target_include_directories(test_openai_privacy_filter_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${PROJECT_SOURCE_DIR}/core
  )
  target_include_directories(test_openai_privacy_filter_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_openai_privacy_filter_task_contract PRIVATE
    trtmc_model_openai_privacy_filter
    trtmc_core
    nlohmann_json::nlohmann_json
  )

  # The tokenizer checks compile the tokenizer alone: no CUDA, engine or model library.
  foreach(test_name IN ITEMS
      test_openai_privacy_filter_tokenizer
      test_openai_privacy_filter_tokenizer_checkpoint
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/openai_privacy_filter/tests/cpp/${test_name}.cpp
      o200k_tokenizer.cpp
    )
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR})
    target_link_libraries(${test_name} PRIVATE nlohmann_json::nlohmann_json)
  endforeach()

  foreach(test_name IN ITEMS
      test_openai_privacy_filter_task_contract
      test_openai_privacy_filter_tokenizer
      test_openai_privacy_filter_tokenizer_checkpoint
  )
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach()
  # The checkpoint tokenizer check needs the pinned tokenizer.json; the family E2E builds and runs it.
  foreach(test_name IN ITEMS
      test_openai_privacy_filter_task_contract
      test_openai_privacy_filter_tokenizer
  )
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES LABELS "cpu")
  endforeach()

  foreach(_language IN ITEMS c cpp)
    add_executable(test_openai_privacy_filter_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/openai_privacy_filter/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_openai_privacy_filter_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_openai_privacy_filter_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_openai_privacy_filter_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_openai_privacy_filter
    test_openai_privacy_filter_sdk_c
    test_openai_privacy_filter_sdk_cpp
  )
endif()
