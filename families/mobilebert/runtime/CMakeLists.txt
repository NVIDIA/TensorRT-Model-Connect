# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_mobilebert SHARED
  wordpiece_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_mobilebert
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_mobilebert SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_mobilebert PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_mobilebert PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_mobilebert PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_mobilebert
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/mobilebert
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_mobilebert_task_contract
    ${PROJECT_SOURCE_DIR}/families/mobilebert/tests/cpp/test_task_contract.cpp
  )
  target_include_directories(test_mobilebert_task_contract PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_mobilebert_task_contract SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_mobilebert_task_contract PRIVATE
    trtmc_model_mobilebert
    trtmc_core
    nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_mobilebert_task_contract PRIVATE
    -Wall -Wextra -Wpedantic -Werror
  )
  add_test(NAME mobilebert_task_contract COMMAND test_mobilebert_task_contract)
  set_tests_properties(mobilebert_task_contract PROPERTIES LABELS "cpu")

  foreach(_language IN ITEMS c cpp)
    add_executable(test_mobilebert_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/mobilebert/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_mobilebert_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_mobilebert_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_mobilebert_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_mobilebert_task_contract
    test_mobilebert_sdk_c test_mobilebert_sdk_cpp
  )
endif()
