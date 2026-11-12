# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_ministral3 SHARED
  tokenizer.cpp
  pipeline.cpp
  plugin.cpp
)

target_include_directories(trtmc_model_ministral3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_ministral3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_ministral3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_ministral3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
)
set_target_properties(trtmc_model_ministral3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_ministral3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/ministral3
)

if(TRTMC_BUILD_TESTS)
  foreach(_test IN ITEMS tokenizer task_contract)
    add_executable(test_ministral3_${_test}
      ${PROJECT_SOURCE_DIR}/families/ministral3/tests/cpp/test_${_test}.cpp
    )
    target_include_directories(test_ministral3_${_test} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_ministral3_${_test} SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
    target_link_libraries(test_ministral3_${_test} PRIVATE
      trtmc_model_ministral3
      trtmc_core
      nlohmann_json::nlohmann_json
    )
    target_compile_options(test_ministral3_${_test} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME ministral3_${_test} COMMAND test_ministral3_${_test}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/families/ministral3
    )
    set_tests_properties(ministral3_${_test} PROPERTIES LABELS "cpu")
  endforeach()

  foreach(_language IN ITEMS c cpp)
    add_executable(test_ministral3_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/ministral3/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_ministral3_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_ministral3_sdk_${_language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_ministral3_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(trtmc_model_ministral3 test_ministral3_sdk_c test_ministral3_sdk_cpp)
endif()
