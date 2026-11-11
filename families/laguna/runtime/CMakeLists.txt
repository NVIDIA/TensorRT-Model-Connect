# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_laguna SHARED
  chat_template.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  sampler.cpp
  tokenizer.cpp
)

target_include_directories(trtmc_model_laguna
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_laguna SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_laguna PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_laguna PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_laguna PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_laguna
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../UNICODE_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/laguna
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_laguna_text_runtime
    ${PROJECT_SOURCE_DIR}/families/laguna/tests/cpp/test_laguna_text_runtime.cpp
    chat_template.cpp
    sampler.cpp
    tokenizer.cpp
  )
  target_include_directories(test_laguna_text_runtime PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_laguna_text_runtime PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_laguna_text_runtime PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_laguna_text_runtime COMMAND test_laguna_text_runtime)

  add_executable(laguna_logits_trace
    ${PROJECT_SOURCE_DIR}/families/laguna/tests/cpp/logits_trace.cpp
  )
  target_include_directories(laguna_logits_trace PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(laguna_logits_trace SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(laguna_logits_trace PRIVATE
    trtmc_runtime
    trtmc_model_laguna
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(laguna_logits_trace PRIVATE -Wall -Wextra -Wpedantic)
endif()
