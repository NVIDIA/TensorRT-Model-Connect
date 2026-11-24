# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_ideogram4 SHARED plugin.cpp tokenizer.cpp)
target_include_directories(trtmc_model_ideogram4 PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_ideogram4 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_ideogram4 PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_ideogram4 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_ideogram4 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_ideogram4 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_ideogram4_tokenizer ${PROJECT_SOURCE_DIR}/families/ideogram4/tests/cpp/test_tokenizer.cpp tokenizer.cpp)
  target_include_directories(test_ideogram4_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_ideogram4_tokenizer PRIVATE nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
  set_target_properties(test_ideogram4_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_ideogram4_tokenizer COMMAND test_ideogram4_tokenizer)
  add_library(ideogram4_request_backend SHARED ${PROJECT_SOURCE_DIR}/families/ideogram4/tests/cpp/request_backend.cpp)
  target_include_directories(ideogram4_request_backend PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(ideogram4_request_backend SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  set_target_properties(ideogram4_request_backend PROPERTIES OUTPUT_NAME trtmc_backend_trt
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/request-runtime")
  add_executable(test_ideogram4_request ${PROJECT_SOURCE_DIR}/families/ideogram4/tests/cpp/test_request.cpp)
  target_include_directories(test_ideogram4_request PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(test_ideogram4_request SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_ideogram4_request PRIVATE trtmc_model_ideogram4 trtmc_core trtmc_c
    nlohmann_json::nlohmann_json Threads::Threads)
  add_test(NAME ideogram4_request COMMAND test_ideogram4_request
    $<TARGET_FILE:trtmc_model_ideogram4> $<TARGET_FILE:ideogram4_request_backend>)
endif()
