# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_molmo2 SHARED bpe_tokenizer.cpp preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_molmo2 PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_molmo2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_molmo2 PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_molmo2 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_molmo2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_molmo2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_molmo2_tokenizer ../tests/cpp/tokenizer_consumer.cpp bpe_tokenizer.cpp)
  target_include_directories(test_molmo2_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_molmo2_tokenizer PRIVATE nlohmann_json::nlohmann_json ICU::uc)
  add_executable(molmo2_qualification ${PROJECT_SOURCE_DIR}/families/molmo2/tests/cpp/qualification.cpp)
  target_include_directories(molmo2_qualification PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(molmo2_qualification PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
  add_dependencies(molmo2_qualification trtmc_model_molmo2 trtmc_backend_trt)
  add_executable(test_molmo2_preprocess ${PROJECT_SOURCE_DIR}/families/molmo2/tests/cpp/test_preprocess.cpp preprocess.cpp bpe_tokenizer.cpp)
  target_include_directories(test_molmo2_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_molmo2_preprocess PRIVATE nlohmann_json::nlohmann_json ICU::uc)
  add_dependencies(test_molmo2_preprocess molmo2_qualification)
  add_test(NAME molmo2_preprocess COMMAND test_molmo2_preprocess)
endif()
