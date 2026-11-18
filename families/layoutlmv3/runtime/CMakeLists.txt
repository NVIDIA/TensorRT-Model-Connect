# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  RESULT_VARIABLE _layoutlmv3_torch_result
  OUTPUT_VARIABLE _layoutlmv3_torch_prefix
  ERROR_VARIABLE _layoutlmv3_torch_error)
if(NOT _layoutlmv3_torch_result EQUAL 0)
  message(FATAL_ERROR "LayoutLMv3 requires torch for its image processor: ${_layoutlmv3_torch_error}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${_layoutlmv3_torch_prefix}")
find_package(Torch REQUIRED)
separate_arguments(_layoutlmv3_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

add_library(trtmc_model_layoutlmv3 SHARED bpe_tokenizer.cpp document_input.cpp image_preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_layoutlmv3 PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_layoutlmv3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(trtmc_model_layoutlmv3 PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ICU::uc ICU::i18n torch_cpu c10)
target_compile_options(trtmc_model_layoutlmv3 PRIVATE
  ${_layoutlmv3_torch_cxx_flags} -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_layoutlmv3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../torch/lib")
install(TARGETS trtmc_model_layoutlmv3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_layoutlmv3_tokenizer
    ${PROJECT_SOURCE_DIR}/families/layoutlmv3/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_layoutlmv3_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_layoutlmv3_tokenizer PRIVATE
    trtmc_model_layoutlmv3 nlohmann_json::nlohmann_json)
  set_target_properties(test_layoutlmv3_tokenizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

if(TRTMC_BUILD_TESTS)
  target_include_directories(test_layoutlmv3_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  add_executable(test_layoutlmv3_runtime ${PROJECT_SOURCE_DIR}/families/layoutlmv3/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_layoutlmv3_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_layoutlmv3_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
