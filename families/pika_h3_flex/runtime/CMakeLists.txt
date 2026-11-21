# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU REQUIRED COMPONENTS uc i18n)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  RESULT_VARIABLE _pika_torch_result
  OUTPUT_VARIABLE _pika_torch_prefix
  ERROR_VARIABLE _pika_torch_error)
if(NOT _pika_torch_result EQUAL 0)
  message(FATAL_ERROR "Pika H3 Flex requires torch for its CPU generator: ${_pika_torch_error}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${_pika_torch_prefix}")
find_package(Torch REQUIRED)
separate_arguments(_pika_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

# Family factories exchange C++ runtime objects with core and the backend.
# Torch must match their toolchain ABI before its flags can be applied here.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch.compiled_with_cxx11_abi()), end='')"
  RESULT_VARIABLE _pika_torch_abi_result OUTPUT_VARIABLE _pika_torch_abi)
if(NOT _pika_torch_abi_result EQUAL 0 OR NOT _pika_torch_abi MATCHES "^[01]$")
  message(FATAL_ERROR "Pika H3 Flex could not determine Torch's C++ ABI")
endif()
include(CheckCXXSourceCompiles)
unset(_pika_torch_abi_matches CACHE)
check_cxx_source_compiles("#include <string>
#if defined(__GLIBCXX__) && _GLIBCXX_USE_CXX11_ABI != ${_pika_torch_abi}
#error Torch C++ ABI differs from the native runtime toolchain
#endif
int main() { return 0; }
" _pika_torch_abi_matches)
if(NOT _pika_torch_abi_matches)
  message(FATAL_ERROR "Pika H3 Flex requires Torch's C++ ABI to match the native runtime toolchain")
endif()

add_library(trtmc_model_pika_h3_flex SHARED bpe_tokenizer.cpp plugin.cpp random.cpp video.cpp)
target_include_directories(trtmc_model_pika_h3_flex PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_pika_h3_flex SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(trtmc_model_pika_h3_flex PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY} ICU::uc ICU::i18n torch_cpu c10)
target_compile_options(trtmc_model_pika_h3_flex PRIVATE ${_pika_torch_cxx_flags} -Wall -Wextra -Wpedantic -ffp-contract=off)
set_target_properties(trtmc_model_pika_h3_flex PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../torch/lib;${TORCH_INSTALL_PREFIX}/lib")
install(TARGETS trtmc_model_pika_h3_flex LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_pika_h3_flex_runtime ${PROJECT_SOURCE_DIR}/families/pika_h3_flex/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_pika_h3_flex_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_pika_h3_flex_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_pika_h3_flex_tokenizer ${PROJECT_SOURCE_DIR}/families/pika_h3_flex/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_pika_h3_flex_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_pika_h3_flex_tokenizer PRIVATE trtmc_model_pika_h3_flex nlohmann_json::nlohmann_json)
  target_compile_options(test_pika_h3_flex_tokenizer PRIVATE ${_pika_torch_cxx_flags})
  set_target_properties(test_pika_h3_flex_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_executable(test_pika_h3_flex_random ${PROJECT_SOURCE_DIR}/families/pika_h3_flex/tests/cpp/test_random.cpp)
  target_include_directories(test_pika_h3_flex_random PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_pika_h3_flex_random PRIVATE trtmc_model_pika_h3_flex nlohmann_json::nlohmann_json)
  target_compile_options(test_pika_h3_flex_random PRIVATE ${_pika_torch_cxx_flags})
  target_compile_options(test_pika_h3_flex_random PRIVATE -ffp-contract=off)
  set_target_properties(test_pika_h3_flex_random PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
