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

# Family factories exchange C++ runtime objects with core and the backend.
# Torch must match their toolchain ABI before its flags can be applied here.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch.compiled_with_cxx11_abi()), end='')"
  RESULT_VARIABLE _layoutlmv3_torch_abi_result OUTPUT_VARIABLE _layoutlmv3_torch_abi)
if(NOT _layoutlmv3_torch_abi_result EQUAL 0 OR NOT _layoutlmv3_torch_abi MATCHES "^[01]$")
  message(FATAL_ERROR "LayoutLMv3 could not determine Torch's C++ ABI")
endif()
include(CheckCXXSourceCompiles)
unset(_layoutlmv3_torch_abi_matches CACHE)
check_cxx_source_compiles("#include <string>
#if defined(__GLIBCXX__) && _GLIBCXX_USE_CXX11_ABI != ${_layoutlmv3_torch_abi}
#error Torch C++ ABI differs from the native runtime toolchain
#endif
int main() { return 0; }
" _layoutlmv3_torch_abi_matches)
if(NOT _layoutlmv3_torch_abi_matches)
  message(FATAL_ERROR "LayoutLMv3 requires Torch's C++ ABI to match the native runtime toolchain")
endif()

add_library(trtmc_model_layoutlmv3 SHARED bpe_tokenizer.cpp document_input.cpp image_preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_layoutlmv3 PRIVATE ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_layoutlmv3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(trtmc_model_layoutlmv3 PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} ICU::uc ICU::i18n torch_cpu c10)
target_compile_options(trtmc_model_layoutlmv3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:${_layoutlmv3_torch_cxx_flags};-Wall;-Wextra;-Wpedantic>")
set_target_properties(trtmc_model_layoutlmv3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../torch/lib;${TORCH_INSTALL_PREFIX}/lib")
install(TARGETS trtmc_model_layoutlmv3 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_layoutlmv3_tokenizer
    ${PROJECT_SOURCE_DIR}/families/layoutlmv3/tests/cpp/test_tokenizer.cpp)
  target_include_directories(test_layoutlmv3_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_layoutlmv3_tokenizer PRIVATE
    trtmc_model_layoutlmv3 nlohmann_json::nlohmann_json)
  target_compile_options(test_layoutlmv3_tokenizer PRIVATE ${_layoutlmv3_torch_cxx_flags})
  set_target_properties(test_layoutlmv3_tokenizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

if(TRTMC_BUILD_TESTS)
  target_include_directories(test_layoutlmv3_tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
  add_executable(test_layoutlmv3_runtime ${PROJECT_SOURCE_DIR}/families/layoutlmv3/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_layoutlmv3_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_layoutlmv3_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

target_sources(trtmc_model_layoutlmv3 PRIVATE fp32_kernels_init.cpp)
if(TARGET trtmc_byok_tvm_ffi)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "from importlib.metadata import PackageNotFoundError, distribution; from pathlib import Path\ntry: print(Path(distribution('nvidia-cublas').locate_file('nvidia/cu13/lib/libcublasLt.so.13')).resolve(), end='')\nexcept PackageNotFoundError: pass"
    RESULT_VARIABLE _layoutlmv3_math_result OUTPUT_VARIABLE _layoutlmv3_math_library
    ERROR_VARIABLE _layoutlmv3_math_error)
  if(NOT _layoutlmv3_math_result EQUAL 0)
    message(FATAL_ERROR "LayoutLMv3 could not inspect its CUDA math dependency: ${_layoutlmv3_math_error}")
  endif()
  if(NOT EXISTS "${_layoutlmv3_math_library}")
    find_library(_layoutlmv3_math_toolkit NAMES cublasLt libcublasLt.so.13
      HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64 REQUIRED)
    set(_layoutlmv3_math_library "${_layoutlmv3_math_toolkit}")
  endif()
  target_sources(trtmc_model_layoutlmv3 PRIVATE fp32_kernels.cu fp32_kernels_ffi.cpp fp32_linear.cpp)
  target_compile_definitions(trtmc_model_layoutlmv3 PRIVATE
    TRTMC_LAYOUTLMV3_CUBLASLT_BUILD_LIBRARY="${_layoutlmv3_math_library}")
  target_include_directories(trtmc_model_layoutlmv3 SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_model_layoutlmv3 PRIVATE
    trtmc_byok_tvm_ffi ${TRTMC_TVM_FFI_LIBRARY} ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS})
  target_compile_definitions(trtmc_model_layoutlmv3 PRIVATE TRTMC_LAYOUTLMV3_HAS_FP32_KERNELS=1)
  target_compile_options(trtmc_model_layoutlmv3 PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
else()
  target_compile_definitions(trtmc_model_layoutlmv3 PRIVATE TRTMC_LAYOUTLMV3_HAS_FP32_KERNELS=0)
endif()
set_property(TARGET trtmc_model_layoutlmv3 APPEND PROPERTY
  INSTALL_RPATH "\$ORIGIN/../../tvm_ffi/lib;${TRTMC_TVM_FFI_LIBRARY_DIR}")
