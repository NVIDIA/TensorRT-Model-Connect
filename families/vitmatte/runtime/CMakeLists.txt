# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(TRTMC_BUILD_TESTS)
  add_executable(test_vitmatte_preprocess ${PROJECT_SOURCE_DIR}/families/vitmatte/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_vitmatte_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  add_test(NAME vitmatte_preprocess COMMAND test_vitmatte_preprocess)
endif()
if(NOT TARGET trtmc_byok_tvm_ffi)
  message(STATUS "Skipping vitmatte: its FP32 operations require TVM-FFI BYOK")
  return()
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  RESULT_VARIABLE _vitmatte_torch_result
  OUTPUT_VARIABLE _vitmatte_torch_prefix
  ERROR_VARIABLE _vitmatte_torch_error
)
if(NOT _vitmatte_torch_result EQUAL 0)
  message(FATAL_ERROR "ViTMatte requires its torch dependency: ${_vitmatte_torch_error}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${_vitmatte_torch_prefix}")
find_package(Torch REQUIRED)
separate_arguments(_vitmatte_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
set_source_files_properties(fp32_kernels.cpp PROPERTIES COMPILE_OPTIONS "${_vitmatte_torch_flags}")

add_library(trtmc_model_vitmatte SHARED plugin.cpp fp32_kernels.cpp)
target_include_directories(trtmc_model_vitmatte PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_vitmatte SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR} ${TRTMC_TVM_FFI_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(trtmc_model_vitmatte PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} trtmc_byok_tvm_ffi ${TRTMC_TVM_FFI_LIBRARY}
  ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
target_compile_options(trtmc_model_vitmatte PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_vitmatte PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../torch/lib")
install(TARGETS trtmc_model_vitmatte LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_vitmatte_runtime ${PROJECT_SOURCE_DIR}/families/vitmatte/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_vitmatte_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_vitmatte_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
