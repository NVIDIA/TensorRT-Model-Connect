# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(TRTMC_BUILD_TESTS)
  add_executable(test_vitmatte_preprocess ${PROJECT_SOURCE_DIR}/families/vitmatte/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_vitmatte_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  add_test(NAME vitmatte_preprocess COMMAND test_vitmatte_preprocess)
endif()
function(_vitmatte_unavailable reason)
  if(DEFINED TRTMC_BUILD_FAMILIES AND NOT "*" IN_LIST TRTMC_BUILD_FAMILIES)
    message(FATAL_ERROR "Selected ViTMatte runtime unavailable: ${reason}")
  endif()
  message(STATUS "Skipping vitmatte: ${reason}")
  add_custom_target(trtmc_model_vitmatte
    COMMAND ${CMAKE_COMMAND} -E echo "ViTMatte runtime unavailable: ${reason}"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM)
endfunction()
if(NOT TARGET trtmc_byok_tvm_ffi)
  _vitmatte_unavailable("its FP32 operations require TVM-FFI BYOK")
  return()
endif()
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  _vitmatte_unavailable("Python3 is required to locate the Torch development libraries")
  return()
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
  RESULT_VARIABLE _vitmatte_torch_result
  OUTPUT_VARIABLE _vitmatte_torch_prefix
  ERROR_VARIABLE _vitmatte_torch_error
)
if(NOT _vitmatte_torch_result EQUAL 0)
  _vitmatte_unavailable("install its Torch dependency in ${Python3_EXECUTABLE}")
  return()
endif()
list(PREPEND CMAKE_PREFIX_PATH "${_vitmatte_torch_prefix}")
find_package(Torch QUIET)
if(NOT Torch_FOUND)
  _vitmatte_unavailable("Torch development libraries were not found")
  return()
endif()
separate_arguments(_vitmatte_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
add_library(trtmc_vitmatte_fp32 OBJECT fp32_kernels.cpp)
set_target_properties(trtmc_vitmatte_fp32 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(trtmc_vitmatte_fp32 PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_vitmatte_fp32 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR} ${TRTMC_TVM_FFI_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS})
target_link_libraries(trtmc_vitmatte_fp32 PRIVATE ${TORCH_LIBRARIES})
target_compile_options(trtmc_vitmatte_fp32 PRIVATE ${_vitmatte_torch_flags} -Wall -Wextra -Wpedantic)
add_library(trtmc_model_vitmatte SHARED plugin.cpp)
target_include_directories(trtmc_model_vitmatte PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_vitmatte SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR} ${TRTMC_TVM_FFI_INCLUDE_DIR})
target_link_libraries(trtmc_model_vitmatte PRIVATE trtmc_core nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY} trtmc_byok_tvm_ffi ${TRTMC_TVM_FFI_LIBRARY} trtmc_vitmatte_fp32
  ${CMAKE_DL_LIBS})
target_compile_options(trtmc_model_vitmatte PRIVATE -Wall -Wextra -Wpedantic)
get_filename_component(_vitmatte_tvm_library_dir "${TRTMC_TVM_FFI_LIBRARY}" DIRECTORY)
set_target_properties(trtmc_model_vitmatte PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN;${TORCH_INSTALL_PREFIX}/lib;${_vitmatte_tvm_library_dir}"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../torch/lib;\$ORIGIN/../../tvm_ffi/lib")
install(TARGETS trtmc_model_vitmatte LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_vitmatte_runtime ${PROJECT_SOURCE_DIR}/families/vitmatte/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_vitmatte_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_vitmatte_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()
