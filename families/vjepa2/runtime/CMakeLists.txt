# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
add_library(trtmc_model_vjepa2 SHARED preprocess.cpp plugin.cpp)
target_include_directories(trtmc_model_vjepa2 PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_vjepa2 SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_vjepa2 PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_vjepa2 PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
set_target_properties(trtmc_model_vjepa2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_vjepa2 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_vjepa2_preprocess ${PROJECT_SOURCE_DIR}/families/vjepa2/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_vjepa2_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_vjepa2_preprocess PRIVATE trtmc_model_vjepa2 nlohmann_json::nlohmann_json)
  add_executable(test_vjepa2_runtime ${PROJECT_SOURCE_DIR}/families/vjepa2/tests/cpp/test_runtime.cpp)
  target_link_libraries(test_vjepa2_runtime PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  set_target_properties(test_vjepa2_preprocess test_vjepa2_runtime PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
target_sources(trtmc_model_vjepa2 PRIVATE fp32_kernels_init.cpp)
if(TARGET trtmc_byok_tvm_ffi)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "from importlib.metadata import PackageNotFoundError, distribution; from pathlib import Path\ntry: print(Path(distribution('nvidia-cublas').locate_file('nvidia/cu13/lib/libcublasLt.so.13')).resolve(), end='')\nexcept PackageNotFoundError: pass"
    RESULT_VARIABLE _vjepa2_math_result OUTPUT_VARIABLE _vjepa2_math_library
    ERROR_VARIABLE _vjepa2_math_error)
  if(NOT _vjepa2_math_result EQUAL 0)
    message(FATAL_ERROR "V-JEPA2 could not inspect its CUDA math dependency: ${_vjepa2_math_error}")
  endif()
  if(NOT EXISTS "${_vjepa2_math_library}")
    find_library(_vjepa2_math_toolkit NAMES cublasLt libcublasLt.so.13
      HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64 REQUIRED)
    set(_vjepa2_math_library "${_vjepa2_math_toolkit}")
  endif()
  target_sources(trtmc_model_vjepa2 PRIVATE fp32_kernels.cu fp32_kernels_ffi.cpp fp32_linear.cpp)
  target_compile_definitions(trtmc_model_vjepa2 PRIVATE
    TRTMC_VJEPA2_CUBLASLT_BUILD_LIBRARY="${_vjepa2_math_library}")
  target_include_directories(trtmc_model_vjepa2 SYSTEM PRIVATE ${TRTMC_TVM_FFI_INCLUDE_DIR})
  target_link_libraries(trtmc_model_vjepa2 PRIVATE
    trtmc_byok_tvm_ffi ${TRTMC_TVM_FFI_LIBRARY} ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS})
  target_compile_definitions(trtmc_model_vjepa2 PRIVATE TRTMC_VJEPA2_HAS_FP32_KERNELS=1)
  target_compile_options(trtmc_model_vjepa2 PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>")
else()
  target_compile_definitions(trtmc_model_vjepa2 PRIVATE TRTMC_VJEPA2_HAS_FP32_KERNELS=0)
endif()
set_property(TARGET trtmc_model_vjepa2 APPEND PROPERTY
  INSTALL_RPATH "\$ORIGIN/../../tvm_ffi/lib")
