# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU QUIET COMPONENTS uc)
if(NOT ICU_FOUND)
  message(FATAL_ERROR "BLIP requires the ICU uc development library; install libicu-dev or select other TRTMC_BUILD_FAMILIES")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from importlib.metadata import PackageNotFoundError, distribution; from pathlib import Path\ntry: print(Path(distribution('nvidia-cublas').locate_file('nvidia/cu13/lib/libcublasLt.so.13')).resolve(), end='')\nexcept PackageNotFoundError: pass"
  RESULT_VARIABLE _blip_math_result
  OUTPUT_VARIABLE _blip_math_library
  ERROR_VARIABLE _blip_math_error
)
if(NOT _blip_math_result EQUAL 0)
  message(FATAL_ERROR "BLIP could not inspect the existing CUDA environment: ${_blip_math_error}")
endif()
if(NOT EXISTS "${_blip_math_library}")
  find_library(_blip_math_library_toolkit NAMES cublasLt libcublasLt.so.13
    HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}/lib64 REQUIRED)
  set(_blip_math_library "${_blip_math_library_toolkit}")
endif()

add_library(trtmc_model_blip SHARED
  image_preprocess.cpp
  wordpiece_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  fp32_kernels.cu
  fp32_linear.cpp
  fp32_plugin.cpp
  fp32_kernels_init.cpp
)
target_include_directories(trtmc_model_blip PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_blip SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${TRTMC_TRT_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_blip PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc
  ${TRTMC_CUDART_LIBRARY} ${TRTMC_TRT_LIBRARY} ${CMAKE_DL_LIBS}
)
target_compile_definitions(trtmc_model_blip PRIVATE
  TRTMC_BLIP_CUBLASLT_BUILD_LIBRARY="${_blip_math_library}"
)
target_compile_options(trtmc_model_blip PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=true;--prec-div=true;--prec-sqrt=true;--ftz=false>"
)
set_target_properties(trtmc_model_blip PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_blip LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_subdirectory("${PROJECT_SOURCE_DIR}/families/blip/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()
