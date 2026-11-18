# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_blip SHARED
  image_preprocess.cpp
  wordpiece_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
  fp32_kernels_init.cpp
)
target_include_directories(trtmc_model_blip PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_blip SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_blip PRIVATE trtmc_core nlohmann_json::nlohmann_json ICU::uc ${TRTMC_CUDART_LIBRARY})
set(_blip_build_rpath "\$ORIGIN")
set(_blip_install_rpath "\$ORIGIN")
if(TARGET trtmc_byok_tvm_ffi)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
    RESULT_VARIABLE _blip_torch_result
    OUTPUT_VARIABLE _blip_torch_prefix
    ERROR_VARIABLE _blip_torch_error
  )
  if(NOT _blip_torch_result EQUAL 0)
    message(FATAL_ERROR "BLIP requires its CUDA PyTorch dependency: ${_blip_torch_error}")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH "${_blip_torch_prefix}")
  find_package(Torch 2.12 REQUIRED)
  separate_arguments(_blip_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_sources(trtmc_model_blip PRIVATE fp32_kernels.cpp)
  target_include_directories(trtmc_model_blip SYSTEM PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_model_blip PRIVATE
    trtmc_byok_tvm_ffi
    ${TRTMC_TVM_FFI_LIBRARY}
    ${TORCH_LIBRARIES}
    ${CMAKE_DL_LIBS}
  )
  target_compile_options(trtmc_model_blip PRIVATE ${_blip_torch_flags})
  target_compile_definitions(trtmc_model_blip PRIVATE TRTMC_BLIP_HAS_FP32_KERNELS=1)
  list(APPEND _blip_build_rpath "${TORCH_INSTALL_PREFIX}/lib")
  list(APPEND _blip_install_rpath "\$ORIGIN/../../torch/lib" "\$ORIGIN/../../tvm_ffi/lib")
else()
  target_compile_definitions(trtmc_model_blip PRIVATE TRTMC_BLIP_HAS_FP32_KERNELS=0)
endif()
target_compile_options(trtmc_model_blip PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_blip PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "${_blip_build_rpath}"
  INSTALL_RPATH "${_blip_install_rpath}"
)
install(TARGETS trtmc_model_blip LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_subdirectory("${PROJECT_SOURCE_DIR}/families/blip/tests" "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()
