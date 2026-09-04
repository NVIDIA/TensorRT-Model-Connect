# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c
    "import torch; print(torch.utils.cmake_prefix_path, end='')"
  RESULT_VARIABLE _sana_wm_torch_result
  OUTPUT_VARIABLE _sana_wm_torch_prefix
  ERROR_VARIABLE _sana_wm_torch_error
)
if(NOT _sana_wm_torch_result EQUAL 0)
  message(FATAL_ERROR
    "SANA-WM requires its torch dependency: ${_sana_wm_torch_error}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${_sana_wm_torch_prefix}")
find_package(Torch REQUIRED)
separate_arguments(_sana_wm_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

add_library(trtmc_model_sana_wm SHARED
  ../native_plugins/sana_wm_gdn_creator.cpp
  ../native_plugins/sana_wm_gdn_plugin.cu
  ../native_plugins/sana_wm_torch_conv2d_plugin.cpp
  ../native_plugins/torch_cuda_bfloat16_math.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sana_wm_bpe_tokenizer.cpp
)

# These exact family plugins currently use TensorRT 11.1's supported
# IPluginV2DynamicExt creator API. Keep its deprecation diagnostics local to
# the plugin implementation instead of weakening warnings for the family DSO.
set_source_files_properties(../native_plugins/sana_wm_gdn_creator.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-deprecated-declarations"
)
set_source_files_properties(../native_plugins/sana_wm_gdn_plugin.cu PROPERTIES
  COMPILE_OPTIONS "-diag-suppress=997"
)

target_include_directories(trtmc_model_sana_wm
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../native_plugins
)
target_include_directories(trtmc_model_sana_wm SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${TRTMC_TRT_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(trtmc_model_sana_wm PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  CUDA::cublas
  CUDA::cublasLt
  ${TRTMC_TRT_LIBRARY}
  ${TORCH_LIBRARIES}
)
target_compile_options(trtmc_model_sana_wm PRIVATE
  ${_sana_wm_torch_cxx_flags}
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_sana_wm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN;${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../torch/lib"
)
install(TARGETS trtmc_model_sana_wm
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
