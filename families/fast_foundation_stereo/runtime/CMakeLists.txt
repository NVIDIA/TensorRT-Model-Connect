# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_fast_foundation_stereo SHARED
  native_plugins/full_volume_leaky_plugin.cu
  native_plugins/geometry_volume_convc1_plugin.cu
  native_plugins/gwc_plugin.cu
  native_plugins/plugin_creators.cpp
  native_plugins/post8_sum_plugin.cu
  native_plugins/spatial_attention_reduce_plugin.cu
  plugin.cpp
  stereo_pipeline.cpp
)

# TensorRT 11.1 keeps IPluginV2DynamicExt supported but marks its creator API
# deprecated and its private final compatibility overloads trigger NVCC #997.
# Scope those third-party diagnostics to the exact plugin translation units.
set_source_files_properties(native_plugins/plugin_creators.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-deprecated-declarations"
)
set_source_files_properties(
  native_plugins/full_volume_leaky_plugin.cu
  native_plugins/geometry_volume_convc1_plugin.cu
  native_plugins/gwc_plugin.cu
  native_plugins/post8_sum_plugin.cu
  native_plugins/spatial_attention_reduce_plugin.cu
  PROPERTIES COMPILE_OPTIONS "-diag-suppress=997"
)

target_include_directories(trtmc_model_fast_foundation_stereo
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/native_plugins
)
target_include_directories(trtmc_model_fast_foundation_stereo SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${TRTMC_TRT_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_fast_foundation_stereo PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
  ${TRTMC_TRT_LIBRARY}
)
target_compile_options(trtmc_model_fast_foundation_stereo PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_fast_foundation_stereo PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_fast_foundation_stereo
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_fast_foundation_stereo_preprocess
    ${PROJECT_SOURCE_DIR}/families/fast_foundation_stereo/tests/cpp/test_fast_foundation_stereo_preprocess.cpp
  )
  target_include_directories(test_fast_foundation_stereo_preprocess PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_fast_foundation_stereo_preprocess PRIVATE
    trtmc_model_fast_foundation_stereo
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_fast_foundation_stereo_preprocess PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(
    NAME fast_foundation_stereo_preprocess
    COMMAND test_fast_foundation_stereo_preprocess
  )
endif()
