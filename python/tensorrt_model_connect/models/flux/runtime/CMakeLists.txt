# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_flux SHARED
  diffusion_helpers.cpp
  gpu_matmul.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  ${TRTMC_MODEL_flux_REGISTRATION_SOURCE}
)
target_include_directories(trtmc_model_flux PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_flux SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
if(NOT TRTMC_CUBLAS_LIBRARY)
  message(FATAL_ERROR "Model 'flux' requires cuBLAS, but it was not found")
endif()
target_link_libraries(trtmc_model_flux PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
  ${TRTMC_CUBLAS_LIBRARY}
)
target_compile_definitions(trtmc_model_flux PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_flux PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_flux PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/flux"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)
if(TRTMC_BUILD_MODEL_KERNELS)
  add_subdirectory(kernels)
endif()

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_flux_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_flux_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_flux_pipeline)
    target_link_libraries(test_flux_pipeline PRIVATE trtmc_model_flux)
  endif()
  trtmc_add_test(test_flux_generation_plan
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_flux_generation_plan.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_flux_denoising_step_seam
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_flux_denoising_step_seam.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  trtmc_add_test(test_flux_batch_utils
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_flux_batch_utils.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
endif()
