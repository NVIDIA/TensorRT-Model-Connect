# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Keep the action surface model-owned. The shared `trtmc` CLI and public
# IPipeline interface do not need OpenPI request or diagnostic concepts.
option(TRTMC_OPENPI_REQUIRE_QUALIFIED_RUNTIME
  "Fail unless the qualified OpenPI GB300 TensorRT runtime can be built" OFF)
if(TRTMC_MODEL_PROOF_MODEL STREQUAL "openpi")
  set(TRTMC_OPENPI_REQUIRE_QUALIFIED_RUNTIME ON CACHE BOOL
    "Fail unless the qualified OpenPI GB300 TensorRT runtime can be built" FORCE)
endif()

add_executable(trtmc_openpi_runner
  "${CMAKE_CURRENT_LIST_DIR}/tool/action_request_json.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tool/main.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tool/qualification_diagnostics.cpp"
)
target_link_libraries(trtmc_openpi_runner PRIVATE trtmc_core)
target_include_directories(trtmc_openpi_runner PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/src"
)
target_include_directories(trtmc_openpi_runner SYSTEM PRIVATE
  "${_trtmc_nlohmann_json_include}"
)
target_compile_options(trtmc_openpi_runner PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_openpi_runner PROPERTIES
  OUTPUT_NAME trtmc-openpi
  BUILD_RPATH "\$ORIGIN"
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH "\$ORIGIN"
)
add_dependencies(trtmc_model_${_trtmc_model} trtmc_openpi_runner)
install(TARGETS trtmc_openpi_runner
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# OpenPI's custom TensorRT plugins are qualified only on the Linux/aarch64
# GB300 stack. Do not compile them on another platform: a successful build
# there would imply support that has not been established.
set(_trtmc_openpi_platform_eligible TRUE)
set(_trtmc_openpi_platform_reason "")
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_trtmc_openpi_platform_eligible FALSE)
  set(_trtmc_openpi_platform_reason
    "requires Linux; detected '${CMAKE_SYSTEM_NAME}'")
elseif(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(_trtmc_openpi_platform_eligible FALSE)
  set(_trtmc_openpi_platform_reason
    "requires aarch64; detected '${CMAKE_SYSTEM_PROCESSOR}'")
elseif(NOT CMAKE_CUDA_COMPILER)
  set(_trtmc_openpi_platform_eligible FALSE)
  set(_trtmc_openpi_platform_reason "requires a CUDA compiler with sm103 support")
elseif(NOT _trtmc_can_build_trt_backend)
  set(_trtmc_openpi_platform_eligible FALSE)
  set(_trtmc_openpi_platform_reason "requires TensorRT headers and libraries")
else()
  _trtmc_header_define_value(_trtmc_openpi_trt_major NV_TENSORRT_MAJOR)
  _trtmc_header_define_value(_trtmc_openpi_trt_minor NV_TENSORRT_MINOR)
  _trtmc_header_define_value(_trtmc_openpi_trt_patch NV_TENSORRT_PATCH)
  _trtmc_header_define_value(_trtmc_openpi_trt_build NV_TENSORRT_BUILD)
  set(_trtmc_openpi_trt_version
    "${_trtmc_openpi_trt_major}.${_trtmc_openpi_trt_minor}."
    "${_trtmc_openpi_trt_patch}.${_trtmc_openpi_trt_build}")
  string(JOIN "" _trtmc_openpi_trt_version ${_trtmc_openpi_trt_version})
  if(NOT _trtmc_openpi_trt_version STREQUAL "11.2.0.113")
    set(_trtmc_openpi_platform_eligible FALSE)
    set(_trtmc_openpi_platform_reason
      "requires TensorRT 11.2.0.113; detected '${_trtmc_openpi_trt_version}'")
  endif()
endif()

if(TRTMC_OPENPI_REQUIRE_QUALIFIED_RUNTIME AND NOT _trtmc_openpi_platform_eligible)
  message(FATAL_ERROR
    "OpenPI native runtime is qualified only for NVIDIA GB300 (sm103) on "
    "Linux/aarch64 with TensorRT 11.2.0.113: ${_trtmc_openpi_platform_reason}")
endif()

if(_trtmc_openpi_platform_eligible)
  set(TRTMC_OPENPI_CUDA_ARCHITECTURES "103" CACHE STRING
    "CUDA architecture for the qualified OpenPI GB300 runtime")
  if(NOT "${TRTMC_OPENPI_CUDA_ARCHITECTURES}" STREQUAL "103")
    message(FATAL_ERROR
      "OpenPI is qualified only for GB300 sm103; "
      "TRTMC_OPENPI_CUDA_ARCHITECTURES must be exactly '103'")
  endif()
  target_sources(trtmc_model_${_trtmc_model} PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/trt_plugins/action_attention_context_plugin.cu"
    "${CMAKE_CURRENT_LIST_DIR}/trt_plugins/action_layer0_mlp_closure_plugin.cu"
    "${CMAKE_CURRENT_LIST_DIR}/trt_plugins/action_output_projection_plugin.cu"
    "${CMAKE_CURRENT_LIST_DIR}/trt_plugins/rms_norm_plugin.cu"
    "${CMAKE_CURRENT_LIST_DIR}/trt_plugins/siglip_attention_residual_plugin.cu"
    "${CMAKE_CURRENT_LIST_DIR}/trt_plugins/siglip_layer_norm_plugin.cu"
  )
  target_include_directories(trtmc_model_${_trtmc_model} SYSTEM PRIVATE
    ${TRTMC_TRT_INCLUDE_DIR}
  )
  target_compile_definitions(trtmc_model_${_trtmc_model} PRIVATE
    TRTMC_OPENPI_RMS_NORM_PLUGIN=1
  )
  target_link_libraries(trtmc_model_${_trtmc_model} PRIVATE
    ${TRTMC_TRT_LIBRARY}
    ${TRTMC_CUBLAS_LIBRARY}
  )
  set_target_properties(trtmc_model_${_trtmc_model} PROPERTIES
    CUDA_ARCHITECTURES "${TRTMC_OPENPI_CUDA_ARCHITECTURES}"
    BUILD_RPATH "\$ORIGIN;\$ORIGIN/../.."
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../.."
  )
  message(STATUS "OpenPI TensorRT plugins: libtrtmc_model_openpi.so")
else()
  message(STATUS
    "OpenPI TensorRT plugin disabled on this unqualified platform: "
    "${_trtmc_openpi_platform_reason}")
endif()
