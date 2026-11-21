# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET TensorRTEdgeLLM::Core)
  set(_phi4_sdk TensorRTEdgeLLM)
  target_compile_definitions(trtmc_model_phi4_multimodal PRIVATE TRTMC_PHI4_EDGE_012=1)
elseif(TARGET EdgeLLM::Core)
  set(_phi4_sdk EdgeLLM)
else()
  return()
endif()
if(NOT TARGET ${_phi4_sdk}::Plugin)
  message(FATAL_ERROR "Phi4 Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
endif()
target_sources(trtmc_model_phi4_multimodal PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
)
target_compile_definitions(trtmc_model_phi4_multimodal PRIVATE TRTMC_HAS_EDGE_LLM=1)
target_link_libraries(trtmc_model_phi4_multimodal PRIVATE ${_phi4_sdk}::Core)
set_target_properties(trtmc_model_phi4_multimodal PROPERTIES
  CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET trtmc_model_phi4_multimodal POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:${_phi4_sdk}::Plugin> $<TARGET_FILE_DIR:trtmc_model_phi4_multimodal>/libNvInfer_edgellm_plugin.so
  VERBATIM
)

# Edge has no visual-feature dump CLI. This helper serves the existing
# Phi4 image-health E2E and is not installed as a product executable.
if(TRTMC_BUILD_TESTS)
  add_executable(phi4_multimodal_edge_vision_features
    "${CMAKE_CURRENT_LIST_DIR}/vision_features.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  if(TARGET TensorRTEdgeLLM::Core)
    target_compile_definitions(phi4_multimodal_edge_vision_features PRIVATE TRTMC_PHI4_EDGE_012=1)
  endif()
  target_link_libraries(phi4_multimodal_edge_vision_features PRIVATE ${_phi4_sdk}::Core ${CMAKE_DL_LIBS})
  set_target_properties(phi4_multimodal_edge_vision_features PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/families/phi4_multimodal"
  )
  add_custom_command(TARGET phi4_multimodal_edge_vision_features POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:${_phi4_sdk}::Plugin>"
      "$<TARGET_FILE_DIR:phi4_multimodal_edge_vision_features>/libNvInfer_edgellm_plugin.so"
    VERBATIM
  )
  add_dependencies(trtmc_model_phi4_multimodal phi4_multimodal_edge_vision_features)
endif()
