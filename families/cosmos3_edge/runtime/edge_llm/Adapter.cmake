# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET TensorRTEdgeLLM::Core)
  set(_cosmos3_edge_sdk TensorRTEdgeLLM)
  target_compile_definitions(trtmc_model_cosmos3_edge PRIVATE TRTMC_COSMOS3_EDGE_012=1)
elseif(TARGET EdgeLLM::Core)
  set(_cosmos3_edge_sdk EdgeLLM)
else()
  return()
endif()
if(NOT TARGET ${_cosmos3_edge_sdk}::Plugin)
  message(FATAL_ERROR "Cosmos3-Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
endif()
target_sources(trtmc_model_cosmos3_edge PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
)
target_compile_definitions(trtmc_model_cosmos3_edge PRIVATE TRTMC_HAS_EDGE_LLM=1)
target_link_libraries(trtmc_model_cosmos3_edge PRIVATE ${_cosmos3_edge_sdk}::Core)
set_target_properties(trtmc_model_cosmos3_edge PROPERTIES
  CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET trtmc_model_cosmos3_edge POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:${_cosmos3_edge_sdk}::Plugin> $<TARGET_FILE_DIR:trtmc_model_cosmos3_edge>/libNvInfer_edgellm_plugin.so
  VERBATIM
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_cosmos3_edge_request
    "${CMAKE_CURRENT_LIST_DIR}/../../tests/cpp/test_edge_request.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu")
  target_include_directories(test_cosmos3_edge_request PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  if(TARGET TensorRTEdgeLLM::Core)
    target_compile_definitions(test_cosmos3_edge_request PRIVATE TRTMC_COSMOS3_EDGE_012=1)
  endif()
  target_link_libraries(test_cosmos3_edge_request PRIVATE trtmc_core ${_cosmos3_edge_sdk}::Core)
  set_target_properties(test_cosmos3_edge_request PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME test_cosmos3_edge_request COMMAND test_cosmos3_edge_request)
endif()
