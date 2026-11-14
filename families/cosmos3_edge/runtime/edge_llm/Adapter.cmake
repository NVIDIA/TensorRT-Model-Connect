# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "Cosmos3-Edge Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_cosmos3_edge PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_compile_definitions(trtmc_model_cosmos3_edge PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_cosmos3_edge PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_cosmos3_edge PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()


if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_cosmos3_edge POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_cosmos3_edge>
    VERBATIM
  )
endif()


if(TRTMC_BUILD_TESTS AND TARGET EdgeLLM::Core)
  add_executable(test_cosmos3_edge_request
    "${CMAKE_CURRENT_LIST_DIR}/../../tests/cpp/test_edge_request.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu")
  target_include_directories(test_cosmos3_edge_request PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_cosmos3_edge_request PRIVATE trtmc_core EdgeLLM::Core)
  set_target_properties(test_cosmos3_edge_request PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_test(NAME test_cosmos3_edge_request COMMAND test_cosmos3_edge_request)
endif()
