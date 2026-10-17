# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "Qwen35 Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_qwen3_5 PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_compile_definitions(trtmc_model_qwen3_5 PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_qwen3_5 PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_qwen3_5 PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_qwen3_5 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_qwen3_5>
    VERBATIM
  )
endif()

# Extend the existing owning native regression executable when Edge is provisioned.
if(TARGET EdgeLLM::Core AND TARGET test_qwen3_5_recurrent_output_initializers)
  target_compile_definitions(test_qwen3_5_recurrent_output_initializers PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(test_qwen3_5_recurrent_output_initializers PRIVATE EdgeLLM::Core)
  target_sources(test_qwen3_5_recurrent_output_initializers PRIVATE "${CMAKE_CURRENT_LIST_DIR}/device_link.cu")
  set_target_properties(test_qwen3_5_recurrent_output_initializers PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()
