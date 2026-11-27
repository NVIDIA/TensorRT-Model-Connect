# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET TensorRTEdgeLLM::Core)
  set(_qwen38_sdk TensorRTEdgeLLM)
  target_compile_definitions(trtmc_model_qwen3_8 PRIVATE TRTMC_QWEN38_EDGE_012=1)
elseif(TARGET EdgeLLM::Core)
  set(_qwen38_sdk EdgeLLM)
else()
  return()
endif()

if(TARGET ${_qwen38_sdk}::Core)
  if(NOT TARGET ${_qwen38_sdk}::Plugin)
    message(FATAL_ERROR "Qwen38 Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_qwen3_8 PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_compile_definitions(trtmc_model_qwen3_8 PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_qwen3_8 PRIVATE ${_qwen38_sdk}::Core)
  set_target_properties(trtmc_model_qwen3_8 PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(TARGET ${_qwen38_sdk}::Core)
  add_custom_command(TARGET trtmc_model_qwen3_8 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:${_qwen38_sdk}::Plugin> $<TARGET_FILE_DIR:trtmc_model_qwen3_8>
    VERBATIM
  )
endif()
