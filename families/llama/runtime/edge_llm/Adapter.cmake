# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
# Each installation links exactly one SDK. Remove the legacy branch after
# the default SDK changes and all family owners have migrated.
if(TARGET TensorRTEdgeLLM::Core)
  set(_llama_sdk TensorRTEdgeLLM)
  target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_LLAMA_EDGE_012=1)
elseif(TARGET EdgeLLM::Core)
  set(_llama_sdk EdgeLLM)
else()
  return()
endif()
if(NOT TARGET ${_llama_sdk}::Plugin)
  message(FATAL_ERROR "Llama Edge adapter requires the complete SDK (Core and Plugin)")
endif()
target_sources(trtmc_model_llama PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
)
target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_HAS_EDGE_LLM=1)
target_link_libraries(trtmc_model_llama PRIVATE ${_llama_sdk}::Core)
set_target_properties(trtmc_model_llama PROPERTIES
  CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET trtmc_model_llama POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:${_llama_sdk}::Plugin> $<TARGET_FILE_DIR:trtmc_model_llama>/libNvInfer_edgellm_plugin.so
  VERBATIM
)
