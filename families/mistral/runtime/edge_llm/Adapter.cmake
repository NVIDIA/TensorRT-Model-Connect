# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
# Only the 0.12 SDK exposes the C++ API used by this owner.
if(NOT TARGET TensorRTEdgeLLM::Core)
  return()
endif()
if(NOT TARGET TensorRTEdgeLLM::Plugin)
  message(FATAL_ERROR "Mistral Edge adapter requires the complete SDK (Core and Plugin)")
endif()
target_sources(trtmc_model_mistral PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
)
target_compile_definitions(trtmc_model_mistral PRIVATE TRTMC_MISTRAL_EDGE_012=1)
target_link_libraries(trtmc_model_mistral PRIVATE TensorRTEdgeLLM::Core)
set_target_properties(trtmc_model_mistral PROPERTIES
  CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
add_custom_command(TARGET trtmc_model_mistral POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:TensorRTEdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_mistral>/libNvInfer_edgellm_plugin.so
  VERBATIM
)
