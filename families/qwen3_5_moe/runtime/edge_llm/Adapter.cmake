# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET TensorRTEdgeLLM::Core)
  set(_qwen35_moe_sdk TensorRTEdgeLLM)
  set(_qwen35_moe_chat TensorRTEdgeLLM::ChatTemplate)
  set(_qwen35_moe_release_define TRTMC_QWEN35_MOE_EDGE_012=1)
elseif(TARGET EdgeLLM::Core)
  set(_qwen35_moe_sdk EdgeLLM)
  set(_qwen35_moe_chat "${EdgeLLM_PREFIX}/lib/libedgellmChatTemplate.a")
endif()

if(_qwen35_moe_sdk)
  target_sources(trtmc_model_qwen3_5_moe PRIVATE ${CMAKE_CURRENT_LIST_DIR}/adapter.cpp ${CMAKE_CURRENT_LIST_DIR}/device_link.cu)
  target_compile_definitions(trtmc_model_qwen3_5_moe PRIVATE TRTMC_HAS_EDGE_LLM=1 ${_qwen35_moe_release_define})
  target_link_libraries(trtmc_model_qwen3_5_moe PRIVATE ${_qwen35_moe_sdk}::Core ${CMAKE_DL_LIBS})
  set_target_properties(trtmc_model_qwen3_5_moe PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET trtmc_model_qwen3_5_moe POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:${_qwen35_moe_sdk}::Plugin> $<TARGET_FILE_DIR:trtmc_model_qwen3_5_moe>/libNvInfer_edgellm_plugin.so VERBATIM)
endif()
