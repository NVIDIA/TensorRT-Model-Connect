# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET EdgeLLM::Core)
  target_sources(trtmc_model_qwen3_5_moe PRIVATE ${CMAKE_CURRENT_LIST_DIR}/adapter.cpp ${CMAKE_CURRENT_LIST_DIR}/device_link.cu)
  target_compile_definitions(trtmc_model_qwen3_5_moe PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_qwen3_5_moe PRIVATE EdgeLLM::Core ${CMAKE_DL_LIBS})
  set_target_properties(trtmc_model_qwen3_5_moe PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET trtmc_model_qwen3_5_moe POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_qwen3_5_moe> VERBATIM)
endif()
