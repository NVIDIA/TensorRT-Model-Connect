# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Optional complete-network offload; all model-specific orchestration stays here.
if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "Gemma Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_gemma PRIVATE "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp" "${CMAKE_CURRENT_LIST_DIR}/device_link.cu")
  target_compile_definitions(trtmc_model_gemma PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_gemma PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_gemma PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET trtmc_model_gemma POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_gemma> VERBATIM)
  if(TRTMC_BUILD_TESTS)
    target_compile_definitions(test_gemma_sampler PRIVATE TRTMC_HAS_EDGE_LLM=1)
    target_link_libraries(test_gemma_sampler PRIVATE EdgeLLM::Core)
  endif()
endif()
