# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "NemotronH Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_nemotron_h PRIVATE "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp" "${CMAKE_CURRENT_LIST_DIR}/device_link.cu")
  target_compile_definitions(trtmc_model_nemotron_h PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_nemotron_h PRIVATE EdgeLLM::Core nlohmann_json::nlohmann_json)
  set_target_properties(trtmc_model_nemotron_h PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET trtmc_model_nemotron_h POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_nemotron_h>
    VERBATIM)
endif()
