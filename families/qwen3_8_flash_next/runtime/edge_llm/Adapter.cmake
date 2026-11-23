# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
if(TARGET TensorRTEdgeLLM::Core AND TRTMC_EDGELLM_RELEASE STREQUAL "0.12.0")
  target_sources(trtmc_model_qwen3_8_flash_next PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/adapter.cpp ${CMAKE_CURRENT_LIST_DIR}/device_link.cu)
  target_compile_definitions(trtmc_model_qwen3_8_flash_next PRIVATE TRTMC_FLASH_NEXT_EDGE=1)
  target_include_directories(trtmc_model_qwen3_8_flash_next SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(trtmc_model_qwen3_8_flash_next PRIVATE TensorRTEdgeLLM::Core ${CMAKE_DL_LIBS} ${TRTMC_CUDART_LIBRARY})
  set_target_properties(trtmc_model_qwen3_8_flash_next PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  add_custom_command(TARGET trtmc_model_qwen3_8_flash_next POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:TensorRTEdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_qwen3_8_flash_next>/libNvInfer_edgellm_plugin.so VERBATIM)
endif()
