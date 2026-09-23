# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "InternVL Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_internvl PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_compile_definitions(trtmc_model_internvl PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_internvl PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_internvl PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()


if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_internvl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_internvl>
    VERBATIM
  )
endif()

# Edge 0.10.1 has no visual-feature dump CLI. This helper serves the existing
# InternVL image-health E2E and is not installed as a product executable.
if(TRTMC_BUILD_TESTS AND TARGET EdgeLLM::Core)
  add_executable(internvl_edge_vision_features
    "${CMAKE_CURRENT_LIST_DIR}/vision_features.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_link_libraries(internvl_edge_vision_features PRIVATE EdgeLLM::Core ${CMAKE_DL_LIBS})
  set_target_properties(internvl_edge_vision_features PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()
