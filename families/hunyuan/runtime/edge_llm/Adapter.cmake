# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "Hunyuan Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_hunyuan PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/tokenizer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_compile_definitions(trtmc_model_hunyuan PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_hunyuan PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_hunyuan PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_hunyuan POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_hunyuan>
    VERBATIM
  )
endif()

# The Python wheel installs only the sdk component; its family DSO lives in bin.
if(TARGET EdgeLLM::Plugin)
  install(CODE "
    file(INSTALL \"$<TARGET_FILE:EdgeLLM::Plugin>\"
      DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}\"
      TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN)
  " COMPONENT sdk)
endif()
