# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  if(NOT TARGET EdgeLLM::Plugin)
    message(FATAL_ERROR "Llama Edge adapter requires the complete EdgeLLM package (Core and Plugin)")
  endif()
  target_sources(trtmc_model_llama PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/adapter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/device_link.cu"
  )
  target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_HAS_EDGE_LLM=1)
  target_link_libraries(trtmc_model_llama PRIVATE EdgeLLM::Core)
  set_target_properties(trtmc_model_llama PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(TARGET EdgeLLM::Core)
  add_custom_command(TARGET trtmc_model_llama POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:EdgeLLM::Plugin> $<TARGET_FILE_DIR:trtmc_model_llama>
    VERBATIM
  )
endif()

# The wheel/source provider is independent of the legacy SDK and vendor headers.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_sources(trtmc_model_llama PRIVATE "${CMAKE_CURRENT_LIST_DIR}/provider_adapter.cpp")
  target_compile_definitions(trtmc_model_llama PRIVATE TRTMC_HAS_EDGE_PROVIDER=1)
  target_link_libraries(trtmc_model_llama PRIVATE ${CMAKE_DL_LIBS})
  foreach(version IN ITEMS 0.10.0 0.11.0)
    string(REPLACE "." "_" suffix "${version}")
    set(provider "trtmc_edge_provider_llama_${suffix}")
    add_library(${provider} SHARED "${PROJECT_SOURCE_DIR}/cmake/edge_llm/provider/provider.cpp")
    target_compile_definitions(${provider} PRIVATE TRTMC_EDGE_PROVIDER_VERSION="${version}")
    target_link_libraries(${provider} PRIVATE nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    set_target_properties(${provider} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    install(TARGETS ${provider} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    add_dependencies(trtmc_model_llama ${provider})
  endforeach()
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/edge_llm/llama")
  configure_file("${PROJECT_SOURCE_DIR}/families/llama/edge_llm/provider.py"
    "${CMAKE_BINARY_DIR}/edge_llm/llama/provider.py" COPYONLY)
  install(FILES "${PROJECT_SOURCE_DIR}/families/llama/edge_llm/provider.py"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/edge_llm/llama")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/edge_llm/provider/installation.py"
    "${CMAKE_BINARY_DIR}/edge_llm/llama/installation.py" COPYONLY)
  install(FILES "${PROJECT_SOURCE_DIR}/cmake/edge_llm/provider/installation.py"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/edge_llm/llama")
endif()
