# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_internvl SHARED
  bpe_tokenizer.cpp
  distributed_runtime.cpp
  image_preprocessor.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  sampler.cpp
)

set_source_files_properties(
  PROPERTIES LANGUAGE CUDA
)
set_source_files_properties(image_preprocessor.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-pedantic;-Wno-unused-function"
)

target_include_directories(trtmc_model_internvl
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_internvl SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_internvl PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_internvl PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_internvl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_internvl
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Complete-network offload is family-owned and absent from native-only builds.
if(TARGET EdgeLLM::Core)
  target_sources(trtmc_model_internvl PRIVATE
    edge_llm/adapter.cpp
    edge_llm/device_link.cu
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
    ${PROJECT_SOURCE_DIR}/families/internvl/tests/cpp/edge_vision_features.cpp
    edge_llm/device_link.cu
  )
  target_link_libraries(internvl_edge_vision_features PRIVATE EdgeLLM::Core ${CMAKE_DL_LIBS})
  set_target_properties(internvl_edge_vision_features PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()
