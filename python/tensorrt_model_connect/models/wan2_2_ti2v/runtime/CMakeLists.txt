# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_wan2_2_ti2v SHARED
  config_schema.cpp
  easycache.cpp
  options.cpp
  pipeline.cpp
  plugin.cpp
  prompt_cleaner.cpp
  runtime_config.cpp
  vae_cache_storage.cpp
  ${TRTMC_MODEL_wan2_2_ti2v_REGISTRATION_SOURCE}
)
if(CMAKE_CUDA_COMPILER)
  target_sources(trtmc_model_wan2_2_ti2v PRIVATE
    torch_cuda_normal.cu
    wan2_2_unipc_cuda.cu
  )
endif()
target_include_directories(trtmc_model_wan2_2_ti2v PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
)
target_include_directories(trtmc_model_wan2_2_ti2v SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
  ${_trtmc_nlohmann_json_include}
  ${PROJECT_SOURCE_DIR}/third_party/stb
)
target_link_libraries(trtmc_model_wan2_2_ti2v PRIVATE
  trtmc_core
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_definitions(trtmc_model_wan2_2_ti2v PRIVATE TRTMC_HAS_TRT=0)
target_compile_options(trtmc_model_wan2_2_ti2v PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_wan2_2_ti2v PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/models/wan2_2_ti2v"
  BUILD_RPATH "\$ORIGIN/../.."
  INSTALL_RPATH "\$ORIGIN/../.."
)

if(TRTMC_BUILD_TESTS)
  trtmc_add_test(test_wan2_2_cuda_rng
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan2_2_cuda_rng.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_wan2_2_cuda_rng)
    target_sources(test_wan2_2_cuda_rng PRIVATE
      torch_cuda_normal.cu
    )
  endif()
  trtmc_add_test(test_wan2_2_easycache
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan2_2_easycache.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_wan2_2_easycache)
    target_sources(test_wan2_2_easycache PRIVATE
      config_schema.cpp
      easycache.cpp
      runtime_config.cpp
    )
  endif()
  trtmc_add_test(test_wan2_2_prompt_cleaner
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan2_2_prompt_cleaner.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_wan2_2_prompt_cleaner)
    target_sources(test_wan2_2_prompt_cleaner PRIVATE
      prompt_cleaner.cpp
    )
  endif()
  trtmc_add_test(test_wan2_2_options
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan2_2_options.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
  )
  if(TARGET test_wan2_2_options)
    target_link_libraries(test_wan2_2_options PRIVATE nlohmann_json::nlohmann_json)
    target_sources(test_wan2_2_options PRIVATE
      options.cpp
    )
  endif()
  trtmc_add_test(test_wan2_2_staged_pipeline
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan2_2_staged_pipeline.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_TRT
    REQUIRES_GPU
  )
  if(TARGET test_wan2_2_staged_pipeline)
    target_link_libraries(test_wan2_2_staged_pipeline PRIVATE trtmc_model_wan2_2_ti2v)
  endif()
  trtmc_add_test(test_wan2_2_unipc_cuda
    SOURCE "${CMAKE_CURRENT_LIST_DIR}/../tests/cpp/test_wan2_2_unipc_cuda.cpp"
    EXTRA_INCLUDES "${CMAKE_CURRENT_LIST_DIR}"
    MODEL_OWNED
    REQUIRES_GPU
  )
  if(TARGET test_wan2_2_unipc_cuda)
    target_sources(test_wan2_2_unipc_cuda PRIVATE
      wan2_2_unipc_cuda.cu
    )
  endif()
endif()
