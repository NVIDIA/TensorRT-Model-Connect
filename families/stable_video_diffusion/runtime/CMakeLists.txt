# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_stable_video_diffusion SHARED
  pipeline.cpp
  plugin.cpp
  preprocess.cpp
  runtime_config.cpp
  sampling.cpp
  torch_random.cpp
)

target_include_directories(trtmc_model_stable_video_diffusion
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_stable_video_diffusion SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_stable_video_diffusion PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
# The host sampler, noise and preprocessing reproduce the reference's float32
# rounding operation by operation, so products must not be fused into FMAs.
target_compile_options(trtmc_model_stable_video_diffusion PRIVATE
  -Wall -Wextra -Wpedantic -ffp-contract=off
)
set_target_properties(trtmc_model_stable_video_diffusion PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_stable_video_diffusion
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  foreach(_test IN ITEMS sampling random preprocess)
    add_executable(test_stable_video_diffusion_${_test}
      ${PROJECT_SOURCE_DIR}/families/stable_video_diffusion/tests/cpp/test_${_test}.cpp
    )
    target_include_directories(test_stable_video_diffusion_${_test} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_stable_video_diffusion_${_test} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(test_stable_video_diffusion_${_test} PRIVATE
      trtmc_model_stable_video_diffusion
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    target_compile_options(test_stable_video_diffusion_${_test} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    add_test(NAME stable_video_diffusion_${_test} COMMAND test_stable_video_diffusion_${_test})
    set_tests_properties(stable_video_diffusion_${_test} PROPERTIES LABELS "cpu")
  endforeach()

  # Public SDK consumers for the family E2E. They link only the public C
  # runtime; building the CPU tests builds them too.
  foreach(_language IN ITEMS c cpp)
    add_executable(test_stable_video_diffusion_sdk_${_language}
      ${PROJECT_SOURCE_DIR}/families/stable_video_diffusion/tests/sdk_consumer.${_language}
    )
    target_link_libraries(test_stable_video_diffusion_sdk_${_language} PRIVATE trtmc_c)
    target_compile_options(test_stable_video_diffusion_sdk_${_language} PRIVATE
      -Wall -Wextra -Wpedantic -Werror
    )
    set_target_properties(test_stable_video_diffusion_sdk_${_language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endforeach()
  add_dependencies(test_stable_video_diffusion_sampling
    test_stable_video_diffusion_sdk_c
    test_stable_video_diffusion_sdk_cpp
  )
endif()
