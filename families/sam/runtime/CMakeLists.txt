# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_sam SHARED
  distributed_runtime.cpp
  plugin.cpp
  plugin_helpers.cpp
  sam_pipeline.cpp
)

target_include_directories(trtmc_model_sam
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_sam SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_sam PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_sam PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_sam PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_sam
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_sam_prompt_seam
    ${PROJECT_SOURCE_DIR}/families/sam/tests/cpp/test_sam_prompt_seam.cpp
  )
  target_include_directories(test_sam_prompt_seam PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_sam_prompt_seam PRIVATE
    trtmc_model_sam
  )
  target_compile_options(test_sam_prompt_seam PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME sam_prompt_seam COMMAND test_sam_prompt_seam)

  # SamPipeline with mock engines: the image embedding's handoff between them.
  add_executable(test_sam_pipeline_handoff
    ${PROJECT_SOURCE_DIR}/families/sam/tests/cpp/test_sam_pipeline_handoff.cpp
  )
  target_include_directories(test_sam_pipeline_handoff PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_sam_pipeline_handoff SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_sam_pipeline_handoff PRIVATE
    trtmc_model_sam trtmc_core ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_sam_pipeline_handoff PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME sam_pipeline_handoff COMMAND test_sam_pipeline_handoff)
  set_tests_properties(sam_pipeline_handoff PROPERTIES LABELS "cpu")

  # The seam tests compare the separable resampling to the per-sample
  # definition bit for bit. A default x86-64 build has no FMA, so it cannot see
  # a multiply-add that would be fused differently where FMA is contracted (as
  # on aarch64); this build contracts them. GCC fuses only when optimizing, so
  # the target sets -O2 whatever the build type. It needs an FMA-capable CPU.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND NOT MSVC)
    add_executable(test_sam_prompt_seam_fma
      ${PROJECT_SOURCE_DIR}/families/sam/tests/cpp/test_sam_prompt_seam.cpp
    )
    target_include_directories(test_sam_prompt_seam_fma PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_compile_options(test_sam_prompt_seam_fma PRIVATE
      -Wall -Wextra -Wpedantic -O2 -mfma -ffp-contract=fast
    )
    add_test(NAME sam_prompt_seam_fma COMMAND test_sam_prompt_seam_fma)
  endif()
endif()
