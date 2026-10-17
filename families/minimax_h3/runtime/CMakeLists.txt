# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_minimax_h3 SHARED
  bpe_tokenizer.cpp
  diffusion_schedule.cpp
  pipeline.cpp
  plugin.cpp
  qwen_reference.cpp
  reference_conditioner.cpp
  reference_denoiser.cpp
  reference_image.cpp
  reference_layout.cpp
  reference_visual_encoder.cpp
  visual_reference.cpp
  torch_cuda_normal.cu
)

set_source_files_properties(
  torch_cuda_normal.cu
  PROPERTIES LANGUAGE CUDA
)

target_include_directories(trtmc_model_minimax_h3
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_minimax_h3 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_minimax_h3 PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
)
target_compile_options(trtmc_model_minimax_h3 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>"
)
set_target_properties(trtmc_model_minimax_h3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minimax_h3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${PROJECT_SOURCE_DIR}/families/minimax_h3/NOTICE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/trtmc_model_minimax_h3
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_minimax_h3_reference_denoiser
    diffusion_schedule.cpp reference_layout.cpp reference_denoiser.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_reference_denoiser.cpp
  )
  target_include_directories(test_minimax_h3_reference_denoiser PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minimax_h3_reference_denoiser SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_minimax_h3_reference_denoiser PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_reference_denoiser COMMAND test_minimax_h3_reference_denoiser)

  # The owning Python GPU oracle drives this probe with a real serialized
  # posterior plan and compares native CPU noise plus backend execution.
  add_executable(minimax_h3_reference_posterior_probe
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/reference_posterior_probe.cpp
  )
  target_include_directories(minimax_h3_reference_posterior_probe PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(minimax_h3_reference_posterior_probe SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(minimax_h3_reference_posterior_probe PRIVATE
    trtmc_model_minimax_h3 trtmc_core ${CMAKE_DL_LIBS}
  )
  add_dependencies(minimax_h3_reference_posterior_probe trtmc_backend_trt)
  set_target_properties(minimax_h3_reference_posterior_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
  target_compile_options(minimax_h3_reference_posterior_probe PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(test_minimax_h3_reference_layout
    reference_layout.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_reference_layout.cpp
  )
  target_include_directories(test_minimax_h3_reference_layout PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_minimax_h3_reference_layout PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_minimax_h3_reference_layout PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_reference_layout COMMAND test_minimax_h3_reference_layout)

  add_executable(test_minimax_h3_reference_visual_encoder
    reference_visual_encoder.cpp
    reference_image.cpp
    visual_reference.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_reference_visual_encoder.cpp
  )
  target_include_directories(test_minimax_h3_reference_visual_encoder PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minimax_h3_reference_visual_encoder SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_minimax_h3_reference_visual_encoder PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_reference_visual_encoder COMMAND test_minimax_h3_reference_visual_encoder)

  add_executable(test_minimax_h3_reference_image
    reference_image.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_reference_image.cpp
  )
  target_include_directories(test_minimax_h3_reference_image PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_minimax_h3_reference_image PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_reference_image COMMAND test_minimax_h3_reference_image)

  add_executable(test_minimax_h3_reference_conditioner
    qwen_reference.cpp
    reference_conditioner.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_reference_conditioner.cpp
  )
  target_include_directories(test_minimax_h3_reference_conditioner PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minimax_h3_reference_conditioner SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_minimax_h3_reference_conditioner PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_reference_conditioner COMMAND test_minimax_h3_reference_conditioner)

  add_executable(test_minimax_h3_qwen_reference
    qwen_reference.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_qwen_reference.cpp
  )
  target_include_directories(test_minimax_h3_qwen_reference PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_minimax_h3_qwen_reference PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_qwen_reference COMMAND test_minimax_h3_qwen_reference)

  # Host-only reference layout/math; no CUDA or model runtime is required.
  add_executable(test_minimax_h3_visual_reference
    visual_reference.cpp
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_visual_reference.cpp
  )
  target_include_directories(test_minimax_h3_visual_reference PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_minimax_h3_visual_reference PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_visual_reference COMMAND test_minimax_h3_visual_reference)

  add_executable(test_minimax_h3_math
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_minimax_h3_math.cpp
  )
  target_include_directories(test_minimax_h3_math PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minimax_h3_math SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_minimax_h3_math PRIVATE trtmc_model_minimax_h3)
  target_compile_options(test_minimax_h3_math PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_math COMMAND test_minimax_h3_math)

  add_executable(test_minimax_h3_cuda_rng
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_minimax_h3_cuda_rng.cpp
  )
  target_include_directories(test_minimax_h3_cuda_rng PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_minimax_h3_cuda_rng SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_link_libraries(test_minimax_h3_cuda_rng PRIVATE
    trtmc_model_minimax_h3
    trtmc_core
    ${TRTMC_CUDART_LIBRARY}
  )
  target_compile_options(test_minimax_h3_cuda_rng PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME test_minimax_h3_cuda_rng COMMAND test_minimax_h3_cuda_rng)
  set_tests_properties(test_minimax_h3_cuda_rng PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )
endif()
