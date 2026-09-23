# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_minimax_h3 SHARED
  bpe_tokenizer.cpp
  pipeline.cpp
  plugin.cpp
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
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_minimax_h3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minimax_h3
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

option(TRTMC_BUILD_MINIMAX_H3_VSA
  "Build the portable FastH3 VSA TVM-FFI kernel" OFF)
option(TRTMC_BUILD_MINIMAX_H3_VSA_SM100A
  "Include the optimized FastH3 sm100a and sm103a VSA kernels" OFF)
if(TRTMC_BUILD_MINIMAX_H3_VSA OR TRTMC_BUILD_MINIMAX_H3_VSA_SM100A)
  if(NOT TRTMC_TVM_FFI_INCLUDE_DIR OR NOT TRTMC_TVM_FFI_LIBRARY)
    message(FATAL_ERROR "FastH3 VSA requires TRTMC_ENABLE_BYOK=ON")
  endif()
  add_library(trtmc_minimax_h3_vsa SHARED
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/kernels/vsa.cu
  )
  target_include_directories(trtmc_minimax_h3_vsa PRIVATE
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/kernels
  )
  target_include_directories(trtmc_minimax_h3_vsa SYSTEM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_minimax_h3_vsa PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    ${TRTMC_TVM_FFI_LIBRARY}
  )
  target_compile_features(trtmc_minimax_h3_vsa PRIVATE cxx_std_17 cuda_std_17)
  target_compile_options(trtmc_minimax_h3_vsa PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra;--generate-code=arch=compute_80,code=compute_80>"
  )
  set_target_properties(trtmc_minimax_h3_vsa PROPERTIES
    CUDA_ARCHITECTURES OFF
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "\$ORIGIN;${TRTMC_TVM_FFI_LIBRARY_DIR}"
    INSTALL_RPATH "\$ORIGIN;${TRTMC_TVM_FFI_LIBRARY_DIR}"
  )
  install(TARGETS trtmc_minimax_h3_vsa
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

  if(TRTMC_BUILD_MINIMAX_H3_VSA_SM100A)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      message(FATAL_ERROR "FastH3 VSA Blackwell optimization requires CUDA 13.0 or newer")
    endif()
    target_compile_definitions(trtmc_minimax_h3_vsa PRIVATE
      TRTMC_MINIMAX_H3_VSA_HAS_BLACKWELL=1
    )
    target_compile_options(trtmc_minimax_h3_vsa PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_100a,code=sm_100a;--generate-code=arch=compute_103a,code=sm_103a>"
    )
  endif()
endif()

if(TRTMC_BUILD_TESTS)
  add_executable(test_minimax_h3_vsa_backend
    ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_minimax_h3_vsa_backend.cpp
  )
  target_include_directories(test_minimax_h3_vsa_backend PRIVATE
    ${PROJECT_SOURCE_DIR}
  )
  target_compile_options(test_minimax_h3_vsa_backend PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME minimax_h3_vsa_backend COMMAND test_minimax_h3_vsa_backend)

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
