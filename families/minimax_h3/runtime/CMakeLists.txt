# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_minimax_h3 SHARED
  bpe_tokenizer.cpp
  conditioning.cpp
  fl2va_runtime.cpp
  pipeline.cpp
  plugin.cpp
  ref2va_runtime.cpp
  super_resolution_runtime.cpp
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
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<PLATFORM_ID:Windows>>>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_minimax_h3 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<0:>"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_minimax_h3
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TRTMC_BUILD_TESTS)
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
  target_compile_options(test_minimax_h3_math PRIVATE
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  )
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
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  )
  add_test(NAME test_minimax_h3_cuda_rng COMMAND test_minimax_h3_cuda_rng)
  set_tests_properties(test_minimax_h3_cuda_rng PROPERTIES
    LABELS gpu
    SKIP_RETURN_CODE 77
  )

  foreach(_trtmc_minimax_test IN ITEMS conditioning fl2va_runtime ref2va_runtime)
    add_executable(test_minimax_h3_${_trtmc_minimax_test}
      ${PROJECT_SOURCE_DIR}/families/minimax_h3/tests/cpp/test_minimax_h3_${_trtmc_minimax_test}.cpp
    )
    target_include_directories(test_minimax_h3_${_trtmc_minimax_test} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_include_directories(test_minimax_h3_${_trtmc_minimax_test} SYSTEM PRIVATE
      ${TRTMC_CUDA_INCLUDE_DIR}
    )
    target_link_libraries(test_minimax_h3_${_trtmc_minimax_test} PRIVATE
      trtmc_model_minimax_h3
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    target_compile_options(test_minimax_h3_${_trtmc_minimax_test} PRIVATE
      "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
    )
    add_test(NAME minimax_h3_${_trtmc_minimax_test}
      COMMAND test_minimax_h3_${_trtmc_minimax_test})
  endforeach()

  if(WIN32)
    set(_trtmc_minimax_test_path
      "$<TARGET_FILE_DIR:trtmc_model_minimax_h3>$<SEMICOLON>$ENV{PATH}")
    foreach(_trtmc_minimax_test_target IN ITEMS
        minimax_h3_math
        test_minimax_h3_cuda_rng
        minimax_h3_conditioning
        minimax_h3_fl2va_runtime
        minimax_h3_ref2va_runtime)
      set_tests_properties(${_trtmc_minimax_test_target} PROPERTIES
        ENVIRONMENT "PATH=${_trtmc_minimax_test_path}")
    endforeach()
  endif()
endif()
