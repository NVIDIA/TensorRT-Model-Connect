# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_whisper SHARED
  distributed_runtime.cpp
  bpe_tokenizer.cpp
  resampler.cpp
  decode_runtime.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  whisper_mel_spectrogram.cpp
)

target_include_directories(trtmc_model_whisper
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_whisper SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_whisper PRIVATE
  trtmc_core
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_whisper PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_whisper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_whisper
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_whisper_mel_spectrogram
    ${PROJECT_SOURCE_DIR}/families/whisper/tests/cpp/test_whisper_mel_spectrogram.cpp
    whisper_mel_spectrogram.cpp
  )
  target_include_directories(test_whisper_mel_spectrogram PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_whisper_mel_spectrogram PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME whisper_mel_spectrogram COMMAND test_whisper_mel_spectrogram)

  add_executable(test_whisper_cross_kv_contract
    ${PROJECT_SOURCE_DIR}/families/whisper/tests/cpp/test_whisper_cross_kv_contract.cpp
  )
  target_include_directories(test_whisper_cross_kv_contract PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_whisper_cross_kv_contract PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME whisper_cross_kv_contract COMMAND test_whisper_cross_kv_contract)

  # CPU CUDA stubs live in the test, so allocation failures can be injected
  # without a GPU or cudart.
  add_executable(test_whisper_cross_kv_alloc
    ${PROJECT_SOURCE_DIR}/families/whisper/tests/cpp/test_whisper_cross_kv_alloc.cpp
  )
  target_include_directories(test_whisper_cross_kv_alloc PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_whisper_cross_kv_alloc SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_compile_options(test_whisper_cross_kv_alloc PRIVATE
    -Wall -Wextra -Wpedantic
  )
  add_test(NAME whisper_cross_kv_alloc COMMAND test_whisper_cross_kv_alloc)

  # The decode step's conditions and order over stand-ins, and the page-locked
  # buffer against CPU CUDA stubs: no GPU or cudart.
  add_executable(test_whisper_decode_step
    ${PROJECT_SOURCE_DIR}/families/whisper/tests/cpp/test_whisper_decode_step.cpp
    decode_runtime.cpp
  )
  target_include_directories(test_whisper_decode_step PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(test_whisper_decode_step SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_compile_options(test_whisper_decode_step PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME whisper_decode_step COMMAND test_whisper_decode_step)

  # The pipeline and its KV cache themselves over a CPU CUDA runtime and
  # stand-in modules, both defined in the test: no GPU, cudart or TensorRT.
  add_executable(test_whisper_decode_pipeline
    ${PROJECT_SOURCE_DIR}/families/whisper/tests/cpp/test_whisper_decode_pipeline.cpp
    ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp
    decode_runtime.cpp
    kv_cache.cpp
    pipeline.cpp
    resampler.cpp
    whisper_mel_spectrogram.cpp
  )
  target_include_directories(test_whisper_decode_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_whisper_decode_pipeline SYSTEM PRIVATE
    ${TRTMC_CUDA_INCLUDE_DIR}
  )
  target_compile_options(test_whisper_decode_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME whisper_decode_pipeline COMMAND test_whisper_decode_pipeline)
endif()

if(TRTMC_BUILD_TESTS)
  # CPU-backed KV-cache test: CUDA calls are served from host memory, and an
  # overlapping copy fails the test.
  add_executable(test_whisper_kv_cache
    ${PROJECT_SOURCE_DIR}/families/whisper/tests/cpp/test_kv_cache.cpp
    kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/core/runtime/primitives/device_tensor.cpp
  )
  target_include_directories(test_whisper_kv_cache PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_include_directories(test_whisper_kv_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_compile_options(test_whisper_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME whisper_kv_cache COMMAND test_whisper_kv_cache)
endif()
