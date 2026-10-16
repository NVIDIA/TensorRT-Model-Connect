# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc)

add_library(trtmc_model_lfm2 SHARED
  bpe_tokenizer.cpp
  byte_level_decoder.cpp
  chat_templates.cpp
  conv_state.cpp
  conversation.cpp
  hybrid_state.cpp
  kv_cache.cpp
  pipeline.cpp
  plugin.cpp
  plugin_helpers.cpp
  pretokenizer.cpp
  sampler.cpp
  sampler_module.cpp
)

target_include_directories(trtmc_model_lfm2
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core/runtime/include
)
target_include_directories(trtmc_model_lfm2 SYSTEM PRIVATE
  ${TRTMC_CUDA_INCLUDE_DIR}
)
target_link_libraries(trtmc_model_lfm2 PRIVATE
  trtmc_core
  ICU::uc
  nlohmann_json::nlohmann_json
  ${TRTMC_CUDART_LIBRARY}
  ${CMAKE_DL_LIBS}
)
target_compile_options(trtmc_model_lfm2 PRIVATE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra>"
)
set_target_properties(trtmc_model_lfm2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN"
  INSTALL_RPATH "\$ORIGIN"
)
install(TARGETS trtmc_model_lfm2
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set(_lfm2_runtime_libraries "[]")
if(TARGET trtmc_byok_tvm_ffi)
  include("${CMAKE_CURRENT_LIST_DIR}/cudnn.cmake")
  if(TARGET lfm2_cudnn)
    add_library(trtmc_lfm2_sdpa_byok SHARED sdpa_byok.cpp mean64_byok.cu)
    target_include_directories(trtmc_lfm2_sdpa_byok SYSTEM PRIVATE
      ${LFM2_CUDNN_FRONTEND_INCLUDE_DIR} ${LFM2_CUDNN_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS} ${TRTMC_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(trtmc_lfm2_sdpa_byok PRIVATE
      lfm2_cudnn CUDA::cudart CUDA::nvrtc ${TRTMC_TVM_FFI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
    add_dependencies(trtmc_lfm2_sdpa_byok lfm2_cudnn_frontend_source lfm2_cudnn_stage)
    target_compile_options(trtmc_lfm2_sdpa_byok PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-O2;--ftz=false;--prec-div=true;--fmad=true>")
    set_target_properties(trtmc_lfm2_sdpa_byok PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
      BUILD_RPATH "\$ORIGIN;${TRTMC_TVM_FFI_LIBRARY_DIR}"
      INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../tvm_ffi/lib;${TRTMC_TVM_FFI_LIBRARY_DIR};/usr/local/cuda/lib64")
    add_dependencies(trtmc_model_lfm2 trtmc_lfm2_sdpa_byok)
    foreach(component IN ITEMS sdk runtime)
      install(TARGETS trtmc_lfm2_sdpa_byok
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component})
      install(FILES ${LFM2_CUDNN_STAGED_LIBRARIES}
        DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component})
      install(FILES sdpa_byok.NOTICE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/trtmc_lfm2_sdpa_byok
        COMPONENT ${component})
      install(FILES "${LFM2_CUDNN_STAGED_LICENSE}"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/trtmc_lfm2_sdpa_byok/cudnn
        RENAME LICENSE.txt COMPONENT ${component})
      install(FILES "${LFM2_CUDNN_FRONTEND_SOURCE_DIR}/LICENSE.txt"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/trtmc_lfm2_sdpa_byok/frontend
        COMPONENT ${component})
      install(FILES "${LFM2_CUDNN_FRONTEND_SOURCE_DIR}/include/cudnn_frontend/thirdparty/nlohmann/LICENSE.MIT"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/trtmc_lfm2_sdpa_byok/frontend/nlohmann
        COMPONENT ${component})
    endforeach()
  endif()
  add_library(trtmc_lfm2_w2_byok SHARED w2_byok.cpp)
  target_include_directories(trtmc_lfm2_w2_byok SYSTEM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_lfm2_w2_byok PRIVATE
    CUDA::cublas
    CUDA::cudart
    ${TRTMC_TVM_FFI_LIBRARY}
  )
  set_target_properties(trtmc_lfm2_w2_byok PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "\$ORIGIN;${TRTMC_TVM_FFI_LIBRARY_DIR}"
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../tvm_ffi/lib;${TRTMC_TVM_FFI_LIBRARY_DIR};/usr/local/cuda/lib64"
  )
  add_library(trtmc_lfm2_sampler_byok SHARED sampler_byok.cu)
  target_include_directories(trtmc_lfm2_sampler_byok SYSTEM PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
  )
  target_link_libraries(trtmc_lfm2_sampler_byok PRIVATE
    CUDA::cudart
    ${TRTMC_TVM_FFI_LIBRARY}
  )
  target_compile_options(trtmc_lfm2_sampler_byok PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-O2;--ftz=false;--prec-div=true;--fmad=true>"
  )
  set_target_properties(trtmc_lfm2_sampler_byok PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    CUDA_RUNTIME_LIBRARY Shared
    BUILD_RPATH "\$ORIGIN;${TRTMC_TVM_FFI_LIBRARY_DIR}"
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../../tvm_ffi/lib;${TRTMC_TVM_FFI_LIBRARY_DIR};/usr/local/cuda/lib64"
  )
  add_dependencies(trtmc_model_lfm2 trtmc_lfm2_sampler_byok trtmc_lfm2_w2_byok trtmc_byok_tvm_ffi)
  set(_lfm2_runtime_library_names
    "$<TARGET_FILE_NAME:trtmc_lfm2_sampler_byok>"
    "$<TARGET_FILE_NAME:trtmc_lfm2_w2_byok>"
    "$<TARGET_FILE_NAME:trtmc_byok_tvm_ffi>")
  if(TARGET trtmc_lfm2_sdpa_byok)
    list(APPEND _lfm2_runtime_library_names
      "$<TARGET_FILE_NAME:trtmc_lfm2_sdpa_byok>" ${LFM2_CUDNN_RUNTIME_NAMES})
  endif()
  string(JOIN "\",\"" _lfm2_runtime_libraries ${_lfm2_runtime_library_names})
  set(_lfm2_runtime_libraries "[\"${_lfm2_runtime_libraries}\"]")
  # Wheels install the SDK component; native installations can select runtime.
  foreach(component IN ITEMS sdk runtime)
    install(TARGETS trtmc_lfm2_sampler_byok trtmc_lfm2_w2_byok
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${component}
    )
    install(FILES sampler_byok.NOTICE
      DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/trtmc_lfm2_sampler_byok
      COMPONENT ${component}
    )
  endforeach()
endif()

set(_lfm2_runtime_manifest "${CMAKE_BINARY_DIR}/families/lfm2/runtime-libraries.json")
file(GENERATE OUTPUT "${_lfm2_runtime_manifest}"
  CONTENT "{\"schema_version\":1,\"libraries\":${_lfm2_runtime_libraries}}\n")
foreach(component IN ITEMS sdk runtime)
  install(FILES "${_lfm2_runtime_manifest}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/families/lfm2 COMPONENT ${component}
  )
endforeach()

if(TRTMC_BUILD_TESTS)
  if(TARGET trtmc_lfm2_w2_byok)
    add_executable(test_lfm2_w2_contract
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm2_w2_contract.cpp)
    target_include_directories(test_lfm2_w2_contract PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(test_lfm2_w2_contract SYSTEM PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS} ${TRTMC_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(test_lfm2_w2_contract PRIVATE
      CUDA::cublas CUDA::cudart ${TRTMC_TVM_FFI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
    set_target_properties(test_lfm2_w2_contract PROPERTIES
      BUILD_RPATH "${CMAKE_BINARY_DIR};${TRTMC_TVM_FFI_LIBRARY_DIR}")
    add_test(NAME test_lfm2_w2_contract COMMAND test_lfm2_w2_contract)
  endif()
  if(TARGET trtmc_lfm2_sdpa_byok)
    add_executable(test_lfm2_sdpa_contract
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm2_sdpa_contract.cpp)
    target_include_directories(test_lfm2_sdpa_contract PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(test_lfm2_sdpa_contract SYSTEM PRIVATE
      ${LFM2_CUDNN_FRONTEND_INCLUDE_DIR} ${LFM2_CUDNN_INCLUDE_DIR}
      ${CUDAToolkit_INCLUDE_DIRS} ${TRTMC_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(test_lfm2_sdpa_contract PRIVATE
      lfm2_cudnn CUDA::cudart CUDA::nvrtc ${TRTMC_TVM_FFI_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)
    add_dependencies(test_lfm2_sdpa_contract lfm2_cudnn_frontend_source lfm2_cudnn_stage)
    set_target_properties(test_lfm2_sdpa_contract PROPERTIES
      BUILD_RPATH "${CMAKE_BINARY_DIR};${TRTMC_TVM_FFI_LIBRARY_DIR}")
    add_test(NAME test_lfm2_sdpa_contract COMMAND test_lfm2_sdpa_contract)
    add_executable(test_lfm2_mean64_contract
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm2_mean64_contract.cu)
    target_include_directories(test_lfm2_mean64_contract PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(test_lfm2_mean64_contract SYSTEM PRIVATE
      ${CUDAToolkit_INCLUDE_DIRS} ${TRTMC_TVM_FFI_INCLUDE_DIR})
    target_link_libraries(test_lfm2_mean64_contract PRIVATE CUDA::cudart ${TRTMC_TVM_FFI_LIBRARY})
    target_compile_options(test_lfm2_mean64_contract PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-O2;--ftz=false;--prec-div=true;--fmad=true>")
    set_target_properties(test_lfm2_mean64_contract PROPERTIES
      CUDA_RUNTIME_LIBRARY Shared
      BUILD_RPATH "${CMAKE_BINARY_DIR};${TRTMC_TVM_FFI_LIBRARY_DIR}")
    add_test(NAME test_lfm2_mean64_contract COMMAND test_lfm2_mean64_contract)
  endif()
  if(TARGET trtmc_backend_trt)
    add_executable(lfm2_logits_probe
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/lfm25_logits_probe.cpp)
    target_include_directories(lfm2_logits_probe PRIVATE
      ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include ${TRTMC_CUDA_INCLUDE_DIR})
    target_link_libraries(lfm2_logits_probe PRIVATE
      trtmc_model_lfm2 trtmc_core trtmc_backend_trt nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
  endif()

  foreach(test_name IN ITEMS
      test_lfm2_byte_level_decoder
      test_lfm2_chat_template
      test_lfm2_pretokenizer
      test_lfm2_runtime
      test_lfm2_sampler
  )
    add_executable(${test_name}
      ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/${test_name}.cpp
    )
    target_include_directories(${test_name} PRIVATE
      ${PROJECT_SOURCE_DIR}
      ${PROJECT_SOURCE_DIR}/core/runtime/include
    )
    target_link_libraries(${test_name} PRIVATE
      trtmc_model_lfm2
      trtmc_core
      ${TRTMC_CUDART_LIBRARY}
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_executable(test_lfm25_bpe_flags
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm25_bpe_flags.cpp)
  target_include_directories(test_lfm25_bpe_flags PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_lfm25_bpe_flags PRIVATE trtmc_model_lfm2 nlohmann_json::nlohmann_json)
  add_test(NAME test_lfm25_bpe_flags COMMAND test_lfm25_bpe_flags
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/fixtures/lfm25-bpe-flags.json)
  add_executable(test_lfm25_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm25_tokenizer.cpp)
  target_include_directories(test_lfm25_tokenizer PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_lfm25_tokenizer PRIVATE trtmc_model_lfm2 nlohmann_json::nlohmann_json)
  add_test(NAME test_lfm25_tokenizer COMMAND test_lfm25_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/fixtures/lfm25-tokenizer.json)
  add_custom_target(test_lfm25_tokenizer_ids)
  add_dependencies(test_lfm25_tokenizer_ids test_lfm25_tokenizer)
  add_test(NAME test_lfm25_tokenizer_ids COMMAND test_lfm25_tokenizer
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/fixtures/lfm25-tokenizer.json --token-ids)
  set_tests_properties(test_lfm25_tokenizer_ids PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties(test_lfm2_runtime PROPERTIES SKIP_RETURN_CODE 77)
  add_executable(test_lfm2_sampler_contract
    ${PROJECT_SOURCE_DIR}/families/lfm2/tests/cpp/test_lfm2_sampler_contract.cpp)
  target_include_directories(test_lfm2_sampler_contract PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_lfm2_sampler_contract PRIVATE
    trtmc_model_lfm2 trtmc_core nlohmann_json::nlohmann_json
    ${TRTMC_CUDART_LIBRARY} ${CMAKE_DL_LIBS})
  add_test(NAME test_lfm2_sampler_contract COMMAND test_lfm2_sampler_contract)
endif()
