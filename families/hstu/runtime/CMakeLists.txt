# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_hstu SHARED pipeline.cpp cached_pipeline.cpp request.cpp cache_policy.cpp
  paged_cache.cpp native_library.cpp config.cpp plugin.cpp)
target_include_directories(trtmc_model_hstu PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_hstu SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_hstu PRIVATE trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_model_hstu PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_hstu PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_hstu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_executable(trtmc_hstu runner.cpp)
target_include_directories(trtmc_hstu PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_hstu PRIVATE trtmc_runtime nlohmann_json::nlohmann_json)
target_compile_options(trtmc_hstu PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_hstu PROPERTIES OUTPUT_NAME trtmc-hstu
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS trtmc_hstu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
# A selectively built family includes its public native command.
add_dependencies(trtmc_model_hstu trtmc_hstu)

if(TRTMC_BUILD_TESTS)
  add_executable(hstu_cache_benchmark ../tests/cpp/cache_benchmark.cpp)
  target_include_directories(hstu_cache_benchmark PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(hstu_cache_benchmark SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(hstu_cache_benchmark PRIVATE
    trtmc_runtime trtmc_core nlohmann_json::nlohmann_json CUDA::cudart)
  target_compile_options(hstu_cache_benchmark PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(hstu_cache_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")

  foreach(driver IN ITEMS cache_sequence session)
    add_executable(hstu_${driver}_runner ../tests/cpp/${driver}_runner.cpp)
    target_include_directories(hstu_${driver}_runner PRIVATE
      ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
    target_include_directories(hstu_${driver}_runner SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
    target_link_libraries(hstu_${driver}_runner PRIVATE
      trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
    target_compile_options(hstu_${driver}_runner PRIVATE -Wall -Wextra -Wpedantic)
    set_target_properties(hstu_${driver}_runner PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")
    # Selective family builds include their E2E drivers. Drivers load the model
    # dynamically and must not depend on the model target in the other direction.
    add_dependencies(trtmc_model_hstu hstu_${driver}_runner)
  endforeach()

  add_executable(test_hstu_cache_policy ../tests/cpp/test_cache_policy.cpp cache_policy.cpp)
  target_include_directories(test_hstu_cache_policy PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_hstu_cache_policy PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hstu_cache_policy COMMAND test_hstu_cache_policy)
  set_tests_properties(hstu_cache_policy PROPERTIES LABELS cpu)

  add_executable(test_hstu_attention_metadata ../tests/cpp/test_attention_metadata.cpp)
  target_include_directories(test_hstu_attention_metadata PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(test_hstu_attention_metadata PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hstu_attention_metadata COMMAND test_hstu_attention_metadata)
  set_tests_properties(hstu_attention_metadata PROPERTIES LABELS cpu)

  add_executable(test_hstu_native_manifest ../tests/cpp/test_native_manifest.cpp)
  target_include_directories(test_hstu_native_manifest PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_hstu_native_manifest SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_hstu_native_manifest PRIVATE trtmc_model_hstu nlohmann_json::nlohmann_json)
  target_compile_options(test_hstu_native_manifest PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hstu_native_manifest COMMAND test_hstu_native_manifest)
  set_tests_properties(hstu_native_manifest PROPERTIES LABELS cpu)

  add_executable(test_hstu_paged_cache ../tests/cpp/test_paged_cache.cpp)
  target_include_directories(test_hstu_paged_cache PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_hstu_paged_cache SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_hstu_paged_cache PRIVATE trtmc_model_hstu)
  target_compile_options(test_hstu_paged_cache PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hstu_paged_cache COMMAND test_hstu_paged_cache)
  set_tests_properties(hstu_paged_cache PROPERTIES LABELS cpu)

  add_executable(test_hstu_pipeline ../tests/cpp/test_pipeline.cpp)
  target_include_directories(test_hstu_pipeline PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_hstu_pipeline SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_hstu_pipeline PRIVATE trtmc_model_hstu)
  target_compile_options(test_hstu_pipeline PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hstu_pipeline COMMAND test_hstu_pipeline)
  set_tests_properties(hstu_pipeline PROPERTIES LABELS cpu)
endif()
