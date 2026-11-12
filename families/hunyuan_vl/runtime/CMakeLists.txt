# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_hunyuan_vl SHARED plugin.cpp tokenizer.cpp media.cpp)
target_include_directories(trtmc_model_hunyuan_vl PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_hunyuan_vl SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_hunyuan_vl PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_hunyuan_vl PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_hunyuan_vl PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_hunyuan_vl LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(hunyuan_vl_inference_test ../tests/cpp/run_model.cpp)
  target_link_libraries(hunyuan_vl_inference_test PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  target_include_directories(hunyuan_vl_inference_test PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  # Preserve helper availability for direct family-target builds.
  add_dependencies(trtmc_model_hunyuan_vl hunyuan_vl_inference_test)
  add_executable(hunyuan_vl_preprocess_probe ../tests/cpp/preprocess_probe.cpp media.cpp tokenizer.cpp)
  target_include_directories(hunyuan_vl_preprocess_probe PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(hunyuan_vl_preprocess_probe PRIVATE
    nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
  add_executable(test_hunyuan_vl_media ../tests/cpp/test_media.cpp media.cpp)
  target_include_directories(test_hunyuan_vl_media PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  add_test(NAME test_hunyuan_vl_media COMMAND test_hunyuan_vl_media)
  add_executable(test_hunyuan_vl_tokenizer ../tests/cpp/test_tokenizer.cpp tokenizer.cpp)
  target_include_directories(test_hunyuan_vl_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_hunyuan_vl_tokenizer PRIVATE
    nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
  add_test(NAME test_hunyuan_vl_tokenizer COMMAND test_hunyuan_vl_tokenizer)
  add_executable(test_hunyuan_vl_request ../tests/cpp/test_request.cpp)
  target_include_directories(test_hunyuan_vl_request PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  add_test(NAME test_hunyuan_vl_request COMMAND test_hunyuan_vl_request)
  add_dependencies(trtmc_model_hunyuan_vl hunyuan_vl_preprocess_probe test_hunyuan_vl_tokenizer)
  # Wheel-backed E2E builds only the owning CTest targets, not the family DSO.
  foreach(_hunyuan_vl_test IN ITEMS test_hunyuan_vl_media test_hunyuan_vl_tokenizer
                                    test_hunyuan_vl_request)
    add_dependencies(${_hunyuan_vl_test} hunyuan_vl_inference_test hunyuan_vl_preprocess_probe)
  endforeach()
endif()
