# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(ICU REQUIRED COMPONENTS uc i18n)
add_library(trtmc_model_smolvlm SHARED plugin.cpp tokenizer.cpp media.cpp prompt.cpp sampler.cpp)
target_include_directories(trtmc_model_smolvlm PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_smolvlm SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_smolvlm PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ICU::uc ICU::i18n ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_smolvlm PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_smolvlm PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_smolvlm LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  set(_smolvlm_tests "${PROJECT_SOURCE_DIR}/families/smolvlm/tests")
  # Public-SDK driver used by the family E2E; not an installed application.
  add_executable(smolvlm_inference_test ${_smolvlm_tests}/cpp/run_model.cpp)
  target_include_directories(smolvlm_inference_test PRIVATE ${PROJECT_SOURCE_DIR}/core/api/include)
  target_link_libraries(smolvlm_inference_test PRIVATE trtmc_c nlohmann_json::nlohmann_json)
  # Native image processing and tokenization, compared with the reference by the E2E.
  add_executable(smolvlm_runtime_probe ${_smolvlm_tests}/cpp/runtime_probe.cpp
    media.cpp prompt.cpp tokenizer.cpp)
  target_include_directories(smolvlm_runtime_probe PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(smolvlm_runtime_probe PRIVATE nlohmann_json::nlohmann_json ICU::uc ICU::i18n)

  add_executable(test_smolvlm_media ${_smolvlm_tests}/cpp/test_media.cpp media.cpp prompt.cpp)
  target_include_directories(test_smolvlm_media PRIVATE
    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_link_libraries(test_smolvlm_media PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(test_smolvlm_media PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_smolvlm_media
    COMMAND test_smolvlm_media ${_smolvlm_tests}/data/media_cases.json
            ${_smolvlm_tests}/data/prompt_cases.json)

  add_executable(test_smolvlm_tokenizer ${_smolvlm_tests}/cpp/test_tokenizer.cpp tokenizer.cpp sampler.cpp)
  target_include_directories(test_smolvlm_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_smolvlm_tokenizer PRIVATE nlohmann_json::nlohmann_json ICU::uc ICU::i18n)
  target_compile_options(test_smolvlm_tokenizer PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_smolvlm_tokenizer
    COMMAND test_smolvlm_tokenizer ${_smolvlm_tests}/data/tokenizer_cases.json)

  # Building the family target also builds its E2E helpers. Wheel-backed E2E
  # builds only the owning CTest targets, so those pull the helpers in too.
  add_dependencies(trtmc_model_smolvlm smolvlm_inference_test smolvlm_runtime_probe)
  add_dependencies(test_smolvlm_media smolvlm_inference_test smolvlm_runtime_probe)
  add_dependencies(test_smolvlm_tokenizer smolvlm_inference_test smolvlm_runtime_probe)
endif()
