# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_phonon SHARED pipeline.cpp plugin.cpp tokenizer.cpp)
target_include_directories(trtmc_model_phonon PRIVATE
  ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_phonon SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_phonon PRIVATE
  trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_phonon PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_phonon PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_phonon LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  foreach(language IN ITEMS c cpp)
    add_executable(test_phonon_sdk_${language} ../tests/sdk_consumer.${language})
    target_link_libraries(test_phonon_sdk_${language} PRIVATE trtmc_c)
    target_compile_options(test_phonon_sdk_${language} PRIVATE -Wall -Wextra -Wpedantic -Werror)
    set_target_properties(test_phonon_sdk_${language} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  endforeach()
  add_executable(test_phonon_tokenizer ../tests/cpp/test_tokenizer.cpp tokenizer.cpp)
  target_include_directories(test_phonon_tokenizer PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_phonon_tokenizer PRIVATE nlohmann_json::nlohmann_json)
  set_target_properties(test_phonon_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME test_phonon_tokenizer COMMAND test_phonon_tokenizer)
endif()
