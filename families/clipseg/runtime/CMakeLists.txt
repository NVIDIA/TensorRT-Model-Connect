# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
find_package(ICU COMPONENTS uc QUIET)
if(NOT ICU_FOUND)
  if(DEFINED TRTMC_BUILD_FAMILIES AND NOT "*" IN_LIST TRTMC_BUILD_FAMILIES)
    message(FATAL_ERROR "Selected CLIPSeg runtime requires ICU uc development packages")
  endif()
  message(STATUS "Skipping clipseg: install ICU development packages for Unicode tokenization")
  add_custom_target(trtmc_model_clipseg
    COMMAND ${CMAKE_COMMAND} -E echo "CLIPSeg runtime requires ICU uc development packages"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM)
  return()
endif()
add_library(trtmc_model_clipseg SHARED bpe_tokenizer.cpp image_preprocess_seam.cpp plugin.cpp)
target_include_directories(trtmc_model_clipseg PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_clipseg SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_clipseg PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY} ICU::uc)
target_compile_options(trtmc_model_clipseg PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_clipseg PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_clipseg LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
  add_executable(test_clipseg_preprocess ${PROJECT_SOURCE_DIR}/families/clipseg/tests/cpp/test_preprocess.cpp)
  target_include_directories(test_clipseg_preprocess PRIVATE ${PROJECT_SOURCE_DIR})
  target_link_libraries(test_clipseg_preprocess PRIVATE trtmc_model_clipseg)
  add_test(NAME clipseg_preprocess COMMAND test_clipseg_preprocess)
  add_executable(test_clipseg_concurrency ${PROJECT_SOURCE_DIR}/families/clipseg/tests/cpp/test_concurrency.cpp)
  target_include_directories(test_clipseg_concurrency PRIVATE ${PROJECT_SOURCE_DIR})
  target_include_directories(test_clipseg_concurrency SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_clipseg_concurrency PRIVATE trtmc_model_clipseg trtmc_core nlohmann_json::nlohmann_json Threads::Threads)
  target_link_options(test_clipseg_concurrency PRIVATE "-Wl,--export-dynamic-symbol=_Znwm")
  add_test(NAME clipseg_concurrency COMMAND test_clipseg_concurrency)
endif()
