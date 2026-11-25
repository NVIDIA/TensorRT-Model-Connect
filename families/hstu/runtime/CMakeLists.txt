# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_hstu SHARED validation.cpp staged_pipeline.cpp history_snapshot.cpp plugin.cpp)
target_include_directories(trtmc_model_hstu PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(trtmc_model_hstu SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(trtmc_model_hstu PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_hstu PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_hstu PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_hstu LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(TRTMC_BUILD_TESTS)
  add_executable(test_hstu_history_snapshot ../tests/cpp/test_history_snapshot.cpp)
  target_include_directories(test_hstu_history_snapshot PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_hstu_history_snapshot SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_hstu_history_snapshot PRIVATE trtmc_model_hstu nlohmann_json::nlohmann_json)
  add_test(NAME hstu_history_snapshot COMMAND test_hstu_history_snapshot)
endif()

add_library(hstu_triton SHARED ../triton/bridge.cpp)
target_include_directories(hstu_triton PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_include_directories(hstu_triton SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
target_link_libraries(hstu_triton PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
set_target_properties(hstu_triton PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")

if(TRTMC_BUILD_TESTS)
  add_executable(test_hstu_inputs ../tests/cpp/test_inputs.cpp)
  target_include_directories(test_hstu_inputs PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
  target_include_directories(test_hstu_inputs SYSTEM PRIVATE ${TRTMC_CUDA_INCLUDE_DIR})
  target_link_libraries(test_hstu_inputs PRIVATE trtmc_model_hstu nlohmann_json::nlohmann_json)
  add_test(NAME hstu_inputs COMMAND test_hstu_inputs)
endif()

install(TARGETS hstu_triton LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE ${CMAKE_CURRENT_SOURCE_DIR}/../UPSTREAM_LICENSE.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/trtmc/families/hstu)

# The family proof consumes this transport while CI isolates the Task DSOs.
if(TRTMC_BUILD_TESTS)
  add_dependencies(trtmc_model_hstu hstu_triton)
endif()
