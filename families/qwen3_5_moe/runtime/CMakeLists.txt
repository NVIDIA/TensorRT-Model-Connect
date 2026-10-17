# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_qwen3_5_moe SHARED plugin.cpp)
target_include_directories(trtmc_model_qwen3_5_moe PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(trtmc_model_qwen3_5_moe PRIVATE trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_model_qwen3_5_moe PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
set_target_properties(trtmc_model_qwen3_5_moe PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_qwen3_5_moe LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

include(edge_llm/Adapter.cmake)

if(TRTMC_BUILD_TESTS)
  foreach(kind contract factory)
    add_executable(test_qwen3_5_moe_edge_${kind} ${PROJECT_SOURCE_DIR}/families/qwen3_5_moe/tests/cpp/test_qwen3_5_moe_edge_${kind}.cpp)
    target_include_directories(test_qwen3_5_moe_edge_${kind} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include ${TRTMC_CUDA_INCLUDE_DIR})
    target_link_libraries(test_qwen3_5_moe_edge_${kind} PRIVATE nlohmann_json::nlohmann_json)
    add_test(NAME test_qwen3_5_moe_edge_${kind} COMMAND test_qwen3_5_moe_edge_${kind})
  endforeach()
  target_link_libraries(test_qwen3_5_moe_edge_factory PRIVATE trtmc_model_qwen3_5_moe trtmc_core)
  if(TARGET EdgeLLM::Core)
    target_compile_definitions(test_qwen3_5_moe_edge_factory PRIVATE TRTMC_HAS_EDGE_LLM=1)
    add_executable(test_qwen3_5_moe_edge_request ${PROJECT_SOURCE_DIR}/families/qwen3_5_moe/tests/cpp/test_qwen3_5_moe_edge_request.cpp)
    target_sources(test_qwen3_5_moe_edge_request PRIVATE edge_llm/device_link.cu)
    set_target_properties(test_qwen3_5_moe_edge_request PROPERTIES
      CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_include_directories(test_qwen3_5_moe_edge_request PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/template")
    configure_file("${PROJECT_SOURCE_DIR}/families/qwen3_5_moe/tests/data/chat_template.txt"
      "${CMAKE_CURRENT_BINARY_DIR}/template/chat_template.jinja" COPYONLY)
    target_compile_definitions(test_qwen3_5_moe_edge_request PRIVATE QWEN35_MOE_EDGE_TEMPLATE="${CMAKE_CURRENT_BINARY_DIR}/template")
    # The standalone template test has a Core/ChatTemplate static-library cycle.
    target_link_libraries(test_qwen3_5_moe_edge_request PRIVATE trtmc_core
      "$<LINK_GROUP:RESCAN,EdgeLLM::Core,${EdgeLLM_PREFIX}/lib/libedgellmChatTemplate.a>")
    add_test(NAME test_qwen3_5_moe_edge_request COMMAND test_qwen3_5_moe_edge_request)
  endif()
endif()

# Explicit model-GPU proof executables, never unprovisioned default CTest cases.
if(TRTMC_BUILD_TESTS)
  add_executable(qwen3_5_moe_edge_inference ${PROJECT_SOURCE_DIR}/families/qwen3_5_moe/tests/cpp/test_qwen3_5_moe_edge_inference.cpp)
  target_link_libraries(qwen3_5_moe_edge_inference PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
endif()
if(TRTMC_BUILD_TESTS AND TARGET EdgeLLM::Core)
  add_executable(qwen3_5_moe_edge_reference ${PROJECT_SOURCE_DIR}/families/qwen3_5_moe/tests/cpp/test_qwen3_5_moe_edge_reference.cpp edge_llm/device_link.cu)
  target_link_libraries(qwen3_5_moe_edge_reference PRIVATE EdgeLLM::Core)
  set_target_properties(qwen3_5_moe_edge_reference PROPERTIES
    CUDA_ARCHITECTURES "${EdgeLLM_CUDA_ARCHITECTURE}" CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
