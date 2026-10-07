# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_model_clef SHARED plugin.cpp record.cpp bpe_tokenizer.cpp media.cpp)
target_include_directories(trtmc_model_clef PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_model_clef PRIVATE trtmc_core nlohmann_json::nlohmann_json ${TRTMC_CUDART_LIBRARY})
target_compile_options(trtmc_model_clef PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_model_clef PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_model_clef LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

add_library(trtmc_cli_clef SHARED cli.cpp media_io.cpp)
target_include_directories(trtmc_cli_clef PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(trtmc_cli_clef PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
target_compile_options(trtmc_cli_clef PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(trtmc_cli_clef PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN")
install(TARGETS trtmc_cli_clef LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TRTMC_BUILD_TESTS)
add_executable(clef_record_probe ../tests/record_probe.cpp record.cpp bpe_tokenizer.cpp)
target_include_directories(clef_record_probe PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(clef_record_probe PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(clef_record_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

add_executable(clef_engine_probe ../tests/engine_probe.cpp)
target_include_directories(clef_engine_probe PRIVATE ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(clef_engine_probe PRIVATE trtmc_backend_trt trtmc_core nlohmann_json::nlohmann_json)
set_target_properties(clef_engine_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")

add_executable(clef_task_probe ../tests/task_probe.cpp media_io.cpp)
target_include_directories(clef_task_probe PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(clef_task_probe PRIVATE trtmc_runtime trtmc_core nlohmann_json::nlohmann_json)
set_target_properties(clef_task_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" BUILD_RPATH "\$ORIGIN")

add_executable(clef_media_probe ../tests/media_probe.cpp media.cpp media_io.cpp record.cpp bpe_tokenizer.cpp)
target_include_directories(clef_media_probe PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/core/runtime/include)
target_link_libraries(clef_media_probe PRIVATE nlohmann_json::nlohmann_json)
set_target_properties(clef_media_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# CI builds the family target before running the family-owned Python tests.
# The task probe loads the model dynamically, so this dependency is acyclic.
add_dependencies(trtmc_model_clef clef_record_probe clef_engine_probe clef_task_probe clef_media_probe)
endif()
