# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_server_native STATIC
  native/entrypoint.cpp
  native/worker.cpp
)
target_include_directories(trtmc_server_native
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(trtmc_server_native
  PRIVATE
    trtmc_runtime
    nlohmann_json::nlohmann_json
)
target_compile_options(trtmc_server_native PRIVATE -Wall -Wextra -Wpedantic)

# Keep source-build CLI smoke tests independent of the caller's working directory
# without embedding checkout paths in the native executable.
add_custom_target(trtmc_server_python ALL
  COMMAND ${CMAKE_COMMAND} -E remove_directory
    "${PROJECT_BINARY_DIR}/server/python/trtmc_server"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/python/trtmc_server"
    "${PROJECT_BINARY_DIR}/server/python/trtmc_server"
  COMMENT "Copying the local server Python module"
)

if(TRTMC_BUILD_TESTS)
  add_executable(test_serve_worker tests/test_serve_worker.cpp)
  target_include_directories(test_serve_worker PRIVATE
    ${PROJECT_SOURCE_DIR}/core/runtime/include
  )
  target_link_libraries(test_serve_worker PRIVATE
    trtmc_server_native
    nlohmann_json::nlohmann_json
  )
  target_compile_options(test_serve_worker PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME serve_worker COMMAND test_serve_worker)
  set_tests_properties(serve_worker PROPERTIES LABELS cpu)

  set(_trtmc_serve_shadow_dir "${PROJECT_BINARY_DIR}/server/tests/cwd-shadow")
  file(MAKE_DIRECTORY "${_trtmc_serve_shadow_dir}/trtmc_server")
  file(WRITE "${_trtmc_serve_shadow_dir}/trtmc_server/__init__.py" "")
  file(WRITE "${_trtmc_serve_shadow_dir}/trtmc_server/__main__.py"
    "print('TRTMC_CWD_SHADOW_EXECUTED')\n"
  )
  add_test(
    NAME serve_cli_ignores_cwd_shadow
    COMMAND $<TARGET_FILE:trtmc> serve --help
  )
  set_tests_properties(serve_cli_ignores_cwd_shadow PROPERTIES
    WORKING_DIRECTORY "${_trtmc_serve_shadow_dir}"
    PASS_REGULAR_EXPRESSION "Serve TensorRT-Model-Connect bundles"
    FAIL_REGULAR_EXPRESSION "TRTMC_CWD_SHADOW_EXECUTED"
    LABELS cpu
  )
endif()
