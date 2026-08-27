# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(trtmc_server_native STATIC
  native/entrypoint.cpp
  native/worker.cpp
)
target_link_libraries(trtmc_server_native PRIVATE trtmc_core)
target_include_directories(trtmc_server_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(trtmc_server_native SYSTEM PRIVATE
  ${_trtmc_nlohmann_json_include}
)
target_compile_options(trtmc_server_native PRIVATE -Wall -Wextra -Wpedantic)
