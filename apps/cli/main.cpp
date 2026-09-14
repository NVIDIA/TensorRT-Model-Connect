/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "native/entrypoint.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "serve")
        return trtmc::server::run_server_frontend(argc - 2, argv + 2);
    if (argc >= 2 && std::string(argv[1]) == "_serve-worker")
        return trtmc::server::run_native_worker(argc - 2, argv + 2);
    return trtmc::cli::run(argc, argv, std::cout, std::cerr);
}
