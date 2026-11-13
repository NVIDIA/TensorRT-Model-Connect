/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"

#include <iostream>

int main(int argc, char** argv) {
    // The executable owns the console: keep result output machine-readable even
    // when loaded libraries write C++ diagnostics to std::cout. Do not change
    // library logger levels or the output behavior of embedded runtime APIs.
    std::ostream result(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());
    const int status = trtmc::cli::run(argc, argv, result, std::cerr);
    result.flush();
    return status;
}
