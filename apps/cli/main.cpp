/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#if defined(_WIN32)
#include "cli/windows_utf8_argv.h"
#endif

#include <cstdlib>
#include <exception>
#include <iostream>

namespace {

int run_cli(int argc, char** argv) {
    // The executable owns the console: keep result output machine-readable even
    // when loaded libraries write C++ diagnostics to std::cout. Do not change
    // library logger levels or the output behavior of embedded runtime APIs.
    std::ostream result(std::cout.rdbuf());
    std::cout.rdbuf(std::cerr.rdbuf());
    const int status = trtmc::cli::run(argc, argv, result, std::cerr);
    result.flush();
    return status;
}

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    try {
        trtmc::cli::Utf8CommandLine command_line(argc, argv);
        return run_cli(command_line.argc(), command_line.argv());
    } catch (const std::exception& error) {
        std::cerr << "Error: unable to decode the Windows command line as UTF-8: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
#else
int main(int argc, char** argv) {
    return run_cli(argc, argv);
}
#endif
