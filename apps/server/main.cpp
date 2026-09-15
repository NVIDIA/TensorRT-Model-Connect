/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/native_worker.h"
#include "trtmc/runtime/family_loader.h"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace {

std::string value_after(int argc, char** argv, int& index, const std::string& option) {
    if (++index >= argc || argv[index][0] == '\0')
        throw std::invalid_argument(option + " requires a value");
    return argv[index];
}

std::uint64_t positive_integer(const std::string& value, const std::string& option) {
    if (value.empty() || value.front() == '-' || value.front() == '+')
        throw std::invalid_argument(option + " must be a positive integer");
    std::size_t used = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " must be a positive integer");
    }
    if (used != value.size() || parsed == 0)
        throw std::invalid_argument(option + " must be a positive integer");
    return parsed;
}

int worker_main(int argc, char** argv) {
    if (argc < 3)
        throw std::invalid_argument("_serve-worker requires a bundle path");
    const std::string bundle = argv[2];
    std::string runtime_root;
    std::string runtime_cache;
    std::uint64_t kv_cache_size = 0;
    bool cuda_graphs = false;
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--runtime-root")
            runtime_root = value_after(argc, argv, index, option);
        else if (option == "--runtime-cache")
            runtime_cache = value_after(argc, argv, index, option);
        else if (option == "--kv-cache-size")
            kv_cache_size = positive_integer(value_after(argc, argv, index, option), option);
        else if (option == "--cuda-graphs")
            cuda_graphs = true;
        else
            throw std::invalid_argument("unknown _serve-worker option: " + option);
    }
    auto task = trtmc::load_task(bundle, runtime_root, kv_cache_size, runtime_cache, cuda_graphs);
    return trtmc::server::run_text_worker(*task, std::cin, std::cout);
}

std::filesystem::path executable_path() {
    std::array<char, 4096> path{};
    const auto length = readlink("/proc/self/exe", path.data(), path.size() - 1U);
    if (length <= 0)
        throw std::runtime_error("cannot resolve the trtmc-server executable");
    path[static_cast<std::size_t>(length)] = '\0';
    return path.data();
}

std::filesystem::path source_python(const std::filesystem::path& executable) {
    const auto directory = executable.parent_path() / "server" / "python";
    std::error_code error;
    return std::filesystem::is_directory(directory / "trtmc_server", error)
               ? directory
               : std::filesystem::path{};
}

int frontend_main(int argc, char** argv) {
    const auto executable = executable_path();
    const auto source = source_python(executable);
    const bool source_build = !source.empty();
    if (source_build) {
        std::string pythonpath = source.string();
        if (const char* existing = std::getenv("PYTHONPATH"); existing && existing[0] != '\0')
            pythonpath += ":" + std::string(existing);
        if (setenv("PYTHONPATH", pythonpath.c_str(), 1) != 0)
            throw std::runtime_error("cannot configure the source-build Python module path");
    }

    std::string python = "python3";
    if (!source_build) {
        const auto installed_python = executable.parent_path() / "python3";
        std::error_code error;
        if (std::filesystem::exists(installed_python, error) && !error)
            python = installed_python.string();
    }
    std::vector<std::string> command{python, source_build ? "-P" : "-I", "-m", "trtmc_server"};
    for (int index = 1; index < argc; ++index)
        command.emplace_back(argv[index]);
    command.emplace_back("--worker-binary");
    command.push_back(executable.string());

    std::vector<char*> arguments;
    arguments.reserve(command.size() + 1U);
    for (auto& argument : command)
        arguments.push_back(argument.data());
    arguments.push_back(nullptr);
    execvp(arguments.front(), arguments.data());
    std::cerr << "trtmc-server: failed to start Python control plane: " << std::strerror(errno)
              << '\n';
    return 127;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "_serve-worker")
            return worker_main(argc, argv);
        return frontend_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "trtmc-server: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
