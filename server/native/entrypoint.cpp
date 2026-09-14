/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native/entrypoint.h"

#include "native/worker.h"
#include "trtmc/runtime/family_loader.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace trtmc::server {
namespace {

struct ByteSizeParts {
    std::string number;
    std::uint64_t multiplier{1};
};

ByteSizeParts split_byte_size(const std::string& text) {
    if (text.size() > 3 && text.compare(text.size() - 3, 3, "GiB") == 0) {
        return {text.substr(0, text.size() - 3), 1024ULL * 1024ULL * 1024ULL};
    }
    if (text.size() > 2 && text.compare(text.size() - 2, 2, "GB") == 0)
        return {text.substr(0, text.size() - 2), 1000ULL * 1000ULL * 1000ULL};
    return {text, 1};
}

std::uint64_t parse_byte_size(const std::string& text) {
    const ByteSizeParts parts = split_byte_size(text);
    if (parts.number.empty())
        throw std::invalid_argument(
            "--kv-cache-size must be integer bytes or a value like 1GB or 1GiB");
    if (!std::all_of(parts.number.begin(), parts.number.end(),
                     [](unsigned char value) { return value >= '0' && value <= '9'; })) {
        throw std::invalid_argument(
            "--kv-cache-size must be integer bytes or a value like 1GB or 1GiB");
    }
    std::size_t consumed = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(parts.number, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("--kv-cache-size is outside its valid range");
    }
    if (value == 0 || consumed != parts.number.size())
        throw std::invalid_argument("--kv-cache-size is outside its valid range");
    if (value > std::numeric_limits<std::uint64_t>::max() / parts.multiplier)
        throw std::invalid_argument("--kv-cache-size is outside its valid range");
    return value * parts.multiplier;
}

std::string take_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc)
        throw std::invalid_argument(option + " requires a value");
    ++index;
    const std::string value = argv[index];
    if (value.empty())
        throw std::invalid_argument(option + " requires a non-empty value");
    return value;
}

NativeWorkerOptions parse_options(int argc, char** argv) {
    if (argc < 1 || argv[0] == nullptr || argv[0][0] == '\0')
        throw std::invalid_argument("_serve-worker requires a .bundle artifact file");

    NativeWorkerOptions options;
    options.bundle_path = argv[0];
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--runtime-root") {
            options.runtime_root = take_value(argc, argv, index, option);
        } else if (option == "--kv-cache-size") {
            options.kv_cache_size_bytes = parse_byte_size(take_value(argc, argv, index, option));
        } else if (option == "--runtime-cache") {
            options.runtime_cache_path = take_value(argc, argv, index, option);
        } else if (option == "--cuda-graphs") {
            options.cuda_graphs = true;
        } else {
            throw std::invalid_argument("unknown _serve-worker option: " + option);
        }
    }
    if (options.runtime_root.empty())
        throw std::invalid_argument("_serve-worker requires --runtime-root DIR");
    return options;
}

std::filesystem::path current_executable_path() {
    std::array<char, 4096> buffer{};
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1U);
    if (length <= 0)
        return {};
    buffer[static_cast<std::size_t>(length)] = '\0';
    return std::filesystem::path(buffer.data());
}

std::string python_executable(const std::filesystem::path& executable) {
    if (!executable.empty()) {
        for (const char* name : {"python3", "python"}) {
            const auto candidate = executable.parent_path() / name;
            if (access(candidate.c_str(), X_OK) == 0)
                return candidate.string();
        }
    }
    return "python3";
}

std::filesystem::path source_server_python(const std::filesystem::path& executable) {
    if (executable.empty())
        return {};
    const auto server_python = executable.parent_path() / "server" / "python";
    std::error_code directory_error;
    if (!std::filesystem::is_directory(server_python / "trtmc_server", directory_error))
        return {};
    return server_python;
}

void configure_source_pythonpath(const std::filesystem::path& server_python) {
    std::string pythonpath = server_python.string();
    if (const char* existing = std::getenv("PYTHONPATH");
        existing != nullptr && existing[0] != '\0')
        pythonpath += ":" + std::string(existing);
    if (setenv("PYTHONPATH", pythonpath.c_str(), 1) != 0)
        throw std::runtime_error("failed to configure the source-build Python module path");
}

} // namespace

int run_server_frontend(int argc, char** argv) {
    const auto executable = current_executable_path();
    if (executable.empty()) {
        std::cerr << "Error: cannot resolve the current trtmc binary for serve\n";
        return EXIT_FAILURE;
    }

    const auto source_python = source_server_python(executable);
    const bool source_mode = !source_python.empty();
    std::vector<std::string> command{python_executable(executable), source_mode ? "-P" : "-I", "-m",
                                     "trtmc_server"};
    bool binary_provided = false;
    for (int index = 0; index < argc; ++index) {
        std::string argument = argv[index];
        if (argument == "--trtmc-binary" || argument.rfind("--trtmc-binary=", 0) == 0)
            binary_provided = true;
        command.push_back(std::move(argument));
    }
    if (!binary_provided) {
        command.emplace_back("--trtmc-binary");
        command.push_back(executable.string());
    }

    std::vector<char*> exec_arguments;
    exec_arguments.reserve(command.size() + 1U);
    for (auto& argument : command)
        exec_arguments.push_back(argument.data());
    exec_arguments.push_back(nullptr);
    try {
        if (source_mode)
            configure_source_pythonpath(source_python);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    execvp(exec_arguments[0], exec_arguments.data());
    std::cerr << "Error: failed to execute Python serving module: " << std::strerror(errno) << '\n';
    return 127;
}

int run_native_worker(const NativeWorkerOptions& options) {
    if (options.bundle_path.empty()) {
        std::cerr << "Error: _serve-worker requires a .bundle artifact file\n";
        return EXIT_FAILURE;
    }
    if (options.runtime_root.empty()) {
        std::cerr << "Error: _serve-worker requires --runtime-root DIR\n";
        return EXIT_FAILURE;
    }

    try {
        auto task =
            load_task(options.bundle_path, options.runtime_root, options.kv_cache_size_bytes,
                      options.runtime_cache_path, options.cuda_graphs);
        if (!task)
            throw std::runtime_error("native worker task is unavailable");
        return serve::run_worker_protocol(*task, std::cin, std::cout);
    } catch (const std::exception& error) {
        // Stderr is the private diagnostic channel; JSONL stdout stays redacted.
        std::cerr << "Error: native worker failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

int run_native_worker(int argc, char** argv) {
    try {
        return run_native_worker(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace trtmc::server
