/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_engine_builder.h"

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using trtmc::sam2::native::Sam2EngineBuildOptions;

[[noreturn]] void usageError(std::string_view message) {
    throw std::invalid_argument(std::string(message) +
                                "\nusage: sam2_native_builder --checkpoint PATH --config PATH "
                                "--output PATH "
                                "[--workspace-bytes N] [--gpu-device N] "
                                "[--created-at YYYY-MM-DDTHH:MM:SSZ]");
}

std::uint64_t parseUnsigned(std::string_view value, const char* option) {
    if (value.empty() || (value.size() > 1U && value.front() == '0'))
        usageError(std::string(option) + " requires a canonical unsigned decimal value");
    std::uint64_t parsed = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
        usageError(std::string(option) + " requires a canonical unsigned decimal value");
    return parsed;
}

std::string currentUtcTimestamp() {
    const std::time_t now = std::time(nullptr);
    if (now == static_cast<std::time_t>(-1))
        throw std::runtime_error("failed to read the build time");
    std::tm utc{};
    if (::gmtime_r(&now, &utc) == nullptr)
        throw std::runtime_error("failed to convert the build time to UTC");
    char value[21]{};
    const int written =
        std::snprintf(value, sizeof(value), "%04d-%02d-%02dT%02d:%02d:%02dZ", utc.tm_year + 1900,
                      utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec);
    if (written != 20)
        throw std::runtime_error("failed to format the build time");
    return value;
}

std::string_view requireValue(int argc, char** argv, int& index, const char* option) {
    if (index + 1 >= argc)
        usageError(std::string(option) + " requires a value");
    ++index;
    const std::string_view value(argv[index]);
    if (value.empty())
        usageError(std::string(option) + " requires a nonempty value");
    return value;
}

Sam2EngineBuildOptions parseArguments(int argc, char** argv) {
    Sam2EngineBuildOptions options;
    bool checkpoint_seen = false;
    bool config_seen = false;
    bool output_seen = false;
    bool workspace_seen = false;
    bool device_seen = false;
    bool created_at_seen = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--help") {
            std::cout << "usage: sam2_native_builder --checkpoint PATH --config PATH "
                         "--output PATH "
                         "[--workspace-bytes N] [--gpu-device N] "
                         "[--created-at YYYY-MM-DDTHH:MM:SSZ]\n";
            std::exit(0);
        }
        if (option == "--checkpoint") {
            if (checkpoint_seen)
                usageError("--checkpoint may be specified only once");
            options.checkpoint_path = requireValue(argc, argv, index, "--checkpoint");
            checkpoint_seen = true;
        } else if (option == "--config") {
            if (config_seen)
                usageError("--config may be specified only once");
            options.source_config_path = requireValue(argc, argv, index, "--config");
            config_seen = true;
        } else if (option == "--output") {
            if (output_seen)
                usageError("--output may be specified only once");
            options.output_path = requireValue(argc, argv, index, "--output");
            output_seen = true;
        } else if (option == "--workspace-bytes") {
            if (workspace_seen)
                usageError("--workspace-bytes may be specified only once");
            options.workspace_bytes = parseUnsigned(
                requireValue(argc, argv, index, "--workspace-bytes"), "--workspace-bytes");
            workspace_seen = true;
        } else if (option == "--gpu-device") {
            if (device_seen)
                usageError("--gpu-device may be specified only once");
            const std::uint64_t device =
                parseUnsigned(requireValue(argc, argv, index, "--gpu-device"), "--gpu-device");
            if (device > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
                usageError("--gpu-device is outside the int32 range");
            options.gpu_device = static_cast<std::int32_t>(device);
            device_seen = true;
        } else if (option == "--created-at") {
            if (created_at_seen)
                usageError("--created-at may be specified only once");
            options.created_at_utc = requireValue(argc, argv, index, "--created-at");
            created_at_seen = true;
        } else {
            usageError(std::string("unsupported option: ") + std::string(option));
        }
    }
    if (!checkpoint_seen || !config_seen || !output_seen)
        usageError("--checkpoint, --config, and --output are required");
    if (!created_at_seen)
        options.created_at_utc = currentUtcTimestamp();
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Sam2EngineBuildOptions options = parseArguments(argc, argv);
        trtmc::sam2::native::buildSam2NativeBundle(options);
        std::cout << "Wrote unqualified SAM2 native bundle: " << options.output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sam2_native_builder: " << error.what() << '\n';
        return 1;
    }
}
