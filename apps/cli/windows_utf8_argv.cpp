/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/windows_utf8_argv.h"

#if !defined(_WIN32)
#error "windows_utf8_argv.cpp is Windows-only"
#endif

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <limits>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace trtmc::cli {

namespace {

int checked_utf16_size(std::wstring_view value) {
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("Windows command-line argument is too large to convert to UTF-8");
    return static_cast<int>(value.size());
}

[[noreturn]] void throw_conversion_error(DWORD error) {
    throw std::runtime_error("WideCharToMultiByte(CP_UTF8) rejected command-line UTF-16 (Windows "
                             "error " +
                             std::to_string(error) + ")");
}

} // namespace

std::string utf8_from_utf16(std::wstring_view value) {
    if (value.empty())
        return {};

    const int input_size = checked_utf16_size(value);
    const int output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
                                                input_size, nullptr, 0, nullptr, nullptr);
    if (output_size <= 0)
        throw_conversion_error(GetLastError());

    std::string result(static_cast<std::size_t>(output_size), '\0');
    const int converted =
        WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value.data(), input_size, result.data(),
                            output_size, nullptr, nullptr);
    if (converted != output_size)
        throw_conversion_error(GetLastError());
    return result;
}

Utf8CommandLine::Utf8CommandLine(int argc, wchar_t* const* argv) : argc_(argc) {
    if (argc < 0 || (argc > 0 && argv == nullptr))
        throw std::invalid_argument("Windows command line has invalid argc/argv metadata");

    storage_.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        if (argv[index] == nullptr)
            throw std::invalid_argument("Windows command line contains a null argument");
        storage_.push_back(utf8_from_utf16(argv[index]));
    }

    pointers_.reserve(static_cast<std::size_t>(argc) + 1U);
    for (auto& argument : storage_)
        pointers_.push_back(argument.data());
    pointers_.push_back(nullptr);
}

} // namespace trtmc::cli
