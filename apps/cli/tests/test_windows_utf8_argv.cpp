/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/windows_utf8_argv.h"

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_multilingual_utf16_converts_to_utf8() {
    const std::wstring multilingual =
        L"Arabic \u0627\u0644\u0639\u0631\u0628\u064a\u0629 | Chinese \u4e2d\u6587 | English | "
        L"French fran\u00e7ais | German Stra\u00dfe | Italian italiano | Japanese "
        L"\u65e5\u672c\u8a9e | "
        L"Korean \ud55c\uad6d\uc5b4 | Portuguese portugu\u00eas | Russian "
        L"\u0440\u0443\u0441\u0441\u043a\u0438\u0439 | Spanish espa\u00f1ol";
    const std::string expected =
        u8"Arabic \u0627\u0644\u0639\u0631\u0628\u064a\u0629 | Chinese \u4e2d\u6587 | English | "
        u8"French fran\u00e7ais | German Stra\u00dfe | Italian italiano | Japanese "
        u8"\u65e5\u672c\u8a9e | "
        u8"Korean \ud55c\uad6d\uc5b4 | Portuguese portugu\u00eas | Russian "
        u8"\u0440\u0443\u0441\u0441\u043a\u0438\u0439 | Spanish espa\u00f1ol";
    check(trtmc::cli::utf8_from_utf16(multilingual) == expected,
          "11-language UTF-16 prompt converts exactly to UTF-8");
}

void test_command_line_preserves_prompt_and_path() {
    wchar_t program[] = L"trtmc.exe";
    wchar_t prompt_flag[] = L"--prompt";
    wchar_t prompt[] = L"\u4e2d\u6587\u63d0\u793a";
    wchar_t output_flag[] = L"--output";
    wchar_t path[] = L"C:\\video\\\u6a21\u578b\\\u8f93\u5165 \u56fe\u7247.mp4";
    wchar_t* wide_argv[] = {program, prompt_flag, prompt, output_flag, path};

    trtmc::cli::Utf8CommandLine command_line(5, wide_argv);
    check(command_line.argc() == 5, "UTF-8 command line preserves argc");
    check(command_line.argv()[5] == nullptr, "UTF-8 command line is null terminated");
    check(std::string(command_line.argv()[2]) == u8"\u4e2d\u6587\u63d0\u793a",
          "UTF-8 command line preserves Chinese prompt");
    check(std::string(command_line.argv()[4]) ==
              u8"C:\\video\\\u6a21\u578b\\\u8f93\u5165 \u56fe\u7247.mp4",
          "UTF-8 command line preserves non-ASCII media path");
}

void test_invalid_utf16_is_rejected() {
    const wchar_t invalid[] = {static_cast<wchar_t>(0xD800), L'\0'};
    bool rejected = false;
    try {
        (void)trtmc::cli::utf8_from_utf16(std::wstring_view(invalid, 1));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    check(rejected, "malformed UTF-16 command-line text is rejected");
}

void test_utf8_process_manifest_and_narrow_filesystem_path() {
    HRSRC resource = FindResourceW(nullptr, MAKEINTRESOURCEW(1), MAKEINTRESOURCEW(24));
    check(resource != nullptr, "test executable contains an embedded process manifest");
    if (resource != nullptr) {
        HGLOBAL loaded = LoadResource(nullptr, resource);
        const DWORD size = SizeofResource(nullptr, resource);
        const void* data = loaded == nullptr ? nullptr : LockResource(loaded);
        const std::string manifest =
            data == nullptr ? std::string{} : std::string(static_cast<const char*>(data), size);
        check(manifest.find("activeCodePage") != std::string::npos &&
                  manifest.find(">UTF-8</activeCodePage>") != std::string::npos,
              "PE manifest declares UTF-8 as the active process code page");
    }
    check(GetACP() == CP_UTF8, "embedded manifest activates the UTF-8 process code page");

    const auto wide_directory =
        std::filesystem::temp_directory_path() /
        (std::wstring(L"trtmc_utf8_path_\u6a21\u578b_") + std::to_wstring(GetCurrentProcessId()));
    const auto wide_file = wide_directory / L"\u8f93\u5165 \u56fe\u7247.txt";
    std::error_code error;
    (void)std::filesystem::remove(wide_file, error);
    error.clear();
    (void)std::filesystem::remove(wide_directory, error);
    error.clear();
    check(std::filesystem::create_directory(wide_directory, error) && !error,
          "wide setup creates a non-ASCII test directory");

    const std::string utf8_directory = trtmc::cli::utf8_from_utf16(wide_directory.native());
    const std::filesystem::path narrow_file =
        std::filesystem::path(utf8_directory) / u8"\u8f93\u5165 \u56fe\u7247.txt";
    {
        std::ofstream output(narrow_file, std::ios::binary | std::ios::trunc);
        output << "utf8-path";
        check(static_cast<bool>(output),
              "UTF-8 narrow std::filesystem path opens a non-ASCII output file");
    }
    check(std::filesystem::exists(wide_file),
          "UTF-8 narrow path resolves to the intended UTF-16 filesystem path");
    {
        std::ifstream input(narrow_file, std::ios::binary);
        std::string value;
        input >> value;
        check(value == "utf8-path", "UTF-8 narrow path reads the non-ASCII file back");
    }

    error.clear();
    check(std::filesystem::remove(wide_file, error) && !error,
          "non-ASCII path test removes its file");
    error.clear();
    check(std::filesystem::remove(wide_directory, error) && !error,
          "non-ASCII path test removes its directory");
}

} // namespace

int main() {
    test_multilingual_utf16_converts_to_utf8();
    test_command_line_preserves_prompt_and_path();
    test_invalid_utf16_is_rejected();
    test_utf8_process_manifest_and_narrow_filesystem_path();
    if (failures == 0)
        std::cout << "All Windows UTF-8 argv tests passed\n";
    return failures == 0 ? 0 : 1;
}
