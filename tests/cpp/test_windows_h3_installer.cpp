/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "installer/windows_h3_installer.h"

#define WIN32_LEAN_AND_MEAN
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <windows.h>

namespace {

namespace fs = std::filesystem;

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
void check_throws(Callable&& callable, const char* label) {
    try {
        callable();
        check(false, label);
    } catch (const std::runtime_error&) {
    }
}

void write_bytes(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("Unable to write installer test file");
}

std::string manifest_row(const fs::path& path, const std::string& relative) {
    return trtmc::installer::sha256_hex(trtmc::installer::sha256_file(path)) + "\t" +
           std::to_string(fs::file_size(path)) + "\t" + relative + "\n";
}

} // namespace

int main() {
    check(trtmc::installer::is_safe_payload_path("bin/trtmc.exe"), "normal path accepted");
    check(trtmc::installer::is_safe_payload_path("models/MiniMax-H3.bundle"),
          "model path accepted");
    for (const std::string& path : {"", "/rooted", "../escape", "bin/../escape", "bin\\trtmc.exe",
                                    "C:/absolute", "bin/con", "bin/file. ", "bin/file."}) {
        check(!trtmc::installer::is_safe_payload_path(path), "unsafe path rejected");
    }

    const auto root = fs::temp_directory_path() /
                      (L"trtmc-h3-installer-test-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                       std::to_wstring(GetTickCount64()));
    const auto payload = root / L"layout" / L"payload";
    const auto manifest = root / L"layout" / L"payload.manifest";
    const auto install = root / L"installed";
    try {
        const std::string marker_name = ".minimax-h3-install-id";
        const std::string marker_contents = "trtmc-minimax-h3-native-install-v1\n";
        write_bytes(payload / L"bin" / L"trtmc.exe", "native-cli");
        write_bytes(payload / fs::u8path(marker_name), marker_contents);
        write_bytes(manifest, manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe") +
                                  manifest_row(payload / fs::u8path(marker_name), marker_name));

        const auto entries = trtmc::installer::read_payload_manifest(manifest);
        check(entries.size() == 2, "manifest entry count");
        trtmc::installer::verify_payload(payload, entries);
        trtmc::installer::install_payload_transactional(payload, install, entries, marker_name,
                                                        marker_contents);
        check(fs::is_regular_file(install / L"bin" / L"trtmc.exe"), "transaction installs payload");
        check(trtmc::installer::installation_marker_matches(install, marker_name, marker_contents),
              "transaction preserves marker");

        write_bytes(payload / L"bin" / L"trtmc.exe", "native-cli-v2");
        write_bytes(manifest, manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe") +
                                  manifest_row(payload / fs::u8path(marker_name), marker_name));
        const auto updated_entries = trtmc::installer::read_payload_manifest(manifest);
        trtmc::installer::install_payload_transactional(payload, install, updated_entries,
                                                        marker_name, marker_contents);
        check(fs::file_size(install / L"bin" / L"trtmc.exe") == 13,
              "transaction replaces an owned install");

        const auto duplicate_manifest = root / L"duplicate.manifest";
        const auto row = manifest_row(payload / L"bin" / L"trtmc.exe", "bin/trtmc.exe");
        write_bytes(duplicate_manifest, row + row);
        check_throws([&] { (void)trtmc::installer::read_payload_manifest(duplicate_manifest); },
                     "duplicate manifest path rejected");

        const auto unrelated = root / L"unrelated";
        write_bytes(unrelated / L"keep.txt", "owner-data");
        check_throws(
            [&] {
                trtmc::installer::install_payload_transactional(payload, unrelated, updated_entries,
                                                                marker_name, marker_contents);
            },
            "unrelated destination rejected");
        check(fs::is_regular_file(unrelated / L"keep.txt"), "unrelated destination remains intact");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);

    if (failures != 0) {
        std::cerr << failures << " Windows H3 installer test(s) failed\n";
        return 1;
    }
    return 0;
}
