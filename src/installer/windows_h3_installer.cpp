/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "installer/windows_h3_installer.h"

#ifndef _WIN32
#error "The native MiniMax-H3 installer support is Windows-only"
#endif

#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>
#include <bcrypt.h>
// clang-format on

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <type_traits>

namespace trtmc::installer {
namespace {

constexpr std::size_t kSha256Bytes = 32;
constexpr std::size_t kSha256HexChars = kSha256Bytes * 2;
constexpr std::size_t kCopyBufferBytes = 8U << 20;

std::runtime_error windows_error(const std::string& operation, DWORD code = GetLastError()) {
    return std::runtime_error(operation + " failed with Windows error " + std::to_string(code));
}

struct AlgorithmCloser {
    void operator()(BCRYPT_ALG_HANDLE handle) const noexcept {
        if (handle != nullptr)
            BCryptCloseAlgorithmProvider(handle, 0);
    }
};

struct HashCloser {
    void operator()(BCRYPT_HASH_HANDLE handle) const noexcept {
        if (handle != nullptr)
            BCryptDestroyHash(handle);
    }
};

using AlgorithmHandle = std::unique_ptr<std::remove_pointer_t<BCRYPT_ALG_HANDLE>, AlgorithmCloser>;
using HashHandle = std::unique_ptr<std::remove_pointer_t<BCRYPT_HASH_HANDLE>, HashCloser>;

bool nt_success(NTSTATUS status) {
    return status >= 0;
}

bool ascii_hex(char value) {
    return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

std::uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9')
        return static_cast<std::uint8_t>(value - '0');
    return static_cast<std::uint8_t>(10 + value - 'a');
}

std::array<std::uint8_t, kSha256Bytes> parse_sha256(const std::string& text) {
    if (text.size() != kSha256HexChars || !std::all_of(text.begin(), text.end(), ascii_hex)) {
        throw std::runtime_error("Payload manifest has a non-canonical SHA-256 digest");
    }
    std::array<std::uint8_t, kSha256Bytes> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>((hex_nibble(text[index * 2]) << 4U) |
                                                  hex_nibble(text[index * 2 + 1]));
    }
    return result;
}

std::uintmax_t parse_size(const std::string& text) {
    if (text.empty())
        throw std::runtime_error("Payload manifest has an empty size");
    std::uintmax_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error("Payload manifest has an invalid decimal size");
    return value;
}

std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool reserved_windows_component(const std::string& component) {
    const auto dot = component.find('.');
    const auto stem = lowercase_ascii(component.substr(0, dot));
    if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul")
        return true;
    if (stem.size() == 4 && (stem.rfind("com", 0) == 0 || stem.rfind("lpt", 0) == 0) &&
        stem[3] >= '1' && stem[3] <= '9') {
        return true;
    }
    return false;
}

std::string read_small_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.eof() && input.fail())
        throw std::runtime_error("Unable to read installation marker");
    return contents.str();
}

void remove_transaction_directory(const std::filesystem::path& path,
                                  const std::filesystem::path& expected_parent) {
    if (path.empty() || path.parent_path() != expected_parent || path == expected_parent)
        throw std::runtime_error("Refusing to remove an unsafe installer transaction path");
    std::error_code error;
    std::filesystem::remove_all(path, error);
    if (error)
        throw std::runtime_error("Unable to clean installer transaction directory: " +
                                 error.message());
}

void materialize_file(const std::filesystem::path& source, const std::filesystem::path& destination,
                      std::uintmax_t size) {
    std::filesystem::create_directories(destination.parent_path());
    // Large model bundles should not consume a second copy when the downloaded
    // layout and installation share an NTFS volume. Removing the layout later
    // simply drops one hard-link; the installed model remains intact.
    if (size >= (1ULL << 30) &&
        CreateHardLinkW(destination.c_str(), source.c_str(), nullptr) != FALSE) {
        return;
    }
    if (!CopyFileW(source.c_str(), destination.c_str(), TRUE))
        throw windows_error("CopyFileW");
}

} // namespace

bool is_safe_payload_path(const std::string& utf8_path) {
    if (utf8_path.empty() || utf8_path.front() == '/' || utf8_path.back() == '/' ||
        utf8_path.find('\\') != std::string::npos || utf8_path.find(':') != std::string::npos ||
        utf8_path.find('\0') != std::string::npos) {
        return false;
    }
    std::size_t begin = 0;
    while (begin < utf8_path.size()) {
        const auto end = utf8_path.find('/', begin);
        const auto component =
            utf8_path.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (component.empty() || component == "." || component == ".." || component.back() == '.' ||
            component.back() == ' ' || reserved_windows_component(component)) {
            return false;
        }
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    const auto path = std::filesystem::u8path(utf8_path);
    return !path.empty() && !path.is_absolute() && !path.has_root_name() &&
           !path.has_root_directory() && path.lexically_normal() == path;
}

std::vector<PayloadEntry> read_payload_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Unable to open payload.manifest");
    std::vector<PayloadEntry> result;
    std::set<std::string> paths;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        const auto first_tab = line.find('\t');
        const auto second_tab =
            first_tab == std::string::npos ? std::string::npos : line.find('\t', first_tab + 1);
        if (first_tab != kSha256HexChars || second_tab == std::string::npos ||
            line.find('\t', second_tab + 1) != std::string::npos) {
            throw std::runtime_error("Malformed payload manifest row " +
                                     std::to_string(line_number));
        }
        const auto relative_utf8 = line.substr(second_tab + 1);
        if (!is_safe_payload_path(relative_utf8))
            throw std::runtime_error("Unsafe payload path on manifest row " +
                                     std::to_string(line_number));
        if (!paths.insert(lowercase_ascii(relative_utf8)).second)
            throw std::runtime_error("Duplicate payload path on manifest row " +
                                     std::to_string(line_number));
        result.push_back(
            PayloadEntry{parse_sha256(line.substr(0, first_tab)),
                         parse_size(line.substr(first_tab + 1, second_tab - first_tab - 1)),
                         std::filesystem::u8path(relative_utf8)});
    }
    if (!input.eof() && input.fail())
        throw std::runtime_error("Unable to read payload.manifest");
    if (result.empty())
        throw std::runtime_error("Payload manifest is empty");
    return result;
}

std::array<std::uint8_t, kSha256Bytes> sha256_file(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE raw_algorithm = nullptr;
    if (!nt_success(BCryptOpenAlgorithmProvider(&raw_algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                                BCRYPT_HASH_REUSABLE_FLAG))) {
        throw std::runtime_error("Unable to initialize Windows SHA-256 provider");
    }
    AlgorithmHandle algorithm(raw_algorithm);
    DWORD object_bytes = 0;
    DWORD returned = 0;
    if (!nt_success(BCryptGetProperty(raw_algorithm, BCRYPT_OBJECT_LENGTH,
                                      reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes),
                                      &returned, 0)) ||
        object_bytes == 0) {
        throw std::runtime_error("Unable to query Windows SHA-256 object size");
    }
    std::vector<std::uint8_t> hash_object(object_bytes);
    BCRYPT_HASH_HANDLE raw_hash = nullptr;
    if (!nt_success(BCryptCreateHash(raw_algorithm, &raw_hash, hash_object.data(), object_bytes,
                                     nullptr, 0, BCRYPT_HASH_REUSABLE_FLAG))) {
        throw std::runtime_error("Unable to create Windows SHA-256 hash");
    }
    HashHandle hash(raw_hash);
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Unable to open payload file: " + path.string());
    std::vector<char> buffer(kCopyBufferBytes);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0 &&
            !nt_success(BCryptHashData(raw_hash, reinterpret_cast<PUCHAR>(buffer.data()),
                                       static_cast<ULONG>(count), 0))) {
            throw std::runtime_error("Windows SHA-256 update failed");
        }
    }
    if (!input.eof())
        throw std::runtime_error("Unable to read payload file: " + path.string());
    std::array<std::uint8_t, kSha256Bytes> result{};
    if (!nt_success(
            BCryptFinishHash(raw_hash, result.data(), static_cast<ULONG>(result.size()), 0))) {
        throw std::runtime_error("Windows SHA-256 finalization failed");
    }
    return result;
}

std::string sha256_hex(const std::array<std::uint8_t, kSha256Bytes>& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(kSha256HexChars, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        result[index * 2] = digits[digest[index] >> 4U];
        result[index * 2 + 1] = digits[digest[index] & 0x0FU];
    }
    return result;
}

void verify_payload(const std::filesystem::path& payload_root,
                    const std::vector<PayloadEntry>& entries) {
    if (!std::filesystem::is_directory(payload_root))
        throw std::runtime_error("Payload directory is missing");
    for (const auto& entry : entries) {
        const auto source = payload_root / entry.relative_path;
        std::error_code error;
        if (!std::filesystem::is_regular_file(source, error) || error)
            throw std::runtime_error("Payload file is missing: " + entry.relative_path.string());
        const auto actual_size = std::filesystem::file_size(source, error);
        if (error || actual_size != entry.size)
            throw std::runtime_error("Payload size mismatch: " + entry.relative_path.string());
        if (sha256_file(source) != entry.sha256)
            throw std::runtime_error("Payload SHA-256 mismatch: " + entry.relative_path.string());
    }
}

bool installation_marker_matches(const std::filesystem::path& install_root,
                                 const std::string& marker_name,
                                 const std::string& marker_contents) {
    if (!is_safe_payload_path(marker_name))
        throw std::runtime_error("Installer marker name is unsafe");
    return read_small_text_file(install_root / std::filesystem::u8path(marker_name)) ==
           marker_contents;
}

void install_payload_transactional(const std::filesystem::path& payload_root,
                                   const std::filesystem::path& install_root,
                                   const std::vector<PayloadEntry>& entries,
                                   const std::string& marker_name,
                                   const std::string& marker_contents) {
    if (install_root.empty() || install_root.filename().empty())
        throw std::runtime_error("Installation directory is invalid");
    verify_payload(payload_root, entries);
    if (std::filesystem::exists(install_root) &&
        !installation_marker_matches(install_root, marker_name, marker_contents)) {
        throw std::runtime_error(
            "Installation directory exists but is not a ModelConnect H3 install");
    }

    const auto parent = install_root.parent_path();
    if (parent.empty())
        throw std::runtime_error("Installation directory must have a parent");
    std::filesystem::create_directories(parent);
    const auto suffix = std::to_wstring(GetCurrentProcessId());
    const auto staging = parent / (install_root.filename().wstring() + L".staging-" + suffix);
    const auto backup = parent / (install_root.filename().wstring() + L".backup-" + suffix);
    if (std::filesystem::exists(staging))
        remove_transaction_directory(staging, parent);
    if (std::filesystem::exists(backup))
        remove_transaction_directory(backup, parent);

    try {
        std::filesystem::create_directory(staging);
        for (const auto& entry : entries) {
            const auto source = payload_root / entry.relative_path;
            const auto destination = staging / entry.relative_path;
            materialize_file(source, destination, entry.size);
            if (std::filesystem::file_size(destination) != entry.size)
                throw std::runtime_error("Installed payload size mismatch: " +
                                         entry.relative_path.string());
        }
        if (!installation_marker_matches(staging, marker_name, marker_contents))
            throw std::runtime_error("Payload installation marker is missing or invalid");

        const bool replacing = std::filesystem::exists(install_root);
        if (replacing)
            std::filesystem::rename(install_root, backup);
        try {
            std::filesystem::rename(staging, install_root);
        } catch (...) {
            if (replacing && std::filesystem::exists(backup) &&
                !std::filesystem::exists(install_root)) {
                std::filesystem::rename(backup, install_root);
            }
            throw;
        }
        if (replacing && std::filesystem::exists(backup))
            remove_transaction_directory(backup, parent);
    } catch (...) {
        if (std::filesystem::exists(staging)) {
            std::error_code ignored;
            std::filesystem::remove_all(staging, ignored);
        }
        throw;
    }
}

} // namespace trtmc::installer
