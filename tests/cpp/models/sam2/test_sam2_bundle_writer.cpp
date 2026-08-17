/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/models/sam2/sam2_engine_contract.h"
#include "tools/sam2_native_builder/bundle_writer.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

extern "C" int __real_fsync(int descriptor);
extern "C" int __real_unlinkat(int directory, const char* path, int flags);

namespace {

using trtmc::sam2::native::BundleMetadata;
using trtmc::sam2::native::BundleSectionView;

enum class DirectoryFsyncFault : int { kNone, kFail, kReplaceThenFail };

std::atomic<DirectoryFsyncFault> g_directory_fsync_fault{DirectoryFsyncFault::kNone};
std::atomic<int> g_directory_fsync_calls{0};
std::atomic<int> g_directory_scan_error{0};
std::atomic<bool> g_temporary_entry_seen{false};
std::atomic<bool> g_fail_next_temporary_unlink{false};
std::atomic<int> g_replacement_error{0};
std::string g_replacement_path;
constexpr std::string_view kReplacementPayload = "replacement";

int replacePublishedDestination() noexcept {
    if (::unlink(g_replacement_path.c_str()) != 0)
        return errno;
    int flags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(g_replacement_path.c_str(), flags, 0600);
    if (descriptor < 0)
        return errno;
    const ssize_t written =
        ::write(descriptor, kReplacementPayload.data(), kReplacementPayload.size());
    const int write_error = written == static_cast<ssize_t>(kReplacementPayload.size()) ? 0 : errno;
    const int close_error = ::close(descriptor) == 0 ? 0 : errno;
    return write_error != 0 ? write_error : close_error;
}

void inspectDirectoryAtFsync(int descriptor) noexcept {
    const int scan_descriptor = ::dup(descriptor);
    if (scan_descriptor < 0) {
        g_directory_scan_error.store(errno, std::memory_order_relaxed);
        return;
    }
    DIR* stream = ::fdopendir(scan_descriptor);
    if (stream == nullptr) {
        const int error = errno;
        (void)::close(scan_descriptor);
        g_directory_scan_error.store(error, std::memory_order_relaxed);
        return;
    }
    errno = 0;
    while (const dirent* entry = ::readdir(stream)) {
        if (std::strncmp(entry->d_name, ".trtmc-sam2-", 12U) == 0)
            g_temporary_entry_seen.store(true, std::memory_order_relaxed);
    }
    const int read_error = errno;
    if (::closedir(stream) != 0 && read_error == 0)
        g_directory_scan_error.store(errno, std::memory_order_relaxed);
    else if (read_error != 0)
        g_directory_scan_error.store(read_error, std::memory_order_relaxed);
}

int wrappedFsync(int descriptor) noexcept {
    struct stat status{};
    if (::fstat(descriptor, &status) != 0 || !S_ISDIR(status.st_mode))
        return __real_fsync(descriptor);

    g_directory_fsync_calls.fetch_add(1, std::memory_order_relaxed);
    inspectDirectoryAtFsync(descriptor);
    const auto fault =
        g_directory_fsync_fault.exchange(DirectoryFsyncFault::kNone, std::memory_order_relaxed);
    if (fault == DirectoryFsyncFault::kReplaceThenFail)
        g_replacement_error.store(replacePublishedDestination(), std::memory_order_relaxed);
    if (fault != DirectoryFsyncFault::kNone) {
        errno = EIO;
        return -1;
    }
    return __real_fsync(descriptor);
}

int wrappedUnlinkat(int directory, const char* path, int flags) noexcept {
    if (path != nullptr && std::strncmp(path, ".trtmc-sam2-", 12U) == 0 &&
        g_fail_next_temporary_unlink.exchange(false, std::memory_order_relaxed)) {
        errno = EIO;
        return -1;
    }
    return __real_unlinkat(directory, path, flags);
}

void resetDirectoryFsyncObservations() noexcept {
    g_directory_fsync_calls.store(0, std::memory_order_relaxed);
    g_directory_scan_error.store(0, std::memory_order_relaxed);
    g_temporary_entry_seen.store(false, std::memory_order_relaxed);
}

std::filesystem::path makeTemporaryDirectory() {
    std::array<char, 64> pattern{};
    const std::string value = "/tmp/trtmc_sam2_bundle_writer_XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    auto* path = ::mkdtemp(pattern.data());
    if (path == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return path;
}

std::uint64_t readU64(std::istream& input) {
    std::uint64_t value = 0;
    for (unsigned int shift = 0; shift < 64U; shift += 8U) {
        const int byte = input.get();
        if (byte < 0)
            throw std::runtime_error("unexpected end of bundle");
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(byte)) << shift;
    }
    return value;
}

std::vector<std::vector<char>> payloads() {
    std::vector<std::vector<char>> result;
    for (std::size_t index = 0; index < trtmc::sam2::kRequiredPlanSections.size(); ++index)
        result.push_back(std::vector<char>(index + 1U, static_cast<char>('A' + index)));
    result.push_back(std::vector<char>{'{', '}'});
    result.push_back(std::vector<char>{'{', '"', 'o', 'k', '"', ':', '1', '}'});
    return result;
}

std::vector<BundleSectionView> sectionViews(const std::vector<std::vector<char>>& data) {
    std::vector<BundleSectionView> result;
    for (std::size_t index = 0; index < trtmc::sam2::kRequiredPlanSections.size(); ++index) {
        result.push_back(
            {trtmc::sam2::kRequiredPlanSections[index], data[index].data(), data[index].size()});
    }
    result.push_back({trtmc::sam2::kConfigSection, data[6].data(), data[6].size()});
    result.push_back({trtmc::sam2::kBuildReceiptSection, data[7].data(), data[7].size()});
    return result;
}

std::string expectThrowsMessage(const std::function<void()>& function, const char* context) {
    try {
        function();
    } catch (const trtmc::sam2::native::BundleWriteError& error) {
        return error.what();
    }
    throw std::runtime_error(std::string("expected BundleWriteError: ") + context);
}

void expectThrows(const std::function<void()>& function, const char* context) {
    (void)expectThrowsMessage(function, context);
}

std::string sha256(const std::vector<char>& data) {
    trtmc::internal::Sha256 hash;
    hash.update(data.data(), data.size());
    return hash.hex_digest();
}

std::string sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to open bundle for full-file hashing");
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                                  std::istreambuf_iterator<char>());
    return sha256(bytes);
}

} // namespace

extern "C" int __wrap_fsync(int descriptor) {
    return wrappedFsync(descriptor);
}
extern "C" int __wrap_unlinkat(int directory, const char* path, int flags) {
    return wrappedUnlinkat(directory, path, flags);
}

int main() {
    const auto directory = makeTemporaryDirectory();
    const auto destination = directory / "sam2.bundle";
    const auto data = payloads();
    const auto sections = sectionViews(data);
    const BundleMetadata metadata{"sam2.1-hiera-small-bbox", "11.1.0.106", "11.1", "NVIDIA L4",
                                  "2026-08-15T00:00:00Z"};

    const auto publication =
        trtmc::sam2::native::writeSam2NativeBundle(destination, metadata, sections);
    if (g_directory_fsync_calls.load(std::memory_order_relaxed) != 1 ||
        g_directory_scan_error.load(std::memory_order_relaxed) != 0 ||
        g_temporary_entry_seen.load(std::memory_order_relaxed)) {
        throw std::runtime_error("successful publication did not fsync its destination directory");
    }
    if (!std::filesystem::is_regular_file(destination))
        throw std::runtime_error("bundle was not published");
    struct stat published_status{};
    if (::stat(destination.c_str(), &published_status) != 0 ||
        publication.sha256 != sha256File(destination) || publication.size_bytes == 0U ||
        publication.size_bytes != static_cast<std::uint64_t>(published_status.st_size) ||
        !publication.filesystem_identity_available ||
        publication.device != static_cast<std::uint64_t>(published_status.st_dev) ||
        publication.inode != static_cast<std::uint64_t>(published_status.st_ino)) {
        throw std::runtime_error("writer did not return facts for the exact published inode");
    }

    std::ifstream input(destination, std::ios::binary);
    const std::array<unsigned char, 8> expected_magic = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', 0};
    std::array<unsigned char, 8> magic{};
    input.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
    if (magic != expected_magic)
        throw std::runtime_error("bundle magic mismatch");
    const auto header_size = readU64(input);
    std::string header(static_cast<std::size_t>(header_size), '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (header.find("\"runtime_strategy\":\"sam2_bbox_video_tracking\"") == std::string::npos)
        throw std::runtime_error("runtime strategy missing from header");
    for (const auto section : trtmc::sam2::kRequiredPlanSections) {
        if (header.find(std::string("\"") + std::string(section) + "\"") == std::string::npos)
            throw std::runtime_error("required plan missing from header");
    }
    std::uint64_t expected_offset = 0;
    for (std::size_t index = 0; index < data.size(); ++index) {
        const std::string_view name = index < trtmc::sam2::kRequiredPlanSections.size()
                                          ? trtmc::sam2::kRequiredPlanSections[index]
                                          : (index == trtmc::sam2::kRequiredPlanSections.size()
                                                 ? trtmc::sam2::kConfigSection
                                                 : trtmc::sam2::kBuildReceiptSection);
        const std::string entry = "\"" + std::string(name) +
                                  "\":{\"offset\":" + std::to_string(expected_offset) +
                                  ",\"size\":" + std::to_string(data[index].size()) +
                                  ",\"sha256\":\"" + sha256(data[index]) + "\"}";
        if (header.find(entry) == std::string::npos)
            throw std::runtime_error("section offset, size, or SHA-256 missing from header");
        expected_offset += data[index].size();
    }
    for (const auto& expected : data) {
        std::vector<char> actual(expected.size());
        input.read(actual.data(), static_cast<std::streamsize>(actual.size()));
        if (actual != expected)
            throw std::runtime_error("bundle section order or payload mismatch");
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unexpected trailing bundle bytes");

    const auto generic_bundle = trtmc::ReadBundleFile(destination.string());
    if (generic_bundle.sections.size() != data.size())
        throw std::runtime_error("generic bundle reader rejected hashed section metadata");
    for (std::size_t index = 0; index < data.size(); ++index) {
        if (generic_bundle.sections[index].data != data[index])
            throw std::runtime_error("generic bundle reader changed a hashed section payload");
    }

    const auto original_sha256 = sha256File(destination);
    expectThrows(
        [&] { trtmc::sam2::native::writeSam2NativeBundle(destination, metadata, sections); },
        "existing destination");
    if (sha256File(destination) != original_sha256)
        throw std::runtime_error("existing destination changed after no-replace publication");

    resetDirectoryFsyncObservations();
    g_directory_fsync_fault.store(DirectoryFsyncFault::kFail, std::memory_order_relaxed);
    const auto failed_sync_destination = directory / "failed-directory-sync.bundle";
    const auto failed_sync_message = expectThrowsMessage(
        [&] {
            trtmc::sam2::native::writeSam2NativeBundle(failed_sync_destination, metadata, sections);
        },
        "destination directory fsync failure");
    if (std::filesystem::exists(failed_sync_destination) ||
        g_directory_fsync_calls.load(std::memory_order_relaxed) != 2 ||
        g_directory_scan_error.load(std::memory_order_relaxed) != 0 ||
        g_temporary_entry_seen.load(std::memory_order_relaxed) ||
        failed_sync_message.find("rollback removed the exact published") == std::string::npos ||
        failed_sync_message.find("post-rollback destination directory fsync succeeded") ==
            std::string::npos) {
        throw std::runtime_error("directory fsync failure did not fail closed and sync rollback");
    }

    resetDirectoryFsyncObservations();
    g_replacement_error.store(0, std::memory_order_relaxed);
    const auto replaced_destination = directory / "replaced-during-directory-sync.bundle";
    g_replacement_path = replaced_destination.string();
    g_directory_fsync_fault.store(DirectoryFsyncFault::kReplaceThenFail, std::memory_order_relaxed);
    const auto replacement_message = expectThrowsMessage(
        [&] {
            trtmc::sam2::native::writeSam2NativeBundle(replaced_destination, metadata, sections);
        },
        "destination replacement during directory fsync failure");
    std::ifstream replacement_input(replaced_destination, std::ios::binary);
    const std::string replacement((std::istreambuf_iterator<char>(replacement_input)),
                                  std::istreambuf_iterator<char>());
    if (g_replacement_error.load(std::memory_order_relaxed) != 0 ||
        g_directory_fsync_calls.load(std::memory_order_relaxed) != 2 ||
        g_directory_scan_error.load(std::memory_order_relaxed) != 0 ||
        g_temporary_entry_seen.load(std::memory_order_relaxed) ||
        replacement != kReplacementPayload ||
        replacement_message.find("no longer matched the published inode") == std::string::npos) {
        throw std::runtime_error("rollback removed or changed a replacement destination");
    }
    std::filesystem::remove(replaced_destination);

    resetDirectoryFsyncObservations();
    g_fail_next_temporary_unlink.store(true, std::memory_order_relaxed);
    const auto failed_temporary_unlink_destination = directory / "failed-temp-unlink.bundle";
    const auto failed_temporary_unlink_message = expectThrowsMessage(
        [&] {
            trtmc::sam2::native::writeSam2NativeBundle(failed_temporary_unlink_destination,
                                                       metadata, sections);
        },
        "temporary-name unlink failure");
    if (std::filesystem::exists(failed_temporary_unlink_destination) ||
        g_directory_fsync_calls.load(std::memory_order_relaxed) != 1 ||
        g_directory_scan_error.load(std::memory_order_relaxed) != 0 ||
        g_temporary_entry_seen.load(std::memory_order_relaxed) ||
        failed_temporary_unlink_message.find("Failed to unlink") == std::string::npos ||
        failed_temporary_unlink_message.find("retry removed the exact") == std::string::npos ||
        failed_temporary_unlink_message.find("rollback removed the exact published") ==
            std::string::npos) {
        throw std::runtime_error("temporary unlink failure did not clean and roll back durably");
    }
    auto missing = sections;
    missing.pop_back();
    expectThrows(
        [&] {
            trtmc::sam2::native::writeSam2NativeBundle(directory / "missing.bundle", metadata,
                                                       missing);
        },
        "missing receipt");
    auto duplicate = sections;
    duplicate.back().name = trtmc::sam2::kConfigSection;
    expectThrows(
        [&] {
            trtmc::sam2::native::writeSam2NativeBundle(directory / "duplicate.bundle", metadata,
                                                       duplicate);
        },
        "duplicate config");

    const auto symlink_destination = directory / "symlink-race.bundle";
    auto legacy_temporary = symlink_destination;
    legacy_temporary += ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    const auto symlink_victim = directory / "symlink-victim";
    std::filesystem::create_symlink(symlink_victim, legacy_temporary);
    trtmc::sam2::native::writeSam2NativeBundle(symlink_destination, metadata, sections);
    if (!std::filesystem::is_regular_file(symlink_destination) ||
        std::filesystem::is_symlink(symlink_destination) ||
        std::filesystem::exists(symlink_victim) || !std::filesystem::is_symlink(legacy_temporary)) {
        throw std::runtime_error("bundle writer followed or published a predictable temp symlink");
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().filename().string().find(".trtmc-sam2-") == 0U)
            throw std::runtime_error("exclusive SAM2 temporary file was not cleaned up");
    }

    std::filesystem::remove_all(directory);
    std::cout << "SAM2 native bundle writer tests passed\n";
    return 0;
}
