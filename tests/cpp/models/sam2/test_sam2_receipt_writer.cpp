/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tools/sam2_native_benchmark/sam2_benchmark_protocol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

extern "C" ssize_t __real_write(int descriptor, const void* data, std::size_t size);
extern "C" int __real_fsync(int descriptor);
extern "C" int __real_linkat(int old_directory, const char* old_path, int new_directory,
                             const char* new_path, int flags);
extern "C" int __real_unlinkat(int directory, const char* path, int flags);

namespace {

namespace benchmark = trtmc::sam2::benchmark;

enum class WriteFault : int { kNone, kPartial, kFail };
enum class DirectoryFsyncFault : int { kNone, kFail, kReplaceThenFail };

std::atomic<WriteFault> g_write_fault{WriteFault::kNone};
std::atomic<bool> g_file_fsync_fault{false};
std::atomic<bool> g_link_fault{false};
std::atomic<bool> g_temporary_unlink_fault{false};
std::atomic<DirectoryFsyncFault> g_directory_fsync_fault{DirectoryFsyncFault::kNone};
std::atomic<int> g_write_calls{0};
std::atomic<int> g_file_fsync_calls{0};
std::atomic<int> g_link_calls{0};
std::atomic<int> g_directory_fsync_calls{0};
std::atomic<int> g_directory_scan_error{0};
std::atomic<bool> g_temporary_entry_seen{false};
std::atomic<int> g_replacement_error{0};
std::string g_replacement_path;
constexpr std::string_view kReplacementPayload = "concurrent replacement";

bool isTemporaryName(const char* path) noexcept {
    return path != nullptr && std::strncmp(path, ".trtmc-sam2-", 12U) == 0;
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
        if (isTemporaryName(entry->d_name))
            g_temporary_entry_seen.store(true, std::memory_order_relaxed);
    }
    const int read_error = errno;
    if (::closedir(stream) != 0 && read_error == 0)
        g_directory_scan_error.store(errno, std::memory_order_relaxed);
    else if (read_error != 0)
        g_directory_scan_error.store(read_error, std::memory_order_relaxed);
}

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
    const auto written =
        __real_write(descriptor, kReplacementPayload.data(), kReplacementPayload.size());
    const int write_error = written == static_cast<ssize_t>(kReplacementPayload.size()) ? 0 : errno;
    const int close_error = ::close(descriptor) == 0 ? 0 : errno;
    return write_error != 0 ? write_error : close_error;
}

ssize_t wrappedWrite(int descriptor, const void* data, std::size_t size) noexcept {
    g_write_calls.fetch_add(1, std::memory_order_relaxed);
    const auto fault = g_write_fault.exchange(WriteFault::kNone, std::memory_order_relaxed);
    if (fault == WriteFault::kFail) {
        errno = EIO;
        return -1;
    }
    const std::size_t write_size =
        fault == WriteFault::kPartial ? std::min<std::size_t>(7U, size) : size;
    return __real_write(descriptor, data, write_size);
}

int wrappedFsync(int descriptor) noexcept {
    struct stat status{};
    if (::fstat(descriptor, &status) != 0)
        return __real_fsync(descriptor);
    if (S_ISREG(status.st_mode)) {
        g_file_fsync_calls.fetch_add(1, std::memory_order_relaxed);
        if (g_file_fsync_fault.exchange(false, std::memory_order_relaxed)) {
            errno = EIO;
            return -1;
        }
        return __real_fsync(descriptor);
    }
    if (!S_ISDIR(status.st_mode))
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

int wrappedLinkat(int old_directory, const char* old_path, int new_directory, const char* new_path,
                  int flags) noexcept {
    g_link_calls.fetch_add(1, std::memory_order_relaxed);
    if (g_link_fault.exchange(false, std::memory_order_relaxed)) {
        errno = EIO;
        return -1;
    }
    return __real_linkat(old_directory, old_path, new_directory, new_path, flags);
}

int wrappedUnlinkat(int directory, const char* path, int flags) noexcept {
    if (isTemporaryName(path) &&
        g_temporary_unlink_fault.exchange(false, std::memory_order_relaxed)) {
        errno = EIO;
        return -1;
    }
    return __real_unlinkat(directory, path, flags);
}

void resetObservations() noexcept {
    g_write_fault.store(WriteFault::kNone, std::memory_order_relaxed);
    g_file_fsync_fault.store(false, std::memory_order_relaxed);
    g_link_fault.store(false, std::memory_order_relaxed);
    g_temporary_unlink_fault.store(false, std::memory_order_relaxed);
    g_directory_fsync_fault.store(DirectoryFsyncFault::kNone, std::memory_order_relaxed);
    g_write_calls.store(0, std::memory_order_relaxed);
    g_file_fsync_calls.store(0, std::memory_order_relaxed);
    g_link_calls.store(0, std::memory_order_relaxed);
    g_directory_fsync_calls.store(0, std::memory_order_relaxed);
    g_directory_scan_error.store(0, std::memory_order_relaxed);
    g_temporary_entry_seen.store(false, std::memory_order_relaxed);
    g_replacement_error.store(0, std::memory_order_relaxed);
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename Exception, typename Function>
void checkThrows(Function&& function, const char* needle, const char* message) {
    static_assert(std::is_base_of<std::exception, Exception>::value);
    try {
        function();
    } catch (const Exception& error) {
        if (std::strstr(error.what(), needle) != nullptr)
            return;
        std::cerr << "FAIL: " << message << " (wrong message: " << error.what() << ")\n";
        std::exit(1);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong exception: " << error.what() << ")\n";
        std::exit(1);
    }
    std::cerr << "FAIL: " << message << " (no exception)\n";
    std::exit(1);
}

std::filesystem::path makeTemporaryDirectory() {
    std::array<char, 64> pattern{};
    constexpr std::string_view value = "/tmp/trtmc_sam2_receipt_writer_XXXXXX";
    std::copy(value.begin(), value.end(), pattern.begin());
    char* result = ::mkdtemp(pattern.data());
    if (result == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return result;
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to read receipt fixture");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

bool hasTemporaryEntry(const std::filesystem::path& directory) {
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (isTemporaryName(entry.path().filename().c_str()))
            return true;
    }
    return false;
}

void checkSuccessfulPublication(const std::filesystem::path& directory, const char* context) {
    check(g_write_calls.load(std::memory_order_relaxed) >= 1, context);
    check(g_file_fsync_calls.load(std::memory_order_relaxed) == 1, context);
    check(g_link_calls.load(std::memory_order_relaxed) >= 1, context);
    check(g_directory_fsync_calls.load(std::memory_order_relaxed) == 1, context);
    check(g_directory_scan_error.load(std::memory_order_relaxed) == 0, context);
    check(!g_temporary_entry_seen.load(std::memory_order_relaxed), context);
    check(!hasTemporaryEntry(directory), context);
}

} // namespace

extern "C" ssize_t __wrap_write(int descriptor, const void* data, std::size_t size) {
    return wrappedWrite(descriptor, data, size);
}

extern "C" int __wrap_fsync(int descriptor) {
    return wrappedFsync(descriptor);
}

extern "C" int __wrap_linkat(int old_directory, const char* old_path, int new_directory,
                             const char* new_path, int flags) {
    return wrappedLinkat(old_directory, old_path, new_directory, new_path, flags);
}

extern "C" int __wrap_unlinkat(int directory, const char* path, int flags) {
    return wrappedUnlinkat(directory, path, flags);
}

int main() {
    constexpr std::string_view q3_contents = "{\"mode\":\"accuracy_only\"}\n";
    constexpr std::string_view regular_contents = "{\"mode\":\"diagnostic_benchmark\"}\n";
    const auto directory = makeTemporaryDirectory();
    const auto q3 = directory / "q3-receipt.json";
    const auto regular = directory / "regular-receipt.json";

    resetObservations();
    benchmark::writeReceiptExclusive(q3, std::string(q3_contents));
    check(readFile(q3) == q3_contents, "Q3 receipt retained its exact bytes");
    checkSuccessfulPublication(directory, "Q3 receipt publication was not durable");

    resetObservations();
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(q3, std::string(regular_contents)); },
        "exclusive destination already exists", "Q3 receipt O_EXCL contract failed");
    check(readFile(q3) == q3_contents && g_write_calls.load(std::memory_order_relaxed) == 0 &&
              g_file_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              g_link_calls.load(std::memory_order_relaxed) == 0 &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              !hasTemporaryEntry(directory),
          "existing Q3 receipt changed or caused publication side effects");

    resetObservations();
    benchmark::writeReceiptExclusive(regular, std::string(regular_contents));
    check(readFile(regular) == regular_contents, "regular receipt retained its exact bytes");
    checkSuccessfulPublication(directory, "regular receipt publication was not durable");

    resetObservations();
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(regular, std::string(q3_contents)); },
        "exclusive destination already exists", "regular receipt O_EXCL contract failed");
    check(readFile(regular) == regular_contents &&
              g_write_calls.load(std::memory_order_relaxed) == 0 &&
              g_file_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              g_link_calls.load(std::memory_order_relaxed) == 0 &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              !hasTemporaryEntry(directory),
          "existing regular receipt changed or caused publication side effects");

    const auto partial = directory / "partial-write.json";
    resetObservations();
    g_write_fault.store(WriteFault::kPartial, std::memory_order_relaxed);
    benchmark::writeReceiptExclusive(partial, std::string(regular_contents));
    check(readFile(partial) == regular_contents &&
              g_write_calls.load(std::memory_order_relaxed) >= 2,
          "partial receipt write was not completed exactly");
    checkSuccessfulPublication(directory, "partial write publication was not durable");

    const auto failed_write = directory / "failed-write.json";
    resetObservations();
    g_write_fault.store(WriteFault::kFail, std::memory_order_relaxed);
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(failed_write, std::string(regular_contents)); },
        "receipt write failed", "receipt write failure did not fail closed");
    check(!std::filesystem::exists(failed_write) && !hasTemporaryEntry(directory) &&
              g_file_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              g_link_calls.load(std::memory_order_relaxed) == 0 &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 1 &&
              !g_temporary_entry_seen.load(std::memory_order_relaxed),
          "receipt write failure left a name or skipped durable cleanup");

    const auto failed_file_sync = directory / "failed-file-fsync.json";
    resetObservations();
    g_file_fsync_fault.store(true, std::memory_order_relaxed);
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(failed_file_sync, std::string(regular_contents)); },
        "Failed to fsync the completed SAM2 benchmark receipt",
        "receipt file fsync failure did not fail closed");
    check(!std::filesystem::exists(failed_file_sync) && !hasTemporaryEntry(directory) &&
              g_file_fsync_calls.load(std::memory_order_relaxed) == 1 &&
              g_link_calls.load(std::memory_order_relaxed) == 0 &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 1 &&
              !g_temporary_entry_seen.load(std::memory_order_relaxed),
          "receipt file fsync failure left a name or skipped durable cleanup");

    const auto failed_link = directory / "failed-link.json";
    resetObservations();
    g_link_fault.store(true, std::memory_order_relaxed);
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(failed_link, std::string(regular_contents)); },
        "Failed to publish the SAM2 benchmark receipt descriptor without overwrite",
        "receipt link failure did not fail closed");
    check(!std::filesystem::exists(failed_link) && !hasTemporaryEntry(directory) &&
              g_link_calls.load(std::memory_order_relaxed) == 1 &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 1 &&
              !g_temporary_entry_seen.load(std::memory_order_relaxed),
          "receipt link failure left a name or skipped durable cleanup");

    const auto failed_temporary_unlink = directory / "failed-temp-unlink.json";
    resetObservations();
    g_temporary_unlink_fault.store(true, std::memory_order_relaxed);
    checkThrows<std::runtime_error>(
        [&] {
            benchmark::writeReceiptExclusive(failed_temporary_unlink,
                                             std::string(regular_contents));
        },
        "Failed to unlink the SAM2 benchmark receipt temporary name",
        "receipt temporary unlink failure did not fail closed");
    check(!std::filesystem::exists(failed_temporary_unlink) && !hasTemporaryEntry(directory) &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 1 &&
              !g_temporary_entry_seen.load(std::memory_order_relaxed),
          "temporary unlink failure did not roll back its exact publication");

    const auto failed_directory_sync = directory / "failed-directory-fsync.json";
    resetObservations();
    g_directory_fsync_fault.store(DirectoryFsyncFault::kFail, std::memory_order_relaxed);
    checkThrows<std::runtime_error>(
        [&] {
            benchmark::writeReceiptExclusive(failed_directory_sync, std::string(regular_contents));
        },
        "destination directory after publication",
        "receipt directory fsync failure did not fail closed");
    check(!std::filesystem::exists(failed_directory_sync) && !hasTemporaryEntry(directory) &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 2 &&
              g_directory_scan_error.load(std::memory_order_relaxed) == 0 &&
              !g_temporary_entry_seen.load(std::memory_order_relaxed),
          "directory fsync failure did not durably roll back its exact publication");

    const auto replaced = directory / "replaced-during-directory-fsync.json";
    resetObservations();
    g_replacement_path = replaced.string();
    g_directory_fsync_fault.store(DirectoryFsyncFault::kReplaceThenFail, std::memory_order_relaxed);
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(replaced, std::string(regular_contents)); },
        "no longer matched the published inode",
        "receipt rollback did not detect a concurrent replacement");
    check(g_replacement_error.load(std::memory_order_relaxed) == 0 &&
              readFile(replaced) == kReplacementPayload &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 2 &&
              g_directory_scan_error.load(std::memory_order_relaxed) == 0 &&
              !g_temporary_entry_seen.load(std::memory_order_relaxed) &&
              !hasTemporaryEntry(directory),
          "receipt rollback removed or changed a concurrent replacement");

    const auto existing = directory / "existing.json";
    constexpr std::string_view existing_payload = "preexisting receipt";
    {
        std::ofstream output(existing, std::ios::binary);
        output << existing_payload;
    }
    resetObservations();
    checkThrows<std::runtime_error>(
        [&] { benchmark::writeReceiptExclusive(existing, std::string(regular_contents)); },
        "exclusive destination already exists",
        "receipt writer did not reject a preexisting destination");
    check(readFile(existing) == existing_payload &&
              g_write_calls.load(std::memory_order_relaxed) == 0 &&
              g_file_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              g_link_calls.load(std::memory_order_relaxed) == 0 &&
              g_directory_fsync_calls.load(std::memory_order_relaxed) == 0 &&
              !hasTemporaryEntry(directory),
          "receipt writer changed a preexisting destination");

    const auto symlink_victim = directory / "symlink-victim.json";
    constexpr std::string_view victim_payload = "symlink victim";
    {
        std::ofstream output(symlink_victim, std::ios::binary);
        output << victim_payload;
    }
    const auto symlink_destination = directory / "symlink-destination.json";
    std::filesystem::create_symlink(symlink_victim, symlink_destination);
    resetObservations();
    checkThrows<std::runtime_error>(
        [&] {
            benchmark::writeReceiptExclusive(symlink_destination, std::string(regular_contents));
        },
        "exclusive destination already exists",
        "receipt writer did not reject an existing destination symlink");
    check(std::filesystem::is_symlink(symlink_destination) &&
              readFile(symlink_victim) == victim_payload && !hasTemporaryEntry(directory),
          "receipt writer followed or changed an existing destination symlink");

    const auto real_parent = directory / "real-parent";
    std::filesystem::create_directory(real_parent);
    const auto parent_symlink = directory / "parent-symlink";
    std::filesystem::create_directory_symlink(real_parent, parent_symlink);
    resetObservations();
    checkThrows<std::runtime_error>(
        [&] {
            benchmark::writeReceiptExclusive(parent_symlink / "through-symlink.json",
                                             std::string(regular_contents));
        },
        "destination directory",
        "receipt writer did not enforce O_NOFOLLOW on its parent directory boundary");
    check(!std::filesystem::exists(real_parent / "through-symlink.json") &&
              !hasTemporaryEntry(real_parent),
          "receipt writer followed a parent directory symlink");

    resetObservations();
    std::filesystem::remove_all(directory);
    std::cout << "SAM2 receipt writer tests passed\n";
    return 0;
}
