/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_cache_persistence.h"

#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace trtmc::internal {

namespace {

namespace fs = std::filesystem;

std::uint64_t current_process_id() {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

class SystemRuntimeCacheFileOperations final : public RuntimeCacheFileOperations {
  public:
    void durable_flush(const fs::path& path) const override {
#if defined(_WIN32)
        HANDLE handle = CreateFileW(path.c_str(), GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            const auto error = static_cast<int>(GetLastError());
            throw std::system_error(error, std::system_category(),
                                    "failed to open RTX runtime cache for durable flush " +
                                        path.string());
        }
        DWORD error = ERROR_SUCCESS;
        if (!FlushFileBuffers(handle))
            error = GetLastError();
        if (!CloseHandle(handle) && error == ERROR_SUCCESS)
            error = GetLastError();
        if (error != ERROR_SUCCESS) {
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "failed to durably flush RTX runtime cache " + path.string());
        }
#else
        // The locked distributable is Windows-only. POSIX rename retains the
        // existing best-effort durability contract for ordinary developer builds.
        (void)path;
#endif
    }

    void atomic_replace(const fs::path& temporary, const fs::path& target) const override {
#if defined(_WIN32)
        if (!MoveFileExW(temporary.c_str(), target.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            const auto error = static_cast<int>(GetLastError());
            throw std::system_error(error, std::system_category(),
                                    "failed to atomically replace RTX runtime cache " +
                                        target.string());
        }
#else
        std::error_code error;
        fs::rename(temporary, target, error);
        if (error) {
            throw std::system_error(error, "failed to atomically replace RTX runtime cache " +
                                               target.string());
        }
#endif
    }
};

} // namespace

const RuntimeCacheFileOperations& system_runtime_cache_file_operations() {
    static const SystemRuntimeCacheFileOperations operations;
    return operations;
}

fs::path runtime_cache_temporary_path(const fs::path& target) {
    fs::path temporary = target;
    temporary += ".tmp." + std::to_string(current_process_id());
    return temporary;
}

void persist_runtime_cache_file(const fs::path& target, const void* data, std::size_t size,
                                const RuntimeCacheFileOperations& operations) {
    if (target.empty())
        throw std::invalid_argument("[trtmc] RTX runtime cache path must not be empty");
    if (size != 0 && data == nullptr)
        throw std::invalid_argument("[trtmc] Serialized RTX runtime cache data is null");
    if (size > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error("[trtmc] RTX runtime cache is too large to persist");

    const fs::path temporary = runtime_cache_temporary_path(target);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("[trtmc] Failed to open RTX runtime cache temporary file: " +
                                     temporary.string());
        }
        if (size != 0)
            output.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        output.flush();
        if (!output) {
            throw std::runtime_error("[trtmc] Failed to write or flush RTX runtime cache: " +
                                     temporary.string());
        }
        output.close();
        if (!output) {
            throw std::runtime_error("[trtmc] Failed to close RTX runtime cache: " +
                                     temporary.string());
        }
        operations.durable_flush(temporary);
        operations.atomic_replace(temporary, target);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

std::uint64_t RuntimeCacheLeaseState::acquire(const char* path) {
    if (path == nullptr || path[0] == '\0')
        throw std::invalid_argument("[trtmc] RTX runtime cache lease requires a path");
    if (path_.empty()) {
        path_ = path;
    } else if (path_ != path) {
        throw std::invalid_argument(
            "[trtmc] RTX backend cannot share one runtime cache across different paths");
    }
    const std::uint64_t lease = next_lease_++;
    if (lease == 0 || !active_.insert(lease).second)
        throw std::runtime_error("[trtmc] RTX runtime cache lease space exhausted");
    return lease;
}

void RuntimeCacheLeaseState::release(std::uint64_t lease, bool cache_materialized,
                                     const std::function<void()>& persist_final_cache) {
    const auto active = active_.find(lease);
    if (lease == 0 || active == active_.end())
        throw std::invalid_argument("[trtmc] RTX runtime cache lease is invalid or inactive");
    if (active_.size() == 1 && cache_materialized)
        persist_final_cache();
    active_.erase(active);
}

bool RuntimeCacheLeaseState::contains(std::uint64_t lease) const noexcept {
    return active_.find(lease) != active_.end();
}

} // namespace trtmc::internal
