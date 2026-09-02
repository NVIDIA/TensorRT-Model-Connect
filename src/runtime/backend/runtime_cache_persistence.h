/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <unordered_set>

namespace trtmc::internal {

// Internal filesystem seam for deterministic failure testing. The production
// implementation uses FlushFileBuffers and MoveFileExW on Windows.
class RuntimeCacheFileOperations {
  public:
    virtual ~RuntimeCacheFileOperations() = default;
    virtual void durable_flush(const std::filesystem::path& path) const = 0;
    virtual void atomic_replace(const std::filesystem::path& temporary,
                                const std::filesystem::path& target) const = 0;
};

const RuntimeCacheFileOperations& system_runtime_cache_file_operations();

std::filesystem::path runtime_cache_temporary_path(const std::filesystem::path& target);

// Write a complete serialized cache to a same-directory temporary file, make
// its contents durable, then atomically replace the target. The temporary is
// removed on every reported failure.
void persist_runtime_cache_file(
    const std::filesystem::path& target, const void* data, std::size_t size,
    const RuntimeCacheFileOperations& operations = system_runtime_cache_file_operations());

// Lease bookkeeping is deliberately independent of TensorRT-RTX so the
// failure contract can be regression-tested without constructing an engine.
// Callers must serialize access to this object.
class RuntimeCacheLeaseState final {
  public:
    std::uint64_t acquire(const char* path);

    // The final active lease is erased only after persistence succeeds. If the
    // callback throws, the exact lease remains active and can be retried.
    void release(std::uint64_t lease, bool cache_materialized,
                 const std::function<void()>& persist_final_cache);

    bool empty() const noexcept { return active_.empty(); }
    std::size_t size() const noexcept { return active_.size(); }
    bool contains(std::uint64_t lease) const noexcept;
    const std::string& path() const noexcept { return path_; }

  private:
    std::string path_;
    std::uint64_t next_lease_{1};
    std::unordered_set<std::uint64_t> active_;
};

} // namespace trtmc::internal
