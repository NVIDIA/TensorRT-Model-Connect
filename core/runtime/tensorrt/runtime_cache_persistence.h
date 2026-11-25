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

// Write a complete serialized cache to a same-directory temporary file, make
// its contents durable, then atomically replace the target. The temporary is
// removed on every reported failure.
void persist_runtime_cache_file(const std::filesystem::path& target, const void* data,
                                std::size_t size);

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
    const std::string& path() const noexcept { return path_; }

  private:
    std::string path_;
    std::uint64_t next_lease_{1};
    std::unordered_set<std::uint64_t> active_;
};

} // namespace trtmc::internal
