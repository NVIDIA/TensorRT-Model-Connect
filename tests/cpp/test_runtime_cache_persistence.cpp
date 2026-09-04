/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_cache_persistence.h"
#include "test_helpers.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to read test file");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void test_only_final_lease_persists_with_durable_atomic_replace() {
    trtmc_test::TempDirGuard directory;
    const fs::path target = fs::path(directory.path()) / "cache.bin";
    trtmc_test::write_file(target, "old-cache");

    trtmc::internal::RuntimeCacheLeaseState leases;
    const std::string target_string = target.string();
    const auto first = leases.acquire(target_string.c_str());
    const auto final = leases.acquire(target_string.c_str());
    int serialization_attempts = 0;
    const std::string new_cache = "new-serialized-cache";
    const auto persist = [&] {
        ++serialization_attempts;
        trtmc::internal::persist_runtime_cache_file(target, new_cache.data(), new_cache.size());
    };

    leases.release(first, true, persist);
    check(serialization_attempts == 0, "non-final lease unexpectedly persisted the cache");
    check(read_file(target) == "old-cache", "non-final lease changed the cache file");
    check(leases.size() == 1, "final cache lease was lost before persistence");

    leases.release(final, true, persist);
    check(serialization_attempts == 1, "final lease did not serialize exactly once");
    check(leases.empty(), "successful final persistence retained its lease");
    check(read_file(target) == new_cache, "atomic cache replacement wrote incorrect bytes");
}

void test_write_failure_preserves_final_lease_for_successful_retry() {
    trtmc_test::TempDirGuard directory;
    const fs::path parent = fs::path(directory.path()) / "not-created";
    const fs::path target = parent / "cache.bin";
    const std::string target_string = target.string();
    const std::string serialized = "cache-after-write-retry";

    trtmc::internal::RuntimeCacheLeaseState leases;
    const auto lease = leases.acquire(target_string.c_str());
    bool failure_observed = false;
    try {
        leases.release(lease, true, [&] {
            trtmc::internal::persist_runtime_cache_file(target, serialized.data(),
                                                        serialized.size());
        });
    } catch (const std::runtime_error& error) {
        failure_observed = std::string(error.what()).find("temporary file") != std::string::npos;
    }

    check(failure_observed, "runtime-cache write failure was not reported");
    check(leases.size() == 1, "runtime-cache write failure consumed the final lease");
    fs::create_directory(parent);
    leases.release(lease, true, [&] {
        trtmc::internal::persist_runtime_cache_file(target, serialized.data(), serialized.size());
    });
    check(leases.empty(), "successful write retry retained the final lease");
    check(read_file(target) == serialized, "write retry persisted incorrect cache bytes");
}

} // namespace

int main() {
    try {
        test_only_final_lease_persists_with_durable_atomic_replace();
        test_write_failure_preserves_final_lease_for_successful_retry();
        std::cout << "Runtime-cache persistence tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime-cache persistence test failed: " << error.what() << '\n';
        return 1;
    }
}
