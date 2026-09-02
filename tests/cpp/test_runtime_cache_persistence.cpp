/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/runtime_cache_persistence.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

class TemporaryDirectory final {
  public:
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned int attempt = 0; attempt != 100; ++attempt) {
            path_ = fs::temp_directory_path() / ("trtmc-runtime-cache-" + std::to_string(nonce) +
                                                 "-" + std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path_, error))
                return;
            if (error && error != std::errc::file_exists)
                throw std::system_error(error, "failed to create runtime-cache test directory");
        }
        throw std::runtime_error("failed to reserve a unique runtime-cache test directory");
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path_, ignored);
    }

    const fs::path& path() const noexcept { return path_; }

  private:
    fs::path path_;
};

void write_file(const fs::path& path, const std::string& value) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
    output.close();
    if (!output)
        throw std::runtime_error("failed to write test file");
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to read test file");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class RejectReplaceOperations final : public trtmc::internal::RuntimeCacheFileOperations {
  public:
    void durable_flush(const fs::path& path) const override {
        ++flush_calls;
        trtmc::internal::system_runtime_cache_file_operations().durable_flush(path);
    }

    void atomic_replace(const fs::path&, const fs::path&) const override {
        ++replace_calls;
        throw std::runtime_error("synthetic atomic replace failure");
    }

    mutable int flush_calls{0};
    mutable int replace_calls{0};
};

void test_only_final_lease_persists_with_durable_atomic_replace() {
    TemporaryDirectory directory;
    const fs::path target = directory.path() / "cache.bin";
    write_file(target, "old-cache");

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
    check(leases.contains(final), "final cache lease was lost before persistence");

    leases.release(final, true, persist);
    check(serialization_attempts == 1, "final lease did not serialize exactly once");
    check(leases.empty(), "successful final persistence retained its lease");
    check(read_file(target) == new_cache, "atomic cache replacement wrote incorrect bytes");
    check(!fs::exists(trtmc::internal::runtime_cache_temporary_path(target)),
          "successful persistence left its temporary file behind");
}

void test_replace_failure_preserves_final_lease_for_successful_retry() {
    TemporaryDirectory directory;
    const fs::path target = directory.path() / "cache.bin";
    write_file(target, "original-cache");

    trtmc::internal::RuntimeCacheLeaseState leases;
    const std::string target_string = target.string();
    const auto lease = leases.acquire(target_string.c_str());
    const std::string replacement = "replacement-cache";
    RejectReplaceOperations reject_replace;
    int serialization_attempts = 0;
    bool failure_observed = false;
    try {
        leases.release(lease, true, [&] {
            ++serialization_attempts;
            trtmc::internal::persist_runtime_cache_file(target, replacement.data(),
                                                        replacement.size(), reject_replace);
        });
    } catch (const std::runtime_error& error) {
        failure_observed =
            std::string(error.what()).find("synthetic atomic replace") != std::string::npos;
    }

    check(failure_observed, "atomic replacement failure was not reported");
    check(reject_replace.flush_calls == 1, "failure path did not durably flush the temporary");
    check(reject_replace.replace_calls == 1, "failure injection did not reach atomic replace");
    check(serialization_attempts == 1, "failed persistence did not attempt serialization once");
    check(leases.contains(lease), "failed atomic replacement consumed the final lease");
    check(read_file(target) == "original-cache", "failed replacement modified the old cache");
    check(!fs::exists(trtmc::internal::runtime_cache_temporary_path(target)),
          "failed atomic replacement left its temporary file behind");

    leases.release(lease, true, [&] {
        ++serialization_attempts;
        trtmc::internal::persist_runtime_cache_file(target, replacement.data(), replacement.size());
    });
    check(serialization_attempts == 2, "retry did not repeat serialization exactly once");
    check(leases.empty(), "successful retry retained the final lease");
    check(read_file(target) == replacement, "successful retry did not replace the cache");
}

void test_write_failure_preserves_final_lease_for_successful_retry() {
    TemporaryDirectory directory;
    const fs::path parent = directory.path() / "not-created";
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
    check(leases.contains(lease), "runtime-cache write failure consumed the final lease");
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
        test_replace_failure_preserves_final_lease_for_successful_retry();
        test_write_failure_preserves_final_lease_for_successful_retry();
        std::cout << "Runtime-cache persistence tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime-cache persistence test failed: " << error.what() << '\n';
        return 1;
    }
}
