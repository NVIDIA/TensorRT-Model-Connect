/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/core/distributed_runtime_detail.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kEnvironmentNames[] = {
    "OMPI_COMM_WORLD_SIZE",       "PMI_SIZE",       "WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",       "PMI_RANK",       "RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
    "MV2_COMM_WORLD_LOCAL_RANK",  "SLURM_LOCALID",  "LOCAL_RANK",
};

class EnvironmentGuard {
  public:
    EnvironmentGuard() {
        for (const char* name : kEnvironmentNames) {
            const char* value = std::getenv(name);
            values_.emplace_back(name, value == nullptr ? std::nullopt
                                                        : std::optional<std::string>(value));
            unsetenv(name);
        }
    }

    ~EnvironmentGuard() {
        for (const auto& [name, value] : values_) {
            if (value.has_value())
                setenv(name.c_str(), value->c_str(), 1);
            else
                unsetenv(name.c_str());
        }
    }

    EnvironmentGuard(const EnvironmentGuard&) = delete;
    EnvironmentGuard& operator=(const EnvironmentGuard&) = delete;

  private:
    std::vector<std::pair<std::string, std::optional<std::string>>> values_;
};

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_generic_multinode_rank_uses_local_device_zero() {
    EnvironmentGuard guard;
    setenv("WORLD_SIZE", "2", 1);
    setenv("RANK", "1", 1);
    setenv("LOCAL_RANK", "0", 1);

    check(trtmc::distributed_runtime_detail::detect_world_size() == 2, "generic world size");
    check(trtmc::distributed_runtime_detail::detect_rank() == 1, "generic global rank");
    check(trtmc::distributed_runtime_detail::detect_local_rank(1) == 0, "generic local rank");
}

void test_mpi_local_rank_precedence() {
    EnvironmentGuard guard;
    setenv("LOCAL_RANK", "5", 1);
    setenv("SLURM_LOCALID", "4", 1);
    setenv("MV2_COMM_WORLD_LOCAL_RANK", "3", 1);
    setenv("MPI_LOCALRANKID", "2", 1);
    setenv("PMI_LOCAL_RANK", "1", 1);
    setenv("OMPI_COMM_WORLD_LOCAL_RANK", "0", 1);

    check(trtmc::distributed_runtime_detail::detect_local_rank(6) == 0,
          "Open MPI local rank has precedence");
}

void test_local_rank_aliases_and_global_fallback() {
    constexpr const char* aliases[] = {
        "PMI_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",  "LOCAL_RANK",
    };
    for (const char* alias : aliases) {
        EnvironmentGuard guard;
        setenv(alias, "1", 1);
        check(trtmc::distributed_runtime_detail::detect_local_rank(7) == 1, alias);
    }

    EnvironmentGuard guard;
    check(trtmc::distributed_runtime_detail::detect_local_rank(7) == 7,
          "global rank fallback preserves single-node behavior");
}

} // namespace

int main() {
    test_generic_multinode_rank_uses_local_device_zero();
    test_mpi_local_rank_precedence();
    test_local_rank_aliases_and_global_fallback();
    return failures == 0 ? 0 : 1;
}
