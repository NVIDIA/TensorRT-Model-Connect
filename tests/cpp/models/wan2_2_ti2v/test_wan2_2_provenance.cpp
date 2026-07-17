/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/plugin_cache.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class EnvironmentSnapshot {
  public:
    explicit EnvironmentSnapshot(const char* name) : name_(name) {
        if (const char* value = std::getenv(name); value != nullptr) {
            present_ = true;
            value_ = value;
        }
    }

    ~EnvironmentSnapshot() {
        if (present_)
            setenv(name_.c_str(), value_.c_str(), 1);
        else
            unsetenv(name_.c_str());
    }

  private:
    std::string name_;
    std::string value_;
    bool present_{false};
};

void test_fixed_creator_set_rejects_conflicting_bytes() {
    const std::string creator_set =
        "wan22-provenance-test-" + std::to_string(static_cast<long long>(getpid()));
    const std::vector<char> first = {'f', 'i', 'r', 's', 't'};
    const std::vector<char> conflicting = {'s', 'e', 'c', 'o', 'n', 'd'};
    trtmc::record_wan22_cuda_plugin_provenance(creator_set, first);
    trtmc::record_wan22_cuda_plugin_provenance(creator_set, first);

    bool rejected = false;
    try {
        trtmc::record_wan22_cuda_plugin_provenance(creator_set, conflicting);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("Conflicting Wan2.2 CUDA plugin bytes") !=
                   std::string::npos;
    }
    check(rejected, "fixed creator set rejects conflicting plugin bytes");
}

void test_override_requires_explicit_development_gate_and_never_strict_mode() {
    constexpr const char* kOverride = "TRTMC_WAN22_TEST_CUDA_PLUGIN_LIBRARY";
    EnvironmentSnapshot override_snapshot(kOverride);
    EnvironmentSnapshot gate_snapshot("TRTMC_WAN22_ALLOW_DEVELOPMENT_PLUGIN_OVERRIDE");
    EnvironmentSnapshot strict_snapshot("TRTMC_MODEL_PLUGIN_STRICT");

    setenv(kOverride, "/tmp/development-only-wan22-plugin.so", 1);
    unsetenv("TRTMC_WAN22_ALLOW_DEVELOPMENT_PLUGIN_OVERRIDE");
    unsetenv("TRTMC_MODEL_PLUGIN_STRICT");
    bool ungated_rejected = false;
    try {
        (void)trtmc::resolve_wan22_cuda_plugin_override(kOverride);
    } catch (const std::runtime_error& error) {
        ungated_rejected = std::string(error.what()).find("ALLOW_DEVELOPMENT_PLUGIN_OVERRIDE=1") !=
                           std::string::npos;
    }
    check(ungated_rejected, "plugin override is rejected without development gate");

    setenv("TRTMC_WAN22_ALLOW_DEVELOPMENT_PLUGIN_OVERRIDE", "1", 1);
    check(trtmc::resolve_wan22_cuda_plugin_override(kOverride) ==
              "/tmp/development-only-wan22-plugin.so",
          "explicit development gate permits override outside strict mode");

    setenv("TRTMC_MODEL_PLUGIN_STRICT", "1", 1);
    bool strict_rejected = false;
    try {
        (void)trtmc::resolve_wan22_cuda_plugin_override(kOverride);
    } catch (const std::runtime_error& error) {
        strict_rejected = std::string(error.what()).find("forbidden") != std::string::npos;
    }
    check(strict_rejected, "strict production mode rejects development plugin override");
}

} // namespace

int main() {
    test_fixed_creator_set_rejects_conflicting_bytes();
    test_override_requires_explicit_development_gate_and_never_strict_mode();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All Wan2.2 provenance tests passed\n";
    return 0;
}
