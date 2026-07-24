/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../qualification/native_dynamic_memory_qualify_schema.h"

#include <cstddef>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_repeat_schema(std::size_t repeat) {
    auto samples = trtmc::qualification::make_sequential_request_samples();
    for (std::size_t index = 0; index < repeat; ++index) {
        samples.push_back({{"request_index", index}});
    }
    check(samples.is_array(), "sequential request samples are a JSON array");
    check(samples.size() == repeat, "sequential request sample count exactly equals repeat");
    check(samples.empty() || samples.front().is_object(),
          "sequential request samples have no leading nested empty array");
}

void test_runtime_phase_memory_schema() {
    auto samples = trtmc::qualification::make_runtime_phase_memory_samples();
    samples.push_back(trtmc::qualification::make_runtime_phase_memory_sample(
        "after runtime KV allocation", 2, 700, 1000, 275));

    nlohmann::json lifetime = {{"label", "measured-load-cycle"}};
    trtmc::qualification::attach_runtime_phase_memory_samples(lifetime, samples);

    check(lifetime.at("runtime_phase_memory_samples").is_array(),
          "runtime phase samples are attached as a JSON array");
    check(lifetime.at("runtime_phase_memory_samples").size() == 1,
          "every captured runtime phase sample is preserved");
    const auto& sample = lifetime.at("runtime_phase_memory_samples").front();
    check(sample.at("phase") == "after runtime KV allocation",
          "runtime phase sample preserves its exact phase");
    check(sample.at("device") == 2, "runtime phase sample preserves the CUDA device");
    check(sample.at("free_bytes") == 700 && sample.at("total_bytes") == 1000,
          "runtime phase sample preserves the exact CUDA snapshot");
    check(sample.at("used_bytes") == 300, "runtime phase sample derives device-wide used bytes");
    check(sample.at("process_used_bytes") == 275,
          "runtime phase sample preserves independent process bytes");
}

} // namespace

int main() {
    test_repeat_schema(1);
    test_repeat_schema(100);
    test_runtime_phase_memory_schema();
    if (failures != 0)
        return 1;
    std::cout << "native dynamic-memory qualification schema checks passed\n";
    return 0;
}
