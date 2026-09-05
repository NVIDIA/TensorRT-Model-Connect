/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_loader.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void write_bundle(const std::filesystem::path& path, const std::string& family,
                  const std::string& task = "time_series_forecast") {
    const std::string header = "{\"format\":1,\"family\":\"" + family + "\",\"task\":\"" + task +
                               "\",\"backend\":\"fake\","
                               "\"sections\":{\"runtime.json\":{\"offset\":0,\"length\":2},"
                               "\"engine.plan\":{\"offset\":2,\"length\":4}}}";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((length >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("{}PLAN", 6);
}

bool load_throws(const std::filesystem::path& bundle, const std::string& runtime_root) {
    try {
        (void)trtmc::load_task(bundle.string(), runtime_root);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

bool rtx_options_throw(const std::filesystem::path& bundle, const std::string& runtime_root) {
    try {
        (void)trtmc::load_task(bundle.string(), runtime_root, 0, "runtime.cache", true);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_family_loader RUNTIME_ROOT\n";
        return 2;
    }
    const std::filesystem::path runtime_root(argv[1]);
    const auto bundle_path = runtime_root / "fake.bundle";
    write_bundle(bundle_path, "fake");

    auto task = trtmc::load_task(bundle_path.string(), runtime_root.string());
    auto* forecast = dynamic_cast<trtmc::ITimeSeriesForecast*>(task.get());
    check(forecast != nullptr, "load returns forecast interface");
    const float values[] = {1.0F, 2.0F, 3.0F};
    const float mask[] = {1.0F, 1.0F, 1.0F};
    const auto result = forecast->forecast({values, mask});
    check(result.values == std::vector<float>({1.0F, 2.0F, 3.0F}), "forecast dispatched");
    check(result.shape == std::vector<std::int64_t>({1, 3}), "forecast shape returned");

    auto second_task = trtmc::load_task(bundle_path.string(), runtime_root.string());
    check(dynamic_cast<trtmc::ITimeSeriesForecast*>(second_task.get()) != nullptr,
          "backend and family DSO cache supports a second load");

    auto sized_task = trtmc::load_task(bundle_path.string(), runtime_root.string(), 4096);
    auto* sized_forecast = dynamic_cast<trtmc::ITimeSeriesForecast*>(sized_task.get());
    const auto sized_result = sized_forecast->forecast({values, mask});
    check(sized_result.shape == std::vector<std::int64_t>({4096, 3}),
          "runtime KV bytes reach the selected family directly");
    check(rtx_options_throw(bundle_path, runtime_root.string()),
          "TensorRT-RTX options reject a non-RTX bundle");

    check(load_throws(bundle_path, ""), "empty runtime root rejected");
    check(load_throws(bundle_path, (runtime_root / "missing").string()),
          "loader does not search outside explicit root");

    const auto unsafe_bundle = runtime_root / "unsafe.bundle";
    write_bundle(unsafe_bundle, "../fake");
    check(load_throws(unsafe_bundle, runtime_root.string()), "unsafe family id rejected");

    const auto mismatch_bundle = runtime_root / "mismatch.bundle";
    write_bundle(mismatch_bundle, "fake", "embedding");
    check(load_throws(mismatch_bundle, runtime_root.string()),
          "factory task must exactly match bundle task");

    std::filesystem::remove(bundle_path);
    std::filesystem::remove(unsafe_bundle);
    std::filesystem::remove(mismatch_bundle);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
