/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_loader.h"

#include <cstdint>
#include <dlfcn.h>
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
                  const std::string& task = "time_series_forecast",
                  const std::string& backend = "fake") {
    const std::string header = "{\"format\":1,\"family\":\"" + family + "\",\"task\":\"" + task +
                               "\",\"backend\":\"" + backend +
                               "\","
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

std::string load_error(const std::filesystem::path& bundle, const std::string& runtime_root) {
    try {
        (void)trtmc::load_task(bundle.string(), runtime_root);
        return {};
    } catch (const std::exception& error) {
        return error.what();
    }
}

bool load_throws(const std::filesystem::path& bundle, const std::string& runtime_root) {
    return !load_error(bundle, runtime_root).empty();
}

bool rtx_options_throw(const std::filesystem::path& bundle, const std::string& runtime_root) {
    try {
        (void)trtmc::load_task(bundle.string(), runtime_root, 0, "runtime.cache", true);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

void check_rtx_options(const std::filesystem::path& runtime_root,
                       const std::string& expected_cache_path, bool expected_cuda_graphs) {
    const auto library_path = runtime_root / "libtrtmc_backend_trt_rtx.so";
    void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    check(handle != nullptr, "fake RTX backend remains loaded");
    if (handle == nullptr)
        return;

    using CachePathFn = const char* (*)();
    using CudaGraphsFn = bool (*)();
    const auto cache_path =
        reinterpret_cast<CachePathFn>(dlsym(handle, "trtmc_test_backend_last_runtime_cache_path"));
    const auto cuda_graphs =
        reinterpret_cast<CudaGraphsFn>(dlsym(handle, "trtmc_test_backend_last_cuda_graphs"));
    check(cache_path != nullptr, "fake RTX cache-path probe is exported");
    check(cuda_graphs != nullptr, "fake RTX CUDA-graphs probe is exported");
    if (cache_path != nullptr)
        check(expected_cache_path == cache_path(), "runtime cache path remains owned by value");
    if (cuda_graphs != nullptr)
        check(cuda_graphs() == expected_cuda_graphs,
              "CUDA-graphs option reaches delayed module creation");
    dlclose(handle);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "usage: test_family_loader RUNTIME_ROOT [--expect-core-mismatch]\n";
        return 2;
    }
    const std::filesystem::path runtime_root(argv[1]);
    const auto bundle_path = runtime_root / "fake.bundle";
    write_bundle(bundle_path, "fake");

    if (argc == 3) {
        const std::string core_error = load_error(bundle_path, runtime_root.string());
        check(std::string(argv[2]) == "--expect-core-mismatch" &&
                  core_error.find("Active libtrtmc_core.so belongs to product build") !=
                      std::string::npos,
              "runtime rejects a core DSO from a different product build");
        std::filesystem::remove(bundle_path);
        return failures;
    }

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

    const auto rtx_bundle = runtime_root / "fake-rtx.bundle";
    write_bundle(rtx_bundle, "fake", "time_series_forecast", "trt_rtx");
    const std::string expected_cache_path = (runtime_root / std::string(256, 'r')).string();
    std::unique_ptr<trtmc::ITask> rtx_task;
    {
        std::string caller_owned_cache_path = expected_cache_path;
        rtx_task = trtmc::load_task(rtx_bundle.string(), runtime_root.string(), 0,
                                    caller_owned_cache_path, true);
    }
    auto* rtx_forecast = dynamic_cast<trtmc::ITimeSeriesForecast*>(rtx_task.get());
    check(rtx_forecast != nullptr, "RTX load returns forecast interface");
    if (rtx_forecast != nullptr)
        (void)rtx_forecast->forecast({values, mask});
    check_rtx_options(runtime_root, expected_cache_path, true);

    const std::string second_cache_path = (runtime_root / std::string(256, 's')).string();
    auto second_rtx_task =
        trtmc::load_task(rtx_bundle.string(), runtime_root.string(), 0, second_cache_path, false);
    auto* second_rtx_forecast = dynamic_cast<trtmc::ITimeSeriesForecast*>(second_rtx_task.get());
    check(second_rtx_forecast != nullptr, "second RTX load returns forecast interface");
    if (second_rtx_forecast != nullptr)
        (void)second_rtx_forecast->forecast({values, mask});
    check_rtx_options(runtime_root, second_cache_path, false);
    if (rtx_forecast != nullptr)
        (void)rtx_forecast->forecast({values, mask});
    check_rtx_options(runtime_root, expected_cache_path, true);

    check(load_throws(bundle_path, ""), "empty runtime root rejected");
    check(load_throws(bundle_path, (runtime_root / "missing").string()),
          "loader does not search outside explicit root");

    const auto wrong_backend_library = runtime_root / "libtrtmc_backend_other.so";
    const auto wrong_backend_bundle = runtime_root / "wrong-backend.bundle";
    std::filesystem::copy_file(runtime_root / "libtrtmc_backend_fake.so", wrong_backend_library,
                               std::filesystem::copy_options::overwrite_existing);
    write_bundle(wrong_backend_bundle, "fake", "time_series_forecast", "other");
    const std::string backend_descriptor_error =
        load_error(wrong_backend_bundle, runtime_root.string());
    check(backend_descriptor_error.find("declares backend 'fake'; expected 'other'") !=
              std::string::npos,
          "explicit root validates backend plugin identity before its factory");

    const auto incompatible_bundle = runtime_root / "incompatible.bundle";
    write_bundle(incompatible_bundle, "fake", "time_series_forecast", "incompatible");
    const std::string incompatible_build_error =
        load_error(incompatible_bundle, runtime_root.string());
    check(incompatible_build_error.find("belongs to product build") != std::string::npos,
          "loader rejects a plugin from a different product build before its factory");

    const auto escaped_root = runtime_root.parent_path() / "escaped-runtime-root";
    std::filesystem::remove_all(escaped_root);
    std::filesystem::create_directories(escaped_root);
    std::filesystem::create_symlink(runtime_root / "libtrtmc_backend_fake.so",
                                    escaped_root / "libtrtmc_backend_fake.so");
    std::filesystem::create_symlink(runtime_root / "libtrtmc_model_fake.so",
                                    escaped_root / "libtrtmc_model_fake.so");
    const std::string escaped_root_error = load_error(bundle_path, escaped_root.string());
    check(escaped_root_error.find("escapes the selected runtime root") != std::string::npos,
          "explicit loading rejects a DSO symlink that escapes the selected root");
    std::filesystem::remove_all(escaped_root);

    const auto wrong_family_library = runtime_root / "libtrtmc_model_other.so";
    const auto wrong_family_bundle = runtime_root / "wrong-family.bundle";
    std::filesystem::copy_file(runtime_root / "libtrtmc_model_fake.so", wrong_family_library,
                               std::filesystem::copy_options::overwrite_existing);
    write_bundle(wrong_family_bundle, "other");
    const std::string family_descriptor_error =
        load_error(wrong_family_bundle, runtime_root.string());
    check(family_descriptor_error.find("declares family 'fake'; expected 'other'") !=
              std::string::npos,
          "explicit root validates family plugin identity before its factory");

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
    std::filesystem::remove(rtx_bundle);
    std::filesystem::remove(wrong_backend_bundle);
    std::filesystem::remove(wrong_backend_library);
    std::filesystem::remove(incompatible_bundle);
    std::filesystem::remove(wrong_family_bundle);
    std::filesystem::remove(wrong_family_library);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
