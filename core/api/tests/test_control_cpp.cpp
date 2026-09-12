/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.hpp"

#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void write_bundle(const std::filesystem::path& path, const std::string& mode,
                  const std::string& backend = "fake") {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = "{\"format\":1,\"family\":\"control_fixture\",\"task\":\"" + mode +
                               "\",\"backend\":\"" + backend +
                               "\",\"sections\":{\"payload\":{\"offset\":0,\"length\":3}}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("A\0B", 3);
}

template <class Invoke>
void rejects(Invoke&& invoke, trtmc_status expected, const char* label) {
    bool rejected = false;
    try {
        invoke();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == expected && error.what()[0] != '\0';
    }
    check(rejected, label);
}

void offline(const std::filesystem::path& root) {
    const auto path = root / "cpp-control-offline.bundle";
    write_bundle(path, "inspect", "not_installed");
    trtmc::BundleMetadata metadata;
    auto bytes = [&] {
        auto bundle = trtmc::Bundle::open(path.string());
        metadata = bundle.info();
        return bundle.read_section("payload");
    }();
    check(metadata.format == 1 && metadata.backend == "not_installed" &&
              metadata.sections.size() == 1 && metadata.sections[0].name == "payload" &&
              metadata.sections[0].length == 3,
          "owned bundle metadata survives closing an offline bundle");
    const auto view = bytes.bytes();
    check(view.size() == 3 && view[0] == 'A' && view[1] == 0 && view[2] == 'B',
          "byte result remains valid after bundle close");
    auto moved = std::move(bytes);
    check(bytes.bytes().empty() && moved.bytes().size() == 3,
          "byte result move transfers ownership");
    auto replacement = trtmc::Bundle::open(path.string()).read_section("payload");
    replacement = std::move(moved);
    check(moved.bytes().empty() && replacement.bytes()[2] == 'B',
          "byte result move assignment works");
}

void lora(const std::filesystem::path& root) {
    const auto path = root / "cpp-control-enabled.bundle";
    const auto disabled = root / "cpp-control-disabled.bundle";
    const auto weights = root / "cpp-control-weights.txt";
    write_bundle(path, "enabled");
    write_bundle(disabled, "disabled");
    std::ofstream(weights) << "cpp-weights";
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    auto owned = [&] {
        auto model = trtmc::Model::load(path.string(), options);
        return std::make_pair(model.lora_adapters(), model.task<trtmc::TextContinuation>());
    }();
    owned.first.load("owned", weights.string());
    auto snapshot = owned.first.list();
    check(snapshot == std::vector<std::string>{"owned"},
          "manager retains its model for load and list");
    auto result = owned.second.run({"Hello"}, {{"lora_adapter_id", "owned"}});
    check(result.text() == "Hello|cpp-weights", "selected adapter is a per-request family option");
    check(owned.second.run({"Hello"}).text() == "Hello", "next request can select the base model");
    owned.first.unload("owned");
    check(owned.first.list().empty() && snapshot == std::vector<std::string>{"owned"},
          "copied inventory remains stable after adapter removal");
    rejects([&] { (void)owned.second.run({"Hello"}, {{"lora_adapter_id", "owned"}}); },
            TRTMC_INVALID_CONFIG, "unloaded adapter cannot be selected");
    rejects([&] { (void)trtmc::Model::load(disabled.string(), options).lora_adapters(); },
            TRTMC_UNSUPPORTED, "wrapper respects family-owned LoRA availability");
    trtmc::load_byok_kernel(weights.string(), "run", "fixture.copy", root.string());
    rejects(
        [&] { trtmc::load_byok_kernel(weights.string(), "absent", "fixture.copy", root.string()); },
        TRTMC_INTERNAL_ERROR, "BYOK extension errors become owned C++ errors");
}

void runtime_options(const std::filesystem::path& root) {
    const auto path = root / "cpp-control-rtx.bundle";
    write_bundle(path, "enabled", "trt_rtx");
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    options.runtime_cache_path = (root / "cache-original").string();
    options.cuda_graphs = true;
    const std::string expected = options.runtime_cache_path;
    auto model = trtmc::Model::load(path.string(), options);
    options.runtime_cache_path.assign("caller-mutated");
    options.cuda_graphs = false;
    (void)model.task<trtmc::TextContinuation>().run({"deferred"});
    void* backend = dlopen((root / "libtrtmc_backend_trt_rtx.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    check(backend != nullptr, "fake RTX backend is available without GPU inference");
    if (backend) {
        const auto cache = reinterpret_cast<const char* (*)()>(
            dlsym(backend, "trtmc_test_backend_last_runtime_cache_path"));
        const auto graphs =
            reinterpret_cast<bool (*)()>(dlsym(backend, "trtmc_test_backend_last_cuda_graphs"));
        check(cache && graphs && expected == cache() && graphs(),
              "runtime options remain owned until delayed module creation");
        dlclose(backend);
    }
    const auto plain = root / "cpp-control-options-reject.bundle";
    write_bundle(plain, "enabled");
    rejects([&] { (void)trtmc::Model::load(plain.string(), options); }, TRTMC_INVALID_ARGUMENT,
            "runtime cache is still rejected for non-RTX bundles");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        offline(root);
        lora(root);
        runtime_options(root);
    } catch (const std::exception& error) {
        std::cerr << "Unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
