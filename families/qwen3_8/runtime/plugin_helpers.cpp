/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_8/runtime/plugin_helpers.h"

#include <dlfcn.h>
#include <stdexcept>

namespace trtmc::qwen3_8 {

namespace {

constexpr const char* kNativePluginLibrary = "libtrtmc_qwen3_8_native_plugin.so";

void* open_native_plugin_library() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&open_native_plugin_library), &info) != 0 &&
        info.dli_fname != nullptr) {
        std::string path(info.dli_fname);
        const auto slash = path.find_last_of('/');
        path = (slash == std::string::npos ? std::string() : path.substr(0, slash + 1)) +
               kNativePluginLibrary;
        if (void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL))
            return handle;
    }
    return dlopen(kNativePluginLibrary, RTLD_NOW | RTLD_GLOBAL);
}

} // namespace

bool load_native_plugin_library() {
    static void* const handle = open_native_plugin_library(); // never closed: keeps the plugins registered
    return handle != nullptr;
}

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::string require_text_section(const BundleReader& bundle, std::string_view name) {
    const auto& data = require_section(bundle, name);
    return {data.begin(), data.end()};
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* label) {
    auto engine = backend.create_module(plan.data(), plan.size(), {});
    if (engine == nullptr || !engine->ok())
        throw std::runtime_error(
            std::string("qwen3_8 failed to load ") + label +
            (load_native_plugin_library()
                 ? ""
                 : std::string(" (an engine built with the cuTile GDN plugins needs ") +
                       kNativePluginLibrary + " next to the runtime library)"));
    engine->set_timing_label(label);
    return engine;
}

std::shared_ptr<ITokenizer> create_tokenizer(const BundleReader& bundle) {
    const auto& data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    if (tokenizer == nullptr)
        throw std::runtime_error("qwen3_8 BPE tokenizer construction failed");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace trtmc::qwen3_8
