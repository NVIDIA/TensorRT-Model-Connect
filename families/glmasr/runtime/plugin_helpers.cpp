/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glmasr/runtime/plugin_helpers.h"

#include <cstring>
#include <stdexcept>
#include <string>
namespace trtmc {
LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options) {
    if (!backend || !plan || plan->empty())
        throw std::runtime_error(std::string("missing ") + label);
    auto module = backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("failed to load ") + label);
    return {std::move(module)};
}
DualProfileModules load_dual_profile_modules(IBackend* backend, const std::vector<char>* plan,
                                             const char* label,
                                             const ModuleCreateOptions& options) {
    auto one = load_trt_module_from_plan(backend, plan, label, options);
    return {nullptr, std::move(one.module)};
}
std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleReader& bundle) {
    const auto data = bundle.read_section("tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("tokenizer.json is not BPE");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}
MelFilterbank load_mel_filterbank(const BundleReader& bundle) {
    const auto data = bundle.read_section("mel_filterbank");
    if (data.size() < 2 * sizeof(std::int32_t))
        throw std::runtime_error("invalid mel filterbank");
    MelFilterbank result;
    std::memcpy(&result.n_freq_bins, data.data(), sizeof(std::int32_t));
    std::memcpy(&result.n_mel_bins, data.data() + sizeof(std::int32_t), sizeof(std::int32_t));
    const auto count = static_cast<std::size_t>(result.n_freq_bins) * result.n_mel_bins;
    if (result.n_freq_bins <= 0 || result.n_mel_bins <= 0 ||
        data.size() != 2 * sizeof(std::int32_t) + count * sizeof(float))
        throw std::runtime_error("invalid mel filterbank dimensions");
    result.data.resize(count);
    std::memcpy(result.data.data(), data.data() + 2 * sizeof(std::int32_t), count * sizeof(float));
    return result;
}
void load_ffi_kernels_from_bundle(const BundleReader&) {}
} // namespace trtmc
