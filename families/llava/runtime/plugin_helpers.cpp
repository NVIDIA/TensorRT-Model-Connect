/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llava/runtime/plugin_helpers.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc {
namespace {

class FramedTokenizer final : public ITokenizer {
  public:
    FramedTokenizer(std::shared_ptr<ITokenizer> tokenizer, std::vector<std::int32_t> prefix,
                    std::vector<std::int32_t> suffix)
        : tokenizer_(std::move(tokenizer)), prefix_(std::move(prefix)), suffix_(std::move(suffix)) {
    }

    std::vector<std::int32_t> encode(const std::string& text) const override {
        auto ids = tokenizer_->encode(text);
        std::vector<std::int32_t> result;
        result.reserve(prefix_.size() + ids.size() + suffix_.size());
        result.insert(result.end(), prefix_.begin(), prefix_.end());
        result.insert(result.end(), ids.begin(), ids.end());
        result.insert(result.end(), suffix_.begin(), suffix_.end());
        return result;
    }
    std::string decode(const std::vector<std::int32_t>& ids) const override {
        return tokenizer_->decode(ids);
    }
    std::int32_t id_for_token(std::string_view token) const override {
        return tokenizer_->id_for_token(token);
    }
    std::string token_for_id(std::int32_t id) const override {
        return tokenizer_->token_for_id(id);
    }

  private:
    std::shared_ptr<ITokenizer> tokenizer_;
    std::vector<std::int32_t> prefix_;
    std::vector<std::int32_t> suffix_;
};

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

std::shared_ptr<ITokenizer> create_tokenizer(const std::vector<char>& data, bool add_special) {
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), add_special);
    if (!tokenizer)
        throw std::runtime_error("tokenizer.json is not BPE");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace

LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options) {
    if (backend == nullptr || plan == nullptr || plan->empty())
        throw std::runtime_error(std::string("bundle is missing ") + label);
    const auto start = std::chrono::steady_clock::now();
    auto module = backend->create_module(plan->data(), plan->size(), options);
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cerr << "[trtmc.load_timing] label=\"" << label << "\" load_deserialize_ms=" << elapsed
              << " plan_bytes=" << plan->size() << '\n';
    if (!module || !module->ok())
        throw std::runtime_error(std::string("failed to load ") + label);
    module->set_timing_label(label);
    return {std::move(module)};
}

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleReader& bundle) {
    const auto& runtime_data = require_section(bundle, "runtime.json");
    const auto runtime = nlohmann::json::parse(runtime_data.begin(), runtime_data.end());
    auto tokenizer = create_tokenizer(require_section(bundle, "tokenizer.json"),
                                      runtime.at("tokenizer_add_special_tokens").get<bool>());
    auto prefix = runtime.at("tokenizer_prefix_ids").get<std::vector<std::int32_t>>();
    auto suffix = runtime.at("tokenizer_suffix_ids").get<std::vector<std::int32_t>>();
    if (prefix.empty() && suffix.empty())
        return tokenizer;
    return std::make_shared<FramedTokenizer>(std::move(tokenizer), std::move(prefix),
                                             std::move(suffix));
}

} // namespace trtmc
