/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lfm2/plugin_helpers.h"

#include "bundle/bundle_view.h"
#include "runtime/models/lfm2/byte_level_decoder.h"
#include "runtime/models/lfm2/chat_templates.h"
#include "runtime/models/lfm2/pretokenizer.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

class SpecialFrameTokenizer final : public ITokenizer {
  public:
    SpecialFrameTokenizer(std::unique_ptr<ITokenizer> inner, std::vector<int32_t> prefix,
                          std::vector<int32_t> suffix)
        : inner_(std::move(inner)), prefix_(std::move(prefix)), suffix_(std::move(suffix)) {}

    std::vector<int32_t> encode(const std::string& text) const override {
        auto encoded = inner_->encode(text);
        std::vector<int32_t> framed;
        framed.reserve(prefix_.size() + encoded.size() + suffix_.size());
        framed.insert(framed.end(), prefix_.begin(), prefix_.end());
        framed.insert(framed.end(), encoded.begin(), encoded.end());
        framed.insert(framed.end(), suffix_.begin(), suffix_.end());
        return framed;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        return inner_->decode(ids);
    }
    int32_t id_for_token(std::string_view token) const override {
        return inner_->id_for_token(token);
    }
    std::string token_for_id(int32_t id) const override { return inner_->token_for_id(id); }

  private:
    std::unique_ptr<ITokenizer> inner_;
    std::vector<int32_t> prefix_;
    std::vector<int32_t> suffix_;
};

bool tokenizer_add_special_tokens(const BundleFile& bundle) {
    if (bundle.info.tokenizer_add_special_tokens_present)
        return bundle.info.tokenizer_add_special_tokens;
    const auto* section = find_section(bundle, "config.json");
    if (section == nullptr)
        return true;
    const std::string text(section->begin(), section->end());
    return extract_json_bool(text, "tokenizer_add_special_tokens", true);
}

std::pair<std::vector<int32_t>, std::vector<int32_t>>
tokenizer_special_frame(const BundleFile& bundle) {
    const auto* section = find_section(bundle, "config.json");
    if (section == nullptr)
        return {};
    const std::string text(section->begin(), section->end());
    return {extract_json_int_array(text, "tokenizer_special_prefix_ids", 32),
            extract_json_int_array(text, "tokenizer_special_suffix_ids", 32)};
}

} // namespace

std::unique_ptr<ITrtModule> load_lfm2_module(IBackend* backend, const std::vector<char>* plan,
                                             const ModuleCreateOptions& options) {
    if (backend == nullptr)
        throw std::runtime_error("LFM2 runtime has no TensorRT backend");
    if (plan == nullptr || plan->empty())
        throw std::runtime_error("LFM2 bundle is missing engine_plan");
    const auto start = std::chrono::steady_clock::now();
    auto module = backend->create_module(plan->data(), plan->size(), options);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "[trtmc.load_timing] label=\"engine_plan\" load_deserialize_ms=" << elapsed
              << " plan_bytes=" << plan->size() << '\n';
    if (!module || !module->ok())
        throw std::runtime_error("Failed to create LFM2 TensorRT module");
    module->set_timing_label("engine_plan");
    return module;
}

std::shared_ptr<ITokenizer> create_lfm2_tokenizer(const BundleFile& bundle) {
    const auto* section = find_section(bundle, "tokenizer.json");
    if (section == nullptr || section->empty())
        throw std::runtime_error("LFM2 bundle is missing tokenizer.json");

    const bool add_special = tokenizer_add_special_tokens(bundle);
    auto [prefix, suffix] = tokenizer_special_frame(bundle);
    bool has_explicit_frame = !prefix.empty() || !suffix.empty();
    const bool use_pinned_pretokenizer =
        lfm2_uses_pinned_split_byte_level_pretokenizer(section->data(), section->size());
    try {
        auto tokenizer =
            CreateBpeTokenizer(section->data(), section->size(),
                               add_special && !has_explicit_frame && !use_pinned_pretokenizer);
        if (!tokenizer)
            throw std::runtime_error("native BPE tokenizer factory returned null");
        if (lfm2_uses_sequence_byte_level_decoder(section->data(), section->size())) {
            tokenizer = lfm2_wrap_byte_level_decoder(std::move(tokenizer));
        }
        if (use_pinned_pretokenizer) {
            tokenizer = lfm2_wrap_pinned_pretokenizer(
                std::move(tokenizer),
                lfm2_tokenizer_added_tokens(section->data(), section->size()));
            if (add_special && !has_explicit_frame) {
                const int32_t bos = tokenizer->id_for_token("<|startoftext|>");
                if (bos < 0)
                    throw std::runtime_error("pinned LFM2 tokenizer has no BOS token");
                prefix = {bos};
                has_explicit_frame = true;
            }
        }
        std::cerr << "[trtmc] Using native LFM2 BPE tokenizer\n";
        if (add_special && has_explicit_frame) {
            return std::make_shared<SpecialFrameTokenizer>(std::move(tokenizer), std::move(prefix),
                                                           std::move(suffix));
        }
        return std::shared_ptr<ITokenizer>(std::move(tokenizer));
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("LFM2 native BPE tokenizer failed: ") + error.what());
    }
}

DType lfm2_state_dtype(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    throw std::runtime_error("LFM2 native runtime supports fp16 or bf16 state precision");
}

std::string lfm2_expand_layer_name(const std::string& pattern, int32_t layer) {
    std::string result = pattern;
    const std::string token = "{i}";
    const std::string value = std::to_string(layer);
    std::size_t offset = 0;
    while ((offset = result.find(token, offset)) != std::string::npos) {
        result.replace(offset, token.size(), value);
        offset += value.size();
    }
    return result;
}

Lfm2KvCacheNames lfm2_kv_names(const BaseConfig& config, int32_t num_attention_layers) {
    Lfm2KvCacheNames names;
    names.position_id = config.io_map.position_id;
    for (int32_t layer = 0; layer < num_attention_layers; ++layer) {
        names.cache_k.push_back(lfm2_expand_layer_name(config.io_map.cache_k_pattern, layer));
        names.cache_v.push_back(lfm2_expand_layer_name(config.io_map.cache_v_pattern, layer));
        names.present_k.push_back(lfm2_expand_layer_name(config.io_map.present_k_pattern, layer));
        names.present_v.push_back(lfm2_expand_layer_name(config.io_map.present_v_pattern, layer));
    }
    return names;
}

void apply_lfm2_chat_template_format(const BundleFile& bundle, Lfm2TextGenConfig& config) {
    std::string source;
    if (const auto* section = find_section(bundle, "tokenizer_config.json");
        section != nullptr && !section->empty()) {
        const std::string text(section->begin(), section->end());
        source = extract_json_string(text, "chat_template", "");
    }
    if (source.empty()) {
        if (const auto* section = find_section(bundle, "chat_template.jinja");
            section != nullptr && !section->empty()) {
            source.assign(section->begin(), section->end());
        }
    }
    config.chat_template_format = lfm2_detect_chat_template_format(source);
}

} // namespace trtmc
