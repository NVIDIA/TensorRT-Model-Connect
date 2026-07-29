/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include "bundle/bundle_view.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace trtmc {

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point start, SteadyClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class SpecialFrameTokenizer final : public ITokenizer {
  public:
    SpecialFrameTokenizer(std::shared_ptr<ITokenizer> inner, std::vector<int32_t> prefix,
                          std::vector<int32_t> suffix)
        : mInner(std::move(inner)), mPrefix(std::move(prefix)), mSuffix(std::move(suffix)) {}

    std::vector<int32_t> encode(const std::string& text) const override {
        auto ids = mInner->encode(text);
        std::vector<int32_t> framed;
        framed.reserve(mPrefix.size() + ids.size() + mSuffix.size());
        framed.insert(framed.end(), mPrefix.begin(), mPrefix.end());
        framed.insert(framed.end(), ids.begin(), ids.end());
        framed.insert(framed.end(), mSuffix.begin(), mSuffix.end());
        return framed;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        return mInner->decode(ids);
    }

    int32_t id_for_token(std::string_view token) const override {
        return mInner->id_for_token(token);
    }

    std::string token_for_id(int32_t id) const override { return mInner->token_for_id(id); }

  private:
    std::shared_ptr<ITokenizer> mInner;
    std::vector<int32_t> mPrefix;
    std::vector<int32_t> mSuffix;
};

struct TokenizerSpecialFrame {
    bool present{false};
    std::vector<int32_t> prefix;
    std::vector<int32_t> suffix;
};

using TokenizerFactory = std::unique_ptr<ITokenizer> (*)(const char*, std::size_t, bool);

void log_trt_load_timing(const char* label, double load_deserialize_ms, std::size_t plan_bytes) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.load_timing] label=\""
         << (label ? label : "engine") << "\" load_deserialize_ms=" << load_deserialize_ms
         << " plan_bytes=" << plan_bytes;
    std::cerr << line.str() << '\n';
}

// Tokenizer helpers.

bool detect_add_special_tokens(const BundleFile& bundle) {
    if (bundle.info.tokenizer_add_special_tokens_present)
        return bundle.info.tokenizer_add_special_tokens;

    auto* config_data = find_section(bundle, "config.json");
    if (!config_data)
        return true;
    std::string cfg_text(config_data->begin(), config_data->end());
    auto pos = cfg_text.find("\"tokenizer_add_special_tokens\"");
    if (pos == std::string::npos)
        return true;
    auto val_pos = cfg_text.find(':', pos);
    if (val_pos == std::string::npos)
        return true;
    auto value_pos = cfg_text.find_first_not_of(" \t\r\n", val_pos + 1);
    if (value_pos == std::string::npos)
        return true;
    if (cfg_text.compare(value_pos, 5, "false") == 0 || cfg_text[value_pos] == '0')
        return false;
    if (cfg_text.compare(value_pos, 4, "true") == 0 || cfg_text[value_pos] == '1')
        return true;
    return true;
}

TokenizerSpecialFrame detect_tokenizer_special_frame(const BundleFile& bundle) {
    TokenizerSpecialFrame frame;
    auto* config_data = find_section(bundle, "config.json");
    if (!config_data)
        return frame;
    std::string cfg_text(config_data->begin(), config_data->end());
    const bool has_prefix = cfg_text.find("\"tokenizer_special_prefix_ids\"") != std::string::npos;
    const bool has_suffix = cfg_text.find("\"tokenizer_special_suffix_ids\"") != std::string::npos;
    if (!has_prefix && !has_suffix)
        return frame;

    frame.present = true;
    frame.prefix = extract_json_int_array(cfg_text, "tokenizer_special_prefix_ids");
    frame.suffix = extract_json_int_array(cfg_text, "tokenizer_special_suffix_ids");
    return frame;
}

std::shared_ptr<ITokenizer> apply_tokenizer_special_frame(std::unique_ptr<ITokenizer> tokenizer,
                                                          const TokenizerSpecialFrame& frame) {
    if (!tokenizer)
        return nullptr;
    std::shared_ptr<ITokenizer> shared(std::move(tokenizer));
    if (!frame.present || (frame.prefix.empty() && frame.suffix.empty()))
        return shared;
    return std::make_shared<SpecialFrameTokenizer>(std::move(shared), frame.prefix, frame.suffix);
}

TokenizerSpecialFrame detect_requested_tokenizer_special_frame(const BundleFile& bundle,
                                                               bool add_special_tokens) {
    if (!add_special_tokens)
        return TokenizerSpecialFrame{};
    return detect_tokenizer_special_frame(bundle);
}

std::shared_ptr<ITokenizer> try_create_native_tokenizer_kind(TokenizerFactory factory,
                                                             const char* data, std::size_t size,
                                                             bool add_special_tokens,
                                                             const TokenizerSpecialFrame& frame,
                                                             const char* label) {
    try {
        auto tok = factory(data, size, add_special_tokens);
        if (!tok)
            return nullptr;
        std::cerr << "[trtmc] Using native " << label << " tokenizer" << std::endl;
        return apply_tokenizer_special_frame(std::move(tok), frame);
    } catch (...) {
        return nullptr;
    }
}

std::shared_ptr<ITokenizer> try_create_native_tokenizer(const BundleFile& bundle,
                                                        bool add_special_tokens) {
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;

    const char* data = tok_data->data();
    std::size_t size = tok_data->size();
    const auto special_frame = detect_requested_tokenizer_special_frame(bundle, add_special_tokens);
    const bool native_add_special = !special_frame.present && add_special_tokens;

    if (auto tokenizer = try_create_native_tokenizer_kind(CreateBpeTokenizer, data, size,
                                                          native_add_special, special_frame, "BPE"))
        return tokenizer;

    if (auto tokenizer = try_create_native_tokenizer_kind(
            CreateWordPieceTokenizer, data, size, native_add_special, special_frame, "WordPiece"))
        return tokenizer;

    return try_create_native_tokenizer_kind(CreateUnigramTokenizer, data, size, native_add_special,
                                            special_frame, "Unigram");
}

} // namespace

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle) {
    bool add_special = detect_add_special_tokens(bundle);
    return try_create_native_tokenizer(bundle, add_special);
}

// TRT module loading (delegated to IBackend).

LoadedModule load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                       const char* label, const ModuleCreateOptions& options) {
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    if (!backend)
        throw std::runtime_error("No backend loaded");

    LoadedModule result;
    const auto t0 = SteadyClock::now();
    result.module = backend->create_module(plan->data(), plan->size(), options);
    const auto t1 = SteadyClock::now();
    log_trt_load_timing(label, elapsed_ms(t0, t1), plan->size());
    if (!result.module || !result.module->ok())
        throw std::runtime_error(std::string("Failed to create ITrtModule for ") + label);
    result.module->set_timing_label(label ? label : "engine");
    return result;
}

} // namespace trtmc
