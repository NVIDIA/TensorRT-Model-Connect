/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include "utils/json_helpers.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace trtmc {

namespace {

struct SequenceProfile {
    int32_t min{0};
    int32_t opt{0};
    int32_t max{0};
};

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

int32_t profile_dim(const TrtModule& module, const std::string& tensor_name,
                    ProfileShapeSelector selector) {
    const auto shape = module.input_profile_shape(tensor_name, 0, selector);
    if (shape.size() != 1 || shape.front() <= 0 ||
        shape.front() > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("InternLM native sequence input '" + tensor_name +
                                 "' must have a positive rank-1 optimization profile");
    }
    return static_cast<int32_t>(shape.front());
}

SequenceProfile sequence_profile(const TrtModule& module, const std::string& tensor_name) {
    if (!module.has_input(tensor_name))
        throw std::runtime_error("InternLM native engine is missing sequence input '" +
                                 tensor_name + "'");
    return SequenceProfile{
        profile_dim(module, tensor_name, ProfileShapeSelector::kMin),
        profile_dim(module, tensor_name, ProfileShapeSelector::kOpt),
        profile_dim(module, tensor_name, ProfileShapeSelector::kMax),
    };
}

void validate_matching_profiles(const SequenceProfile& tokens, const SequenceProfile& positions) {
    if (tokens.min != positions.min || tokens.opt != positions.opt || tokens.max != positions.max) {
        throw std::runtime_error("InternLM native token_id and position_id profiles must match");
    }
}

void validate_decode_profile(const SequenceProfile& profile) {
    if (profile.min != 1 || profile.opt != 1 || profile.max != 1)
        throw std::runtime_error("InternLM decode profile must have min=opt=max=1");
}

void validate_prefill_profile(const SequenceProfile& profile, int32_t capacity) {
    if (profile.min != 1 || profile.opt <= 1 || profile.opt > profile.max ||
        profile.max > capacity) {
        throw std::runtime_error(
            "InternLM prefill profile must satisfy min=1 < opt <= max <= KV capacity");
    }
}

} // namespace

int32_t validate_internlm_native_sequence_profile(const TrtModule& module,
                                                  const std::string& token_name,
                                                  const std::string& position_name,
                                                  int32_t capacity, InternlmEngineRole role) {
    if (module.optimization_profile_count() != 1)
        throw std::runtime_error("InternLM native split engines must contain exactly one profile");
    if (module.profile_idx() != 0)
        throw std::runtime_error("InternLM native split engines must use profile index 0");

    const auto tokens = sequence_profile(module, token_name);
    const auto positions = sequence_profile(module, position_name);
    validate_matching_profiles(tokens, positions);
    if (role == InternlmEngineRole::kDecode)
        validate_decode_profile(tokens);
    else
        validate_prefill_profile(tokens, capacity);
    return tokens.max;
}

void log_trt_load_timing(const char* label, double load_deserialize_ms, std::size_t plan_bytes) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.load_timing] label=\""
         << (label ? label : "engine") << "\" load_deserialize_ms=" << load_deserialize_ms
         << " plan_bytes=" << plan_bytes;
    std::cerr << line.str() << '\n';
}

// Tokenizer helpers.

static bool detect_add_special_tokens(const BundleFile& bundle) {
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

namespace {

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

} // namespace

static std::shared_ptr<ITokenizer> try_create_native_tokenizer(const BundleFile& bundle,
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

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle) {
    bool add_special = detect_add_special_tokens(bundle);
    return try_create_native_tokenizer(bundle, add_special);
}

DType cache_dtype_from_precision(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
}

} // namespace trtmc
