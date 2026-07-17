/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
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

struct Utf8Unit {
    char32_t codepoint;
    std::size_t size;
};

Utf8Unit decode_utf8_unit(std::string_view text, std::size_t offset) {
    const auto first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80)
        return {first, 1};
    if ((first & 0xE0U) == 0xC0U && offset + 1 < text.size()) {
        return {static_cast<char32_t>((first & 0x1FU) << 6U) |
                    (static_cast<unsigned char>(text[offset + 1]) & 0x3FU),
                2};
    }
    if ((first & 0xF0U) == 0xE0U && offset + 2 < text.size()) {
        return {static_cast<char32_t>((first & 0x0FU) << 12U) |
                    static_cast<char32_t>((static_cast<unsigned char>(text[offset + 1]) & 0x3FU)
                                          << 6U) |
                    (static_cast<unsigned char>(text[offset + 2]) & 0x3FU),
                3};
    }
    if ((first & 0xF8U) == 0xF0U && offset + 3 < text.size()) {
        return {static_cast<char32_t>((first & 0x07U) << 18U) |
                    static_cast<char32_t>((static_cast<unsigned char>(text[offset + 1]) & 0x3FU)
                                          << 12U) |
                    static_cast<char32_t>((static_cast<unsigned char>(text[offset + 2]) & 0x3FU)
                                          << 6U) |
                    (static_cast<unsigned char>(text[offset + 3]) & 0x3FU),
                4};
    }
    return {0xFFFD, 1};
}

bool is_clip_whitespace(char32_t codepoint) {
    static constexpr std::array<char32_t, 13> isolated_whitespace{
        ' ', '\t', '\n', '\r', 0x0B, 0x0C, 0x85, 0xA0, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000};
    return (codepoint >= 0x2000 && codepoint <= 0x200A) ||
           std::find(isolated_whitespace.begin(), isolated_whitespace.end(), codepoint) !=
               isolated_whitespace.end();
}

struct UnicodeRange {
    char32_t first;
    char32_t last;
};

bool is_clip_letter(char32_t codepoint) {
    static constexpr std::array<UnicodeRange, 30> ranges{{
        {'A', 'Z'},         {'a', 'z'},         {0xB5, 0xB5},     {0xC0, 0xD6},
        {0xD8, 0xF6},       {0xF8, 0x2AF},      {0x370, 0x481},   {0x48A, 0x52F},
        {0x531, 0x588},     {0x600, 0x6FF},     {0x900, 0x97F},   {0xE00, 0xE7F},
        {0x10A0, 0x10C5},   {0x13A0, 0x13F5},   {0x1C90, 0x1CBF}, {0x1D00, 0x1D2B},
        {0x1D6B, 0x1D77},   {0x1D79, 0x1D9A},   {0x1E00, 0x1F15}, {0x1F18, 0x1F1D},
        {0x1F20, 0x1F45},   {0x2C00, 0x2C5F},   {0x3040, 0x309F}, {0x30A0, 0x30FF},
        {0x3400, 0x4DBF},   {0x4E00, 0x9FFF},   {0xAC00, 0xD7AF}, {0xFB00, 0xFDFF},
        {0x10000, 0x1007F}, {0x20000, 0x2FA1F},
    }};
    return std::any_of(ranges.begin(), ranges.end(), [codepoint](const auto& range) {
        return codepoint >= range.first && codepoint <= range.last;
    });
}

bool is_clip_number(char32_t codepoint) {
    static constexpr std::array<UnicodeRange, 12> ranges{{
        {'0', '9'},
        {0x660, 0x669},
        {0x6F0, 0x6F9},
        {0x7C0, 0x7C9},
        {0x966, 0x96F},
        {0x9E6, 0x9EF},
        {0xA66, 0xA6F},
        {0xAE6, 0xAEF},
        {0xB66, 0xB6F},
        {0xBE6, 0xBEF},
        {0xC66, 0xC6F},
        {0xFF10, 0xFF19},
    }};
    return std::any_of(ranges.begin(), ranges.end(), [codepoint](const auto& range) {
        return codepoint >= range.first && codepoint <= range.last;
    });
}

std::size_t clip_contraction_size(std::string_view text, std::size_t offset) {
    static constexpr std::array<std::string_view, 7> contractions{"'re", "'ve", "'ll", "'s",
                                                                  "'t",  "'m",  "'d"};
    for (const auto contraction : contractions) {
        if (text.substr(offset, contraction.size()) == contraction)
            return contraction.size();
    }
    return 0;
}

enum class ClipPieceKind { kWhitespace, kLetter, kNumber, kOther };

ClipPieceKind clip_piece_kind(char32_t codepoint) {
    if (is_clip_whitespace(codepoint))
        return ClipPieceKind::kWhitespace;
    if (is_clip_letter(codepoint))
        return ClipPieceKind::kLetter;
    if (is_clip_number(codepoint))
        return ClipPieceKind::kNumber;
    return ClipPieceKind::kOther;
}

std::size_t clip_piece_end(std::string_view text, std::size_t offset, ClipPieceKind kind) {
    auto end = offset + decode_utf8_unit(text, offset).size;
    if (kind == ClipPieceKind::kNumber)
        return end;
    while (end < text.size()) {
        const auto next = decode_utf8_unit(text, end);
        if (clip_piece_kind(next.codepoint) != kind || clip_contraction_size(text, end) > 0)
            break;
        end += next.size;
    }
    return end;
}

char ascii_lower(unsigned char byte) {
    return byte >= 'A' && byte <= 'Z' ? static_cast<char>(byte - 'A' + 'a')
                                      : static_cast<char>(byte);
}

class Sam3ClipTokenizer final : public ITokenizer {
  public:
    explicit Sam3ClipTokenizer(std::shared_ptr<ITokenizer> inner) : mInner(std::move(inner)) {}

    std::vector<int32_t> encode(const std::string& text) const override {
        std::vector<int32_t> ids;
        for (std::size_t offset = 0; offset < text.size();) {
            const auto unit = decode_utf8_unit(text, offset);
            const auto kind = clip_piece_kind(unit.codepoint);
            if (kind == ClipPieceKind::kWhitespace) {
                offset += unit.size;
                continue;
            }

            const auto contraction_size = clip_contraction_size(text, offset);
            if (contraction_size > 0) {
                append_encoded(text.substr(offset, contraction_size), ids);
                offset += contraction_size;
                continue;
            }

            const auto begin = offset;
            offset = clip_piece_end(text, offset, kind);
            auto piece = text.substr(begin, offset - begin);
            std::transform(piece.begin(), piece.end(), piece.begin(), ascii_lower);
            append_encoded(piece, ids);
        }
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const override {
        return mInner->decode(ids);
    }

    int32_t id_for_token(std::string_view token) const override {
        return mInner->id_for_token(token);
    }

    std::string token_for_id(int32_t id) const override { return mInner->token_for_id(id); }

  private:
    void append_encoded(const std::string& piece, std::vector<int32_t>& ids) const {
        const auto encoded = mInner->encode(piece);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
    }

    std::shared_ptr<ITokenizer> mInner;
};

struct TokenizerSpecialFrame {
    bool present{false};
    std::vector<int32_t> prefix;
    std::vector<int32_t> suffix;
};

using TokenizerFactory = std::unique_ptr<ITokenizer> (*)(const char*, std::size_t, bool);

} // namespace

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

std::shared_ptr<ITokenizer> apply_tokenizer_special_frame(std::shared_ptr<ITokenizer> tokenizer,
                                                          const TokenizerSpecialFrame& frame) {
    if (!tokenizer)
        return nullptr;
    if (!frame.present || (frame.prefix.empty() && frame.suffix.empty()))
        return tokenizer;
    return std::make_shared<SpecialFrameTokenizer>(std::move(tokenizer), frame.prefix,
                                                   frame.suffix);
}

TokenizerSpecialFrame detect_requested_tokenizer_special_frame(const BundleFile& bundle,
                                                               bool add_special_tokens) {
    if (!add_special_tokens)
        return TokenizerSpecialFrame{};
    return detect_tokenizer_special_frame(bundle);
}

bool is_sam3_clip_pre_tokenizer(const nlohmann::json& pre_tokenizer) {
    static constexpr std::string_view split_regex =
        R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)";
    if (pre_tokenizer.value("type", "") != "Sequence" || !pre_tokenizer.contains("pretokenizers") ||
        pre_tokenizer["pretokenizers"].size() != 2)
        return false;
    const auto& split = pre_tokenizer["pretokenizers"][0];
    const auto& byte_level = pre_tokenizer["pretokenizers"][1];
    return split.value("type", "") == "Split" && split.value("behavior", "") == "Removed" &&
           split.value("invert", false) && split.contains("pattern") &&
           split["pattern"].value("Regex", "") == split_regex &&
           byte_level.value("type", "") == "ByteLevel" &&
           !byte_level.value("add_prefix_space", true);
}

bool is_sam3_clip_normalizer(const nlohmann::json& normalizer) {
    if (normalizer.value("type", "") != "Sequence" || !normalizer.contains("normalizers") ||
        normalizer["normalizers"].size() != 3)
        return false;
    const auto& parts = normalizer["normalizers"];
    const auto& replace = parts[1];
    return parts[0].value("type", "") == "NFC" && replace.value("type", "") == "Replace" &&
           replace.value("content", "") == " " && replace.contains("pattern") &&
           replace["pattern"].value("Regex", "") == "\\s+" &&
           parts[2].value("type", "") == "Lowercase";
}

bool uses_sam3_clip_tokenizer_contract(const BundleFile& bundle) {
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return false;
    try {
        const auto tokenizer = nlohmann::json::parse(tok_data->begin(), tok_data->end());
        if (!tokenizer.contains("model") || !tokenizer.contains("normalizer") ||
            !tokenizer.contains("pre_tokenizer"))
            return false;
        const auto& model = tokenizer["model"];
        return model.value("type", "") == "BPE" &&
               model.value("end_of_word_suffix", "") == "</w>" &&
               is_sam3_clip_normalizer(tokenizer["normalizer"]) &&
               is_sam3_clip_pre_tokenizer(tokenizer["pre_tokenizer"]);
    } catch (const nlohmann::json::exception&) {
        return false;
    }
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
        std::shared_ptr<ITokenizer> shared(std::move(tok));
        return apply_tokenizer_special_frame(std::move(shared), frame);
    } catch (...) {
        return nullptr;
    }
}

} // namespace

bool is_bpe_tokenizer_json(const BundleFile& bundle) {
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return false;
    // Quick string search — avoid full JSON parse just for type detection
    std::string_view json(tok_data->data(), tok_data->size());
    return json.find("\"type\":\"BPE\"") != std::string_view::npos ||
           json.find("\"type\": \"BPE\"") != std::string_view::npos;
}

std::shared_ptr<ITokenizer> try_create_native_bpe(const BundleFile& bundle, bool add_special,
                                                  bool throw_on_failure) {
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;
    try {
        auto tok = CreateBpeTokenizer(tok_data->data(), tok_data->size(), add_special);
        if (tok) {
            std::cerr << "[trtmc] Using native BPE tokenizer" << std::endl;
        }
        return tok;
    } catch (const std::exception& e) {
        // "Not a BPE tokenizer" -> non-BPE model (WordPiece, Unigram), allow fallback
        std::string msg = e.what();
        bool is_non_bpe = msg.find("Not a BPE") != std::string::npos;

        if (throw_on_failure || (!is_non_bpe && is_bpe_tokenizer_json(bundle))) {
            // BPE model but native failed -> error, no silent fallback
            throw std::runtime_error(std::string("Native BPE tokenizer failed for BPE model: ") +
                                     e.what());
        }
        std::cerr << "[trtmc] Native BPE unavailable (" << e.what()
                  << "), falling back to HF Python" << std::endl;
    }
    return nullptr;
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

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle) {
    const bool add_special = detect_add_special_tokens(bundle);
    // The generic bundle header records what AutoTokenizer.encode() reported
    // at build time.  SAM3's text tower has a stricter, model-owned CLIP
    // contract: an explicit prefix/suffix frame in config.json is
    // authoritative even if that generic header bit is false.  This also
    // makes builds independent of whether Transformers was importable while
    // the bundle was produced.
    const auto special_frame = detect_tokenizer_special_frame(bundle);
    if (!uses_sam3_clip_tokenizer_contract(bundle) || (add_special && !special_frame.present))
        return try_create_native_tokenizer(bundle, add_special);

    auto tokenizer = try_create_native_tokenizer(bundle, /*add_special_tokens=*/false);
    if (!tokenizer)
        return nullptr;
    tokenizer = std::make_shared<Sam3ClipTokenizer>(std::move(tokenizer));
    return apply_tokenizer_special_frame(std::move(tokenizer), special_frame);
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

LoadedModule try_load_trt_module_from_plan(IBackend* backend, const std::vector<char>* plan,
                                           const char* label, const ModuleCreateOptions& options) {
    if (!plan || plan->empty())
        return LoadedModule{};
    try {
        return load_trt_module_from_plan(backend, plan, label, options);
    } catch (...) {
        std::cerr << "[trtmc] WARNING: failed to load optional engine: " << label << std::endl;
        return LoadedModule{};
    }
}

std::unique_ptr<ITrtModule> extract_optional_module(IBackend* backend,
                                                    const std::vector<char>* plan,
                                                    const char* label,
                                                    const ModuleCreateOptions& options) {
    auto loaded = try_load_trt_module_from_plan(backend, plan, label, options);
    if (loaded.module && loaded.module->ok())
        return std::move(loaded.module);
    return nullptr;
}

// Dual-profile module loading (delegated to IBackend).

DualProfileModules load_dual_profile_modules(IBackend* backend, const std::vector<char>* plan,
                                             const char* label,
                                             const ModuleCreateOptions& options) {
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    if (!backend)
        throw std::runtime_error("No backend loaded");

    const auto t0 = SteadyClock::now();
    auto pair = backend->create_dual_profile_modules(plan->data(), plan->size(), options);
    const auto t1 = SteadyClock::now();
    log_trt_load_timing(label, elapsed_ms(t0, t1), plan->size());
    if (!pair.decode || !pair.decode->ok())
        throw std::runtime_error(std::string("Failed to create dual-profile modules for ") + label);

    DualProfileModules out;
    out.prefill = std::move(pair.prefill);
    out.decode = std::move(pair.decode);
    if (out.prefill)
        out.prefill->set_timing_label(std::string(label ? label : "engine") + ":prefill");
    if (out.decode)
        out.decode->set_timing_label(std::string(label ? label : "engine") + ":decode");
    return out;
}

// Config helpers.

int32_t compute_kv_dim(const BaseConfig& cfg) {
    int32_t hd = (cfg.head_dim > 0) ? cfg.head_dim
                                    : ((cfg.num_heads > 0) ? cfg.hidden_size / cfg.num_heads : 128);
    int32_t kv_heads = (cfg.num_kv_heads > 0) ? cfg.num_kv_heads : cfg.num_heads;
    return kv_heads * hd;
}

DType cache_dtype_from_precision(const std::string& precision) {
    if (precision == "fp16")
        return DType::kFloat16;
    if (precision == "bf16")
        return DType::kBFloat16;
    return DType::kFloat32;
}

// Section data conversion.

std::vector<float> section_to_floats(const std::vector<char>* sec) {
    if (!sec || sec->empty())
        return {};
    std::size_t count = sec->size() / sizeof(float);
    std::vector<float> out(count);
    std::memcpy(out.data(), sec->data(), count * sizeof(float));
    return out;
}

std::vector<int32_t> section_to_int32s(const std::vector<char>* sec) {
    if (!sec || sec->empty())
        return {};
    std::size_t count = sec->size() / sizeof(int32_t);
    std::vector<int32_t> out(count);
    std::memcpy(out.data(), sec->data(), count * sizeof(int32_t));
    return out;
}

bool has_section_data(const std::vector<char>* d) {
    return d && !d->empty();
}

MelFilterbank load_mel_filterbank(const BundleFile& bundle) {
    MelFilterbank fb;
    const auto* data = find_section(bundle, "mel_filterbank");
    if (data == nullptr || data->empty())
        return fb;

    // Format: [n_freq_bins(int32), n_mel_bins(int32), float32 data...]
    if (data->size() < 2 * sizeof(int32_t))
        return fb;

    int32_t header[2] = {0, 0};
    std::memcpy(header, data->data(), sizeof(header));
    fb.n_freq_bins = header[0];
    fb.n_mel_bins = header[1];

    if (fb.n_freq_bins <= 0 || fb.n_mel_bins <= 0)
        return fb;

    const auto expected_data_size = static_cast<std::size_t>(fb.n_freq_bins) *
                                    static_cast<std::size_t>(fb.n_mel_bins) * sizeof(float);
    const auto payload_offset = 2 * sizeof(int32_t);
    if (data->size() < payload_offset + expected_data_size) {
        fb.n_freq_bins = 0;
        fb.n_mel_bins = 0;
        return fb;
    }

    fb.data.resize(static_cast<std::size_t>(fb.n_freq_bins) * fb.n_mel_bins);
    std::memcpy(fb.data.data(), data->data() + payload_offset, expected_data_size);
    return fb;
}

std::unique_ptr<ITokenizer> create_clip_tokenizer_from_bundle(const BundleFile& bundle) {
    auto* tok_data = find_section(bundle, "clip_tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;
    try {
        auto tok =
            CreateBpeTokenizer(tok_data->data(), tok_data->size(), /*add_special_tokens=*/true);
        if (tok)
            std::cerr << "[trtmc] Using native BPE CLIP tokenizer" << std::endl;
        return tok;
    } catch (const std::exception& e) {
        std::cerr << "[trtmc] WARNING: CLIP tokenizer failed: " << e.what() << std::endl;
    }
    return nullptr;
}

} // namespace trtmc
