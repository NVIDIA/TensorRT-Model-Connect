/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <cctype>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#if TRTMC_HAS_TVM_FFI
#include "plugins/tvm_ffi_module_loader.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#endif

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

// ─── FFI kernel loading ───

std::string sanitize_kernel_filename_component(const std::string& global_name) {
    if (global_name.empty())
        throw std::invalid_argument("Kernel global name must not be empty");
    std::string safe_name;
    safe_name.reserve(global_name.size());
    for (const char c : global_name) {
        const bool allowed =
            std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '-';
        safe_name.push_back(allowed ? c : '_');
    }
    return safe_name;
}

#if TRTMC_HAS_TVM_FFI

namespace {

// Write a bundle section to a temporary .so file, returning the path.
std::string write_kernel_so_to_temp(const std::string& global_name, const char* data,
                                    std::size_t size) {
    const std::string safe_name = sanitize_kernel_filename_component(global_name).substr(0, 64);
    std::string pattern = "/tmp/trtmc_kernel_" + safe_name + "_XXXXXX.so";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int fd = ::mkstemps(writable.data(), 3);
    if (fd < 0)
        throw std::runtime_error("Failed to create a unique kernel temp file");

    std::size_t written = 0;
    while (written < size) {
        const auto count = ::write(fd, data + written, size - written);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            ::close(fd);
            std::remove(writable.data());
            throw std::runtime_error("Failed to write the complete kernel temp file");
        }
        written += static_cast<std::size_t>(count);
    }
    if (::close(fd) != 0) {
        std::remove(writable.data());
        throw std::runtime_error("Failed to close kernel temp file");
    }
    return writable.data();
}

// Load a single kernel entry from the manifest and register it via TVM-FFI.
void load_single_kernel(const BundleFile& bundle, const std::string& obj) {
    std::string global_name = extract_json_string(obj, "global_name", "");
    std::string func_name = extract_json_string(obj, "func_name", "run");
    std::string section_name = extract_json_string(obj, "section", "");

    if (global_name.empty() || section_name.empty())
        return;

    const auto* so_sec = find_section(bundle, section_name);
    if (!so_sec || so_sec->empty()) {
        std::cerr << "[ffi] Kernel .so section not found: " << section_name << '\n';
        return;
    }

    std::string tmp_path = write_kernel_so_to_temp(global_name, so_sec->data(), so_sec->size());
    bool loaded = false;
    try {
        loaded = load_tvm_ffi_module_func(tmp_path, func_name, global_name);
    } catch (...) {
        std::remove(tmp_path.c_str());
        throw;
    }
    std::remove(tmp_path.c_str());
    if (loaded) {
        std::cerr << "[ffi] Loaded kernel: " << global_name << '\n';
    } else {
        std::cerr << "[ffi] Failed to load kernel: " << global_name << " from " << section_name
                  << '\n';
    }
}

// Find the "kernels" JSON array bounds within the manifest string.
// Returns {start_after_bracket, closing_bracket} or {npos, npos}.
std::pair<std::size_t, std::size_t> find_kernels_array_bounds(const std::string& s) {
    auto pos = s.find("\"kernels\"");
    if (pos == std::string::npos)
        return {std::string::npos, std::string::npos};
    auto arr_start = s.find('[', pos);
    if (arr_start == std::string::npos)
        return {std::string::npos, std::string::npos};
    auto arr_end = s.find(']', arr_start);
    return {arr_start + 1, arr_end};
}

} // namespace

#endif // TRTMC_HAS_TVM_FFI

void load_ffi_kernels_from_bundle(const BundleFile& bundle) {
#if TRTMC_HAS_TVM_FFI
    const auto* manifest_sec = find_section(bundle, "kernel_manifest.json");
    if (!manifest_sec)
        return;

    std::string manifest_str(manifest_sec->begin(), manifest_sec->end());
    auto [cur, arr_end] = find_kernels_array_bounds(manifest_str);
    if (cur == std::string::npos || arr_end == std::string::npos)
        return;

    while (cur < arr_end) {
        auto obj_start = manifest_str.find('{', cur);
        if (obj_start == std::string::npos || obj_start >= arr_end)
            break;
        auto obj_end = manifest_str.find('}', obj_start);
        if (obj_end == std::string::npos)
            break;

        load_single_kernel(bundle, manifest_str.substr(obj_start, obj_end - obj_start + 1));
        cur = obj_end + 1;
    }
#else
    (void)bundle;
#endif
}

} // namespace trtmc
