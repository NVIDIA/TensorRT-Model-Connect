/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugin_helpers.h"

#include "trtmc/runtime/trt_backend.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#if TRTMC_HAS_TVM_FFI
#include "plugins/tvm_ffi_module_loader.h"
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

int32_t cache_rows_from_shape(const std::vector<int64_t>& shape) {
    const std::size_t row_index = shape.size() == 4 ? 2 : 0;
    if (shape.empty() || row_index >= shape.size())
        return -1;
    const int64_t rows = shape[row_index];
    if (rows <= 0 || rows > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(rows);
}

} // namespace

void log_trt_load_timing(const char* label, double load_deserialize_ms, std::size_t plan_bytes) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(6) << "[trtmc.load_timing] label=\""
         << (label ? label : "engine") << "\" load_deserialize_ms=" << load_deserialize_ms
         << " plan_bytes=" << plan_bytes;
    std::cerr << line.str() << '\n';
}

std::vector<int32_t> select_decoder_profile_rows(const std::vector<int32_t>& ordered_profile_rows,
                                                 int32_t runtime_rows) {
    if (runtime_rows <= 0 || ordered_profile_rows.empty()) {
        throw std::invalid_argument(
            "Decoder profile selection requires positive runtime rows and profiles");
    }
    int32_t previous_rows = 0;
    for (const int32_t profile_rows : ordered_profile_rows) {
        if (profile_rows <= 0 || profile_rows < previous_rows) {
            throw std::invalid_argument("Decoder KV profile rows must be positive and ordered");
        }
        previous_rows = profile_rows;
    }

    std::vector<int32_t> selected;
    selected.reserve(ordered_profile_rows.size());
    for (const int32_t profile_rows : ordered_profile_rows) {
        selected.push_back(profile_rows);
        if (profile_rows >= runtime_rows)
            break;
    }
    if (selected.back() < runtime_rows) {
        throw std::runtime_error("No decoder KV profile can cover the selected runtime capacity");
    }
    return selected;
}

bool cache_input_supports_runtime_rows(const TrtModule& module, const std::string& tensor_name) {
    if (!module.input_is_dynamic(tensor_name))
        return false;
    const int32_t num_profiles = module.optimization_profile_count();
    for (int32_t profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
        const auto min_shape =
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMin);
        const auto max_shape =
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax);
        if (!min_shape.empty() && !max_shape.empty() && min_shape.front() > 0 &&
            max_shape.front() > min_shape.front()) {
            return true;
        }
    }
    return false;
}

int32_t decoder_profile_cache_rows(const TrtModule& module, const std::string& tensor_name,
                                   int32_t profile_idx, int32_t fallback_rows) {
    if (!module.input_is_dynamic(tensor_name)) {
        const int32_t static_rows = cache_rows_from_shape(module.tensor_shape(tensor_name));
        if (static_rows > 0)
            return static_rows;
    }
    if (profile_idx >= 0 && profile_idx < module.optimization_profile_count()) {
        const int32_t max_rows = cache_rows_from_shape(
            module.input_profile_shape(tensor_name, profile_idx, ProfileShapeSelector::kMax));
        if (max_rows > 0)
            return max_rows;
    }
    return fallback_rows;
}

DecoderProfileRoles detect_decoder_profile_roles(const TrtModule& module,
                                                 const std::string& token_id_name,
                                                 const std::string& cache_k_name,
                                                 int32_t fallback_rows) {
    const auto token_max_length = [&](int32_t profile_idx) -> int32_t {
        const auto shape =
            module.input_profile_shape(token_id_name, profile_idx, ProfileShapeSelector::kMax);
        if (shape.empty() || shape.front() <= 0 ||
            shape.front() > std::numeric_limits<int32_t>::max()) {
            return -1;
        }
        return static_cast<int32_t>(shape.front());
    };

    DecoderProfileRoles roles;
    const int32_t num_profiles = module.optimization_profile_count();
    if (num_profiles <= 0) {
        roles.decode_profiles.push_back(DecoderProfileInfo{0, fallback_rows});
        return roles;
    }

    for (int32_t profile_idx = 0; profile_idx < num_profiles; ++profile_idx) {
        const int32_t token_max = token_max_length(profile_idx);
        if (token_max > 1) {
            if (token_max > roles.prefill_max_length) {
                roles.prefill_profile_idx = profile_idx;
                roles.prefill_max_length = token_max;
            }
            continue;
        }
        roles.decode_profiles.push_back(DecoderProfileInfo{
            profile_idx,
            decoder_profile_cache_rows(module, cache_k_name, profile_idx, fallback_rows)});
    }

    if (roles.decode_profiles.empty()) {
        const int32_t fallback_profile =
            roles.prefill_profile_idx >= 0 ? roles.prefill_profile_idx : 0;
        roles.decode_profiles.push_back(DecoderProfileInfo{
            fallback_profile,
            decoder_profile_cache_rows(module, cache_k_name, fallback_profile, fallback_rows)});
    }
    return roles;
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
                  << "); runtime tokenization remains native-only" << std::endl;
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

// ─── FFI kernel loading ───

#if TRTMC_HAS_TVM_FFI

namespace {

// Write a bundle section to a temporary .so file, returning the path.
std::string write_kernel_so_to_temp(const std::string& global_name, const char* data,
                                    std::size_t size) {
    std::string safe_name = global_name;
    for (auto& c : safe_name) {
        if (c == '.')
            c = '_';
    }
    std::string tmp_path = "/tmp/trtmc_kernel_" + safe_name + ".so";
    std::ofstream ofs(tmp_path, std::ios::binary);
    ofs.write(data, static_cast<std::streamsize>(size));
    return tmp_path;
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
    if (load_tvm_ffi_module_func(tmp_path, func_name, global_name)) {
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
