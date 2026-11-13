/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/parakeet_tdt/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::parakeet_tdt {
namespace {
std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (!section || section->length == 0)
        throw std::runtime_error(std::string("Parakeet TDT missing section: ") + name);
    return reader.read_section(name);
}
std::unique_ptr<ITrtModule> load_engine(const FamilyContext& context, const char* name) {
    auto plan = require_section(context.reader, name);
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error(std::string("Parakeet TDT failed to load ") + name);
    return module;
}
} // namespace
} // namespace trtmc::parakeet_tdt

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    using namespace trtmc::parakeet_tdt;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("parakeet_tdt does not support --kv-cache-size");
    if (context.reader.info().family != "parakeet_tdt" ||
        context.reader.info().task != "speech_transcription" ||
        context.reader.info().backend != "trt" || std::string(context.backend.name()) != "trt")
        throw std::invalid_argument("Parakeet TDT bundle identity does not match runtime");
    const auto data = require_section(context.reader, "runtime.json");
    const auto raw = nlohmann::json::parse(data.begin(), data.end());
    TdtConfig config;
    const std::pair<const char*, int> dimensions[] = {
        {"tensor_parallel_size", 1},
        {"mel_sampling_rate", 16000},
        {"num_mel_bins", 128},
        {"mel_n_fft", 512},
        {"mel_win_length", 400},
        {"mel_hop_length", 160},
        {"mel_chunk_length", 30},
        {"mel_length", 3000},
        {"tdt_encoder_hidden_size", 1024},
        {"tdt_pred_hidden_size", 640},
        {"tdt_pred_num_layers", 2},
        {"tdt_vocab_size", 8192},
        {"tdt_blank_id", 8192},
        {"tdt_encoder_layers", 24},
        {"max_source_positions", 375},
        {"subsampling_factor", 8},
        {"tdt_max_symbols_per_step", 10},
        {"tdt_att_context_left", -1},
        {"tdt_att_context_right", -1},
    };
    for (const auto& [name, expected] : dimensions) {
        if (!raw.at(name).is_number_integer() || raw.at(name) != expected)
            throw std::invalid_argument(std::string("unsupported Parakeet TDT dimension: ") + name);
    }
    if (raw.at("tdt_duration_values") != nlohmann::json({0, 1, 2, 3, 4}) ||
        raw.at("tdt_causal_downsampling") != false || raw.at("mel_normalize") != "per_feature" ||
        std::abs(raw.at("mel_preemph").get<double>() - 0.97) > 1e-6)
        throw std::invalid_argument("unsupported Parakeet TDT frontend or duration policy");

    const auto mel = require_section(context.reader, "mel_filterbank");
    constexpr size_t values = 257 * 128;
    if (mel.size() != 8 + values * sizeof(float))
        throw std::invalid_argument("invalid Parakeet TDT mel filterbank size");
    MelFilterbank filters;
    std::memcpy(&filters.n_freq_bins, mel.data(), 4);
    std::memcpy(&filters.n_mel_bins, mel.data() + 4, 4);
    validate_tdt_mel_geometry(config, filters.n_freq_bins, filters.n_mel_bins);
    filters.data.resize(values);
    std::memcpy(filters.data.data(), mel.data() + 8, values * sizeof(float));
    for (float value : filters.data)
        if (!std::isfinite(value))
            throw std::invalid_argument("nonfinite Parakeet mel coefficient");
    const auto tokenizer_data = require_section(context.reader, "tokenizer.json");
    const auto tokenizer_json = nlohmann::json::parse(tokenizer_data.begin(), tokenizer_data.end());
    if (tokenizer_json.at("model").at("type") != "BPE" ||
        tokenizer_json.at("decoder").at("type") != "Metaspace" ||
        tokenizer_json.at("decoder").at("replacement") != "\xe2\x96\x81" ||
        tokenizer_json.at("decoder").at("prepend_scheme") != "always")
        throw std::invalid_argument(
            "Parakeet TDT requires its native BPE tokenizer with Metaspace decoding");
    auto tokenizer = CreateBpeTokenizer(tokenizer_data.data(), tokenizer_data.size(), false);
    if (!tokenizer)
        throw std::invalid_argument("Parakeet TDT requires its native BPE tokenizer");
    auto encoder = load_engine(context, "encoder.plan");
    auto predictor = load_engine(context, "predictor.plan");
    auto joint = load_engine(context, "joint.plan");
    return new TdtPipeline(std::move(encoder), std::move(predictor), std::move(joint),
                           std::move(config), std::move(filters), std::move(tokenizer));
}
