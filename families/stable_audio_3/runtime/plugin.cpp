/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio_3/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::stable_audio_3 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

RuntimeConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    RuntimeConfig config;
    config.sample_rate = json.at("sample_rate").get<std::int32_t>();
    config.audio_channels = json.at("audio_channels").get<std::int32_t>();
    config.latent_dim = json.at("latent_dim").get<std::int32_t>();
    config.prompt_tokens = json.at("prompt_tokens").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    config.context_tokens = json.at("context_tokens").get<std::int32_t>();
    config.cond_dim = json.at("cond_dim").get<std::int32_t>();
    config.time_precision = json.at("time_precision").get<std::string>();
    config.limits.sample_rate = static_cast<double>(config.sample_rate);
    config.limits.downsampling_ratio = json.at("downsampling_ratio").get<std::int64_t>();
    config.limits.latent_align = json.at("latent_align").get<std::int64_t>();
    config.limits.max_latents = json.at("max_latents").get<std::int64_t>();
    config.limits.headroom_seconds = json.at("headroom_seconds").get<double>();
    config.limits.headroom_latents = json.at("headroom_latents").get<std::int64_t>();
    config.shift.anchor_length = json.at("shift_anchor_length").get<float>();
    config.shift.anchor_logsnr = json.at("shift_anchor_logsnr").get<float>();
    config.shift.rate = json.at("shift_rate").get<float>();
    config.shift.logsnr_end = json.at("shift_logsnr_end").get<float>();
    if (config.time_precision != "fp16" && config.time_precision != "fp32")
        throw std::runtime_error("stable_audio_3 runtime.json names an unknown time precision");
    if (config.sample_rate <= 0 || config.audio_channels <= 0 || config.latent_dim <= 0 ||
        config.prompt_tokens <= 0 || config.context_tokens != config.prompt_tokens + 1 ||
        config.cond_dim <= 0 || config.limits.downsampling_ratio <= 0 ||
        config.limits.latent_align <= 0 || config.limits.max_latents <= 0 ||
        config.limits.headroom_latents < 0 || config.shift.anchor_length <= 0.0F)
        throw std::runtime_error("stable_audio_3 runtime.json does not match its contract");
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what, cudaStream_t stream) {
    ModuleCreateOptions options{};
    options.stream = stream;
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("stable_audio_3 " + std::string(what) + " engine failed to load");
    engine->set_timing_label(what);
    return engine;
}

} // namespace
} // namespace trtmc::stable_audio_3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    namespace sa3 = trtmc::stable_audio_3;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("stable_audio_3 does not support --kv-cache-size");
    if (context.reader.info().task != std::string(trtmc::internal::ITextToAudio::kTask))
        throw std::runtime_error("stable_audio_3 does not implement task: " +
                                 context.reader.info().task);
    auto config = sa3::parse_config(sa3::require_section(context.reader, "runtime.json"));
    const auto tokenizer_json = sa3::require_section(context.reader, "tokenizer.json");
    // The tokenizer's post-processor adds no BOS/EOS, matching the reference prompt encoding.
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer)
        throw std::runtime_error("stable_audio_3 could not build its tokenizer");

    auto conditioner =
        sa3::load_engine(context.backend, sa3::require_section(context.reader, "conditioner.plan"),
                         "conditioner", nullptr);
    const auto stream = conditioner->stream();
    auto dit = sa3::load_engine(context.backend, sa3::require_section(context.reader, "dit.plan"),
                                "dit", stream);
    auto decoder = sa3::load_engine(
        context.backend, sa3::require_section(context.reader, "decoder.plan"), "decoder", stream);
    return new sa3::Pipeline(std::move(conditioner), std::move(dit), std::move(decoder),
                             std::move(tokenizer), std::move(config));
}
