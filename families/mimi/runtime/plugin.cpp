/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/audio.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::mimi {
namespace {
class Pipeline final : public internal::IModel, public internal::IAudioCodebookTokensToAudio {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, nlohmann::json cfg)
        : engine_(std::move(engine)), cfg_(std::move(cfg)) {}
    const char* task() const noexcept override { return "audio_codebook_tokens_to_audio"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IAudioCodebookTokensToAudio>(*this)};
    }
    internal::AudioResult run(const internal::AudioCodebookTokensToAudioRequest& request,
                              internal::ConfigView config) override {
        internal::validate_config({}, config);
        const auto& input = request.codes;
        const uint64_t frames = cfg_.at("frames"), books = cfg_.at("codebooks"),
                       samples = cfg_.at("samples");
        if (input.frames != frames || input.codebooks != books ||
            input.tokens.size() != frames * books || !input.tokens.data())
            throw std::invalid_argument("Mimi input differs from its built codebook token profile");
        for (const auto value : input.tokens)
            if (value < 0 || value >= cfg_.at("vocabulary_size").get<int32_t>())
                throw std::invalid_argument("Mimi codec token lies outside its vocabulary");
        Tensor tensor;
        tensor.data = const_cast<int32_t*>(input.tokens.data());
        tensor.shape = {static_cast<int64_t>(books), static_cast<int64_t>(frames)};
        tensor.dtype = DType::kInt32;
        const auto outputs = engine_->forward({{"codes", tensor}});
        const auto found = outputs.find("waveform");
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
            found->second.numel() != samples || !found->second.data)
            throw std::runtime_error("Invalid Mimi waveform output");
        const auto* data = static_cast<const float*>(found->second.data);
        internal::AudioResult result;
        result.samples.assign(data, data + samples);
        for (const float value : result.samples)
            if (!std::isfinite(value))
                throw std::runtime_error("Mimi returned a nonfinite sample");
        result.sample_rate = cfg_.at("sample_rate");
        result.channels = 1;
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> engine_;
    nlohmann::json cfg_;
};
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length)
        throw std::runtime_error(std::string("Mimi bundle lacks ") + name);
    return reader.read_section(name);
}
} // namespace
} // namespace trtmc::mimi
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("mimi does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "audio_codebook_tokens_to_audio")
        throw std::invalid_argument("Mimi requires codec token decoding without an external cache");
    const auto raw = mimi::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int64_t frames = cfg.at("frames"), books = cfg.at("codebooks"),
                  samples = cfg.at("samples"), rate = cfg.at("sample_rate");
    if (frames <= 0 || frames > 125 || books <= 0 || books > 64 ||
        cfg.at("vocabulary_size").get<int64_t>() <= 0 || samples <= 0 || rate <= 0 ||
        rate > UINT32_MAX)
        throw std::runtime_error("Invalid Mimi audio profile");
    const auto plan = mimi::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("Mimi engine failed to load");
    return new mimi::Pipeline(std::move(engine), cfg);
}
