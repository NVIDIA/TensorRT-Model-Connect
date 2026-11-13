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

namespace trtmc::speecht5_hifigan {
namespace {
class Pipeline final : public internal::IModel, public internal::ISpectrogramToAudio {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, nlohmann::json cfg)
        : engine_(std::move(engine)), cfg_(std::move(cfg)) {}
    const char* task() const noexcept override { return "spectrogram_to_audio"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISpectrogramToAudio>(*this)};
    }
    internal::AudioResult run(const internal::SpectrogramToAudioRequest& request,
                              internal::ConfigView config) override {
        internal::validate_config({}, config);
        const auto& input = request.spectrogram;
        const uint64_t frames = cfg_.at("frames"), bins = cfg_.at("bins"),
                       samples = cfg_.at("samples");
        if (input.rows != frames || input.columns != bins || input.values.size() != frames * bins ||
            !input.values.data())
            throw std::invalid_argument("HiFiGAN input differs from its built spectrogram profile");
        for (const float value : input.values)
            if (!std::isfinite(value))
                throw std::invalid_argument("HiFiGAN spectrogram contains a nonfinite value");
        Tensor tensor;
        tensor.data = const_cast<float*>(input.values.data());
        tensor.shape = {static_cast<int64_t>(frames), static_cast<int64_t>(bins)};
        tensor.dtype = DType::kFloat32;
        const auto outputs = engine_->forward({{"spectrogram", tensor}});
        const auto found = outputs.find("waveform");
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
            found->second.numel() != samples || !found->second.data)
            throw std::runtime_error("Invalid HiFiGAN waveform output");
        const auto* data = static_cast<const float*>(found->second.data);
        internal::AudioResult result;
        result.samples.assign(data, data + samples);
        for (const float value : result.samples)
            if (!std::isfinite(value))
                throw std::runtime_error("HiFiGAN returned a nonfinite sample");
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
        throw std::runtime_error(std::string("HiFiGAN bundle lacks ") + name);
    return reader.read_section(name);
}
} // namespace
} // namespace trtmc::speecht5_hifigan
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "spectrogram_to_audio")
        throw std::invalid_argument(
            "HiFiGAN requires spectrogram-to-audio without an external cache");
    const auto raw = speecht5_hifigan::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int64_t frames = cfg.at("frames"), bins = cfg.at("bins"), samples = cfg.at("samples"),
                  rate = cfg.at("sample_rate");
    if (frames <= 0 || frames > 1024 || bins <= 0 || samples <= 0 || rate <= 0 || rate > UINT32_MAX)
        throw std::runtime_error("Invalid HiFiGAN audio profile");
    const auto plan = speecht5_hifigan::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("HiFiGAN engine failed to load");
    return new speecht5_hifigan::Pipeline(std::move(engine), cfg);
}
