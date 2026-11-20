/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::wav2vec2_bert {
namespace {
class Pipeline final : public internal::IModel, public internal::ISpectrogramToFrameFeatures {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, nlohmann::json cfg)
        : engine_(std::move(engine)), cfg_(std::move(cfg)) {}
    const char* task() const noexcept override { return "spectrogram_to_frame_features"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISpectrogramToFrameFeatures>(*this)};
    }
    internal::SpectrogramFrameFeaturesResult
    run(const internal::SpectrogramToFrameFeaturesRequest& request,
        internal::ConfigView config) override {
        internal::validate_config({}, config);
        const auto& input = request.spectrogram;
        const uint64_t frames = cfg_.at("frames"), bins = cfg_.at("bins"),
                       width = cfg_.at("hidden_size");
        if (input.rows != frames || input.columns != bins || input.values.size() != frames * bins ||
            !input.values.data())
            throw std::invalid_argument(
                "W2v-BERT input differs from its normalized spectral frame profile");
        for (const float value : input.values)
            if (!std::isfinite(value))
                throw std::invalid_argument("W2v-BERT input contains nonfinite features");
        Tensor tensor;
        tensor.data = const_cast<float*>(input.values.data());
        tensor.shape = {static_cast<int64_t>(frames), static_cast<int64_t>(bins)};
        tensor.dtype = DType::kFloat32;
        const auto outputs = engine_->forward({{"spectrogram", tensor}});
        const auto found = outputs.find("features");
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
            found->second.shape !=
                std::vector<int64_t>{static_cast<int64_t>(frames), static_cast<int64_t>(width)} ||
            found->second.numel() != frames * width || !found->second.data)
            throw std::runtime_error("Invalid W2v-BERT frame feature output");
        const auto* values = static_cast<const float*>(found->second.data);
        internal::SpectrogramFrameFeaturesResult result;
        result.features.rows = result.source_frame_count = frames;
        result.features.columns = width;
        result.features.values.assign(values, values + frames * width);
        for (uint64_t index = 0; index < frames; ++index)
            result.input_frame_indices.push_back(index);
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> engine_;
    nlohmann::json cfg_;
};
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length)
        throw std::runtime_error(std::string("W2v-BERT bundle lacks ") + name);
    return reader.read_section(name);
}
} // namespace
} // namespace trtmc::wav2vec2_bert
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("wav2vec2_bert does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "spectrogram_to_frame_features")
        throw std::invalid_argument(
            "W2v-BERT requires spectral frame features without an external cache");
    const auto raw = wav2vec2_bert::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int64_t frames = cfg.at("frames"), bins = cfg.at("bins"), width = cfg.at("hidden_size");
    if (frames <= 0 || frames > 512 || bins <= 0 || bins > 4096 || width <= 0 || width > 16384)
        throw std::runtime_error("Invalid W2v-BERT spectral encoder profile");
    const auto plan = wav2vec2_bert::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("W2v-BERT engine failed to load");
    if (!engine->has_input("spectrogram") ||
        engine->tensor_dtype("spectrogram") != DType::kFloat32 ||
        engine->tensor_shape("spectrogram") != std::vector<int64_t>{frames, bins})
        throw std::runtime_error("W2v-BERT spectrogram binding differs from its runtime profile");
    return new wav2vec2_bert::Pipeline(std::move(engine), cfg);
}
