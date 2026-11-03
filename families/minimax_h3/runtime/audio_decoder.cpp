/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/audio_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::minimax_h3 {
namespace {
constexpr int64_t kStereo = 2, kLatentChannels = 32, kHop = 800;
constexpr const char* kInput = "normalized_audio_latents";
constexpr const char* kOutput = "waveform";

void validate_module(const ITrtModule& module, int64_t frames) {
    if (!module.ok() || !module.has_input(kInput) || !module.has_output(kOutput) ||
        module.input_info().size() != 1 || module.output_info().size() != 1 ||
        module.tensor_dtype(kInput) != DType::kFloat32 ||
        module.tensor_dtype(kOutput) != DType::kFloat32 ||
        module.tensor_shape(kInput) != std::vector<int64_t>({kStereo, kLatentChannels, frames}) ||
        module.tensor_shape(kOutput) != std::vector<int64_t>({kStereo, 1, frames * kHop}))
        throw std::runtime_error("MiniMax-H3 audio decoder plan ABI mismatch");
}
} // namespace

AudioDecoder::AudioDecoder(ReferenceModuleLoader loader, int32_t frames, cudaStream_t stream,
                           AudioDecoderResidency residency)
    : loader_(std::move(loader)), frames_(frames), stream_(stream), residency_(residency) {
    if (!loader_ || frames <= 0 ||
        static_cast<uint64_t>(frames) >
            std::numeric_limits<std::size_t>::max() / (kStereo * kHop * sizeof(float)) ||
        (residency != AudioDecoderResidency::Staged &&
         residency != AudioDecoderResidency::Resident))
        throw std::invalid_argument("MiniMax-H3 audio decoder has invalid configuration");
}

DecodedH3Audio AudioDecoder::run(const std::vector<float>& rows) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto frames = static_cast<std::size_t>(frames_);
    if (rows.size() != kStereo * frames * kLatentChannels ||
        !std::all_of(rows.begin(), rows.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("MiniMax-H3 audio decoder requires finite generated FP32 rows");
    try {
        if (!module_) {
            module_ = loader_("audio_vae.plan", stream_);
            if (!module_)
                throw std::runtime_error("MiniMax-H3 audio decoder could not load its plan");
            validate_module(*module_, frames_);
        }
        std::vector<float> latents(rows.size());
        for (std::size_t stereo = 0; stereo < kStereo; ++stereo)
            for (std::size_t time = 0; time < frames; ++time)
                for (std::size_t channel = 0; channel < kLatentChannels; ++channel)
                    latents[(stereo * kLatentChannels + channel) * frames + time] =
                        rows[(stereo * frames + time) * kLatentChannels + channel];
        TensorMap inputs;
        inputs.emplace(
            kInput, Tensor{latents.data(), {kStereo, kLatentChannels, frames_}, DType::kFloat32});
        const auto outputs = module_->forward(inputs);
        const auto found = outputs.find(kOutput);
        const auto samples = frames * kHop;
        if (found == outputs.end() || !found->second.data ||
            found->second.dtype != DType::kFloat32 ||
            found->second.shape !=
                std::vector<int64_t>({kStereo, 1, static_cast<int64_t>(samples)}))
            throw std::runtime_error("MiniMax-H3 audio decoder returned an invalid waveform");
        module_->sync();
        const auto* waveform = static_cast<const float*>(found->second.data);
        DecodedH3Audio result;
        result.interleaved_samples.resize(kStereo * samples);
        for (std::size_t time = 0; time < samples; ++time)
            for (std::size_t channel = 0; channel < kStereo; ++channel) {
                const float value = waveform[channel * samples + time];
                if (!std::isfinite(value) || value < -1.0F || value > 1.0F)
                    throw std::runtime_error(
                        "MiniMax-H3 audio decoder returned a non-finite or unclamped sample");
                result.interleaved_samples[time * kStereo + channel] = value;
            }
        if (residency_ == AudioDecoderResidency::Staged)
            module_.reset();
        return result;
    } catch (...) {
        module_.reset();
        throw;
    }
}

} // namespace trtmc::minimax_h3
