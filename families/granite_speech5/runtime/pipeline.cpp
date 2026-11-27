/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/granite_speech5/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::granite_speech5 {
namespace {

constexpr const char* kInput = "waveform";
constexpr const char* kOutput = "frame_ids";

void require_io(const ITrtModule& module, const char* name, bool input, DType dtype) {
    const bool present = input ? module.has_input(name) : module.has_output(name);
    if (!present || module.tensor_dtype(name) != dtype)
        throw std::runtime_error(std::string("granite_speech5 encoder engine lacks the ") +
                                 (input ? "input " : "output ") + name + " of the bundle's type");
}

} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> encoder, Geometry geometry, Vocabulary vocabulary)
    : encoder_(std::move(encoder)), geometry_(geometry), vocabulary_(std::move(vocabulary)) {
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("granite_speech5 needs a loaded encoder engine");
    if (vocabulary_.size() != static_cast<std::size_t>(geometry_.vocab_size))
        throw std::runtime_error("granite_speech5 vocabulary does not match the CTC head's width");
    // Exactly one rank-1 input and one rank-1 output: an extra required input or another output
    // layout would load here and fail only on the first transcription.
    const auto inputs = encoder_->input_info();
    const auto outputs = encoder_->output_info();
    if (inputs.size() != 1 || inputs.front().name != kInput || outputs.size() != 1 ||
        outputs.front().name != kOutput || outputs.front().shape.size() != 1 ||
        encoder_->input_rank(kInput) != 1 || encoder_->optimization_profile_count() != 1)
        throw std::runtime_error(
            "granite_speech5 encoder engine must take one rank-1 waveform in one "
            "profile and return only rank-1 frame_ids");
    require_io(*encoder_, kInput, true, DType::kFloat32);
    require_io(*encoder_, kOutput, false, DType::kInt32);
    // Every clip the bundle accepts must fit the engine's profile, so no accepted request fails
    // inside TensorRT.
    const auto low = encoder_->input_profile_shape(kInput, 0, ProfileShapeSelector::kMin);
    const auto high = encoder_->input_profile_shape(kInput, 0, ProfileShapeSelector::kMax);
    if (low.size() != 1 || high.size() != 1 || low[0] > geometry_.min_samples + geometry_.n_fft ||
        high[0] < geometry_.max_samples + 1 + geometry_.n_fft)
        throw std::runtime_error(
            "granite_speech5 encoder engine profile does not cover the bundle's sample range");
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ISpeechTranscription>(*this)};
}

TextResult Pipeline::run(const internal::SpeechTranscriptionRequest& request,
                         internal::ConfigView options) {
    internal::validate_config({}, options);
    if (request.source_language)
        throw std::invalid_argument(
            "granite_speech5 transcribes the spoken language without a language hint; "
            "omit source_language");
    const internal::AudioView& audio = request.audio;
    if (audio.samples.data() == nullptr || audio.samples.empty())
        throw std::invalid_argument("granite_speech5 requires float32 PCM input");
    const auto rate = audio.sample_rate.value_or(static_cast<std::uint32_t>(geometry_.sample_rate));
    if (rate != static_cast<std::uint32_t>(geometry_.sample_rate))
        throw std::invalid_argument("granite_speech5 requires " +
                                    std::to_string(geometry_.sample_rate) + " Hz audio, not " +
                                    std::to_string(rate) + " Hz; it ships no resampler");
    const auto channels = audio.channels == 0 ? 1U : audio.channels;
    const auto values = static_cast<std::uint64_t>(audio.samples.size());
    if (values % channels != 0)
        throw std::invalid_argument("granite_speech5 input is not a whole number of audio frames");
    const auto count = static_cast<std::int64_t>(values / channels);
    // Checked before any copy: the engine's profile is built for this sample range.
    if (count < geometry_.min_samples || count > geometry_.max_samples)
        throw std::invalid_argument(
            "granite_speech5 accepts " + std::to_string(geometry_.min_samples) + " to " +
            std::to_string(geometry_.max_samples) + " samples, not " + std::to_string(count));
    if (!std::all_of(audio.samples.begin(), audio.samples.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("granite_speech5 received a non-finite audio sample");

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    auto waveform = pad_waveform(mix_mono(audio.samples.data(), values, channels), geometry_);
    std::vector<std::int32_t> frame_ids;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Tensor input{
            waveform.data(), {static_cast<std::int64_t>(waveform.size())}, DType::kFloat32};
        const auto outputs = encoder_->forward({{kInput, input}});
        const auto found = outputs.find(kOutput);
        const auto frames = geometry_.output_frames(count);
        // The decoder reads one id per CTC frame: the count the geometry implies is checked on
        // every call, not only that some ids came back. It cannot be checked at load: output_info()
        // reports a dynamic output at the profile's longest input, like a fixed one.
        if (found == outputs.end() || found->second.dtype != DType::kInt32 ||
            found->second.data == nullptr ||
            found->second.shape != std::vector<std::int64_t>{frames})
            throw std::runtime_error("granite_speech5 encoder engine did not return " +
                                     std::to_string(frames) + " int32 frame ids");
        const auto* begin = static_cast<const std::int32_t*>(found->second.data);
        frame_ids.assign(begin, begin + frames);
    }
    const auto after_encoder = Clock::now();
    for (const auto id : frame_ids)
        if (id < 0 || id >= geometry_.vocab_size)
            throw std::runtime_error("granite_speech5 encoder engine returned the frame id " +
                                     std::to_string(id) + " outside the vocabulary");
    TextResult result;
    result.token_ids = collapse(frame_ids.data(), static_cast<std::int64_t>(frame_ids.size()),
                                vocabulary_.blank_id());
    result.text = vocabulary_.decode(result.token_ids);
    result.prefill_ms = std::chrono::duration<double, std::milli>(after_encoder - start).count();
    result.decode_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - after_encoder).count();
    return result;
}

} // namespace trtmc::granite_speech5
