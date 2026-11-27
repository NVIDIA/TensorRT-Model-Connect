/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../request.h"
#include "contract.h"

#include <cstring>
#ifdef TRTMC_NEMOTRON_OMNI_EDGE_012
#include "../resampler.h"

#include <tensorrt_edgellm/runtime/audioUtils.h>
#include <tensorrt_edgellm/runtime/llmRuntimeUtils.h>
#include <tensorrt_edgellm/runtime/streaming.h>
#else
#include <edgellm/cpp/runtime/llmRuntimeUtils.h>
#include <edgellm/cpp/runtime/streaming.h>
#endif

namespace trtmc::nemotron_omni::edge_llm {
using namespace internal;

using EdgeDataType = decltype(trt_edgellm::rt::Tensor{}.getDataType());

inline void validate_capacity(int prompt, int input_limit, int capacity, int64_t budget) {
    if (prompt <= 0 || prompt > input_limit || budget <= 0 || budget > int64_t(capacity) - prompt)
        throw std::invalid_argument("Nemotron Omni prompt and generation exceed bundle capacity");
}
/// Validate complete responses before exposing any partial output, even early EOS.
inline void validate_response(const trt_edgellm::rt::LLMGenerationResponse& response,
                              int input_limit, int capacity, std::int64_t requested_budget) {
    if (response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
        response.inputTokenCounts.size() != 1 || response.finishReasons.size() != 1 ||
        response.outputIds.front().empty() || requested_budget <= 0 ||
        response.outputIds.front().size() > static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Nemotron Omni Edge returned an invalid generation response");
    validate_capacity(response.inputTokenCounts.front(), input_limit, capacity, requested_budget);
    const auto reason = response.finishReasons.front();
    if (reason != trt_edgellm::rt::FinishReason::kEndId &&
        reason != trt_edgellm::rt::FinishReason::kLength)
        throw std::runtime_error("Nemotron Omni Edge generation did not complete successfully");
    if (reason == trt_edgellm::rt::FinishReason::kLength &&
        response.outputIds.front().size() != static_cast<std::uint64_t>(requested_budget))
        throw std::runtime_error("Nemotron Omni Edge clipped the original generation budget");
}

/// Validate the source view before allocating or reading its byte storage.
inline size_t image_elements(ImageView image) {
    if (!image.data || image.height == 0 || image.width == 0 || image.channels != 3 ||
        (image.format != ImageFormat::UInt8 && image.format != ImageFormat::Float32))
        throw std::invalid_argument("Nemotron Omni requires nonempty uint8 or float32 HWC RGB");
    const size_t item = image.format == ImageFormat::Float32 ? sizeof(float) : 1;
    const size_t maximum =
        std::min(size_t(std::numeric_limits<int64_t>::max()), std::numeric_limits<size_t>::max()) /
        item;
    if (size_t(image.width) > maximum / 3 ||
        size_t(image.height) > maximum / (size_t(image.width) * 3))
        throw std::invalid_argument("Nemotron Omni image dimensions overflow");
    const size_t count = size_t(image.height) * image.width * 3;
    if (image.byte_size < count * item)
        throw std::invalid_argument("Nemotron Omni image storage is shorter than its dimensions");
    return count;
}

inline trt_edgellm::rt::imageUtils::ImageData image_data(Span<const ImageView> frames, bool video) {
    if (frames.empty())
        throw std::invalid_argument("Nemotron Omni video requires frames");
    const auto first = frames[0];
    const size_t elements = image_elements(first);
    if (frames.size() > size_t(std::numeric_limits<int64_t>::max()) / elements)
        throw std::invalid_argument("Nemotron Omni video dimensions overflow");
    for (const auto& frame : frames) {
        image_elements(frame);
        if (frame.height != first.height || frame.width != first.width)
            throw std::invalid_argument("Nemotron Omni video requires equal frame dimensions");
    }
    trt_edgellm::rt::Tensor tensor({int64_t(frames.size()), first.height, first.width, 3},
                                   trt_edgellm::rt::DeviceType::kCPU, EdgeDataType::kUINT8);
    auto* output = static_cast<uint8_t*>(tensor.rawPointer());
    for (const auto& frame : frames) {
        if (frame.format == ImageFormat::UInt8)
            std::memcpy(output, frame.data, elements);
        else {
            const auto* input = static_cast<const float*>(frame.data);
            for (size_t i = 0; i < elements; ++i) {
                if (!std::isfinite(input[i]) || input[i] < 0 || input[i] > 1)
                    throw std::invalid_argument(
                        "Nemotron Omni float images require finite values in [0, 1]");
                output[i] = uint8_t(std::round(input[i] * 255.0F));
            }
        }
        output += elements;
    }
    trt_edgellm::rt::imageUtils::ImageData result(std::move(tensor));
    result.isVideo = video;
    return result;
}

#ifdef TRTMC_NEMOTRON_OMNI_EDGE_012
using EdgePart = std::variant<TextPartView, ImageView, AudioView>;

inline auto audio_data(AudioView audio) {
    const auto rate = audio.sample_rate.value_or(16000);
    if (!rate || rate > uint32_t(INT32_MAX) || !audio.channels || audio.samples.empty() ||
        audio.samples.size() % audio.channels != 0 ||
        audio.samples.size() / audio.channels > size_t(INT32_MAX))
        throw std::invalid_argument("Omni audio requires a sample rate and complete PCM frames");
    const auto frames = audio.samples.size() / audio.channels;
    // The acoustic builder profile ends at 6000 steps, with a 160-sample hop.
    if (frames * uint64_t(16000) / rate >= 6001ULL * 160)
        throw std::invalid_argument("Omni audio exceeds the acoustic profile; it is not truncated");
    std::vector<float> mono(frames);
    for (size_t i = 0; i < frames; ++i) {
        double sum = 0;
        for (size_t ch = 0; ch < audio.channels; ++ch) {
            float sample = audio.samples[i * audio.channels + ch];
            if (!std::isfinite(sample))
                throw std::invalid_argument("Omni PCM must be finite");
            sum += sample;
        }
        mono[i] = float(sum / audio.channels);
    }
    if (rate != 16000)
        mono = resample_pcm(mono.data(), int32_t(mono.size()), int32_t(rate), 16000);
    auto samples = std::make_shared<trt_edgellm::rt::Tensor>(
        trt_edgellm::rt::Coords{int64_t(mono.size())}, trt_edgellm::rt::DeviceType::kCPU,
        EdgeDataType::kFLOAT);
    std::memcpy(samples->rawPointer(), mono.data(), mono.size() * sizeof(float));
    return trt_edgellm::rt::audioUtils::wrapPcm(std::move(samples), 16000);
}
#else
using EdgePart = ImageVideoTextPart;
#endif
using EdgeMessage = MediaMessage<EdgePart>;

/// Translate typed content in caller order without synthesizing media or timestamps.
inline trt_edgellm::rt::LLMGenerationRequest
make_request(Span<const EdgeMessage> messages, const TextGenerationConfig& cfg, int capacity) {
    validate_request(cfg);
    if (cfg.min_p != 0 || cfg.repetition_penalty != 1 || cfg.top_k < 0 || cfg.top_p <= 0)
        throw std::invalid_argument(
            "Requested sampling control is not exposed by Nemotron Omni offload");
    if (messages.empty() || cfg.max_new_tokens >= capacity)
        throw std::invalid_argument(
            "Nemotron Omni requires a prompt and room for the output budget");
    if (!cfg.use_chat_template && (messages.size() != 1 || !cfg.system_prompt.empty()))
        throw std::invalid_argument(
            "Nemotron Omni multi-message prompts require the chat template");
    trt_edgellm::rt::LLMGenerationRequest result{};
    result.requests.resize(1);
    auto& request = result.requests.front();
    if (!cfg.system_prompt.empty()) {
        if (messages[0].role == MessageRole::System)
            throw std::invalid_argument("Supply one Nemotron Omni system message, not two");
        auto& system = request.messages.emplace_back();
        system.role = "system";
        system.contents.push_back({"text", cfg.system_prompt});
    }
    for (const auto& message : messages) {
        auto& target = request.messages.emplace_back();
        target.role = role_name(message.role);
        target.contentIsArray = true;
        for (const auto& part : message.parts) {
            if (const auto* text = std::get_if<TextPartView>(&part)) {
                target.contents.push_back({"text", std::string(text->text)});
                continue;
            }
            if (message.role == MessageRole::Assistant || message.role == MessageRole::System)
                throw std::invalid_argument(
                    "Nemotron Omni assistant and system history must contain text");
            if (const auto* image = std::get_if<ImageView>(&part)) {
                target.contents.push_back({"image", ""});
                request.imageBuffers.push_back(image_data({image, 1}, false));
#ifdef TRTMC_NEMOTRON_OMNI_EDGE_012
            } else if (const auto* audio = std::get_if<AudioView>(&part)) {
                target.contents.push_back({"audio", ""});
                request.audioBuffers.push_back(audio_data(*audio));
#endif
            } else {
                throw std::invalid_argument(
                    "Nemotron Omni Edge video is unavailable; build task=video_text_to_text "
                    "to use the native TensorRT implementation");
            }
        }
    }
    result.applyChatTemplate = cfg.use_chat_template;
    result.enableThinking = cfg.enable_thinking;
    result.temperature = cfg.temperature;
#ifdef TRTMC_NEMOTRON_OMNI_EDGE_012
    // With an active nucleus, native sampling does not cap the default top_k=1.
    result.topK = cfg.top_k == 1 && cfg.top_p < 1.0F - 1e-6F ? 0 : cfg.top_k;
#else
    result.topK = cfg.top_k;
#endif
    result.topP = cfg.top_p;
    result.maxGenerateLength = cfg.max_new_tokens;
    if (cfg.seed >= 0)
        result.samplingSeed = uint64_t(cfg.seed);
    return result;
}
} // namespace trtmc::nemotron_omni::edge_llm
