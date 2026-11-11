/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cohere_asr/runtime/pipeline.h"

#include "families/cohere_asr/runtime/frontend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
namespace trtmc::cohere_asr {
Pipeline::Pipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
                   Config config)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder)), config_(std::move(config)) {
    if (!encoder_ || !encoder_->ok() || !decoder_ || !decoder_->ok())
        throw std::runtime_error("invalid Cohere ASR engines");
}
internal::TextResult Pipeline::run(const internal::SpeechTranscriptionRequest& request,
                                   internal::ConfigView config) {
    if (!config.empty())
        throw internal::ConfigError("cohere_asr has no runtime Config fields");
    const auto language = request.source_language.value_or("ar");
    if (language != "ar" && language != "en")
        throw std::invalid_argument("cohere_asr supports source_language ar or en");
    const auto& audio = request.audio;
    if (audio.channels != 1 || audio.sample_rate.value_or(16000) != 16000 ||
        !audio.samples.data() || audio.samples.size() < 512 ||
        audio.samples.size() > static_cast<std::size_t>(config_.max_samples))
        throw std::invalid_argument(
            "cohere_asr requires mono 16 kHz PCM within the bundle's 35 second capacity");
    const auto begin = std::chrono::steady_clock::now();
    auto mel =
        features(audio.samples.data(), audio.samples.size(), config_.window, config_.filters);
    const auto frames = audio.samples.size() / 160;
    Tensor input{mel.data(), {1, 128, static_cast<int64_t>(frames)}, DType::kFloat32};
    auto outputs = encoder_->forward({{"features", input}});
    const auto found = outputs.find("encoded");
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.shape !=
            std::vector<int64_t>{1, static_cast<int64_t>((frames + 7) / 8), 1280} ||
        !found->second.data)
        throw std::runtime_error("cohere_asr encoder returned invalid output");
    // Host outputs belong to the encoder module and remain valid during decoder execution.
    const auto encoded = found->second;
    auto tokens = language == "ar" ? config_.arabic_prompt : config_.english_prompt;
    internal::TextResult result;
    bool ended = false;
    while (tokens.size() < static_cast<std::size_t>(config_.max_tokens)) {
        Tensor token_input{tokens.data(), {1, static_cast<int64_t>(tokens.size())}, DType::kInt32};
        auto values = decoder_->forward({{"tokens", token_input}, {"encoded", encoded}});
        const auto found_logits = values.find("log_probs");
        if (found_logits == values.end() || found_logits->second.dtype != DType::kFloat32 ||
            found_logits->second.numel() != config_.pieces.size() || !found_logits->second.data)
            throw std::runtime_error("cohere_asr decoder returned invalid output");
        const auto* logits = static_cast<const float*>(found_logits->second.data);
        if (!std::all_of(logits, logits + config_.pieces.size(),
                         [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("cohere_asr decoder returned non-finite log probabilities");
        const auto next =
            static_cast<int32_t>(std::max_element(logits, logits + config_.pieces.size()) - logits);
        if (next == config_.eos_id) {
            ended = true;
            break;
        }
        result.token_ids.push_back(next);
        tokens.push_back(next);
    }
    if (!ended)
        throw std::runtime_error("cohere_asr output exceeds bundle token capacity; rebuild with a "
                                 "larger max_sequence_length");
    for (auto id : result.token_ids) {
        if (config_.special.count(id))
            continue;
        const auto& piece = config_.pieces.at(id);
        if (piece.size() == 6 && piece.substr(0, 3) == "<0x" && piece.back() == '>') {
            result.text.push_back(static_cast<char>(std::stoi(piece.substr(3, 2), nullptr, 16)));
            continue;
        }
        result.text += piece;
    }
    const std::string marker = "\xe2\x96\x81";
    for (std::size_t at = 0; (at = result.text.find(marker, at)) != std::string::npos; ++at)
        result.text.replace(at, marker.size(), " ");
    if (!result.text.empty() && result.text.front() == ' ')
        result.text.erase(result.text.begin());
    result.decode_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    return result;
}
} // namespace trtmc::cohere_asr
