/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/task.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

static_assert(std::is_abstract_v<trtmc::ITextGeneration>);
static_assert(std::is_abstract_v<trtmc::IVisionLanguageGeneration>);
static_assert(std::is_abstract_v<trtmc::IImageGeneration>);
static_assert(std::is_abstract_v<trtmc::IImageEditing>);
static_assert(std::is_abstract_v<trtmc::IImageBatchGeneration>);
static_assert(std::is_abstract_v<trtmc::IWorldModelGeneration>);
static_assert(std::is_abstract_v<trtmc::IAudioGeneration>);
static_assert(std::is_abstract_v<trtmc::IStreamingAudioGeneration>);
static_assert(std::is_abstract_v<trtmc::ITranscription>);
static_assert(std::is_abstract_v<trtmc::IBatchTranscription>);
static_assert(std::is_abstract_v<trtmc::IStreamingTranscription>);
static_assert(std::is_abstract_v<trtmc::ISpeechToSpeech>);
static_assert(std::is_abstract_v<trtmc::ISpeechSessionProvider>);
static_assert(std::is_abstract_v<trtmc::ISpeechBatchSessionProvider>);
static_assert(std::is_abstract_v<trtmc::ISpeechToolSessionProvider>);
static_assert(std::is_abstract_v<trtmc::IEmbedding>);
static_assert(std::is_abstract_v<trtmc::IEncoding>);
static_assert(std::is_abstract_v<trtmc::IReranking>);
static_assert(std::is_abstract_v<trtmc::ISegmentation>);
static_assert(std::is_abstract_v<trtmc::IPointPromptedSegmentation>);
static_assert(std::is_abstract_v<trtmc::ITextPromptedSegmentation>);
static_assert(std::is_abstract_v<trtmc::IStereoDisparity>);
static_assert(std::is_abstract_v<trtmc::IImageClassification>);
static_assert(std::is_abstract_v<trtmc::IPoseHypothesisRefinement>);
static_assert(std::is_abstract_v<trtmc::IImageFeatureExtractor>);
static_assert(std::is_abstract_v<trtmc::IVideoSegmentation>);
static_assert(std::is_abstract_v<trtmc::IVideoSegmentationSession>);
static_assert(std::is_abstract_v<trtmc::INeuralOperator>);
static_assert(std::is_abstract_v<trtmc::ITimeSeriesForecast>);
static_assert(std::is_abstract_v<trtmc::IRobotControl>);
static_assert(std::is_abstract_v<trtmc::ILoraAdapterManager>);
static_assert(!std::is_same_v<trtmc::TextGenerationConfig, trtmc::ImageGenerationConfig>);
static_assert(!std::is_same_v<trtmc::AudioGenerationConfig, trtmc::SpeechToSpeechConfig>);

namespace {

class TextAndEmbedding final : public trtmc::ITextGeneration, public trtmc::IEmbedding {
  public:
    const char* task() const noexcept override { return trtmc::ITextGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return 8; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig&) override {
        return {prompt, {1}};
    }

    trtmc::EmbeddingResult embed(const std::string& text) override {
        return {{static_cast<float>(text.size())}, 1};
    }
};

void test_robot_observation() {
    const float pixels[] = {0.0F, 0.5F, 1.0F};
    const float state[] = {0.25F};
    const trtmc::RobotObservation observation{{pixels, 3}, 1, 1, 3, {state, 1}};
    if (observation.image_pixels.size() != 3 || observation.state.size() != 1)
        throw std::runtime_error("RobotObservation did not preserve its input spans");
}

} // namespace

int main() {
    trtmc::AudioGenerationConfig audio_config;
    if (audio_config.talker_max_new_tokens != 0)
        return 1;
    audio_config.talker_max_new_tokens = 32;
    if (audio_config.talker_max_new_tokens != 32)
        return 1;

    const float forecast_values[] = {1.0F, 2.0F};
    const float forecast_mask[] = {1.0F, 1.0F};
    const trtmc::ForecastRequest forecast_request{{forecast_values, 2}, {forecast_mask, 2}, 7};
    if (forecast_request.frequency != 7)
        return 1;

    test_robot_observation();

    std::unique_ptr<trtmc::ITask> task = std::make_unique<TextAndEmbedding>();
    if (std::string(task->task()) != trtmc::ITextGeneration::kTask)
        return 1;

    auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
    auto* embedding = dynamic_cast<trtmc::IEmbedding*>(task.get());
    if (text == nullptr || embedding == nullptr)
        return 1;
    if (text->generate("hello").text != "hello")
        return 1;
    if (embedding->embed("abc").data != std::vector<float>{3.0F})
        return 1;

    std::cerr << "ALL PASSED\n";
    return 0;
}
