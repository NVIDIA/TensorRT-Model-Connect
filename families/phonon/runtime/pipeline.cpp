/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phonon/runtime/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace trtmc::phonon {
namespace {
using namespace internal;
const ConfigField fields[] = {
    {"temperature", ConfigKind::F64, ConfigValue{0.3},
     "Gaussian noise variance; zero is deterministic"},
    {"seed", ConfigKind::I64, ConfigValue{std::int64_t{0}}, "Sampling seed in [0, 4294967295]"},
};
Tensor tensor(std::vector<float>& v, std::vector<std::int64_t> shape) {
    return {v.data(), std::move(shape), DType::kFloat32};
}
std::vector<float> output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto& t = outputs.at(name);
    if (!t.data || t.dtype != DType::kFloat32 || t.numel() != count)
        throw std::runtime_error(std::string("Phonon invalid engine output: ") + name);
    const auto* data = static_cast<const float*>(t.data);
    std::vector<float> result(data, data + count);
    for (float value : result)
        if (!std::isfinite(value))
            throw std::runtime_error("Phonon engine produced nonfinite output");
    return result;
}
} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> backbone, std::unique_ptr<ITrtModule> flow,
                   std::unique_ptr<ITrtModule> codec, Assets assets, Tokenizer tokenizer)
    : backbone_(std::move(backbone)), flow_(std::move(flow)), codec_(std::move(codec)),
      assets_(std::move(assets)), tokenizer_(std::move(tokenizer)) {}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ITextToSpeech>(*this, fields)};
}

internal::AudioResult Pipeline::run(const internal::TextToSpeechRequest& request,
                                    internal::ConfigView config) {
    using namespace internal;
    validate_config(fields, config);
    const auto temperature = *config_get<double>(config, fields, "temperature");
    const auto seed = *config_get<std::int64_t>(config, fields, "seed");
    if (!std::isfinite(temperature) || temperature < 0 || temperature > 10 || seed < 0 ||
        seed > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()))
        throw std::invalid_argument(
            "Phonon requires temperature in [0,10] and seed in [0,4294967295]");
    if (request.language && *request.language != "en")
        throw std::invalid_argument("Phonon 31m English supports language=en only");
    const auto start = std::chrono::steady_clock::now();
    const auto tokens = tokenizer_.encode(request.text);
    auto embeddings = assets_.voice;
    for (auto id : tokens)
        embeddings.insert(embeddings.end(), assets_.text.begin() + id * 512,
                          assets_.text.begin() + (id + 1) * 512);
    embeddings.insert(embeddings.end(), assets_.bos.begin(), assets_.bos.end());
    const auto prompt_length = embeddings.size() / 512;
    if (prompt_length >= static_cast<std::size_t>(assets_.maximum))
        throw std::invalid_argument("Phonon text and voice exceed the bundle sequence budget");
    const auto frame_budget = std::min<std::size_t>(
        assets_.maximum - prompt_length,
        static_cast<std::size_t>(std::ceil((tokens.size() / 3.0 + 2.0) * 12.5)));
    const auto prepared = Tokenizer::prepare(request.text);
    const auto words = 1 + std::count(prepared.begin(), prepared.end(), ' ');
    const int tail = words <= 4 ? 3 : 1;
    int remaining = -1;
    std::mt19937 generator(static_cast<std::uint32_t>(seed));
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> latents;
    bool ended = false;
    for (std::size_t frame = 0; frame < frame_budget; ++frame) {
        auto outputs = backbone_->forward(
            {{"embeddings",
              tensor(embeddings, {1, static_cast<std::int64_t>(embeddings.size() / 512), 512})}});
        auto condition = output(outputs, "condition", 512);
        const auto eos = output(outputs, "eos", 1)[0];
        std::vector<float> noise(32);
        for (auto& v : noise)
            v = normal(generator) * static_cast<float>(std::sqrt(temperature));
        auto sample = flow_->forward(
            {{"noise", tensor(noise, {1, 32})}, {"condition", tensor(condition, {1, 512})}});
        const auto latent = output(sample, "latent", 32);
        latents.insert(latents.end(), latent.begin(), latent.end());
        if (remaining < 0 && eos > 4.0F)
            remaining = tail;
        if (remaining == 0) {
            ended = true;
            break;
        }
        if (remaining > 0)
            --remaining;
        for (std::size_t row = 0; row < 512; ++row) {
            float value = assets_.condition[row];
            for (std::size_t col = 0; col < 32; ++col)
                value += assets_.projection[row * 32 + col] * latent[col];
            embeddings.push_back(value);
        }
    }
    if (!ended)
        throw std::runtime_error("Phonon did not finish speech within its frame/sequence budget");
    const auto frames = static_cast<std::int64_t>(latents.size() / 32);
    auto audio = codec_->forward({{"latents", tensor(latents, {1, frames, 32})}});
    internal::AudioResult result;
    result.samples = output(audio, "pcm", static_cast<std::size_t>(frames) * 1920);
    result.sample_rate = 24000;
    result.channels = 1;
    result.inference_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}
} // namespace trtmc::phonon
