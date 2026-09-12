/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <regex>
#include <stdexcept>

namespace trtmc::cosyvoice3 {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error("CosyVoice3: " + message);
}
template <class T>
Tensor tensor(std::vector<T>& values, std::vector<int64_t> shape, DType dtype = DType::kFloat32) {
    return {values.data(), std::move(shape), dtype};
}
std::vector<float> output(const TensorMap& result, const std::string& name,
                          const std::vector<int64_t>& shape) {
    const auto& t = result.at(name);
    require(t.dtype == DType::kFloat32 && t.shape == shape && t.data, "invalid output " + name);
    auto ptr = static_cast<const float*>(t.data);
    std::vector<float> values(ptr, ptr + t.numel());
    require(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }),
            "nonfinite output " + name);
    return values;
}
TensorMap run(ITrtModule& module, const TensorMap& inputs) {
    require(module.input_info().size() == inputs.size(), "unexpected engine inputs");
    for (const auto& [name, t] : inputs) {
        require(module.has_input(name) && module.tensor_dtype(name) == t.dtype,
                "input dtype/name mismatch: " + name);
        const auto lo = module.input_profile_shape(name, 0, ProfileShapeSelector::kMin);
        const auto hi = module.input_profile_shape(name, 0, ProfileShapeSelector::kMax);
        require(lo.size() == t.shape.size() && hi.size() == t.shape.size(), "input rank: " + name);
        for (size_t i = 0; i < lo.size(); ++i)
            require(t.shape[i] >= lo[i] && t.shape[i] <= hi[i],
                    "input outside engine profile: " + name);
        if (t.dtype == DType::kFloat32 && t.numel()) {
            auto values = static_cast<const float*>(t.data);
            require(values && std::all_of(values, values + t.numel(),
                                          [](float x) { return std::isfinite(x); }),
                    "nonfinite input " + name);
        }
    }
    return module.forward(inputs);
}
void plain(const std::string& text, bool allow_empty = false) {
    require((allow_empty || text.find_first_not_of(" \t\r\n") != std::string::npos) &&
                !std::regex_search(text, std::regex("<[^>]*>|\\[[^\\]]*\\]")),
            "expected plain text, without control/phoneme tags");
}
} // namespace

int sample(const std::vector<float>& logits, const std::vector<int32_t>& history, int minimum,
           bool greedy, const std::function<double()>& draw) {
    require(logits.size() == 6761, "expected 6761 speech logits");
    std::vector<double> p(logits.begin(), logits.end());
    for (double x : p)
        require(std::isfinite(x), "nonfinite speech logits");
    if (static_cast<int>(history.size()) < minimum)
        p[6561] = -std::numeric_limits<double>::infinity();
    auto maximum = std::max_element(p.begin(), p.end());
    if (greedy)
        return static_cast<int>(maximum - p.begin());
    double top = *maximum, total = 0;
    for (double& x : p) {
        x = std::exp(x - top);
        total += x;
    }
    for (double& x : p)
        x /= total;
    std::vector<int> order(p.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
    double mass = 0;
    size_t count = 0;
    do {
        mass += p[order[count++]];
    } while (mass < .8 && count < 25);
    auto choose = [&](const std::vector<int>& ids, size_t n, double sum) {
        require(sum > 0 && std::isfinite(sum), "sampling has zero probability mass");
        double u = draw();
        require(u >= 0 && u < 1, "sampling draw must be in [0,1)");
        double cumulative = 0;
        for (size_t i = 0; i < n; ++i) {
            cumulative += p[ids[i]];
            if (u * sum < cumulative)
                return ids[i];
        }
        return ids[n - 1];
    };
    int token = choose(order, count, mass);
    auto start = history.begin() + (history.size() > 10 ? history.size() - 10 : 0);
    if (std::find(start, history.end(), token) != history.end()) {
        p[token] = 0;
        std::iota(order.begin(), order.end(), 0);
        token = choose(order, order.size(), std::accumulate(p.begin(), p.end(), 0.0));
    }
    return token;
}

std::vector<int32_t> pack(const ITokenizer& tokenizer, const Settings& settings, const Voice& voice,
                          const std::string& text) {
    plain(text);
    plain(settings.instruction);
    plain(settings.transcript, true);
    require(settings.transcript.empty() || !voice.tokens.empty(),
            "transcript needs reference tokens");
    auto prompt = tokenizer.encode(settings.instruction + "<|endofprompt|>" + settings.transcript);
    auto target = tokenizer.encode(text);
    require(!target.empty() && std::find(prompt.begin(), prompt.end(), 151646) != prompt.end(),
            "invalid tokenizer/instruction boundary");
    std::vector<int32_t> ids{151936 + 6561};
    prompt.insert(prompt.end(), target.begin(), target.end());
    for (int id : prompt) {
        require(id >= 0 && id < 151936, "text token out of range");
        ids.push_back(id);
    }
    ids.push_back(151936 + 6563);
    if (!settings.transcript.empty())
        for (int id : voice.tokens)
            ids.push_back(151936 + id);
    return ids;
}

Pipeline::Pipeline(Settings settings, std::unique_ptr<ITokenizer> tokenizer, ModuleFactory factory,
                   nlohmann::json coefficients)
    : settings_(std::move(settings)), tokenizer_(std::move(tokenizer)),
      factory_(std::move(factory)), coefficients_(std::move(coefficients)) {
    require(tokenizer_ && factory_, "missing tokenizer or module factory");
    require(settings_.max_context > 0 && settings_.max_tokens > 0, "invalid capacity");
    require(!coefficients_.empty(), "missing reference frontend coefficients");
    require(settings_.total_tokens > 0, "invalid combined speech capacity");
}

AudioResult Pipeline::generate_audio(const std::string&, const AudioGenerationConfig&) {
    throw std::runtime_error("CosyVoice3: this bundle requires reference audio for each request");
}

AudioResult Pipeline::generate_audio_with_reference(const std::string& text,
                                                    const AudioReference& reference,
                                                    const AudioGenerationConfig& cfg) {
    plain(text);
    plain(reference.transcript, true);
    require(cfg.max_new_tokens > 0, "max_new_tokens must be positive");
    require(cfg.talker_max_new_tokens == 0, "talker_max_new_tokens is not supported");
    auto features = reference_features(reference, coefficients_);
    Voice voice;
    {
        auto campplus = factory_("campplus");
        auto result = run(
            *campplus, {{"features", tensor(features.speaker, {1, features.speaker_frames, 80})}});
        voice.speaker = output(result, "speaker", {1, 192});
        double norm = 0;
        for (float x : voice.speaker)
            norm += double(x) * x;
        require(norm > 0, "zero reference speaker embedding");
    }
    {
        auto tokenizer = factory_("speech_tokenizer");
        auto result = run(*tokenizer,
                          {{"features", tensor(features.tokens, {1, 128, features.token_frames})}});
        const auto& tokens = result.at("tokens");
        int count = (features.token_frames + 3) / 4;
        require(tokens.dtype == DType::kInt32 && tokens.shape == std::vector<int64_t>({1, count}) &&
                    tokens.data,
                "invalid reference speech tokens");
        count = std::min(count, features.mel_frames / 2);
        require(count > 0 && count < settings_.total_tokens,
                "reference leaves no speech capacity; use a larger profile");
        auto data = static_cast<const int32_t*>(tokens.data);
        voice.tokens.assign(data, data + count);
        for (int token : voice.tokens)
            require(token >= 0 && token < 6561, "reference token outside vocabulary");
        voice.features.assign(features.mel.begin(), features.mel.begin() + count * 160);
    }
    auto settings = settings_;
    settings.transcript = reference.transcript;
    settings.max_tokens =
        std::min(settings.max_tokens, settings.total_tokens - int(voice.tokens.size()));
    return synthesize(text, cfg, settings, std::move(voice));
}

AudioResult Pipeline::synthesize(const std::string& text, const AudioGenerationConfig& cfg,
                                 const Settings& settings, Voice voice) {
    require(cfg.talker_max_new_tokens == 0, "talker_max_new_tokens is not supported");
    auto ids = pack(*tokenizer_, settings, voice, text);
    int text_length = static_cast<int>(tokenizer_->encode(text).size());
    require(cfg.max_new_tokens > 0, "max_new_tokens must be positive");
    int limit = std::min(cfg.max_new_tokens, text_length * 20);
    require(limit <= settings.max_tokens && ids.size() + limit - 1 <= size_t(settings.max_context),
            "requested generation exceeds bundle capacity; lower max_new_tokens or rebuild");
    // Native RNG is deliberately not advertised as NumPy/Torch seed-equivalent.
    std::mt19937_64 rng(cfg.seed < 0 ? 2512 : cfg.seed);
    auto draw = [&]() { return std::generate_canonical<double, 53>(rng); };
    std::vector<int32_t> speech;
    {
        auto llm = factory_("llm");
        std::vector<float> keys(1), values(1); // nonnull pointers for zero-length cache
        int past = 0;
        bool stopped = false;
        for (int step = 0; step < limit; ++step) {
            int n = static_cast<int>(ids.size()), total = past + n;
            std::vector<int32_t> positions(n);
            std::iota(positions.begin(), positions.end(), past);
            std::vector<uint8_t> mask(size_t(n) * total);
            for (int q = 0; q < n; ++q)
                for (int k = 0; k < total; ++k)
                    mask[size_t(q) * total + k] = k <= past + q;
            auto result = run(*llm, {{"ids", tensor(ids, {1, n}, DType::kInt32)},
                                     {"positions", tensor(positions, {n}, DType::kInt32)},
                                     {"mask", tensor(mask, {1, 1, n, total}, DType::kBool)},
                                     {"keys", tensor(keys, {24, 2, past, 64})},
                                     {"values", tensor(values, {24, 2, past, 64})}});
            auto logits = output(result, "logits", {1, 1, 6761});
            keys = output(result, "present_keys", {24, 2, total, 64});
            values = output(result, "present_values", {24, 2, total, 64});
            past = total;
            int token =
                sample(logits, speech, std::min(text_length * 2, limit), settings.greedy, draw);
            if (token >= 6561) {
                stopped = true;
                break;
            }
            speech.push_back(token);
            ids = {151936 + token};
        }
        require(stopped && !speech.empty(),
                "LLM stopped empty or reached length limit without EOS");
    }
    const int prompt_frames = static_cast<int>(voice.tokens.size()) * 2;
    const int frames = prompt_frames + static_cast<int>(speech.size()) * 2;
    const size_t plane = size_t(80) * frames;
    std::vector<float> mu, spks;
    {
        auto conditioner = factory_("conditioning");
        auto tokens = voice.tokens;
        tokens.insert(tokens.end(), speech.begin(), speech.end());
        auto result = run(*conditioner,
                          {{"tokens", tensor(tokens, {1, int64_t(tokens.size())}, DType::kInt32)},
                           {"speaker", tensor(voice.speaker, {1, 192})}});
        mu = output(result, "mu", {1, 80, frames});
        spks = output(result, "spks", {1, 80});
    }
    std::vector<float> mel(size_t(80) * (frames - prompt_frames));
    {
        auto flow = factory_("flow");
        mu.resize(2 * plane, 0);
        spks.resize(160, 0);
        std::vector<float> cond(2 * plane, 0), mask(2 * frames, 1), x(2 * plane), t(2);
        std::vector<int32_t> positions(frames);
        std::iota(positions.begin(), positions.end(), 0);
        std::normal_distribution<float> gaussian;
        for (size_t i = 0; i < plane; ++i)
            x[i] = gaussian(rng);
        for (int c = 0; c < 80; ++c)
            for (int f = 0; f < prompt_frames; ++f)
                cond[size_t(c) * frames + f] = voice.features[size_t(f) * 80 + c];
        for (int step = 0; step < 10; ++step) {
            t[0] = t[1] = 1.f - std::cos((step / 10.f) * 1.5707963267948966f);
            float dt = (1.f - std::cos(((step + 1) / 10.f) * 1.5707963267948966f)) - t[0];
            std::copy_n(x.begin(), plane, x.begin() + plane);
            auto result = run(*flow, {{"x", tensor(x, {2, 80, frames})},
                                      {"mask", tensor(mask, {2, 1, frames})},
                                      {"mu", tensor(mu, {2, 80, frames})},
                                      {"t", tensor(t, {2})},
                                      {"spks", tensor(spks, {2, 80})},
                                      {"cond", tensor(cond, {2, 80, frames})},
                                      {"positions", tensor(positions, {frames}, DType::kInt32)}});
            auto velocity = output(result, "velocity", {2, 80, frames});
            for (size_t i = 0; i < plane; ++i)
                x[i] += dt * (1.7f * velocity[i] - .7f * velocity[plane + i]);
        }
        for (int c = 0; c < 80; ++c)
            std::copy_n(x.begin() + size_t(c) * frames + prompt_frames, frames - prompt_frames,
                        mel.begin() + size_t(c) * (frames - prompt_frames));
    }
    auto hift = factory_("hift");
    int target_frames = frames - prompt_frames, samples = target_frames * 480;
    std::vector<float> noise(size_t(9) * samples);
    for (float& x : noise)
        x = std::generate_canonical<float, 24>(rng);
    auto result = run(*hift, {{"mel", tensor(mel, {1, 80, target_frames})},
                              {"noise", tensor(noise, {1, 9, samples})}});
    AudioResult audio;
    audio.samples = output(result, "audio", {1, samples});
    audio.num_samples = samples;
    audio.sample_rate = 24000;
    return audio;
}
} // namespace trtmc::cosyvoice3
