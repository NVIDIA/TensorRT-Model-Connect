/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/pipeline.h"

#include "families/vibevoice/runtime/engine_io.h"
#include "families/vibevoice/runtime/voice.h"
#include "trtmc/internal/config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::vibevoice {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("vibevoice ") + what + ": " +
                                 cudaGetErrorString(status));
}

const void* require_output(ITrtModule& module, const char* name) {
    const void* pointer = module.device_ptr(name);
    if (pointer == nullptr)
        throw std::runtime_error(std::string("vibevoice engine has no output ") + name);
    return pointer;
}

// The named tensors of a loaded module, with the bounds of its own profile.
EngineTensors engine_tensors(const ITrtModule& module, const std::vector<std::string>& names) {
    EngineTensors tensors;
    for (const auto& name : names) {
        const bool input = module.has_input(name);
        if (!input && !module.has_output(name))
            continue;
        EngineTensor tensor;
        tensor.input = input;
        tensor.dtype = module.tensor_dtype(name);
        tensor.dynamic = input && module.input_is_dynamic(name);
        if (tensor.dynamic) {
            tensor.profile_min =
                module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMin);
            tensor.profile_max =
                module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMax);
        } else {
            tensor.shape = module.tensor_shape(name);
        }
        tensors.emplace(name, std::move(tensor));
    }
    return tensors;
}

} // namespace

Sequence::Sequence(const RuntimeConfig& config, cudaStream_t stream)
    : storage_({config.num_layers, 2, config.capacity,
                static_cast<std::int64_t>(config.num_kv_heads) * config.head_dim},
               config.bfloat16 ? DType::kBFloat16 : DType::kFloat32, stream),
      row_bytes_(static_cast<std::size_t>(config.num_kv_heads) * config.head_dim *
                 (config.bfloat16 ? 2U : 4U)),
      capacity_(config.capacity) {
    if (!storage_.ok())
        throw std::runtime_error("vibevoice could not allocate its KV cache");
}

void* Sequence::row(std::int32_t layer, std::int32_t kind, std::int32_t position) {
    const auto index = (static_cast<std::size_t>(layer) * 2U + static_cast<std::size_t>(kind)) *
                           static_cast<std::size_t>(capacity_) +
                       static_cast<std::size_t>(position);
    return static_cast<char*>(storage_.data()) + index * row_bytes_;
}

void* Sequence::key(std::int32_t layer, std::int32_t position) {
    return row(layer, 0, position);
}
void* Sequence::value(std::int32_t layer, std::int32_t position) {
    return row(layer, 1, position);
}

VibeVoicePipeline::VibeVoicePipeline(Engines engines, RuntimeConfig config,
                                     std::unique_ptr<ITokenizer> tokenizer,
                                     std::vector<std::string> special_tokens)
    : engines_(std::move(engines)), config_(std::move(config)), tokenizer_(std::move(tokenizer)),
      special_tokens_(std::move(special_tokens)),
      stream_(engines_.lm_decode ? engines_.lm_decode->stream() : nullptr),
      positive_(config_, stream_), negative_(config_, stream_), unconditional_(config_, stream_) {
    if (!engines_.lm_prefill || !engines_.lm_decode || !engines_.embed || !engines_.head ||
        !engines_.feedback || !engines_.voice || !tokenizer_)
        throw std::runtime_error("vibevoice pipeline is missing an engine or its tokenizer");
    // Every buffer below and every binding at run time is sized from runtime.json.
    check_lm_io(engine_tensors(*engines_.lm_prefill, lm_tensor_names(config_)), config_,
                config_.prefill_chunk);
    check_lm_io(engine_tensors(*engines_.lm_decode, lm_tensor_names(config_)), config_, 1);
    check_embed_io(engine_tensors(*engines_.embed, embed_tensor_names()), config_);
    check_head_io(engine_tensors(*engines_.head, head_tensor_names()), config_);
    check_feedback_io(engine_tensors(*engines_.feedback, feedback_tensor_names()), config_);
    check_voice_io(engine_tensors(*engines_.voice, voice_tensor_names()), config_);
    const auto type = config_.bfloat16 ? DType::kBFloat16 : DType::kFloat32;
    const std::int64_t hidden = config_.hidden_size;
    prompt_ = DeviceTensor({config_.capacity, hidden}, type, stream_);
    special_embeds_ = DeviceTensor({4, hidden}, type, stream_);
    step_input_ = DeviceTensor({1, hidden}, type, stream_);
    conditions_ = DeviceTensor({2, hidden}, type, stream_);
    rope_cos_ = DeviceTensor({config_.prefill_chunk, config_.head_dim}, DType::kFloat32, stream_);
    rope_sin_ = DeviceTensor({config_.prefill_chunk, config_.head_dim}, DType::kFloat32, stream_);
    for (auto& state : states_)
        state = DeviceTensor({config_.stream_state_size}, type, stream_);
    embed_tokens(config_.control_token_ids, special_embeds_.data());
    check(cudaStreamSynchronize(stream_), "embedding the control tokens");

    fields_ = {
        {"seed", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
         "Seed of the diffusion noise and the voice-prompt sampling noise."},
        {"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
         "Most generated tokens, speech frames and control tokens together; by default "
         "generation runs until the model ends it or the KV cache is full."},
        {"cfg_scale", internal::ConfigKind::F64,
         internal::ConfigValue{config_.default_guidance_scale},
         "Classifier-free guidance scale of the diffusion head and of the control tokens."},
        {"diffusion_steps", internal::ConfigKind::I64,
         internal::ConfigValue{config_.default_diffusion_steps},
         "DPM-Solver++ steps per speech frame."},
        {"voice_prompts", internal::ConfigKind::StringList,
         internal::ConfigValue{Span<const std::string_view>{}},
         "24 kHz mono WAV files giving the voices of speakers 0, 1, ... in order; without them the "
         "model chooses the voices."},
    };
}

std::vector<internal::TaskInstance> VibeVoicePipeline::task_bindings() {
    return {internal::bind<internal::ITextToSpeech>(
        *this, Span<const internal::ConfigField>{fields_.data(), fields_.size()})};
}

std::int64_t VibeVoicePipeline::element_bytes() const noexcept {
    return config_.bfloat16 ? 2 : 4;
}

VibeVoicePipeline::Options VibeVoicePipeline::resolve_options(internal::ConfigView config) const {
    const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
    internal::validate_config(fields, config);
    Options options;
    options.seed =
        static_cast<std::uint64_t>(*internal::config_get<std::int64_t>(config, fields, "seed"));
    if (const auto limit = internal::config_get<std::int64_t>(config, fields, "max_new_tokens")) {
        if (*limit <= 0)
            throw internal::ConfigError("vibevoice max_new_tokens must be positive");
        options.max_new_tokens = *limit;
    }
    options.guidance_scale = *internal::config_get<double>(config, fields, "cfg_scale");
    // The guidance is applied in float32 (and bfloat16).
    if (!std::isfinite(options.guidance_scale) || options.guidance_scale < 0.0 ||
        options.guidance_scale > std::numeric_limits<float>::max())
        throw internal::ConfigError(
            "vibevoice cfg_scale must be a finite non-negative float32 number");
    options.diffusion_steps =
        *internal::config_get<std::int64_t>(config, fields, "diffusion_steps");
    if (options.diffusion_steps <= 0 ||
        options.diffusion_steps > static_cast<std::int64_t>(config_.train_sigmas.size()))
        throw internal::ConfigError("vibevoice diffusion_steps must lie in [1, " +
                                    std::to_string(config_.train_sigmas.size()) + "]");
    try {
        check_schedule(config_.train_sigmas, static_cast<std::int32_t>(options.diffusion_steps));
    } catch (const std::invalid_argument& error) {
        throw internal::ConfigError(error.what());
    }
    const auto prompts =
        internal::config_get<Span<const std::string_view>>(config, fields, "voice_prompts");
    for (const auto path : *prompts)
        options.voice_prompts.emplace_back(path);
    return options;
}

const void* VibeVoicePipeline::special_row(std::int32_t token) const {
    const auto found =
        std::find(config_.control_token_ids.begin(), config_.control_token_ids.end(), token);
    if (found == config_.control_token_ids.end())
        throw std::logic_error("vibevoice has no embedding row for token " + std::to_string(token));
    const auto index = static_cast<std::size_t>(found - config_.control_token_ids.begin());
    return static_cast<const char*>(special_embeds_.data()) +
           index * static_cast<std::size_t>(config_.hidden_size * element_bytes());
}

void VibeVoicePipeline::embed_tokens(const std::vector<std::int32_t>& ids, void* destination) {
    const auto row_bytes = static_cast<std::size_t>(config_.hidden_size * element_bytes());
    for (std::size_t start = 0; start < ids.size();
         start += static_cast<std::size_t>(config_.prefill_chunk)) {
        const auto count =
            std::min(ids.size() - start, static_cast<std::size_t>(config_.prefill_chunk));
        std::vector<std::int32_t> chunk(ids.begin() + static_cast<std::ptrdiff_t>(start),
                                        ids.begin() + static_cast<std::ptrdiff_t>(start + count));
        engines_.embed->forward_async(
            {{"token_ids",
              Tensor{chunk.data(), {static_cast<std::int64_t>(count)}, DType::kInt32}}});
        check(cudaMemcpyAsync(static_cast<char*>(destination) + start * row_bytes,
                              require_output(*engines_.embed, "embeds"), count * row_bytes,
                              cudaMemcpyDeviceToDevice, stream_),
              "copying token embeddings");
        check(cudaStreamSynchronize(stream_), "embedding tokens");
    }
}

void VibeVoicePipeline::encode_voice(const std::vector<float>& samples, NoiseStream& noise,
                                     void* destination) {
    const auto frames = static_cast<std::int64_t>(samples.size()) / config_.hop_length;
    std::vector<float> scale{noise.normal()};
    std::vector<float> draws(static_cast<std::size_t>(frames * config_.latent_size));
    for (auto& draw : draws)
        draw = noise.normal();
    auto audio = samples;
    engines_.voice->forward_async(
        {{"audio", Tensor{audio.data(),
                          {1, 1, 1, static_cast<std::int64_t>(audio.size())},
                          DType::kFloat32}},
         {"noise", Tensor{draws.data(), {1, frames, config_.latent_size}, DType::kFloat32}},
         {"noise_scale", Tensor{scale.data(), {1}, DType::kFloat32}}});
    check(cudaMemcpyAsync(destination, require_output(*engines_.voice, "embeds"),
                          static_cast<std::size_t>(frames * config_.hidden_size * element_bytes()),
                          cudaMemcpyDeviceToDevice, stream_),
          "copying voice embeddings");
    check(cudaStreamSynchronize(stream_), "encoding a voice prompt");
}

VibeVoicePipeline::Decision VibeVoicePipeline::forward(ITrtModule& module, Sequence& sequence,
                                                       const void* inputs, std::int32_t rows) {
    if (rows <= 0 || rows > config_.prefill_chunk || sequence.length + rows > config_.capacity)
        throw std::runtime_error(
            "vibevoice sequence exceeds its KV cache capacity (max_sequence_length)");
    const auto half = static_cast<std::size_t>(config_.head_dim / 2);
    std::vector<float> cosines(static_cast<std::size_t>(rows) * config_.head_dim);
    std::vector<float> sines(cosines.size());
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto position = static_cast<float>(sequence.length + row);
        for (std::size_t index = 0; index < half; ++index) {
            // float32 angle, as the reference forms it; cos/sin rounded from double.
            const double angle = static_cast<double>(position * config_.rope_inv_freq[index]);
            const auto offset = static_cast<std::size_t>(row) * config_.head_dim;
            cosines[offset + index] = cosines[offset + index + half] =
                static_cast<float>(std::cos(angle));
            sines[offset + index] = sines[offset + index + half] =
                static_cast<float>(std::sin(angle));
        }
    }
    const auto table_bytes = cosines.size() * sizeof(float);
    check(cudaMemcpyAsync(rope_cos_.data(), cosines.data(), table_bytes, cudaMemcpyHostToDevice,
                          stream_),
          "uploading rotary tables");
    check(cudaMemcpyAsync(rope_sin_.data(), sines.data(), table_bytes, cudaMemcpyHostToDevice,
                          stream_),
          "uploading rotary tables");
    module.bind_external("inputs_embeds", const_cast<void*>(inputs),
                         {1, rows, config_.hidden_size});
    module.bind_external("rope_cos", rope_cos_.data(), {rows, config_.head_dim});
    module.bind_external("rope_sin", rope_sin_.data(), {rows, config_.head_dim});
    for (std::int32_t layer = 0; layer < config_.num_layers; ++layer) {
        const auto suffix = std::to_string(layer);
        const std::vector<std::int64_t> past{1, sequence.length, config_.num_kv_heads,
                                             config_.head_dim};
        module.bind_external("past_key_" + suffix, sequence.key(layer, 0), past);
        module.bind_external("past_value_" + suffix, sequence.value(layer, 0), past);
        module.bind_external("present_key_" + suffix, sequence.key(layer, sequence.length));
        module.bind_external("present_value_" + suffix, sequence.value(layer, sequence.length));
    }
    module.forward_device_async({});
    Decision decision;
    decision.logits.resize(config_.control_token_ids.size());
    check(cudaMemcpyAsync(decision.logits.data(), require_output(module, "control_logits"),
                          decision.logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
          "reading control logits");
    check(cudaMemcpyAsync(&decision.normalizer, require_output(module, "log_normalizer"),
                          sizeof(float), cudaMemcpyDeviceToHost, stream_),
          "reading the log normalizer");
    check(cudaStreamSynchronize(stream_), "running the language model");
    sequence.length += rows;
    return decision;
}

void VibeVoicePipeline::copy_hidden(ITrtModule& module, std::int32_t condition_row) {
    const auto bytes = static_cast<std::size_t>(config_.hidden_size * element_bytes());
    check(cudaMemcpyAsync(static_cast<char*>(conditions_.data()) +
                              static_cast<std::size_t>(condition_row) * bytes,
                          require_output(module, "hidden"), bytes, cudaMemcpyDeviceToDevice,
                          stream_),
          "copying a diffusion condition");
}

std::int32_t VibeVoicePipeline::choose(const Decision& conditional, const Decision* unconditional,
                                       double guidance_scale) const {
    // The reference restricts the choice to the four control tokens and takes the
    // first maximum in token-id order. With guidance it scores
    // s * (logp_cond - logp_uncond) + logp_uncond, the unconditional
    // log-probabilities held in the model's type.
    const auto scale = static_cast<float>(guidance_scale);
    std::size_t best = 0;
    float best_score = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < conditional.logits.size(); ++index) {
        float score = conditional.logits[index];
        if (unconditional != nullptr) {
            const float cond = conditional.logits[index] - conditional.normalizer;
            float uncond = unconditional->logits[index] - unconditional->normalizer;
            if (config_.bfloat16)
                uncond = round_bf16(uncond);
            score = scale * (cond - uncond) + uncond;
        }
        if (!std::isfinite(score))
            throw std::runtime_error("vibevoice control-token score is not finite");
        if (score > best_score) {
            best_score = score;
            best = index;
        }
    }
    return config_.control_token_ids[best];
}

std::vector<float> VibeVoicePipeline::sample_latent(NoiseStream& noise, DpmSolver& solver,
                                                    double guidance_scale) {
    const auto width = static_cast<std::size_t>(config_.latent_size);
    std::vector<float> sample(width);
    for (auto& value : sample) {
        value = noise.normal();
        if (config_.bfloat16)
            value = round_bf16(value);
    }
    const auto scale = static_cast<float>(guidance_scale);
    solver.reset();
    std::vector<float> pair(2 * width);
    std::vector<float> velocity(2 * width);
    std::vector<float> guided(width);
    engines_.head->bind_external("condition", conditions_.data(), {2, config_.hidden_size});
    for (std::int32_t step = 0; step < solver.steps(); ++step) {
        std::copy(sample.begin(), sample.end(), pair.begin());
        std::copy(sample.begin(), sample.end(), pair.begin() + static_cast<std::ptrdiff_t>(width));
        const auto timestep =
            static_cast<float>(solver.timesteps()[static_cast<std::size_t>(step)]);
        std::vector<float> timesteps{timestep, timestep};
        engines_.head->forward_async(
            {{"noisy_latent", Tensor{pair.data(), {2, config_.latent_size}, DType::kFloat32}},
             {"timestep", Tensor{timesteps.data(), {2}, DType::kFloat32}}});
        check(cudaMemcpyAsync(velocity.data(), require_output(*engines_.head, "velocity"),
                              velocity.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
              "reading the diffusion head");
        check(cudaStreamSynchronize(stream_), "running the diffusion head");
        for (std::size_t index = 0; index < width; ++index) {
            const float cond = velocity[index];
            const float uncond = velocity[width + index];
            if (config_.bfloat16)
                guided[index] = round_bf16(uncond + round_bf16(scale * round_bf16(cond - uncond)));
            else
                guided[index] = uncond + scale * (cond - uncond);
            if (!std::isfinite(guided[index]))
                throw std::runtime_error(
                    "vibevoice diffusion guidance is not finite; lower cfg_scale");
        }
        solver.step(step, guided, sample);
    }
    return sample;
}

void VibeVoicePipeline::feed_back(const std::vector<float>& latent, std::vector<float>& audio,
                                  int& state) {
    auto& feedback = *engines_.feedback;
    const std::vector<std::int64_t> state_shape{config_.stream_state_size};
    feedback.bind_external("state_in", states_[state].data(), state_shape);
    feedback.bind_external("state_out", states_[1 - state].data());
    auto input = latent;
    feedback.forward_async(
        {{"latent", Tensor{input.data(), {1, config_.latent_size}, DType::kFloat32}}});
    const auto offset = audio.size();
    audio.resize(offset + static_cast<std::size_t>(config_.hop_length));
    check(cudaMemcpyAsync(audio.data() + offset, require_output(feedback, "audio"),
                          static_cast<std::size_t>(config_.hop_length) * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_),
          "reading generated audio");
    check(cudaMemcpyAsync(step_input_.data(), require_output(feedback, "next_embed"),
                          static_cast<std::size_t>(config_.hidden_size * element_bytes()),
                          cudaMemcpyDeviceToDevice, stream_),
          "copying the next input embedding");
    check(cudaStreamSynchronize(stream_), "running the feedback step");
    if (!all_finite(audio.data() + offset, static_cast<std::size_t>(config_.hop_length)))
        throw std::runtime_error("vibevoice synthesized a NaN or infinite audio sample");
    state = 1 - state;
}

internal::AudioResult VibeVoicePipeline::run(const internal::TextToSpeechRequest& request,
                                             internal::ConfigView config) {
    const auto start = Clock::now();
    if (request.language)
        throw std::invalid_argument(
            "vibevoice has no language control; the language follows the text");
    const auto options = resolve_options(config);
    // The prompt holds every turn's text and leaves a position for generation, and every
    // rendered turn takes at least one position: the parser refuses text or turns that
    // cannot fit from their size and count, before copying, rendering or tokenizing them.
    const auto budget = static_cast<std::size_t>(config_.capacity) - 1U;
    std::vector<Turn> turns;
    try {
        turns = parse_script(request.text, tokenizer_->max_text_bytes(budget), budget);
    } catch (const std::length_error& error) {
        throw std::invalid_argument(std::string(error.what()) +
                                    "; it cannot fit max_sequence_length " +
                                    std::to_string(config_.capacity));
    }
    std::set<std::int32_t> speakers;
    for (const auto& turn : turns) {
        speakers.insert(turn.speaker);
        for (const auto& token : special_tokens_) {
            if (turn.text.find(token) != std::string::npos)
                throw std::invalid_argument("vibevoice text must not contain the special token " +
                                            token);
        }
    }

    std::vector<std::vector<float>> voices;
    std::vector<std::int64_t> voice_frames;
    // Each voice frame takes one prompt position: a prompt's voices together are read only
    // as far as the positions left allow.
    std::int64_t frames_left = static_cast<std::int64_t>(budget);
    for (std::size_t speaker = 0; speaker < options.voice_prompts.size(); ++speaker) {
        if (speakers.count(static_cast<std::int32_t>(speaker)) == 0)
            throw std::invalid_argument("vibevoice voice prompt " + std::to_string(speaker) +
                                        " has no matching 'Speaker " + std::to_string(speaker) +
                                        "' in the text");
        auto wav = read_wav(options.voice_prompts[speaker],
                            std::min(config_.max_voice_samples, frames_left * config_.hop_length));
        if (wav.channels != 1)
            throw std::invalid_argument("vibevoice voice prompts must be mono");
        if (wav.sample_rate != static_cast<std::uint32_t>(config_.sample_rate))
            throw std::invalid_argument("vibevoice voice prompts must be sampled at " +
                                        std::to_string(config_.sample_rate) + " Hz");
        auto prepared = prepare_voice(std::move(wav.samples), config_.target_dbfs,
                                      config_.normalize_eps, config_.hop_length);
        if (static_cast<std::int64_t>(prepared.size()) > config_.max_voice_samples)
            throw std::invalid_argument(
                "vibevoice voice prompts are limited to " +
                std::to_string(config_.max_voice_samples / config_.sample_rate) + " seconds");
        voice_frames.push_back(static_cast<std::int64_t>(prepared.size()) / config_.hop_length);
        frames_left -= voice_frames.back();
        voices.push_back(std::move(prepared));
    }

    // The prompt leaves at least one position for generation; tokenization stops as
    // soon as the prompt cannot fit.
    std::vector<std::int32_t> ids;
    try {
        ids = tokenizer_->encode_within(render_prompt(config_.frame, turns, voice_frames),
                                        static_cast<std::size_t>(config_.capacity) - 1U);
    } catch (const std::length_error&) {
        throw std::invalid_argument(
            "vibevoice prompt needs more than " + std::to_string(config_.capacity - 1) +
            " tokens and does not fit max_sequence_length " + std::to_string(config_.capacity));
    }
    if (ids.empty() || ids.back() != config_.audio_bos_token_id)
        throw std::runtime_error("vibevoice prompt does not end with the speech-start token");

    NoiseStream noise(options.seed);
    embed_tokens(ids, prompt_.data());
    const auto row_bytes = static_cast<std::size_t>(config_.hidden_size * element_bytes());
    std::size_t cursor = 0;
    for (std::size_t voice = 0; voice < voices.size(); ++voice) {
        while (cursor < ids.size() && ids[cursor] != config_.audio_token_id)
            ++cursor;
        std::size_t end = cursor;
        while (end < ids.size() && ids[end] == config_.audio_token_id)
            ++end;
        if (static_cast<std::int64_t>(end - cursor) != voice_frames[voice])
            throw std::runtime_error("vibevoice voice placeholders do not match the voice prompt");
        encode_voice(voices[voice], noise, static_cast<char*>(prompt_.data()) + cursor * row_bytes);
        cursor = end;
    }

    positive_.length = negative_.length = unconditional_.length = 0;
    Decision conditional;
    for (std::size_t offset = 0; offset < ids.size();
         offset += static_cast<std::size_t>(config_.prefill_chunk)) {
        const auto rows =
            std::min(ids.size() - offset, static_cast<std::size_t>(config_.prefill_chunk));
        conditional = forward(*engines_.lm_prefill, positive_,
                              static_cast<const char*>(prompt_.data()) + offset * row_bytes,
                              static_cast<std::int32_t>(rows));
    }
    copy_hidden(*engines_.lm_prefill, 0);
    check(cudaMemcpyAsync(step_input_.data(), special_row(config_.audio_bos_token_id), row_bytes,
                          cudaMemcpyDeviceToDevice, stream_),
          "seeding the step input");

    const bool token_guidance = options.guidance_scale != 1.0;
    DpmSolver solver(config_.train_sigmas, static_cast<std::int32_t>(options.diffusion_steps),
                     config_.bfloat16);
    for (auto& state : states_)
        check(cudaMemsetAsync(state.data(), 0, state.nbytes(), stream_), "clearing stream states");
    int state = 0;
    std::int32_t previous = config_.audio_bos_token_id;
    const std::int64_t limit =
        options.max_new_tokens > 0 ? options.max_new_tokens : config_.capacity - positive_.length;
    std::vector<float> audio;
    const double setup_ms = elapsed_ms(start);
    const auto generation_start = Clock::now();
    for (std::int64_t generated = 0; generated < limit;) {
        Decision unconditional;
        if (token_guidance)
            unconditional = forward(*engines_.lm_decode, unconditional_, special_row(previous), 1);
        const auto token =
            choose(conditional, token_guidance ? &unconditional : nullptr, options.guidance_scale);
        ++generated;
        if (token == config_.eos_token_id)
            break;
        if (token == config_.audio_bos_token_id)
            negative_.length = 0; // a new speech segment restarts the unconditional branch
        if (token == config_.audio_token_id) {
            forward(*engines_.lm_decode, negative_, step_input_.data(), 1);
            copy_hidden(*engines_.lm_decode, 1);
            feed_back(sample_latent(noise, solver, options.guidance_scale), audio, state);
        } else {
            check(cudaMemcpyAsync(step_input_.data(), special_row(token), row_bytes,
                                  cudaMemcpyDeviceToDevice, stream_),
                  "selecting a control-token embedding");
        }
        previous = token;
        if (generated >= limit || positive_.length >= config_.capacity)
            break;
        conditional = forward(*engines_.lm_decode, positive_, step_input_.data(), 1);
        copy_hidden(*engines_.lm_decode, 0);
    }
    if (audio.empty())
        throw std::runtime_error("vibevoice generated no speech for this text");
    internal::AudioResult result;
    result.samples = std::move(audio);
    result.sample_rate = static_cast<std::uint32_t>(config_.sample_rate);
    result.channels = 1;
    result.setup_ms = setup_ms;
    result.inference_ms = elapsed_ms(generation_start);
    return result;
}

} // namespace trtmc::vibevoice
