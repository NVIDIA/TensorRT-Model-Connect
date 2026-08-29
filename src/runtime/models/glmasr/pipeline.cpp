/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/glmasr/pipeline.h"

#include "runtime/models/glmasr/glmasr_mel_spectrogram.h"
#include "runtime/models/glmasr/glmasr_prompt_plan.h"
#include "utils/wav_reader.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

namespace {

// Pick the highest-scoring vocabulary entry from a logits output.
int32_t host_argmax_logits(const TensorMap& outputs, int32_t vocab_size) {
    const auto found = outputs.find("logits");
    if (found == outputs.end() || found->second.data == nullptr)
        return -1;
    const auto* logits = static_cast<const float*>(found->second.data);
    int32_t best = 0;
    float best_value = logits[0];
    for (int32_t index = 1; index < vocab_size; ++index) {
        if (logits[index] > best_value) {
            best_value = logits[index];
            best = index;
        }
    }
    return best;
}

} // namespace

GlmAsrPipeline::GlmAsrPipeline(std::unique_ptr<TrtModule> encoder,
                               std::unique_ptr<TrtModule> decoder,
                               std::unique_ptr<GlmAsrInferenceState> state, GlmAsrConfig config,
                               MelFilterbank mel_filterbank, cudaStream_t stream,
                               std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder)), state_(std::move(state)),
      config_(std::move(config)), mel_filterbank_(std::move(mel_filterbank)), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {}

GlmAsrPipeline::~GlmAsrPipeline() = default;

std::vector<float> GlmAsrPipeline::run_audio_encoder(const std::vector<float>& mel) {
    const auto expected = static_cast<std::size_t>(config_.mel_num_bins) * config_.mel_length;
    std::vector<float> padded(expected, 0.0F);
    std::copy_n(mel.data(), std::min(mel.size(), expected), padded.data());

    // The encoder declares an FP32 mel input at every build precision, so no
    // conversion is needed here. The runtime's host-to-device copy is untyped.
    TensorMap inputs;
    inputs["mel_features"] =
        Tensor{padded.data(), {config_.mel_num_bins, config_.mel_length}, DType::kFloat32};

    ++stats_.encoder_launches;
    TensorMap outputs = encoder_->forward(inputs);
    const auto found = outputs.find("audio_embeddings");
    if (found == outputs.end() || found->second.data == nullptr)
        throw std::runtime_error("GlmAsrPipeline: encoder produced no audio_embeddings");

    // The encoder casts audio_embeddings back to FP32 before marking it.
    const auto count =
        static_cast<std::size_t>(config_.max_audio_embeddings) * config_.audio_embedding_size;
    std::vector<float> embeddings(count, 0.0F);
    std::memcpy(embeddings.data(), found->second.data, count * sizeof(float));
    return embeddings;
}

std::vector<int32_t> GlmAsrPipeline::tokenize_instruction() const {
    if (!tokenizer_)
        return {};
    return tokenizer_->encode(config_.transcription_prompt);
}

int32_t GlmAsrPipeline::run_decode_step(int32_t token_id, const float* audio_embed) {
    TensorMap inputs;
    int32_t token = token_id;
    inputs["token_id"] = Tensor{&token, {1}, DType::kInt32};

    // The decode engine carries the embed-input contract too, so a placeholder
    // position substitutes its projected audio frame one token at a time.
    const auto hidden = config_.audio_embedding_size;
    std::vector<float> embed_row;
    float use_embed = 0.0F;
    if (decoder_->has_input("input_embed")) {
        embed_row.assign(static_cast<std::size_t>(hidden), 0.0F);
        if (audio_embed != nullptr) {
            std::copy_n(audio_embed, hidden, embed_row.data());
            use_embed = 1.0F;
        }
        inputs["input_embed"] = Tensor{embed_row.data(), {1, hidden}, DType::kFloat32};
        inputs["use_input_embed"] = Tensor{&use_embed, {1}, DType::kFloat32};
    }
    state_->prepare_step(inputs);

    ++stats_.decode_launches;
    TensorMap outputs = decoder_->forward(inputs);
    state_->advance();
    return host_argmax_logits(outputs, config_.vocab_size);
}

std::vector<int32_t> GlmAsrPipeline::run_decoder(const std::vector<int32_t>& input_ids,
                                                 const std::vector<float>& audio_embeds,
                                                 int32_t audio_offset, int32_t num_audio_embeddings,
                                                 int32_t max_new_tokens) {
    if (input_ids.empty() || max_new_tokens <= 0)
        return {};

    state_->reset();
    state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));
    state_->bind_to(*decoder_);

    // Replay the prompt one token at a time, feeding each placeholder position
    // its projected audio frame through the decoder's embed-input contract.
    int32_t next_token = -1;
    const auto hidden = static_cast<std::size_t>(config_.audio_embedding_size);
    for (std::size_t index = 0; index < input_ids.size(); ++index) {
        const auto audio_index = static_cast<int64_t>(index) - audio_offset;
        const bool is_audio = audio_index >= 0 && audio_index < num_audio_embeddings;
        const float* embed =
            is_audio ? audio_embeds.data() + static_cast<std::size_t>(audio_index) * hidden
                     : nullptr;
        next_token = run_decode_step(input_ids[index], embed);
    }
    state_->mark_prefill_complete();

    std::vector<int32_t> generated;
    generated.reserve(static_cast<std::size_t>(max_new_tokens));
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        if (next_token < 0 || next_token == config_.eos_token_id)
            break;
        generated.push_back(next_token);
        next_token = run_decode_step(next_token);
    }
    return generated;
}

TextResult GlmAsrPipeline::transcribe(const float* audio_samples, int32_t num_samples,
                                      int32_t max_new_tokens, int32_t input_sample_rate) {
    stats_ = {};
    if (audio_samples == nullptr || num_samples <= 0)
        return TextResult{"", {}};

    const float* samples = audio_samples;
    int32_t sample_count = num_samples;
    std::vector<float> resampled;
    if (input_sample_rate > 0 && input_sample_rate != config_.mel_sampling_rate) {
        resampled = resample_linear(audio_samples, num_samples, input_sample_rate,
                                    config_.mel_sampling_rate);
        samples = resampled.data();
        sample_count = static_cast<int32_t>(resampled.size());
    }

    if (mel_filterbank_.data.empty())
        throw std::runtime_error("GlmAsrPipeline: bundle carries no mel filterbank");

    const glmasr::MelResult mel = glmasr::extract_mel_spectrogram(
        samples, sample_count, mel_filterbank_.data.data(), mel_filterbank_.n_freq_bins,
        mel_filterbank_.n_mel_bins, config_.mel_n_fft, config_.mel_hop_length,
        config_.mel_chunk_length, config_.mel_sampling_rate);
    if (mel.data.empty())
        return TextResult{"", {}};

    // The engine always consumes a padded chunk, but only the embeddings the
    // real audio covers become prompt placeholders. This mirrors the reference
    // processor, which derives the placeholder count from the unpadded length.
    const int32_t real_mel_frames =
        glmasr::mel_frames_for_samples(sample_count, config_.mel_hop_length);
    const int32_t num_audio_embeddings =
        std::min(glmasr::audio_embedding_count(real_mel_frames, config_.audio_merge_factor),
                 config_.max_audio_embeddings);

    const std::vector<float> audio_embeds = run_audio_encoder(mel.data);

    const auto plan =
        glmasr::build_prompt_plan(config_, tokenize_instruction(), num_audio_embeddings);
    stats_.prompt_tokens = static_cast<int32_t>(plan.input_ids.size());
    stats_.audio_embeddings = plan.num_audio_embeddings;

    // One placeholder per audio embedding means the prompt grows with the clip.
    // Overflowing the compiled cache silently returns nonsense, so refuse
    // instead and name the numbers the caller needs to fix the build.
    if (config_.max_cache_length > 0 && stats_.prompt_tokens >= config_.max_cache_length) {
        std::cerr << "[glmasr] Audio needs " << stats_.audio_embeddings << " embeddings, giving a "
                  << stats_.prompt_tokens << " token prompt, but the bundle's KV cache holds "
                  << config_.max_cache_length
                  << ". Rebuild with a larger --max-cache-length or shorten the audio."
                  << std::endl;
        return TextResult{"", {}};
    }

    const std::vector<int32_t> generated =
        run_decoder(plan.input_ids, audio_embeds, plan.audio_offset, plan.num_audio_embeddings,
                    max_new_tokens > 0 ? max_new_tokens : default_max_new_tokens());

    // TRTMC_GLMASR_DEBUG reports how the audio mapped onto the prompt, which is
    // what to check first when a transcript comes back empty or truncated.
    if (std::getenv("TRTMC_GLMASR_DEBUG") != nullptr) {
        std::cerr << "[glmasr] prompt_tokens=" << stats_.prompt_tokens
                  << " audio_embeddings=" << stats_.audio_embeddings
                  << " audio_offset=" << plan.audio_offset
                  << " decode_launches=" << stats_.decode_launches
                  << " generated=" << generated.size() << std::endl;
    }

    std::string text;
    if (tokenizer_ && !generated.empty())
        text = tokenizer_->decode(generated);
    return TextResult{std::move(text), generated};
}

} // namespace trtmc
