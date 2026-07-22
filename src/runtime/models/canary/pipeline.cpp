/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/canary/pipeline.h"

#include "plugin_helpers.h"
#include "runtime/models/canary/canary_cross_kv_apply.h"
#include "runtime/models/canary/canary_cross_kv_plan.h"
#include "runtime/models/canary/canary_decode_policy.h"
#include "runtime/models/canary/canary_host_plan.h"
#include "runtime/models/canary/canary_mel_spectrogram.h"
#include "runtime/models/canary/canary_request.h"
#include "runtime/models/canary/decode_runtime.h"
#include "trtmc/tokenizer.h"
#include "utils/wav_reader.h"

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace trtmc {

namespace {

using CanaryClock = std::chrono::steady_clock;

bool canary_stage_timing_enabled() {
    const char* value = std::getenv("TRTMC_CANARY_STAGE_TIMING");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

double elapsed_ms(CanaryClock::time_point start, CanaryClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string decode_canary_tokens(const ITokenizer& tokenizer, const std::vector<int32_t>& token_ids,
                                 int32_t eot_token_id) {
    auto content_end = token_ids.end();
    while (content_end != token_ids.begin() && *(content_end - 1) == eot_token_id)
        --content_end;
    return tokenizer.decode(std::vector<int32_t>(token_ids.begin(), content_end));
}

struct CanaryStageTimestamps {
    CanaryClock::time_point transcribe_start;
    CanaryClock::time_point resample_end;
    CanaryClock::time_point mel_end;
    CanaryClock::time_point encoder_end;
    CanaryClock::time_point cross_kv_end;
    CanaryClock::time_point decoder_end;
    CanaryClock::time_point tokenize_end;
};

bool is_canary_control_token_start(const std::string& text, std::size_t position) {
    return text[position] == '<' && position + 1 < text.size() && text[position + 1] == '|';
}

bool is_canary_control_token_end(const std::string& text, std::size_t position) {
    return text[position] == '>' && position > 0 && text[position - 1] == '|';
}

std::string remove_punctuation_outside_control_tokens(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size());
    bool in_control_token = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (!in_control_token && is_canary_control_token_start(text, i))
            in_control_token = true;
        if (in_control_token || std::ispunct(ch) == 0)
            cleaned.push_back(static_cast<char>(ch));
        if (in_control_token && is_canary_control_token_end(text, i))
            in_control_token = false;
    }
    return cleaned;
}

void report_canary_stage_timing(bool enabled, const CanaryStageTimestamps& timestamps) {
    if (!enabled)
        return;
    std::ostringstream timing;
    timing << "[trtmc.canary_timing.json] {\"resample_ms\":"
           << elapsed_ms(timestamps.transcribe_start, timestamps.resample_end)
           << ",\"mel_ms\":" << elapsed_ms(timestamps.resample_end, timestamps.mel_end)
           << ",\"encoder_ms\":" << elapsed_ms(timestamps.mel_end, timestamps.encoder_end)
           << ",\"cross_kv_ms\":" << elapsed_ms(timestamps.encoder_end, timestamps.cross_kv_end)
           << ",\"decoder_ms\":" << elapsed_ms(timestamps.cross_kv_end, timestamps.decoder_end)
           << ",\"tokenizer_ms\":" << elapsed_ms(timestamps.decoder_end, timestamps.tokenize_end)
           << ",\"total_ms\":" << elapsed_ms(timestamps.transcribe_start, timestamps.tokenize_end)
           << '}';
    std::cerr << timing.str() << std::endl;
}

void validate_canary_audio_input(const float* audio_data, int32_t num_samples) {
    if (audio_data == nullptr || num_samples <= 0) {
        throw std::invalid_argument("Canary transcription requires non-empty audio samples");
    }
}

void validate_canary_output_budget(const CanaryConfig& model, const TranscriptionConfig& cfg,
                                   std::size_t prompt_tokens) {
    const int32_t available_output_tokens =
        model.max_target_positions - static_cast<int32_t>(prompt_tokens);
    if (available_output_tokens <= 0 || cfg.max_output_tokens > available_output_tokens) {
        throw std::invalid_argument("Canary max_output_tokens must be in [1, " +
                                    std::to_string(std::max(available_output_tokens, 0)) +
                                    "] after accounting for the decoder prompt");
    }
}

double validate_canary_input_duration(int32_t num_samples, int32_t sample_rate,
                                      const TranscriptionConfig& cfg) {
    const double duration_seconds =
        static_cast<double>(num_samples) / static_cast<double>(sample_rate);
    if (cfg.max_input_duration_seconds > 0.0F &&
        duration_seconds > static_cast<double>(cfg.max_input_duration_seconds) + 1.0e-6) {
        throw std::invalid_argument("Canary input duration " + std::to_string(duration_seconds) +
                                    " seconds exceeds max_input_duration_seconds=" +
                                    std::to_string(cfg.max_input_duration_seconds));
    }
    return duration_seconds;
}

double resolve_canary_segment_duration(double input_duration_seconds, double model_segment_seconds,
                                       const TranscriptionConfig& cfg) {
    const double segment_seconds = cfg.segment_duration_seconds > 0.0F
                                       ? static_cast<double>(cfg.segment_duration_seconds)
                                       : model_segment_seconds;
    if (segment_seconds <= 0.0 || segment_seconds > model_segment_seconds) {
        throw std::invalid_argument(
            "Canary segment_duration_seconds must be > 0 and <= the bundle limit of " +
            std::to_string(model_segment_seconds) + " seconds");
    }
    if (cfg.segment_duration_seconds <= 0.0F && input_duration_seconds > model_segment_seconds) {
        throw std::invalid_argument(
            "Canary input exceeds the bundle's single-segment limit of " +
            std::to_string(model_segment_seconds) +
            " seconds; set segment_duration_seconds to enable segmented decoding");
    }
    return segment_seconds;
}

int32_t canary_segment_sample_count(double segment_seconds, int32_t sample_rate) {
    const int64_t segment_samples =
        static_cast<int64_t>(std::llround(segment_seconds * static_cast<double>(sample_rate)));
    if (segment_samples <= 0 || segment_samples > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("Canary segment duration produces an invalid sample count");
    }
    return static_cast<int32_t>(segment_samples);
}

void append_canary_transcription_segment(TextResult& combined, TextResult segment, int64_t offset,
                                         int32_t count, int32_t sample_rate,
                                         const TranscriptionConfig& cfg) {
    if (!cfg.punctuation) {
        segment.text = remove_punctuation_outside_control_tokens(segment.text);
    }
    if (cfg.timestamps) {
        TranscriptionSegment timed;
        timed.start_seconds = static_cast<double>(offset) / static_cast<double>(sample_rate);
        timed.end_seconds = static_cast<double>(offset + count) / static_cast<double>(sample_rate);
        timed.text = segment.text;
        timed.token_ids = segment.token_ids;
        combined.segments.push_back(std::move(timed));
    }
    if (!combined.text.empty() && !segment.text.empty()) {
        combined.text += '\n';
    }
    combined.text += segment.text;
    combined.token_ids.insert(combined.token_ids.end(), segment.token_ids.begin(),
                              segment.token_ids.end());
    combined.prefill_ms += segment.prefill_ms;
    combined.decode_ms += segment.decode_ms;
}

int32_t canary_module_batch_capacity(const TrtModule& module, const std::string& input_name) {
    if (!module.has_input(input_name) || !module.input_is_dynamic(input_name))
        return 1;
    const auto shape =
        module.input_profile_shape(input_name, module.profile_idx(), ProfileShapeSelector::kMax);
    if (shape.empty() || shape.front() <= 0 ||
        shape.front() > std::numeric_limits<int32_t>::max()) {
        return 1;
    }
    return static_cast<int32_t>(shape.front());
}

void validate_canary_pipeline_components(const TrtModule* encoder, const TrtModule* decoder,
                                         const CanaryInferenceState* state) {
    if (encoder == nullptr || !encoder->ok())
        throw std::runtime_error("CanaryPipeline: invalid encoder module");
    if (decoder == nullptr || !decoder->ok())
        throw std::runtime_error("CanaryPipeline: invalid decoder module");
    if (state == nullptr || !state->ok())
        throw std::runtime_error("CanaryPipeline: invalid inference state");
}

void* allocate_canary_cross_kv_buffer(int32_t num_decoder_layers, std::size_t bytes) {
    if (num_decoder_layers <= 0 || bytes == 0)
        return nullptr;
    void* buffer = nullptr;
    const auto status = cudaMalloc(&buffer, bytes);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CanaryPipeline: cross-attention allocation failed: ") +
            cudaGetErrorString(status));
    }
    return buffer;
}

struct CanaryBatchSegment {
    std::size_t request_index{0};
    int64_t offset{0};
    int32_t count{0};
    int32_t sample_rate{0};
    std::vector<int32_t> initial_tokens;
    int32_t max_output_tokens{0};
    int32_t beam_size{1};
};

struct CanaryPreparedSegment {
    canary::MelResult mel;
    int32_t actual_encoder_length{0};
};

struct CanaryBatchWorkGroups {
    std::unordered_map<int32_t, std::vector<std::size_t>> by_beam_size;
    std::vector<int32_t> beam_order;
};

struct CanaryPreparedBatchChunk {
    std::vector<std::size_t> valid_indices;
    std::vector<std::vector<float>> mel_batch;
    std::vector<int32_t> valid_frames;
    std::vector<int32_t> actual_encoder_lengths;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<int32_t> output_limits;
    int32_t mel_bins{0};
    int32_t mel_frames{0};
};

std::vector<CanaryBatchSegment>
build_canary_batch_work(const std::vector<TranscriptionRequest>& requests,
                        const CanaryConfig& canary_config, ITokenizer* tokenizer,
                        int32_t mel_sampling_rate, int32_t mel_chunk_length) {
    std::vector<CanaryBatchSegment> work;
    for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
        const auto& request = requests[request_index];
        validate_canary_audio_input(request.audio_samples.data(),
                                    static_cast<int32_t>(request.audio_samples.size()));
        auto initial_tokens = make_canary_request_tokens(canary_config, request.config, tokenizer);
        validate_canary_output_budget(canary_config, request.config, initial_tokens.size());

        const int32_t sample_rate = request.config.input_sample_rate > 0
                                        ? request.config.input_sample_rate
                                        : mel_sampling_rate;
        const auto sample_count = static_cast<int32_t>(request.audio_samples.size());
        const double duration_seconds =
            validate_canary_input_duration(sample_count, sample_rate, request.config);
        const double segment_seconds = resolve_canary_segment_duration(
            duration_seconds, static_cast<double>(mel_chunk_length), request.config);
        const int32_t segment_samples = canary_segment_sample_count(segment_seconds, sample_rate);

        for (int64_t offset = 0; offset < sample_count; offset += segment_samples) {
            const int32_t count = static_cast<int32_t>(
                std::min<int64_t>(segment_samples, static_cast<int64_t>(sample_count) - offset));
            work.push_back({request_index, offset, count, sample_rate, initial_tokens,
                            request.config.max_output_tokens, request.config.beam_size});
        }
    }
    return work;
}

CanaryBatchWorkGroups group_canary_batch_work(const std::vector<CanaryBatchSegment>& work) {
    CanaryBatchWorkGroups groups;
    for (std::size_t index = 0; index < work.size(); ++index) {
        const int32_t beam_size = work[index].beam_size;
        auto [it, inserted] = groups.by_beam_size.try_emplace(beam_size);
        if (inserted)
            groups.beam_order.push_back(beam_size);
        it->second.push_back(index);
    }
    return groups;
}

CanaryPreparedSegment
prepare_canary_batch_segment(const CanaryBatchSegment& item, const TranscriptionRequest& request,
                             const MelFilterbank* mel_filterbank, int32_t mel_n_fft,
                             int32_t mel_win_length, int32_t mel_hop_length,
                             int32_t mel_chunk_length, int32_t mel_sampling_rate, float mel_preemph,
                             bool mel_normalize_per_feature, const CanaryConfig& canary_config) {
    const float* samples = request.audio_samples.data() + item.offset;
    int32_t sample_count = item.count;
    std::vector<float> resampled;
    if (item.sample_rate != mel_sampling_rate) {
        resampled = resample_linear(samples, sample_count, item.sample_rate, mel_sampling_rate);
        quantize_canary_pcm16_inplace(resampled);
        samples = resampled.data();
        sample_count = static_cast<int32_t>(resampled.size());
    }

    CanaryPreparedSegment prepared;
    if (mel_filterbank != nullptr && !mel_filterbank->data.empty()) {
        prepared.mel = canary::extract_mel_spectrogram(
            samples, sample_count, mel_filterbank->data.data(), mel_filterbank->n_freq_bins,
            mel_filterbank->n_mel_bins, mel_n_fft, mel_win_length, mel_hop_length, mel_chunk_length,
            mel_sampling_rate, mel_preemph, mel_normalize_per_feature);
    }
    if (!prepared.mel.data.empty()) {
        const int32_t valid_frames =
            prepared.mel.valid_frames > 0 ? prepared.mel.valid_frames : prepared.mel.n_frames;
        prepared.actual_encoder_length = compute_canary_actual_encoder_length(
            valid_frames, resolve_canary_expected_mel_length(canary_config),
            canary_config.max_source_positions);
    }
    return prepared;
}

std::vector<std::future<CanaryPreparedSegment>> launch_canary_batch_preparation(
    const std::vector<std::size_t>& indices, std::size_t chunk_start, std::size_t chunk_end,
    const std::vector<CanaryBatchSegment>& work, const std::vector<TranscriptionRequest>& requests,
    const MelFilterbank* mel_filterbank, int32_t mel_n_fft, int32_t mel_win_length,
    int32_t mel_hop_length, int32_t mel_chunk_length, int32_t mel_sampling_rate, float mel_preemph,
    bool mel_normalize_per_feature, const CanaryConfig& canary_config) {
    std::vector<std::future<CanaryPreparedSegment>> futures;
    futures.reserve(chunk_end - chunk_start);
    for (std::size_t cursor = chunk_start; cursor < chunk_end; ++cursor) {
        const std::size_t index = indices[cursor];
        futures.push_back(std::async(
            std::launch::async, [&requests, &work, index, mel_filterbank, mel_n_fft, mel_win_length,
                                 mel_hop_length, mel_chunk_length, mel_sampling_rate, mel_preemph,
                                 mel_normalize_per_feature, &canary_config] {
                const auto& item = work[index];
                return prepare_canary_batch_segment(
                    item, requests[item.request_index], mel_filterbank, mel_n_fft, mel_win_length,
                    mel_hop_length, mel_chunk_length, mel_sampling_rate, mel_preemph,
                    mel_normalize_per_feature, canary_config);
            }));
    }
    return futures;
}

void set_or_validate_canary_mel_shape(const CanaryPreparedSegment& prepared,
                                      CanaryPreparedBatchChunk& chunk) {
    if (chunk.mel_batch.empty()) {
        chunk.mel_bins = prepared.mel.n_mels;
        chunk.mel_frames = prepared.mel.n_frames;
        return;
    }
    if (prepared.mel.n_mels != chunk.mel_bins || prepared.mel.n_frames != chunk.mel_frames) {
        throw std::runtime_error("Canary mel batch contains mismatched shapes");
    }
}

CanaryPreparedBatchChunk collect_canary_prepared_batch_chunk(
    std::vector<std::future<CanaryPreparedSegment>>& futures,
    const std::vector<std::size_t>& indices, std::size_t chunk_start,
    const std::vector<CanaryBatchSegment>& work, std::vector<TextResult>& segment_results) {
    CanaryPreparedBatchChunk chunk;
    for (std::size_t future_index = 0; future_index < futures.size(); ++future_index) {
        const std::size_t index = indices[chunk_start + future_index];
        auto prepared = futures[future_index].get();
        if (prepared.mel.data.empty()) {
            segment_results[index] = TextResult{"[mel extraction failed]", {}};
            continue;
        }
        set_or_validate_canary_mel_shape(prepared, chunk);
        chunk.valid_indices.push_back(index);
        chunk.valid_frames.push_back(prepared.mel.valid_frames > 0 ? prepared.mel.valid_frames
                                                                   : prepared.mel.n_frames);
        chunk.actual_encoder_lengths.push_back(prepared.actual_encoder_length);
        chunk.mel_batch.push_back(std::move(prepared.mel.data));
        chunk.prompts.push_back(work[index].initial_tokens);
        chunk.output_limits.push_back(work[index].max_output_tokens);
    }
    return chunk;
}

void store_canary_batch_decodes(std::vector<std::vector<int32_t>>& output_ids,
                                const std::vector<std::size_t>& valid_indices,
                                ITokenizer* tokenizer, int32_t eot_token_id,
                                std::vector<TextResult>& segment_results) {
    for (std::size_t batch = 0; batch < valid_indices.size(); ++batch) {
        TextResult result;
        result.token_ids = std::move(output_ids[batch]);
        if (tokenizer != nullptr && !result.token_ids.empty())
            result.text = decode_canary_tokens(*tokenizer, result.token_ids, eot_token_id);
        segment_results[valid_indices[batch]] = std::move(result);
    }
}

void report_canary_batch_timing(CanaryClock::time_point begin, std::size_t batch_size,
                                int32_t beam_size) {
    if (!canary_stage_timing_enabled())
        return;
    const double total = elapsed_ms(begin, CanaryClock::now());
    std::cerr << "[trtmc.canary_batch_timing.json] {\"batch_size\":" << batch_size
              << ",\"beam_size\":" << beam_size << ",\"total_ms\":" << total
              << ",\"per_sample_ms\":" << total / static_cast<double>(batch_size) << '}'
              << std::endl;
}

std::vector<TextResult>
merge_canary_batch_segments(const std::vector<TranscriptionRequest>& requests,
                            const std::vector<CanaryBatchSegment>& work,
                            std::vector<TextResult>& segment_results) {
    std::vector<TextResult> results(requests.size());
    for (std::size_t index = 0; index < work.size(); ++index) {
        const auto& item = work[index];
        append_canary_transcription_segment(
            results[item.request_index], std::move(segment_results[index]), item.offset, item.count,
            item.sample_rate, requests[item.request_index].config);
    }
    return results;
}

struct CanaryEncoderBatchMask {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

std::vector<float> pack_canary_mel_batch(const std::vector<std::vector<float>>& mel_data,
                                         std::size_t sample_values) {
    std::vector<float> packed(mel_data.size() * sample_values, 0.0F);
    for (std::size_t batch = 0; batch < mel_data.size(); ++batch) {
        if (mel_data[batch].size() != sample_values)
            throw std::invalid_argument("Canary encoder batch mel shape mismatch");
        std::copy(mel_data[batch].begin(), mel_data[batch].end(),
                  packed.begin() + static_cast<std::ptrdiff_t>(batch * sample_values));
    }
    return packed;
}

void copy_canary_encoder_mask_row(const std::vector<float>& row, bool full_attention_mask,
                                  int32_t encoder_sequence_length, float* output) {
    if (!full_attention_mask) {
        std::copy(row.begin(), row.end(), output);
        return;
    }
    for (int32_t query = 0; query < encoder_sequence_length; ++query) {
        std::copy(row.begin(), row.end(),
                  output + static_cast<std::ptrdiff_t>(query) * encoder_sequence_length);
    }
}

CanaryEncoderBatchMask make_canary_encoder_batch_mask(const std::vector<int32_t>& valid_mel_frames,
                                                      int32_t expected_mel_length,
                                                      int32_t encoder_sequence_length,
                                                      const std::vector<int64_t>& engine_shape) {
    const bool full_attention_mask =
        engine_shape.size() == 4 && engine_shape[2] == encoder_sequence_length;
    const std::size_t values_per_sample =
        full_attention_mask ? static_cast<std::size_t>(encoder_sequence_length) *
                                  static_cast<std::size_t>(encoder_sequence_length)
                            : static_cast<std::size_t>(encoder_sequence_length);

    CanaryEncoderBatchMask mask;
    mask.values.resize(valid_mel_frames.size() * values_per_sample);
    for (std::size_t batch = 0; batch < valid_mel_frames.size(); ++batch) {
        int32_t actual_encoder_length = compute_canary_actual_encoder_length(
            valid_mel_frames[batch], expected_mel_length, encoder_sequence_length);
        if (actual_encoder_length <= 0)
            actual_encoder_length = encoder_sequence_length;
        const auto row =
            build_canary_encoder_mask_values(encoder_sequence_length, actual_encoder_length);
        copy_canary_encoder_mask_row(row, full_attention_mask, encoder_sequence_length,
                                     mask.values.data() +
                                         static_cast<std::ptrdiff_t>(batch * values_per_sample));
    }
    mask.shape = full_attention_mask
                     ? std::vector<int64_t>{static_cast<int64_t>(valid_mel_frames.size()), 1,
                                            encoder_sequence_length, encoder_sequence_length}
                     : std::vector<int64_t>{static_cast<int64_t>(valid_mel_frames.size()), 1, 1,
                                            encoder_sequence_length};
    return mask;
}

void zero_pad_canary_encoder_batch(uint8_t* encoder_output,
                                   const std::vector<int32_t>& actual_encoder_lengths,
                                   int32_t max_source_positions, int32_t hidden_size,
                                   std::size_t sample_bytes, cudaStream_t stream) {
    if (max_source_positions <= 0 || hidden_size <= 0)
        return;
    const std::size_t sample_elements =
        static_cast<std::size_t>(max_source_positions) * static_cast<std::size_t>(hidden_size);
    if (sample_bytes == 0 || sample_bytes % sample_elements != 0)
        throw std::invalid_argument("Canary encoder output has an invalid byte size");
    const std::size_t element_size = sample_bytes / sample_elements;
    for (std::size_t sample = 0; sample < actual_encoder_lengths.size(); ++sample) {
        const auto plan = make_canary_cross_kv_plan(max_source_positions, hidden_size,
                                                    actual_encoder_lengths[sample], element_size);
        if (!plan.zero_pad_encoder_output || plan.pad_bytes == 0)
            continue;
        const auto status = cudaMemsetAsync(
            encoder_output + sample * sample_bytes + plan.valid_bytes, 0, plan.pad_bytes, stream);
        if (status != cudaSuccess)
            throw std::runtime_error("Canary encoder padding failed");
    }
}

void copy_canary_cross_attention_lanes(uint8_t* cross_attention, const uint8_t* encoder_output,
                                       const std::vector<int32_t>& lane_to_sample,
                                       std::size_t sample_count, std::size_t sample_bytes,
                                       cudaStream_t stream) {
    for (std::size_t lane = 0; lane < lane_to_sample.size(); ++lane) {
        const int32_t sample = lane_to_sample[lane];
        if (sample < 0 || static_cast<std::size_t>(sample) >= sample_count)
            throw std::out_of_range("Canary cross-attention sample lane is out of range");
        const auto status =
            cudaMemcpyAsync(cross_attention + lane * sample_bytes,
                            encoder_output + static_cast<std::size_t>(sample) * sample_bytes,
                            sample_bytes, cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess)
            throw std::runtime_error("Canary cross-attention copy failed");
    }
}

void bind_canary_cross_attention_layers(TrtModule& decoder, int32_t num_decoder_layers,
                                        void* cross_kv, const std::vector<int64_t>& cross_shape) {
    for (int32_t layer = 0; layer < num_decoder_layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        if (decoder.input_rank("cross_k" + suffix) == 3) {
            decoder.bind_external("cross_k" + suffix, cross_kv, cross_shape);
            decoder.bind_external("cross_v" + suffix, cross_kv, cross_shape);
        } else {
            decoder.bind_external("cross_k" + suffix, cross_kv);
            decoder.bind_external("cross_v" + suffix, cross_kv);
        }
    }
}

std::size_t validate_canary_batch_prompts(const std::vector<std::vector<int32_t>>& initial_tokens) {
    if (initial_tokens.empty())
        return 0;
    const std::size_t prompt_length = initial_tokens.front().size();
    for (const auto& prompt : initial_tokens) {
        if (prompt.size() != prompt_length)
            throw std::invalid_argument("Canary batch prompts must have equal lengths");
    }
    return prompt_length;
}

bool update_canary_greedy_batch_step(const std::vector<float>& logits, std::size_t vocab_size,
                                     int32_t step, const std::vector<int32_t>& max_new_tokens,
                                     int32_t eot_token_id, std::vector<int32_t>& tokens,
                                     std::vector<std::vector<int32_t>>& output,
                                     std::vector<bool>& finished) {
    bool all_finished = true;
    for (std::size_t batch = 0; batch < tokens.size(); ++batch) {
        if (finished[batch] || step >= max_new_tokens[batch]) {
            tokens[batch] = eot_token_id;
            finished[batch] = true;
            continue;
        }
        const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(batch * vocab_size);
        const auto end = begin + static_cast<std::ptrdiff_t>(vocab_size);
        tokens[batch] = static_cast<int32_t>(std::distance(begin, std::max_element(begin, end)));
        output[batch].push_back(tokens[batch]);
        finished[batch] = tokens[batch] == eot_token_id || step + 1 >= max_new_tokens[batch];
        all_finished = all_finished && finished[batch];
    }
    return all_finished;
}

using CanaryBeamBatch = std::vector<std::vector<CanaryBeamHypothesis>>;

struct CanaryBeamBatchStep {
    CanaryBeamBatch beams;
    std::vector<int32_t> parent_lanes;
    std::vector<int32_t> tokens;
    bool any_active{false};
};

CanaryBeamBatch initialize_canary_beam_batch(const std::vector<float>& logits,
                                             int32_t request_batch, std::size_t vocab_size) {
    CanaryBeamBatch beams(static_cast<std::size_t>(request_batch));
    for (int32_t batch = 0; batch < request_batch; ++batch) {
        CanaryBeamHypothesis initial;
        initial.state_slot = batch;
        const auto begin = logits.begin() + static_cast<std::ptrdiff_t>(batch) *
                                                static_cast<std::ptrdiff_t>(vocab_size);
        initial.logits.assign(begin, begin + static_cast<std::ptrdiff_t>(vocab_size));
        beams[static_cast<std::size_t>(batch)].push_back(std::move(initial));
    }
    return beams;
}

bool canary_beam_candidate_is_active(const CanaryBeamCandidate& candidate, bool sample_finished,
                                     bool final_step) {
    return !sample_finished && !candidate.hypothesis.finished && !final_step &&
           candidate.parent_slot >= 0;
}

bool append_canary_next_beam_sample(const std::vector<CanaryBeamHypothesis>& current_beams,
                                    int32_t max_new_tokens, int32_t step, int32_t batch,
                                    int32_t beam_size, int32_t eot_token_id,
                                    std::vector<int32_t>& parent_lanes,
                                    std::vector<int32_t>& tokens,
                                    std::vector<CanaryBeamHypothesis>& next_beams) {
    CanaryDecodeLoopResult status;
    std::vector<CanaryBeamCandidate> candidates;
    bool sample_finished = false;
    if (!collect_canary_beam_candidates(current_beams, eot_token_id, beam_size, candidates,
                                        sample_finished, status)) {
        throw std::runtime_error("Canary batched beam search failed: " + status.error);
    }
    rank_canary_beam_candidates(candidates, beam_size, CanaryDefaultBeamLengthPenalty);
    const bool final_step = step + 1 >= max_new_tokens;
    for (int32_t beam = 0; beam < beam_size; ++beam) {
        const int32_t lane = batch * beam_size + beam;
        auto& candidate = candidates.at(static_cast<std::size_t>(beam));
        const bool active = canary_beam_candidate_is_active(candidate, sample_finished, final_step);
        if (active) {
            parent_lanes[static_cast<std::size_t>(lane)] = candidate.parent_slot;
            tokens[static_cast<std::size_t>(lane)] = candidate.token;
            candidate.hypothesis.state_slot = lane;
        } else {
            candidate.hypothesis.state_slot = -1;
        }
        next_beams.push_back(std::move(candidate.hypothesis));
    }
    return std::any_of(next_beams.begin(), next_beams.end(),
                       [](const CanaryBeamHypothesis& beam) { return beam.state_slot >= 0; });
}

CanaryBeamBatchStep make_canary_beam_batch_step(const CanaryBeamBatch& beams,
                                                const std::vector<int32_t>& max_new_tokens,
                                                int32_t step, int32_t beam_size,
                                                int32_t eot_token_id) {
    const int32_t request_batch = static_cast<int32_t>(beams.size());
    const auto decoder_lanes = static_cast<std::size_t>(request_batch * beam_size);
    CanaryBeamBatchStep next;
    next.beams.resize(beams.size());
    next.parent_lanes.assign(decoder_lanes, 0);
    next.tokens.assign(decoder_lanes, eot_token_id);
    for (int32_t batch = 0; batch < request_batch; ++batch) {
        next.any_active |= append_canary_next_beam_sample(
            beams[static_cast<std::size_t>(batch)], max_new_tokens[static_cast<std::size_t>(batch)],
            step, batch, beam_size, eot_token_id, next.parent_lanes, next.tokens,
            next.beams[static_cast<std::size_t>(batch)]);
    }
    return next;
}

std::vector<int32_t> make_canary_beam_lane_to_sample(int32_t decoder_lanes, int32_t beam_size) {
    std::vector<int32_t> lane_to_sample(static_cast<std::size_t>(decoder_lanes));
    for (int32_t lane = 0; lane < decoder_lanes; ++lane)
        lane_to_sample[static_cast<std::size_t>(lane)] = lane / beam_size;
    return lane_to_sample;
}

void update_canary_beam_batch_logits(CanaryBeamBatch& beams, const std::vector<float>& logits,
                                     int32_t beam_size, std::size_t vocab_size) {
    for (std::size_t batch = 0; batch < beams.size(); ++batch) {
        for (int32_t beam = 0; beam < beam_size; ++beam) {
            const auto lane = static_cast<std::ptrdiff_t>(
                batch * static_cast<std::size_t>(beam_size) + static_cast<std::size_t>(beam));
            auto& hypothesis = beams[batch][static_cast<std::size_t>(beam)];
            if (hypothesis.state_slot < 0)
                continue;
            const auto begin = logits.begin() + lane * static_cast<std::ptrdiff_t>(vocab_size);
            hypothesis.logits.assign(begin, begin + static_cast<std::ptrdiff_t>(vocab_size));
        }
    }
}

std::vector<std::vector<int32_t>> take_canary_beam_batch_output(CanaryBeamBatch& beams) {
    std::vector<std::vector<int32_t>> output(beams.size());
    for (std::size_t batch = 0; batch < beams.size(); ++batch) {
        if (!beams[batch].empty())
            output[batch] = std::move(beams[batch].front().output_ids);
    }
    return output;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// CanaryPipeline
// ═══════════════════════════════════════════════════════════════════════════

CanaryPipeline::CanaryPipeline(
    std::unique_ptr<TrtModule> encoder, std::unique_ptr<TrtModule> decoder,
    std::unique_ptr<CanaryInferenceState> state, CanaryConfig canary_config, int32_t hidden_size,
    int32_t num_decoder_layers, MelFilterbank mel_fb, int32_t mel_n_fft, int32_t mel_win_length,
    int32_t mel_hop_length, int32_t mel_chunk_length, int32_t mel_sampling_rate, float mel_preemph,
    bool mel_normalize_per_feature, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder)), state_(std::move(state)),
      canary_config_(std::move(canary_config)), hidden_size_(hidden_size),
      num_decoder_layers_(num_decoder_layers),
      mel_fb_(std::make_unique<MelFilterbank>(std::move(mel_fb))), mel_n_fft_(mel_n_fft),
      mel_win_length_(mel_win_length), mel_hop_length_(mel_hop_length),
      mel_chunk_length_(mel_chunk_length), mel_sampling_rate_(mel_sampling_rate),
      mel_preemph_(mel_preemph), mel_normalize_per_feature_(mel_normalize_per_feature),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {
    validate_canary_pipeline_components(encoder_.get(), decoder_.get(), state_.get());

    // Decoder shapes are stable within each dynamic cache-row bucket. Capture
    // the TensorRT enqueue on the first step and replay it until a shape change
    // invalidates the graph and triggers a recapture.
    if (!canary_config_.disable_cuda_graph)
        decoder_->enable_cuda_graph();

    encoder_batch_capacity_ = canary_module_batch_capacity(*encoder_, "mel_features");
    decoder_lane_capacity_ = canary_module_batch_capacity(*decoder_, "token_id");
    if (batch_cache().batch_capacity() < decoder_lane_capacity_) {
        throw std::runtime_error(
            "CanaryPipeline: inference-state batch capacity is smaller than decoder profile");
    }

    // All decoder layers consume the same raw encoder output and perform their
    // own cross-attention projections, so one stable external buffer can back
    // every cross_k/cross_v input.
    const DType encoder_output_dtype = encoder_->tensor_dtype("encoder_output");
    const DType cross_input_dtype = decoder_->tensor_dtype("cross_k_0");
    if (encoder_output_dtype != cross_input_dtype) {
        throw std::runtime_error(
            "CanaryPipeline: encoder output and decoder cross-attention dtypes differ");
    }
    cross_kv_sample_bytes_ = static_cast<std::size_t>(canary_config_.max_source_positions) *
                             static_cast<std::size_t>(hidden_size_) * dtype_size(cross_input_dtype);
    const std::size_t cross_bytes =
        static_cast<std::size_t>(decoder_lane_capacity_) * cross_kv_sample_bytes_;
    cross_kv_ptr_ = allocate_canary_cross_kv_buffer(num_decoder_layers_, cross_bytes);
}

CanaryPipeline::~CanaryPipeline() {
    if (cross_kv_ptr_)
        cudaFree(cross_kv_ptr_);
}

TextResult CanaryPipeline::transcribe(const float* audio_data, int32_t num_samples,
                                      int32_t max_new_tokens, int32_t input_sample_rate) {
    TranscriptionConfig cfg;
    cfg.max_output_tokens = max_new_tokens;
    cfg.input_sample_rate = input_sample_rate;
    return transcribe(audio_data, num_samples, cfg);
}

TextResult CanaryPipeline::transcribe(const float* audio_data, int32_t num_samples,
                                      const TranscriptionConfig& cfg) {
    validate_canary_audio_input(audio_data, num_samples);

    auto initial_tokens = make_canary_request_tokens(canary_config_, cfg, tokenizer_.get());
    validate_canary_output_budget(canary_config_, cfg, initial_tokens.size());

    const int32_t sample_rate =
        cfg.input_sample_rate > 0 ? cfg.input_sample_rate : mel_sampling_rate_;
    const double duration_seconds = validate_canary_input_duration(num_samples, sample_rate, cfg);
    const double model_segment_seconds = static_cast<double>(mel_chunk_length_);
    const double segment_seconds =
        resolve_canary_segment_duration(duration_seconds, model_segment_seconds, cfg);
    const int32_t segment_samples = canary_segment_sample_count(segment_seconds, sample_rate);

    TextResult combined;
    for (int64_t offset = 0; offset < num_samples; offset += segment_samples) {
        const int32_t count = static_cast<int32_t>(
            std::min<int64_t>(segment_samples, static_cast<int64_t>(num_samples) - offset));
        auto segment = transcribe_segment(audio_data + offset, count, sample_rate, initial_tokens,
                                          cfg.max_output_tokens, cfg.beam_size);
        append_canary_transcription_segment(combined, std::move(segment), offset, count,
                                            sample_rate, cfg);
    }
    return combined;
}

std::vector<TextResult>
CanaryPipeline::transcribe_batch(const std::vector<TranscriptionRequest>& requests) {
    if (requests.empty())
        return {};

    auto work = build_canary_batch_work(requests, canary_config_, tokenizer_.get(),
                                        mel_sampling_rate_, mel_chunk_length_);

    std::vector<TextResult> segment_results(work.size());
    const auto groups = group_canary_batch_work(work);

    for (const int32_t beam_size : groups.beam_order) {
        const auto& indices = groups.by_beam_size.at(beam_size);
        if (beam_size > decoder_lane_capacity_) {
            for (const std::size_t index : indices) {
                const auto& item = work[index];
                const auto& audio = requests[item.request_index].audio_samples;
                segment_results[index] =
                    transcribe_segment(audio.data() + item.offset, item.count, item.sample_rate,
                                       item.initial_tokens, item.max_output_tokens, item.beam_size);
            }
            continue;
        }

        const int32_t decoder_request_capacity =
            std::max(decoder_lane_capacity_ / std::max(beam_size, 1), 1);
        const std::size_t chunk_capacity =
            static_cast<std::size_t>(std::min(encoder_batch_capacity_, decoder_request_capacity));
        for (std::size_t chunk_start = 0; chunk_start < indices.size();
             chunk_start += chunk_capacity) {
            const std::size_t chunk_end = std::min(chunk_start + chunk_capacity, indices.size());
            const auto chunk_begin_time = CanaryClock::now();

            auto futures = launch_canary_batch_preparation(
                indices, chunk_start, chunk_end, work, requests, mel_fb_.get(), mel_n_fft_,
                mel_win_length_, mel_hop_length_, mel_chunk_length_, mel_sampling_rate_,
                mel_preemph_, mel_normalize_per_feature_, canary_config_);
            auto chunk = collect_canary_prepared_batch_chunk(futures, indices, chunk_start, work,
                                                             segment_results);
            if (chunk.valid_indices.empty())
                continue;

            run_encoder_batch(chunk.mel_batch, chunk.mel_bins, chunk.mel_frames,
                              chunk.valid_frames);
            auto output_ids = run_decoder_batch(chunk.prompts, chunk.output_limits, beam_size,
                                                chunk.actual_encoder_lengths);
            store_canary_batch_decodes(output_ids, chunk.valid_indices, tokenizer_.get(),
                                       canary_config_.eot_token_id, segment_results);
            report_canary_batch_timing(chunk_begin_time, chunk.valid_indices.size(), beam_size);
        }
    }
    return merge_canary_batch_segments(requests, work, segment_results);
}

TextResult CanaryPipeline::transcribe_segment(const float* audio_data, int32_t num_samples,
                                              int32_t input_sample_rate,
                                              const std::vector<int32_t>& initial_tokens,
                                              int32_t max_output_tokens, int32_t beam_size) {
    const bool report_stage_timing = canary_stage_timing_enabled();
    const auto transcribe_start = CanaryClock::now();

    // Step 0: Resample if needed
    const float* samples_ptr = audio_data;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;

    if (input_sample_rate > 0 && input_sample_rate != mel_sampling_rate_) {
        std::cerr << "[canary] Resampling audio from " << input_sample_rate << " Hz to "
                  << mel_sampling_rate_ << " Hz" << std::endl;
        resampled_buf =
            resample_linear(audio_data, num_samples, input_sample_rate, mel_sampling_rate_);
        // The NeMo file-based reference serializes its resampled signal as
        // PCM16 before feature extraction. Mirror that model-local boundary so
        // low-amplitude inputs reach the frontend with identical quantization.
        quantize_canary_pcm16_inplace(resampled_buf);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
    }
    const auto resample_end = CanaryClock::now();

    // Step 1: Extract mel spectrogram
    canary::MelResult mel;
    if (mel_fb_ && !mel_fb_->data.empty()) {
        mel = canary::extract_mel_spectrogram(
            samples_ptr, samples_count, mel_fb_->data.data(), mel_fb_->n_freq_bins,
            mel_fb_->n_mel_bins, mel_n_fft_, mel_win_length_, mel_hop_length_, mel_chunk_length_,
            mel_sampling_rate_, mel_preemph_, mel_normalize_per_feature_);
    }
    const auto mel_end = CanaryClock::now();

    if (mel.data.empty()) {
        return TextResult{"[mel extraction failed]", {}};
    }

    // Step 2: Run encoder. The mel is chunk-padded, so mel.n_frames is the full
    // (padded) length; mel.valid_frames is the real audio length, used to mask
    // the padded tail in self-attention and to zero-pad the cross-attention K/V.
    const int32_t valid_mel_frames = mel.valid_frames > 0 ? mel.valid_frames : mel.n_frames;
    std::cerr << "[canary] Running encoder ..." << std::endl;
    run_encoder(mel.data.data(), mel.n_mels, mel.n_frames, valid_mel_frames);
    const auto encoder_end = CanaryClock::now();

    // Compute actual encoder sequence length for masking
    const int32_t mel_full = resolve_canary_expected_mel_length(canary_config_);
    int32_t actual_enc_seq_len = compute_canary_actual_encoder_length(
        valid_mel_frames, mel_full, canary_config_.max_source_positions);
    if (actual_enc_seq_len > 0) {
        std::cerr << "[canary] Actual encoder seq len: " << actual_enc_seq_len << " / "
                  << canary_config_.max_source_positions << std::endl;
    }

    // Step 3: Set up cross-attention K/V
    std::cerr << "[canary] Computing cross-attention K/V ..." << std::endl;
    setup_cross_attention(actual_enc_seq_len);
    const auto cross_kv_end = CanaryClock::now();

    // Step 4: Run decoder
    std::cerr << "[canary] Running decoder ..." << std::endl;
    auto output_ids = run_decoder(initial_tokens, max_output_tokens, beam_size);
    const auto decoder_end = CanaryClock::now();

    // Step 5: Decode token IDs
    TextResult out;
    out.token_ids = std::move(output_ids);
    if (tokenizer_ && !out.token_ids.empty()) {
        out.text = decode_canary_tokens(*tokenizer_, out.token_ids, canary_config_.eot_token_id);
    }
    const auto tokenize_end = CanaryClock::now();

    report_canary_stage_timing(report_stage_timing,
                               {transcribe_start, resample_end, mel_end, encoder_end, cross_kv_end,
                                decoder_end, tokenize_end});
    return out;
}

void CanaryPipeline::run_encoder(const float* mel_data, int32_t mel_bins, int32_t mel_length,
                                 int32_t valid_mel_frames) {
    if (encoder_->input_rank("mel_features") == 3) {
        const std::size_t mel_size =
            static_cast<std::size_t>(mel_bins) * static_cast<std::size_t>(mel_length);
        run_encoder_batch({std::vector<float>(mel_data, mel_data + mel_size)}, mel_bins, mel_length,
                          {valid_mel_frames});
        return;
    }

    const int32_t expected_length = resolve_canary_expected_mel_length(canary_config_);
    const std::size_t mel_size =
        static_cast<std::size_t>(mel_bins) * static_cast<std::size_t>(expected_length);

    // Prepare mel input (pad if needed)
    std::vector<float> mel_host;
    if (mel_length == expected_length) {
        mel_host.assign(mel_data, mel_data + mel_size);
    } else {
        mel_host = build_canary_padded_mel_input(mel_data, mel_bins, mel_length, expected_length);
    }

    // Build input TensorMap
    TensorMap inputs;
    Tensor mel_tensor;
    mel_tensor.data = mel_host.data();
    mel_tensor.shape = {mel_bins, expected_length};
    mel_tensor.dtype = DType::kFloat32;
    inputs["mel_features"] = mel_tensor;

    // Optional encoder_mask input
    const int32_t enc_seq = canary_config_.max_source_positions;
    std::vector<float> enc_mask;
    if (encoder_->has_input("encoder_mask")) {
        int32_t actual_enc =
            compute_canary_actual_encoder_length(valid_mel_frames, expected_length, enc_seq);
        if (actual_enc <= 0)
            actual_enc = enc_seq;
        enc_mask = build_canary_encoder_mask_values(enc_seq, actual_enc);

        Tensor mask_tensor;
        mask_tensor.data = enc_mask.data();
        mask_tensor.shape = {static_cast<int64_t>(enc_mask.size())};
        mask_tensor.dtype = DType::kFloat32;
        inputs["encoder_mask"] = mask_tensor;
    }

    // Run encoder (we need the output to stay on device, so use forward_async + sync)
    encoder_->forward_async(inputs);
    encoder_->sync();
}

void CanaryPipeline::run_encoder_batch(const std::vector<std::vector<float>>& mel_data,
                                       int32_t mel_bins, int32_t mel_length,
                                       const std::vector<int32_t>& valid_mel_frames) {
    if (mel_data.empty() || mel_data.size() != valid_mel_frames.size())
        throw std::invalid_argument("Canary encoder batch has inconsistent inputs");
    if (encoder_->input_rank("mel_features") != 3) {
        if (mel_data.size() != 1)
            throw std::invalid_argument("Legacy Canary encoder only supports batch size 1");
        run_encoder(mel_data.front().data(), mel_bins, mel_length, valid_mel_frames.front());
        return;
    }
    if (mel_data.size() > static_cast<std::size_t>(encoder_batch_capacity_))
        throw std::invalid_argument("Canary encoder batch exceeds engine capacity");

    const int32_t expected_length = resolve_canary_expected_mel_length(canary_config_);
    if (mel_length != expected_length)
        throw std::invalid_argument("Canary encoder batch requires padded mel frames");
    const std::size_t sample_values =
        static_cast<std::size_t>(mel_bins) * static_cast<std::size_t>(expected_length);
    std::vector<float> packed_mel = pack_canary_mel_batch(mel_data, sample_values);

    TensorMap inputs;
    Tensor mel_tensor;
    mel_tensor.data = packed_mel.data();
    mel_tensor.shape = {static_cast<int64_t>(mel_data.size()), mel_bins, expected_length};
    mel_tensor.dtype = DType::kFloat32;
    inputs["mel_features"] = mel_tensor;

    CanaryEncoderBatchMask packed_mask;
    if (encoder_->has_input("encoder_mask")) {
        const int32_t enc_seq = canary_config_.max_source_positions;
        packed_mask = make_canary_encoder_batch_mask(valid_mel_frames, expected_length, enc_seq,
                                                     encoder_->tensor_shape("encoder_mask"));

        Tensor mask_tensor;
        mask_tensor.data = packed_mask.values.data();
        mask_tensor.shape = packed_mask.shape;
        mask_tensor.dtype = DType::kFloat32;
        inputs["encoder_mask"] = mask_tensor;
    }

    encoder_->forward_async(inputs);
    encoder_->sync();
}

void CanaryPipeline::setup_cross_attention(int32_t actual_enc_seq_len) {
    setup_cross_attention({actual_enc_seq_len}, {0});
}

void CanaryPipeline::setup_cross_attention(const std::vector<int32_t>& actual_enc_seq_lens,
                                           const std::vector<int32_t>& lane_to_sample) {
    if (num_decoder_layers_ <= 0)
        return;
    if (cross_kv_ptr_ == nullptr || lane_to_sample.empty() ||
        lane_to_sample.size() > static_cast<std::size_t>(decoder_lane_capacity_)) {
        throw std::invalid_argument("Canary cross-attention batch is invalid");
    }

    auto* encoder_output = static_cast<uint8_t*>(encoder_->device_ptr("encoder_output"));
    zero_pad_canary_encoder_batch(encoder_output, actual_enc_seq_lens,
                                  canary_config_.max_source_positions, hidden_size_,
                                  cross_kv_sample_bytes_, stream_);
    copy_canary_cross_attention_lanes(static_cast<uint8_t*>(cross_kv_ptr_), encoder_output,
                                      lane_to_sample, actual_enc_seq_lens.size(),
                                      cross_kv_sample_bytes_, stream_);

    const std::vector<int64_t> cross_shape{static_cast<int64_t>(lane_to_sample.size()),
                                           canary_config_.max_source_positions, hidden_size_};
    bind_canary_cross_attention_layers(*decoder_, num_decoder_layers_, cross_kv_ptr_, cross_shape);
    decoder_->sync();

    // NeMo masks encoder padding in decoder cross-attention. Zeroing padded
    // encoder rows is not sufficient because the decoder K/V projections have
    // biases and would turn those rows back into non-zero attention keys.
    const int32_t max_source_positions = canary_config_.max_source_positions;
    cross_attention_mask_.clear();
    cross_attention_mask_.reserve(lane_to_sample.size() *
                                  static_cast<std::size_t>(max_source_positions));
    for (const int32_t sample : lane_to_sample) {
        if (sample < 0 || static_cast<std::size_t>(sample) >= actual_enc_seq_lens.size())
            throw std::invalid_argument("Canary cross-attention lane maps to an invalid sample");
        const int32_t actual_enc_seq_len = actual_enc_seq_lens[static_cast<std::size_t>(sample)];
        const int32_t valid_enc_seq_len =
            actual_enc_seq_len > 0 ? actual_enc_seq_len : max_source_positions;
        auto mask = build_canary_encoder_mask_values(max_source_positions, valid_enc_seq_len);
        cross_attention_mask_.insert(cross_attention_mask_.end(), mask.begin(), mask.end());
    }
}

std::vector<int32_t> CanaryPipeline::run_decoder(const std::vector<int32_t>& initial_tokens,
                                                 int32_t max_new_tokens, int32_t beam_size) {
    batch_cache().set_batch_size(1);
    if (beam_size > 1)
        return run_beam_decoder(initial_tokens, max_new_tokens, beam_size);

    state_->reset();
    state_->bind_to(*decoder_);

    const int32_t eot_id = canary_config_.eot_token_id;

    auto result = run_canary_decode_loop(
        initial_tokens, max_new_tokens, eot_id,
        [this](int32_t token, std::vector<float>& logits, std::string&) {
            run_decoder_step(token, logits);
            return true;
        },
        [](const std::vector<float>& logits) { return canary_select_argmax_token(logits); });

    if (result.prefill_failed) {
        std::cerr << "[canary] Prefill step failed: " << result.error << std::endl;
    } else if (result.decode_failed) {
        std::cerr << "[canary] Decode step failed: " << result.error << std::endl;
    }

    return result.output_ids;
}

std::vector<std::vector<int32_t>>
CanaryPipeline::run_decoder_batch(const std::vector<std::vector<int32_t>>& initial_tokens,
                                  const std::vector<int32_t>& max_new_tokens, int32_t beam_size,
                                  const std::vector<int32_t>& actual_enc_seq_lens) {
    if (initial_tokens.empty() || initial_tokens.size() != max_new_tokens.size() ||
        initial_tokens.size() != actual_enc_seq_lens.size()) {
        throw std::invalid_argument("Canary decoder batch has inconsistent inputs");
    }
    std::vector<int32_t> identity(initial_tokens.size());
    std::iota(identity.begin(), identity.end(), 0);
    setup_cross_attention(actual_enc_seq_lens, identity);
    if (beam_size <= 1)
        return run_greedy_decoder_batch(initial_tokens, max_new_tokens);
    return run_beam_decoder_batch(initial_tokens, max_new_tokens, beam_size, actual_enc_seq_lens);
}

std::vector<std::vector<int32_t>>
CanaryPipeline::run_greedy_decoder_batch(const std::vector<std::vector<int32_t>>& initial_tokens,
                                         const std::vector<int32_t>& max_new_tokens) {
    const std::size_t batch_size = initial_tokens.size();
    const std::size_t prompt_length = validate_canary_batch_prompts(initial_tokens);
    if (prompt_length == 0)
        return std::vector<std::vector<int32_t>>(batch_size);

    auto& cache = batch_cache();
    cache.set_batch_size(static_cast<int32_t>(batch_size));
    state_->reset();
    state_->bind_to(*decoder_);

    std::vector<float> logits;
    std::vector<int32_t> tokens(batch_size);
    for (std::size_t position = 0; position < prompt_length; ++position) {
        for (std::size_t batch = 0; batch < batch_size; ++batch)
            tokens[batch] = initial_tokens[batch][position];
        run_decoder_step_batch(tokens, logits);
    }

    const std::size_t vocab_size = logits.size() / batch_size;
    if (vocab_size == 0 || vocab_size * batch_size != logits.size())
        throw std::runtime_error("Canary batched decoder returned invalid logits");

    std::vector<std::vector<int32_t>> output(batch_size);
    std::vector<bool> finished(batch_size, false);
    const int32_t max_steps = *std::max_element(max_new_tokens.begin(), max_new_tokens.end());
    for (int32_t step = 0; step < max_steps; ++step) {
        if (update_canary_greedy_batch_step(logits, vocab_size, step, max_new_tokens,
                                            canary_config_.eot_token_id, tokens, output,
                                            finished)) {
            break;
        }
        run_decoder_step_batch(tokens, logits);
    }
    return output;
}

std::vector<std::vector<int32_t>>
CanaryPipeline::run_beam_decoder_batch(const std::vector<std::vector<int32_t>>& initial_tokens,
                                       const std::vector<int32_t>& max_new_tokens,
                                       int32_t beam_size,
                                       const std::vector<int32_t>& actual_enc_seq_lens) {
    const int32_t request_batch = static_cast<int32_t>(initial_tokens.size());
    const int32_t decoder_lanes = request_batch * beam_size;
    if (decoder_lanes > decoder_lane_capacity_)
        throw std::invalid_argument("Canary batched beam search exceeds decoder lane capacity");

    const std::size_t prompt_length = validate_canary_batch_prompts(initial_tokens);

    auto& scratch = batch_cache();
    scratch.set_batch_size(request_batch);
    state_->reset();
    state_->bind_to(*decoder_);
    std::vector<int32_t> tokens(static_cast<std::size_t>(request_batch));
    std::vector<float> logits;
    for (std::size_t position = 0; position < prompt_length; ++position) {
        for (int32_t batch = 0; batch < request_batch; ++batch)
            tokens[static_cast<std::size_t>(batch)] =
                initial_tokens[static_cast<std::size_t>(batch)][position];
        run_decoder_step_batch(tokens, logits);
    }

    const std::size_t vocab_size = logits.size() / static_cast<std::size_t>(request_batch);
    if (vocab_size == 0 || vocab_size * static_cast<std::size_t>(request_batch) != logits.size())
        throw std::runtime_error("Canary batched beam decoder returned invalid logits");

    ensure_batch_beam_state();
    auto* persistent = dynamic_cast<CanaryKvCache*>(batch_beam_state_.get());
    if (persistent == nullptr)
        throw std::runtime_error("Canary batched beam state has the wrong type");
    persistent->set_batch_size(request_batch);
    persistent->copy_from(*state_);

    auto beams = initialize_canary_beam_batch(logits, request_batch, vocab_size);

    const auto beam_lane_to_sample = make_canary_beam_lane_to_sample(decoder_lanes, beam_size);
    setup_cross_attention(actual_enc_seq_lens, beam_lane_to_sample);

    const int32_t max_steps = *std::max_element(max_new_tokens.begin(), max_new_tokens.end());
    for (int32_t step = 0; step < max_steps; ++step) {
        auto next = make_canary_beam_batch_step(beams, max_new_tokens, step, beam_size,
                                                canary_config_.eot_token_id);
        beams = std::move(next.beams);
        if (!next.any_active)
            break;

        scratch.copy_lanes_from(*persistent, next.parent_lanes);
        state_->bind_to(*decoder_);
        run_decoder_step_batch(next.tokens, logits);
        persistent->copy_from(*state_);
        update_canary_beam_batch_logits(beams, logits, beam_size, vocab_size);
    }
    return take_canary_beam_batch_output(beams);
}

std::vector<int32_t> CanaryPipeline::run_beam_decoder(const std::vector<int32_t>& initial_tokens,
                                                      int32_t max_new_tokens, int32_t beam_size) {
    ensure_beam_state_capacity(beam_size);
    auto result = run_canary_beam_search(
        initial_tokens, max_new_tokens, canary_config_.eot_token_id, beam_size,
        CanaryDefaultBeamLengthPenalty,
        [this](const std::vector<int32_t>& prefix, std::vector<float>& logits, std::string& error) {
            try {
                state_->reset();
                state_->bind_to(*decoder_);
                for (const int32_t token : prefix)
                    run_decoder_step(token, logits);
                beam_states_a_.front()->copy_from(*state_);
                return true;
            } catch (const std::exception& e) {
                error = e.what();
                return false;
            }
        },
        [this](int32_t generation, int32_t parent_slot, int32_t child_slot, int32_t token,
               std::vector<float>& logits, std::string& error) {
            try {
                auto& parents = generation % 2 == 0 ? beam_states_a_ : beam_states_b_;
                auto& children = generation % 2 == 0 ? beam_states_b_ : beam_states_a_;
                state_->copy_from(*parents.at(static_cast<std::size_t>(parent_slot)));
                run_decoder_step(token, logits);
                children.at(static_cast<std::size_t>(child_slot))->copy_from(*state_);
                return true;
            } catch (const std::exception& e) {
                error = e.what();
                return false;
            }
        });
    if (result.prefill_failed || result.decode_failed)
        throw std::runtime_error("Canary beam search failed: " + result.error);
    return result.output_ids;
}

void CanaryPipeline::ensure_beam_state_capacity(int32_t beam_size) {
    const auto target = static_cast<std::size_t>(beam_size);
    while (beam_states_a_.size() < target) {
        auto state_a = state_->create_empty();
        auto state_b = state_->create_empty();
        if (!state_a || !state_b || !state_a->ok() || !state_b->ok())
            throw std::runtime_error("CanaryPipeline: failed to allocate beam inference state");
        beam_states_a_.push_back(std::move(state_a));
        beam_states_b_.push_back(std::move(state_b));
    }
}

CanaryKvCache& CanaryPipeline::batch_cache() {
    auto* cache = dynamic_cast<CanaryKvCache*>(state_.get());
    if (cache == nullptr)
        throw std::runtime_error("CanaryPipeline requires CanaryKvCache for request batching");
    return *cache;
}

const CanaryKvCache& CanaryPipeline::batch_cache() const {
    const auto* cache = dynamic_cast<const CanaryKvCache*>(state_.get());
    if (cache == nullptr)
        throw std::runtime_error("CanaryPipeline requires CanaryKvCache for request batching");
    return *cache;
}

void CanaryPipeline::ensure_batch_beam_state() {
    if (!batch_beam_state_) {
        batch_beam_state_ = state_->create_empty();
        if (!batch_beam_state_ || !batch_beam_state_->ok())
            throw std::runtime_error("CanaryPipeline: failed to allocate batched beam state");
    }
}

void CanaryPipeline::run_decoder_step(int32_t token_id, std::vector<float>& logits) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (decoder_->has_input("cross_attention_mask")) {
        const auto source_positions = static_cast<std::size_t>(canary_config_.max_source_positions);
        if (cross_attention_mask_.size() != source_positions)
            throw std::runtime_error(
                "Canary cross-attention mask has an invalid single-lane shape");
        Tensor cross_mask_tensor;
        cross_mask_tensor.data = cross_attention_mask_.data();
        cross_mask_tensor.shape = {1, 1, canary_config_.max_source_positions};
        cross_mask_tensor.dtype = DType::kFloat32;
        inputs["cross_attention_mask"] = cross_mask_tensor;
    }
    state_->prepare_step(inputs);

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end()) {
        throw std::runtime_error("CanaryPipeline: no 'logits' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
}

void CanaryPipeline::run_decoder_step_batch(const std::vector<int32_t>& token_ids,
                                            std::vector<float>& logits) {
    if (token_ids.empty())
        throw std::invalid_argument("Canary decoder step requires at least one lane");
    batch_cache().set_batch_size(static_cast<int32_t>(token_ids.size()));

    Tensor token_tensor;
    token_tensor.data = const_cast<int32_t*>(token_ids.data());
    token_tensor.shape = {static_cast<int64_t>(token_ids.size())};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    if (decoder_->has_input("cross_attention_mask")) {
        const auto source_positions = static_cast<std::size_t>(canary_config_.max_source_positions);
        if (cross_attention_mask_.size() != token_ids.size() * source_positions)
            throw std::runtime_error("Canary cross-attention mask has an invalid batched shape");
        Tensor cross_mask_tensor;
        cross_mask_tensor.data = cross_attention_mask_.data();
        cross_mask_tensor.shape = {static_cast<int64_t>(token_ids.size()), 1,
                                   canary_config_.max_source_positions};
        cross_mask_tensor.dtype = DType::kFloat32;
        inputs["cross_attention_mask"] = cross_mask_tensor;
    }
    state_->prepare_step(inputs);
    TensorMap outputs = decoder_->forward(inputs);
    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("CanaryPipeline: no 'logits' output");

    const auto& logits_tensor = it->second;
    const auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));
    state_->advance();
}

} // namespace trtmc
