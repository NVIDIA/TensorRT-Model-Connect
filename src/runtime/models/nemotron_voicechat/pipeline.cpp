/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/pipeline.h"

#include "runtime/models/nemotron_voicechat/audio_helpers.h"
#include "runtime/models/nemotron_voicechat/codec_reconstruction.h"
#include "runtime/models/nemotron_voicechat/function_channel.h"
#include "runtime/models/nemotron_voicechat/session_state.h"
#include "runtime/models/nemotron_voicechat/thinker_hybrid_state.h"
#include "runtime/models/nemotron_voicechat/thinker_kv_cache.h"
#include "runtime/models/nemotron_voicechat/thinker_mamba_state.h"
#include "trtmc/runtime/device_tensor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace trtmc {

namespace voicechat = nemotron_voicechat;

voicechat::StreamingMelStep voicechat::make_streaming_mel_step(bool first_step,
                                                               int32_t next_mel_frame,
                                                               int32_t available_mel_frames,
                                                               bool final) {
    if (next_mel_frame < 0 || available_mel_frames < 0)
        throw std::invalid_argument("VoiceChat mel frame positions must be non-negative");
    StreamingMelStep step;
    step.history_frames = first_step ? 0 : 9;
    step.requested_new_frames = first_step ? 1 : 8;
    step.engine_frames = first_step ? 1 : 17;
    const int32_t available_new = std::max(0, available_mel_frames - next_mel_frame);
    step.valid_new_frames = std::min(step.requested_new_frames, available_new);
    if (!final && step.valid_new_frames != step.requested_new_frames)
        throw std::runtime_error("VoiceChat streaming mel step is not ready");
    return step;
}

int32_t voicechat::streaming_frontend_capacity_seconds(const Config& config) {
    if (config.tts_max_cache_length <= 0 || config.input_samples_per_frame <= 0 ||
        config.input_sample_rate <= 0)
        throw std::invalid_argument("VoiceChat frontend capacity requires positive dimensions");
    const int64_t samples =
        static_cast<int64_t>(config.tts_max_cache_length) * config.input_samples_per_frame;
    return static_cast<int32_t>((samples + config.input_sample_rate - 1) /
                                config.input_sample_rate) +
           1;
}

namespace {

Tensor tensor(void* data, std::vector<int64_t> shape, DType dtype) {
    return Tensor{data, std::move(shape), dtype};
}

void require_module(const std::unique_ptr<TrtModule>& module, const char* label) {
    if (!module || !module->ok())
        throw std::runtime_error(std::string("NemotronVoiceChat: invalid ") + label + " module");
}

int32_t argmax(const Tensor& values) {
    if (values.dtype != DType::kFloat32 || values.data == nullptr || values.numel() == 0)
        throw std::runtime_error("NemotronVoiceChat: expected non-empty FP32 logits");
    const auto* first = static_cast<const float*>(values.data);
    return static_cast<int32_t>(
        std::distance(first, std::max_element(first, first + values.numel())));
}

std::string normalize_rnnt_text(const std::vector<int32_t>& ids,
                                const std::vector<std::string>& vocabulary) {
    std::string text;
    for (const int32_t id : ids) {
        if (id >= 0 && static_cast<std::size_t>(id) < vocabulary.size())
            text += vocabulary[static_cast<std::size_t>(id)];
    }
    constexpr std::string_view marker = "\xE2\x96\x81"; // U+2581 SentencePiece boundary.
    std::size_t at = 0;
    while ((at = text.find(marker, at)) != std::string::npos) {
        text.replace(at, marker.size(), " ");
        ++at;
    }
    std::string compact;
    compact.reserve(text.size());
    bool previous_space = true;
    for (const char ch : text) {
        if (ch == ' ') {
            if (!previous_space)
                compact.push_back(ch);
            previous_space = true;
        } else {
            compact.push_back(ch);
            previous_space = false;
        }
    }
    if (!compact.empty() && compact.back() == ' ')
        compact.pop_back();
    return compact;
}

std::vector<float> resample_frame(const std::vector<float>& input, int32_t source_rate,
                                  int32_t target_rate) {
    if (source_rate == target_rate || input.empty())
        return input;
    const auto output_size = static_cast<std::size_t>(
        std::llround(static_cast<double>(input.size()) * static_cast<double>(target_rate) /
                     static_cast<double>(source_rate)));
    std::vector<float> output(output_size, 0.0F);
    if (input.size() == 1) {
        std::fill(output.begin(), output.end(), input.front());
        return output;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const double source = static_cast<double>(index) * source_rate / target_rate;
        const auto left = std::min(static_cast<std::size_t>(source), input.size() - 1);
        const auto right = std::min(left + 1, input.size() - 1);
        const float fraction = static_cast<float>(source - static_cast<double>(left));
        output[index] = input[left] + fraction * (input[right] - input[left]);
    }
    return output;
}

class StreamingLinearResampler {
  public:
    StreamingLinearResampler(int32_t source_rate, int32_t target_rate)
        : source_rate_(source_rate), target_rate_(target_rate) {
        if (source_rate_ <= 0 || target_rate_ <= 0)
            throw std::invalid_argument("VoiceChat resampler rates must be positive");
    }

    void append(const float* samples, int32_t count) {
        if (count < 0 || (count > 0 && samples == nullptr))
            throw std::invalid_argument("VoiceChat resampler received invalid samples");
        if (count > 0)
            source_.insert(source_.end(), samples, samples + count);
    }

    std::vector<float> drain(bool final) {
        if (source_rate_ == target_rate_) {
            std::vector<float> result(source_.begin() + static_cast<std::ptrdiff_t>(produced_),
                                      source_.end());
            produced_ = source_.size();
            return result;
        }

        const std::size_t available =
            final ? static_cast<std::size_t>(std::llround(static_cast<double>(source_.size()) *
                                                          target_rate_ / source_rate_))
                  : stable_output_count();
        std::vector<float> result;
        if (available <= produced_)
            return result;
        result.reserve(available - produced_);
        for (std::size_t output_index = produced_; output_index < available; ++output_index) {
            const double source_position =
                static_cast<double>(output_index) * source_rate_ / target_rate_;
            const auto left = std::min(static_cast<std::size_t>(source_position),
                                       source_.empty() ? 0U : source_.size() - 1U);
            const auto right = std::min(left + 1U, source_.empty() ? 0U : source_.size() - 1U);
            const float fraction = static_cast<float>(source_position - static_cast<double>(left));
            const float left_value = source_.empty() ? 0.0F : source_[left];
            const float right_value = source_.empty() ? left_value : source_[right];
            result.push_back(left_value + fraction * (right_value - left_value));
        }
        produced_ = available;
        return result;
    }

    void reset() {
        source_.clear();
        produced_ = 0;
    }

  private:
    std::size_t stable_output_count() const {
        if (source_.size() < 2)
            return 0;
        // j * source_rate / target_rate must have both floor and ceil samples.
        const double exclusive = static_cast<double>(source_.size() - 1) * target_rate_ /
                                 static_cast<double>(source_rate_);
        return static_cast<std::size_t>(std::ceil(exclusive));
    }

    int32_t source_rate_{0};
    int32_t target_rate_{0};
    std::vector<float> source_;
    std::size_t produced_{0};
};

class TtsCacheState {
  public:
    TtsCacheState(TrtModule& module, const voicechat::Config& config,
                  const VoiceChatTtsPrompt& prompt, int32_t seed)
        : module_(module), config_(config), prompt_(prompt), stream_(module.stream()),
          seed_(static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed))), rng_(seed_),
          uniform_(std::nextafter(0.0F, 1.0F), std::nextafter(1.0F, 0.0F)), normal_(0.0F, 1.0F) {
        if (config_.tts_num_layers <= 0 || config_.tts_max_cache_length <= 0 ||
            config_.tts_kv_width <= 0)
            throw std::invalid_argument("VoiceChat TTS cache dimensions must be positive");
        const DType dtype = module_.tensor_dtype("cache_k_0");
        cache_dtype_ = dtype;
        cache_k_.reserve(static_cast<std::size_t>(config_.tts_num_layers));
        cache_v_.reserve(static_cast<std::size_t>(config_.tts_num_layers));
        present_k_.reserve(static_cast<std::size_t>(config_.tts_num_layers));
        present_v_.reserve(static_cast<std::size_t>(config_.tts_num_layers));
        for (int32_t layer = 0; layer < config_.tts_num_layers; ++layer) {
            cache_k_.emplace_back(
                std::vector<int64_t>{2, config_.tts_max_cache_length, config_.tts_kv_width}, dtype,
                stream_);
            cache_v_.emplace_back(
                std::vector<int64_t>{2, config_.tts_max_cache_length, config_.tts_kv_width}, dtype,
                stream_);
            present_k_.emplace_back(std::vector<int64_t>{2, 1, config_.tts_kv_width}, dtype,
                                    stream_);
            present_v_.emplace_back(std::vector<int64_t>{2, 1, config_.tts_kv_width}, dtype,
                                    stream_);
            if (!cache_k_.back().ok() || !cache_v_.back().ok() || !present_k_.back().ok() ||
                !present_v_.back().ok())
                throw std::runtime_error("VoiceChat failed to allocate EAR-TTS cache");
        }
        attention_mask_.resize(static_cast<std::size_t>(config_.tts_max_cache_length) + 1U,
                               -10000.0F);
        mixture_uniform_.resize(static_cast<std::size_t>(config_.tts_num_refinement_steps) *
                                config_.tts_mog_num_predictions);
        mog_noise_.resize(static_cast<std::size_t>(config_.tts_num_refinement_steps) * 512U);
        audio_prompt_latent_.resize(static_cast<std::size_t>(config_.tts_hidden_size), 0.0F);
        previous_codes_ = prompt_.first_codes;
    }

    std::vector<int32_t> step(int32_t subword_id, bool agent_idle) {
        std::vector<int32_t> input_codes = previous_codes_;
        if (subword_id == config_.eos_token_id)
            input_codes = prompt_.silence_codes;
        auto generated = enqueue(subword_id, 1.0F, input_codes, nullptr, 0.0F, 0.0F, position_);
        previous_codes_ = generated;
        if (subword_id == config_.eos_token_id ||
            (subword_id == config_.pad_token_id && agent_idle))
            previous_codes_ = prompt_.silence_codes;
        return generated;
    }

    void reset_and_warmup() {
        position_ = 0;
        rng_.seed(seed_);
        if (prompt_.first_codes.size() != static_cast<std::size_t>(config_.tts_num_quantizers) ||
            prompt_.silence_codes.size() != static_cast<std::size_t>(config_.tts_num_quantizers))
            throw std::runtime_error("VoiceChat bundle has invalid TTS code assets");
        previous_codes_ = prompt_.first_codes;
        for (int32_t step_index = 0; step_index < prompt_.warmup_steps; ++step_index) {
            const auto offset = static_cast<std::size_t>(step_index) * config_.tts_hidden_size;
            if (offset + static_cast<std::size_t>(config_.tts_hidden_size) >
                prompt_.aria_embeddings.size())
                throw std::runtime_error("VoiceChat bundle has truncated Aria warmup embeddings");
            const int32_t subword = prompt_.subword_ids.at(static_cast<std::size_t>(step_index));
            const float mask = prompt_.subword_mask.at(static_cast<std::size_t>(step_index));
            const float prompt_mode =
                prompt_.audio_prompt_mode.at(static_cast<std::size_t>(step_index));
            const float bos = prompt_.bos_flags.at(static_cast<std::size_t>(step_index));
            const int32_t position = prompt_.position_ids.at(static_cast<std::size_t>(step_index));
            const auto& warmup_codes =
                prompt_mode == 0.0F ? prompt_.silence_codes : previous_codes_;
            (void)enqueue(subword, mask, warmup_codes, prompt_.aria_embeddings.data() + offset,
                          prompt_mode, bos, position);
        }
        if (position_ != prompt_.first_generation_position)
            throw std::runtime_error("VoiceChat TTS warmup position does not match its recipe");
        // NeMo ignores every warmup prediction and feeds the checkpoint's PAD
        // frame into the first real generation step.
        previous_codes_ = prompt_.first_codes;
        // Reference warmup builds KV only; its RVQ head does not consume the
        // generation RNG. Re-seed after the ignored warmup outputs so live
        // position 37 starts from the model-card seed.
        rng_.seed(seed_);
    }

  private:
    void validate_enqueue_inputs(const std::vector<int32_t>& previous_codes,
                                 int32_t position_id) const {
        if (position_id != position_)
            throw std::runtime_error("VoiceChat EAR-TTS received a non-contiguous position");
        if (previous_codes.size() != static_cast<std::size_t>(config_.tts_num_quantizers))
            throw std::runtime_error("VoiceChat EAR-TTS previous-code width mismatch");
        if (position_ >= config_.tts_max_cache_length)
            throw std::runtime_error("VoiceChat EAR-TTS cache exhausted");
    }

    void prepare_attention_mask() {
        std::fill(attention_mask_.begin(), attention_mask_.end(), -10000.0F);
        std::fill(attention_mask_.begin(),
                  attention_mask_.begin() + static_cast<std::ptrdiff_t>(position_), 0.0F);
        attention_mask_.back() = 0.0F;
    }

    void sample_refinement_noise() {
        // NeMo samples one Gumbel-uniform row and then one Gaussian-noise
        // row at each of the eight RVQ refinement points.
        for (int32_t refinement = 0; refinement < config_.tts_num_refinement_steps; ++refinement) {
            const auto uniform_offset =
                static_cast<std::size_t>(refinement) * config_.tts_mog_num_predictions;
            for (int32_t index = 0; index < config_.tts_mog_num_predictions; ++index)
                mixture_uniform_[uniform_offset + static_cast<std::size_t>(index)] = uniform_(rng_);
            const auto noise_offset = static_cast<std::size_t>(refinement) * 512U;
            for (int32_t index = 0; index < 512; ++index)
                mog_noise_[noise_offset + static_cast<std::size_t>(index)] = normal_(rng_);
        }
    }

    void prepare_prompt_embedding(const float* prompt_embedding) {
        if (prompt_embedding != nullptr) {
            std::copy_n(prompt_embedding, config_.tts_hidden_size, audio_prompt_latent_.begin());
            return;
        }
        std::fill(audio_prompt_latent_.begin(), audio_prompt_latent_.end(), 0.0F);
    }

    std::vector<int32_t> extract_generated_codes(const TensorMap& outputs) const {
        const auto codes = outputs.find("rvq_codes");
        if (codes == outputs.end() || codes->second.dtype != DType::kInt32 ||
            codes->second.numel() != static_cast<std::size_t>(config_.tts_num_quantizers))
            throw std::runtime_error("VoiceChat EAR-TTS missing rvq_codes output");
        const auto* first = static_cast<const int32_t*>(codes->second.data);
        return std::vector<int32_t>(first, first + config_.tts_num_quantizers);
    }

    void bind_cache() {
        const std::vector<int64_t> cache_shape{2, config_.tts_max_cache_length,
                                               config_.tts_kv_width};
        for (int32_t layer = 0; layer < config_.tts_num_layers; ++layer) {
            const auto suffix = "_" + std::to_string(layer);
            const auto index = static_cast<std::size_t>(layer);
            module_.bind_external("cache_k" + suffix, cache_k_[index].data(), cache_shape);
            module_.bind_external("cache_v" + suffix, cache_v_[index].data(), cache_shape);
            module_.bind_external("present_k" + suffix, present_k_[index].data());
            module_.bind_external("present_v" + suffix, present_v_[index].data());
        }
    }

    void append_present() {
        if (position_ >= config_.tts_max_cache_length)
            throw std::runtime_error("VoiceChat EAR-TTS cache exhausted");
        const std::size_t row_bytes =
            static_cast<std::size_t>(config_.tts_kv_width) * dtype_size(cache_dtype_);
        const std::size_t batch_stride =
            static_cast<std::size_t>(config_.tts_max_cache_length) * row_bytes;
        const std::size_t row_offset = static_cast<std::size_t>(position_) * row_bytes;
        for (int32_t layer = 0; layer < config_.tts_num_layers; ++layer) {
            const auto index = static_cast<std::size_t>(layer);
            auto* dst_k = static_cast<std::byte*>(cache_k_[index].data());
            auto* dst_v = static_cast<std::byte*>(cache_v_[index].data());
            const auto* src_k = static_cast<const std::byte*>(present_k_[index].data());
            const auto* src_v = static_cast<const std::byte*>(present_v_[index].data());
            cudaMemcpyAsync(dst_k + row_offset, src_k, row_bytes, cudaMemcpyDeviceToDevice,
                            stream_);
            cudaMemcpyAsync(dst_k + batch_stride + row_offset, src_k + row_bytes, row_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
            cudaMemcpyAsync(dst_v + row_offset, src_v, row_bytes, cudaMemcpyDeviceToDevice,
                            stream_);
            cudaMemcpyAsync(dst_v + batch_stride + row_offset, src_v + row_bytes, row_bytes,
                            cudaMemcpyDeviceToDevice, stream_);
        }
        ++position_;
    }

    std::vector<int32_t> enqueue(int32_t subword_id, float subword_mask,
                                 const std::vector<int32_t>& previous_codes,
                                 const float* prompt_embedding, float prompt_mode, float bos_flag,
                                 int32_t position_id) {
        validate_enqueue_inputs(previous_codes, position_id);
        prepare_attention_mask();
        sample_refinement_noise();
        prepare_prompt_embedding(prompt_embedding);

        bind_cache();
        TensorMap inputs;
        inputs["prev_codes"] = tensor(const_cast<int32_t*>(previous_codes.data()),
                                      {config_.tts_num_quantizers}, DType::kInt32);
        inputs["subword_id"] = tensor(&subword_id, {1}, DType::kInt32);
        inputs["subword_mask"] = tensor(&subword_mask, {1}, DType::kFloat32);
        inputs["position_id"] = tensor(&position_id, {1}, DType::kInt32);
        inputs["attention_mask"] = tensor(
            attention_mask_.data(), {1, 1, 1, config_.tts_max_cache_length + 1}, DType::kFloat32);
        inputs["mixture_uniform"] = tensor(
            mixture_uniform_.data(),
            {config_.tts_num_refinement_steps, config_.tts_mog_num_predictions}, DType::kFloat32);
        inputs["mog_noise"] =
            tensor(mog_noise_.data(), {config_.tts_num_refinement_steps, 512}, DType::kFloat32);
        inputs["audio_prompt_latent"] =
            tensor(audio_prompt_latent_.data(), {config_.tts_hidden_size}, DType::kFloat32);
        inputs["audio_prompt_mode"] = tensor(&prompt_mode, {1}, DType::kFloat32);
        inputs["bos_flag"] = tensor(&bos_flag, {1}, DType::kFloat32);

        const auto outputs = module_.forward(inputs);
        auto generated = extract_generated_codes(outputs);
        append_present();
        return generated;
    }

    TrtModule& module_;
    const voicechat::Config& config_;
    const VoiceChatTtsPrompt& prompt_;
    cudaStream_t stream_{nullptr};
    DType cache_dtype_{DType::kFloat32};
    std::vector<DeviceTensor> cache_k_;
    std::vector<DeviceTensor> cache_v_;
    std::vector<DeviceTensor> present_k_;
    std::vector<DeviceTensor> present_v_;
    std::vector<float> attention_mask_;
    std::vector<float> mixture_uniform_;
    std::vector<float> mog_noise_;
    std::vector<float> audio_prompt_latent_;
    std::vector<int32_t> previous_codes_;
    int32_t position_{0};
    std::uint64_t seed_{0};
    std::mt19937_64 rng_;
    std::uniform_real_distribution<float> uniform_;
    std::normal_distribution<float> normal_;
};

} // namespace

class NemotronVoiceChatRuntime {
  public:
    NemotronVoiceChatRuntime(std::unique_ptr<TrtModule> thinker,
                             std::unique_ptr<TrtModule> perception_stream_first,
                             std::unique_ptr<TrtModule> perception_stream,
                             std::unique_ptr<TrtModule> rnnt_predictor,
                             std::unique_ptr<TrtModule> rnnt_joint, std::unique_ptr<TrtModule> tts,
                             std::unique_ptr<TrtModule> codec, voicechat::Config config,
                             VoiceChatAssets assets, std::shared_ptr<ITokenizer> tokenizer)
        : thinker(std::move(thinker)), perception_stream_first(std::move(perception_stream_first)),
          perception_stream(std::move(perception_stream)),
          rnnt_predictor(std::move(rnnt_predictor)), rnnt_joint(std::move(rnnt_joint)),
          tts(std::move(tts)), codec(std::move(codec)), config(std::move(config)),
          assets(std::move(assets)), tokenizer(std::move(tokenizer)) {
        require_module(this->thinker, "thinker");
        require_module(this->perception_stream_first, "first-step perception");
        require_module(this->perception_stream, "streaming perception");
        require_module(this->rnnt_predictor, "RNNT predictor");
        require_module(this->rnnt_joint, "RNNT joint");
        require_module(this->tts, "EAR-TTS");
        require_module(this->codec, "RVQ codec");
        if (!this->tokenizer)
            throw std::runtime_error("NemotronVoiceChat: native text tokenizer is required");
        if (this->assets.mel_filterbank.empty() || this->assets.mel_freq_bins <= 0 ||
            this->assets.mel_bins != this->config.mel_num_bins)
            throw std::runtime_error("NemotronVoiceChat: invalid mel filterbank asset");
        if (this->assets.mel_window.size() != static_cast<std::size_t>(this->config.mel_win_length))
            throw std::runtime_error("NemotronVoiceChat: checkpoint mel window size mismatch");
        if (this->assets.rnnt_vocabulary.size() !=
            static_cast<std::size_t>(this->config.rnnt_vocab_size))
            throw std::runtime_error("NemotronVoiceChat: RNNT vocabulary size mismatch");
    }

    std::unique_ptr<TrtModule> thinker;
    std::unique_ptr<TrtModule> perception_stream_first;
    std::unique_ptr<TrtModule> perception_stream;
    std::unique_ptr<TrtModule> rnnt_predictor;
    std::unique_ptr<TrtModule> rnnt_joint;
    std::unique_ptr<TrtModule> tts;
    std::unique_ptr<TrtModule> codec;
    voicechat::Config config;
    VoiceChatAssets assets;
    std::shared_ptr<ITokenizer> tokenizer;
    std::mutex inference_mutex;
};

namespace {

bool live_policy_limits_are_valid(const voicechat::Config& config) {
    return config.max_pending_input_ms > 0 && config.max_pending_events > 0 &&
           config.stream_tick_ms > 0 && config.function_max_response_tokens > 0 &&
           config.function_max_async_steps > 0 && config.function_tool_timeout_ms > 0 &&
           config.function_on_hold_min_pad_frames >= 0;
}

enum class SpeechSessionMode { kLive, kBatch };

class NemotronVoiceChatSession final : public ISpeechSession,
                                       public ISpeechRealtimeControl,
                                       public ISpeechToolSession {
  private:
    enum class WorkKind {
        kAudio,
        kFinish,
        kReset,
        kTick,
        kFunctionStep,
        kOnHoldStep,
        kToolResponse,
        kFunctionResponseStep,
        kToolTimeout,
        kCommitInput,
        kCreateResponse,
        kClearInput,
        kCancelResponse,
        kTruncateResponse,
    };

    struct WorkItem {
        WorkKind kind{WorkKind::kAudio};
        std::uint64_t work_epoch{0};
        std::uint64_t serial{0};
        std::vector<float> audio;
        std::uint64_t tool_epoch{0};
        std::uint64_t response_epoch{0};
        std::int64_t played_output_samples{0};
        bool create_response{true};
        std::vector<int32_t> forced_function_tokens;
        std::chrono::steady_clock::time_point enqueued_at{};
    };

    struct PendingToolCall {
        voicechat::FunctionCall call;
        std::optional<std::string> output;
    };

    struct PerceptionFrameOutputs {
        std::vector<float> rnnt_frame;
        std::vector<float> projected_audio;
    };

    struct RnntFrameActivity {
        bool emitted_speech_token{false};
    };

    struct ModelFrameDecision {
        int32_t text_token{0};
        int32_t function_token{0};
        std::optional<std::uint64_t> output_epoch;
        voicechat::FunctionChannelObservation function_observation;
        bool function_silent_step{false};
    };

    struct ThinkerReplayStep {
        int32_t text_token{0};
        int32_t timeline_token{0};
        int32_t function_token{0};
        std::vector<float> audio_embedding;
        bool use_audio{false};
    };

    struct TtsReplayStep {
        int32_t text_token{0};
        bool agent_idle{true};
    };

    struct ModelStateMarker {
        std::size_t thinker_steps{0};
        std::size_t tts_steps{0};
        std::size_t codec_steps{0};
        std::size_t timeline_steps{0};
        std::vector<int32_t> agent_text_tokens;
        int32_t previous_text_token{0};
        int32_t previous_function_token{0};
        int32_t agent_turn_frames{0};
        int32_t agent_turn_text_tokens{0};
        std::int64_t frame_index{0};
        bool agent_idle{true};
    };

    struct ResponseCheckpoint {
        ModelStateMarker model;
        std::int64_t response_end_sample{0};
    };

  public:
    NemotronVoiceChatSession(std::shared_ptr<NemotronVoiceChatRuntime> runtime,
                             SpeechSessionConfig session_config, SpeechSessionMode mode,
                             std::optional<SpeechToolSessionConfig> tool_config = std::nullopt)
        : runtime_(std::move(runtime)), session_config_(std::move(session_config)), mode_(mode),
          tool_config_(std::move(tool_config)), scheduler_(conversation_.epoch()),
          resampler_(session_config_.input_sample_rate, runtime_->config.input_sample_rate),
          mel_(runtime_->assets.mel_filterbank.data(), runtime_->assets.mel_freq_bins,
               runtime_->assets.mel_bins, mel_options(runtime_->config),
               runtime_->config.input_sample_rate, runtime_->assets.mel_window.data(),
               static_cast<int32_t>(runtime_->assets.mel_window.size())),
          tts_state_(*runtime_->tts, runtime_->config, runtime_->assets.tts_prompt,
                     session_config_.seed),
          function_channel_(static_cast<std::size_t>(runtime_->config.function_max_call_tokens)),
          turn_detector_(make_turn_policy(runtime_->config)) {
        if (session_config_.input_sample_rate <= 0)
            throw std::invalid_argument("VoiceChat input_sample_rate must be positive");
        if (session_config_.output_sample_rate == 0)
            session_config_.output_sample_rate = runtime_->config.output_sample_rate;
        if (session_config_.output_sample_rate <= 0)
            throw std::invalid_argument("VoiceChat output_sample_rate must be positive");
        if (session_config_.finish_tail_frames < -1)
            throw std::invalid_argument("VoiceChat finish_tail_frames must be -1 or non-negative");
        if (is_live())
            validate_live_config();
        initialize_queue_policies();
        initialize_tool_config();

        worker_ = std::thread([this] { worker_loop(); });
        std::unique_lock<std::mutex> lock(mutex_);
        initialized_cv_.wait(lock, [this] { return worker_initialized_ || worker_error_; });
        if (worker_error_) {
            const auto error = worker_error_;
            stop_requested_ = true;
            lock.unlock();
            work_cv_.notify_all();
            worker_.join();
            std::rethrow_exception(error);
        }
    }

    ~NemotronVoiceChatSession() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_requested_ = true;
            (void)work_epochs_.invalidate();
            work_queue_.clear();
        }
        work_cv_.notify_all();
        event_cv_.notify_all();
        if (worker_.joinable())
            worker_.join();
    }

    void append_audio(const float* samples, int32_t count) override {
        if (count < 0 || (count > 0 && samples == nullptr))
            throw std::invalid_argument("VoiceChat append_audio received invalid samples");
        if (count == 0)
            return;

        const auto chunk_samples = std::max<int32_t>(1, session_config_.input_sample_rate *
                                                            runtime_->config.stream_tick_ms / 1000);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            enqueue_audio_locked(samples, count, chunk_samples);
        }
        work_cv_.notify_one();
    }

    void finish_input() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rethrow_worker_error_locked();
            if (reset_in_progress_)
                throw std::logic_error("VoiceChat session reset is still in progress");
            if (public_input_finished_)
                return;
            if (!conversation_.can_accept_audio())
                throw std::logic_error("VoiceChat session cannot finish cancelled input");
            public_input_finished_ = true;
            conversation_.finish_input();
            WorkItem work;
            work.kind = WorkKind::kFinish;
            work.work_epoch = work_epochs_.current();
            work_queue_.push_back(std::move(work));
        }
        work_cv_.notify_one();
    }

    std::vector<SpeechSessionEvent> take_events() override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto events = std::move(events_);
        events_.clear();
        queued_output_audio_samples_ = 0;
        return events;
    }

    std::vector<SpeechSessionEvent> wait_events(int32_t timeout_ms) override {
        if (timeout_ms < -1)
            throw std::invalid_argument("speech event timeout must be -1 or non-negative");
        std::unique_lock<std::mutex> lock(mutex_);
        const auto ready = [this] {
            return !events_.empty() ||
                   voicechat::event_wait_is_terminal(conversation_.phase(),
                                                     worker_input_finished_) ||
                   worker_done_ || static_cast<bool>(worker_error_);
        };
        if (timeout_ms < 0)
            event_cv_.wait(lock, ready);
        else if (timeout_ms > 0)
            (void)event_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), ready);
        auto events = std::move(events_);
        events_.clear();
        queued_output_audio_samples_ = 0;
        return events;
    }

    void cancel() override {
        std::unique_lock<std::mutex> control_lock(reset_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto interrupted_epoch = conversation_.epoch();
            conversation_.cancel();
            (void)work_epochs_.invalidate();
            public_input_finished_ = true;
            purge_inference_work_locked();
            clear_pending_tools_locked();
            erase_interrupted_agent_output_locked(interrupted_epoch);
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kCancelled;
            event.epoch = conversation_.epoch();
            event.sequence = conversation_.next_sequence();
            enqueue_event_locked(std::move(event));
        }
        work_cv_.notify_all();
        event_cv_.notify_all();
    }

    void reset() override {
        std::unique_lock<std::mutex> reset_lock(reset_mutex_);
        std::uint64_t serial = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rethrow_worker_error_locked();
            conversation_.reset();
            const auto work_epoch = work_epochs_.invalidate();
            reset_in_progress_ = true;
            public_input_finished_ = false;
            worker_input_finished_ = false;
            purge_inference_work_locked();
            clear_pending_tools_locked();
            events_.clear();
            queued_output_audio_samples_ = 0;
            WorkItem work;
            work.kind = WorkKind::kReset;
            work.work_epoch = work_epoch;
            work.serial = serial = ++requested_reset_serial_;
            work_queue_.push_back(std::move(work));
        }
        work_cv_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        reset_cv_.wait(lock, [this, serial] {
            return completed_reset_serial_ >= serial || worker_done_ || worker_error_;
        });
        rethrow_worker_error_locked();
    }

    SpeechSessionConfig config() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return session_config_;
    }

    void submit_tool_response(std::uint64_t epoch, const std::string& call_id,
                              const std::string& output) override {
        if (!tool_config_)
            throw std::logic_error("VoiceChat session was not created with tools");
        validate_tool_response_size(output);

        bool queued;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queued = submit_tool_response_locked(epoch, call_id, output);
        }
        if (queued)
            work_cv_.notify_one();
    }

    void commit_input_turn(bool create_response) override {
        WorkItem work;
        work.kind = WorkKind::kCommitInput;
        work.create_response = create_response;
        run_control(std::move(work), false, false);
    }

    void clear_pending_input() override {
        WorkItem work;
        work.kind = WorkKind::kClearInput;
        run_control(std::move(work), true);
    }

    void create_response() override {
        WorkItem work;
        work.kind = WorkKind::kCreateResponse;
        run_control(std::move(work));
    }

    void cancel_response() override {
        WorkItem work;
        work.kind = WorkKind::kCancelResponse;
        run_control(std::move(work));
    }

    void truncate_response(std::uint64_t epoch, std::int64_t played_output_samples) override {
        WorkItem work;
        work.kind = WorkKind::kTruncateResponse;
        work.response_epoch = epoch;
        work.played_output_samples = played_output_samples;
        run_control(std::move(work));
    }

  private:
    bool is_live() const noexcept { return mode_ == SpeechSessionMode::kLive; }

    static voicechat::RnntTurnPolicy make_turn_policy(const voicechat::Config& config) {
        voicechat::RnntTurnPolicy policy;
        policy.first_utterance_min_speech_frames = config.rnnt_min_speech_frames_first_turn;
        policy.subsequent_utterance_min_speech_frames = config.rnnt_min_speech_frames;
        policy.end_of_utterance_blank_frames = config.rnnt_eou_frames;
        policy.beginning_of_utterance_speech_frames = config.rnnt_bou_frames;
        return policy;
    }

    void validate_live_config() const {
        const auto& config = runtime_->config;
        if (!live_policy_limits_are_valid(config))
            throw std::invalid_argument("VoiceChat live-session policy must be positive");
    }

    void enqueue_audio_locked(const float* samples, int32_t count, int32_t chunk_samples) {
        rethrow_worker_error_locked();
        if (reset_in_progress_)
            throw std::logic_error("VoiceChat session reset is still in progress");
        if (!conversation_.can_accept_audio())
            throw std::logic_error("VoiceChat session is not accepting audio; call reset()");
        if (!input_queue_budget_->reserve(static_cast<std::size_t>(count)))
            throw std::overflow_error("VoiceChat pending input exceeded its session queue bound");

        const std::size_t initial_queue_size = work_queue_.size();
        try {
            const auto work_epoch = work_epochs_.current();
            auto frame_time = std::chrono::steady_clock::now();
            for (int32_t offset = 0; offset < count; offset += chunk_samples) {
                const int32_t size = std::min(chunk_samples, count - offset);
                WorkItem work;
                work.kind = WorkKind::kAudio;
                work.work_epoch = work_epoch;
                work.enqueued_at = frame_time;
                work.audio.assign(samples + offset, samples + offset + size);
                work_queue_.push_back(std::move(work));
                frame_time += std::chrono::milliseconds(runtime_->config.stream_tick_ms);
            }
        } catch (...) {
            while (work_queue_.size() > initial_queue_size)
                work_queue_.pop_back();
            input_queue_budget_->release(static_cast<std::size_t>(count));
            throw;
        }
    }

    std::vector<int32_t> encode_tool_response(std::string_view output) const {
        return voicechat::build_tool_response_tokens(output,
                                                     [this](std::string_view text) {
                                                         return runtime_->tokenizer->encode(
                                                             std::string(text));
                                                     })
            .token_ids;
    }

    void validate_tool_response_size(std::string_view output) const {
        if (encode_tool_response(output).size() >
            static_cast<std::size_t>(runtime_->config.function_max_response_tokens))
            throw std::invalid_argument("VoiceChat tool response exceeds its token bound");
    }

    PendingToolCall& pending_tool_call_locked(std::uint64_t epoch, const std::string& call_id) {
        const auto pending =
            std::find_if(pending_tool_calls_.begin(), pending_tool_calls_.end(),
                         [&](const PendingToolCall& call) {
                             return call.call.call_id == call_id && epoch == pending_tool_epoch_;
                         });
        if (pending == pending_tool_calls_.end())
            throw std::invalid_argument("VoiceChat tool response does not match a pending call");
        if (std::chrono::steady_clock::now() >= tool_response_deadline_)
            throw std::invalid_argument("VoiceChat tool response arrived after its deadline");
        if (tool_response_work_queued_)
            throw std::logic_error("VoiceChat tool response cycle is already closing");
        if (pending->output.has_value())
            throw std::logic_error("VoiceChat tool response was already submitted");
        return *pending;
    }

    bool all_tool_responses_ready_locked() const {
        return std::all_of(pending_tool_calls_.begin(), pending_tool_calls_.end(),
                           [](const PendingToolCall& call) { return call.output.has_value(); });
    }

    std::vector<int32_t> combined_tool_response_tokens_locked(PendingToolCall& submitted) const {
        auto tokens = encode_tool_response(pending_tool_results_json_locked());
        if (tokens.size() >
            static_cast<std::size_t>(runtime_->config.function_max_response_tokens)) {
            submitted.output.reset();
            throw std::invalid_argument(
                "VoiceChat combined tool responses exceed their token bound");
        }
        return tokens;
    }

    void queue_tool_response_locked(std::uint64_t epoch, std::vector<int32_t> tokens) {
        WorkItem work;
        work.kind = WorkKind::kToolResponse;
        work.work_epoch = work_epochs_.current();
        work.tool_epoch = epoch;
        work.forced_function_tokens = std::move(tokens);
        work_queue_.push_front(std::move(work));
        tool_response_work_queued_ = true;
    }

    bool submit_tool_response_locked(std::uint64_t epoch, const std::string& call_id,
                                     const std::string& output) {
        rethrow_worker_error_locked();
        if (reset_in_progress_)
            throw std::logic_error("VoiceChat session reset is still in progress");
        auto& pending = pending_tool_call_locked(epoch, call_id);
        pending.output = output;
        if (!all_tool_responses_ready_locked())
            return false;
        queue_tool_response_locked(epoch, combined_tool_response_tokens_locked(pending));
        return true;
    }

    void initialize_queue_policies() {
        const auto capacity_ms =
            is_live() && session_config_.enable_barge_in
                ? static_cast<std::size_t>(runtime_->config.max_pending_input_ms)
                : static_cast<std::size_t>(
                      voicechat::streaming_frontend_capacity_seconds(runtime_->config)) *
                      1000U;
        const auto input_capacity =
            static_cast<std::size_t>(session_config_.input_sample_rate) * capacity_ms / 1000U;
        input_queue_budget_.emplace(std::max<std::size_t>(input_capacity, 1U));
        const auto output_frame_samples =
            static_cast<std::size_t>(std::max(session_config_.output_sample_rate, 1)) *
                static_cast<std::size_t>(runtime_->config.stream_tick_ms) / 1000U +
            1U;
        output_capacity_.emplace(static_cast<std::size_t>(runtime_->config.max_pending_events),
                                 static_cast<std::size_t>(runtime_->config.max_pending_events) *
                                     output_frame_samples);
    }

    void initialize_tool_config() {
        if (!tool_config_)
            return;
        function_tools_ = voicechat::FunctionToolCatalog::from_json(tool_config_->tools_json);
        if (function_tools_.tools().empty())
            throw std::invalid_argument("VoiceChat tool sessions require at least one tool");
        if (!tool_config_->on_hold_messages_json.empty()) {
            on_hold_messages_ =
                voicechat::FunctionOnHoldMessages::from_json(tool_config_->on_hold_messages_json);
        }
        if (runtime_->tokenizer->id_for_token("<SPECIAL_20>") != voicechat::kFunctionSotcTokenId ||
            runtime_->tokenizer->id_for_token("<SPECIAL_21>") != voicechat::kFunctionEotcTokenId ||
            runtime_->tokenizer->id_for_token("<SPECIAL_22>") != voicechat::kFunctionEotrTokenId)
            throw std::invalid_argument("VoiceChat tokenizer function markers do not match config");
    }

    void rethrow_worker_error_locked() const {
        if (worker_error_)
            std::rethrow_exception(worker_error_);
    }

    static bool is_control_work(WorkKind kind) {
        return kind == WorkKind::kCommitInput || kind == WorkKind::kCreateResponse ||
               kind == WorkKind::kClearInput || kind == WorkKind::kCancelResponse ||
               kind == WorkKind::kTruncateResponse;
    }

    void erase_queued_input_locked() {
        std::size_t released_audio = 0;
        work_queue_.erase(std::remove_if(work_queue_.begin(), work_queue_.end(),
                                         [&](const WorkItem& item) {
                                             if (item.kind != WorkKind::kAudio &&
                                                 item.kind != WorkKind::kTick)
                                                 return false;
                                             released_audio += item.audio.size();
                                             return true;
                                         }),
                          work_queue_.end());
        if (released_audio != 0)
            input_queue_budget_->release(released_audio);
    }

    void run_control(WorkItem work, bool clear_queued_audio = false, bool priority = true) {
        if (!is_live())
            throw std::logic_error("VoiceChat realtime controls require a live session");
        std::unique_lock<std::mutex> control_lock(reset_mutex_);
        std::uint64_t serial = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rethrow_worker_error_locked();
            if (reset_in_progress_)
                throw std::logic_error("VoiceChat session reset is still in progress");
            if (public_input_finished_)
                throw std::logic_error("VoiceChat realtime controls require an open input stream");
            if (clear_queued_audio)
                erase_queued_input_locked();
            work.work_epoch = work_epochs_.current();
            work.serial = serial = ++requested_control_serial_;
            if (priority)
                work_queue_.push_front(std::move(work));
            else
                work_queue_.push_back(std::move(work));
        }
        work_cv_.notify_all();

        std::unique_lock<std::mutex> lock(mutex_);
        reset_cv_.wait(lock, [this, serial] {
            return completed_control_serial_ >= serial || worker_done_ || worker_error_;
        });
        rethrow_worker_error_locked();
        if (control_error_serial_ == serial && control_error_)
            std::rethrow_exception(control_error_);
    }

    void acknowledge_control(std::uint64_t serial, std::exception_ptr error = {}) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_control_serial_ = std::max(completed_control_serial_, serial);
            control_error_serial_ = serial;
            control_error_ = std::move(error);
        }
        reset_cv_.notify_all();
    }

    void purge_inference_work_locked() {
        std::size_t released_audio = 0;
        for (const auto& item : work_queue_) {
            if (item.kind == WorkKind::kAudio)
                released_audio += item.audio.size();
        }
        work_queue_.erase(
            std::remove_if(work_queue_.begin(), work_queue_.end(),
                           [](const WorkItem& item) { return item.kind != WorkKind::kReset; }),
            work_queue_.end());
        if (released_audio != 0)
            input_queue_budget_->release(released_audio);
    }

    void enqueue_event_locked(SpeechSessionEvent event) {
        output_capacity_->validate(events_.size(), queued_output_audio_samples_, event);
        queued_output_audio_samples_ += event.audio_samples.size();
        events_.push_back(std::move(event));
    }

    void clear_pending_tools_locked() {
        pending_tool_calls_.clear();
        pending_tool_epoch_ = 0;
        tool_response_work_queued_ = false;
        tool_response_deadline_ = {};
    }

    void recompute_output_audio_samples_locked() {
        queued_output_audio_samples_ = 0;
        for (const auto& event : events_)
            queued_output_audio_samples_ += event.audio_samples.size();
    }

    void erase_interrupted_agent_output_locked(std::uint64_t interrupted_epoch) {
        std::int64_t removed_audio_samples = 0;
        for (const auto& event : events_) {
            if (event.epoch == interrupted_epoch && voicechat::is_agent_output_event(event.kind))
                removed_audio_samples += static_cast<std::int64_t>(event.audio_samples.size());
        }
        events_.erase(std::remove_if(events_.begin(), events_.end(),
                                     [&](const SpeechSessionEvent& event) {
                                         return event.epoch == interrupted_epoch &&
                                                voicechat::is_agent_output_event(event.kind);
                                     }),
                      events_.end());
        output_sample_cursor_ =
            std::max<std::int64_t>(0, output_sample_cursor_ - removed_audio_samples);
        recompute_output_audio_samples_locked();
    }

    void erase_response_output_locked(std::uint64_t response_epoch) {
        events_.erase(std::remove_if(events_.begin(), events_.end(),
                                     [&](const auto& event) {
                                         return event.epoch == response_epoch &&
                                                voicechat::is_agent_output_event(event.kind);
                                     }),
                      events_.end());
        recompute_output_audio_samples_locked();
    }

    static bool is_response_work(WorkKind kind) {
        return kind == WorkKind::kFunctionStep || kind == WorkKind::kOnHoldStep ||
               kind == WorkKind::kToolResponse || kind == WorkKind::kFunctionResponseStep ||
               kind == WorkKind::kToolTimeout;
    }

    void purge_response_work_locked() {
        work_queue_.erase(
            std::remove_if(work_queue_.begin(), work_queue_.end(),
                           [](const WorkItem& item) { return is_response_work(item.kind); }),
            work_queue_.end());
    }

    bool work_is_current(std::uint64_t work_epoch) const {
        return work_epochs_.accepts(work_epoch);
    }

    std::optional<std::uint64_t> accepted_output_epoch(std::uint64_t work_epoch) const {
        if (!work_is_current(work_epoch))
            return std::nullopt;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto epoch = conversation_.epoch();
        if (!work_is_current(work_epoch) || !conversation_.accepts_output(epoch))
            return std::nullopt;
        return epoch;
    }

    bool agent_reply_active(std::uint64_t work_epoch) const {
        if (!work_is_current(work_epoch))
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        return work_is_current(work_epoch) &&
               conversation_.phase() == voicechat::ConversationPhase::kAgentSpeaking;
    }

    bool publish_agent_event(SpeechSessionEvent event, std::uint64_t work_epoch,
                             std::uint64_t output_epoch) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch) || !conversation_.accepts_output(output_epoch))
                return false;
            event.epoch = output_epoch;
            event.sequence = conversation_.next_sequence();
            enqueue_event_locked(std::move(event));
        }
        event_cv_.notify_all();
        return true;
    }

    bool publish_current_event(SpeechSessionEvent event, std::uint64_t work_epoch) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch))
                return false;
            event.epoch = conversation_.epoch();
            event.sequence = conversation_.next_sequence();
            enqueue_event_locked(std::move(event));
        }
        event_cv_.notify_all();
        return true;
    }

    void publish_input_finished(std::uint64_t work_epoch) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch))
                return;
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kInputFinished;
            event.epoch = conversation_.epoch();
            event.sequence = conversation_.next_sequence();
            enqueue_event_locked(std::move(event));
            worker_input_finished_ = true;
        }
        event_cv_.notify_all();
    }

    void acknowledge_reset(std::uint64_t serial) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_reset_serial_ = std::max(completed_reset_serial_, serial);
            reset_in_progress_ = false;
        }
        reset_cv_.notify_all();
    }

    bool tool_response_pending_locked() const { return !pending_tool_calls_.empty(); }

    bool tool_response_pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tool_response_pending_locked();
    }

    bool clock_needed_locked() const {
        if (!is_live() || public_input_finished_ || tool_response_pending_locked())
            return false;
        return conversation_.phase() == voicechat::ConversationPhase::kAgentSpeaking ||
               turn_detector_.utterance_active() || turn_detector_.speech_frames() > 0;
    }

    bool work_is_processable_locked(const WorkItem& work) const {
        if (!tool_response_pending_locked())
            return true;
        return work.kind == WorkKind::kAudio || work.kind == WorkKind::kOnHoldStep ||
               work.kind == WorkKind::kToolResponse ||
               work.kind == WorkKind::kFunctionResponseStep || work.kind == WorkKind::kReset ||
               is_control_work(work.kind);
    }

    bool has_processable_work_locked() const {
        return std::any_of(work_queue_.begin(), work_queue_.end(), [this](const WorkItem& work) {
            return work_is_processable_locked(work);
        });
    }

    std::optional<WorkItem> take_processable_work_locked() {
        const auto selected =
            std::find_if(work_queue_.begin(), work_queue_.end(),
                         [this](const WorkItem& work) { return work_is_processable_locked(work); });
        if (selected == work_queue_.end())
            return std::nullopt;
        WorkItem work = std::move(*selected);
        work_queue_.erase(selected);
        if (work.kind == WorkKind::kAudio)
            input_queue_budget_->release(work.audio.size());
        return work;
    }

    std::optional<WorkItem> take_tool_control_work_locked() {
        const auto selected =
            std::find_if(work_queue_.begin(), work_queue_.end(), [](const WorkItem& work) {
                return work.kind == WorkKind::kToolResponse || work.kind == WorkKind::kReset;
            });
        if (selected == work_queue_.end())
            return std::nullopt;
        WorkItem work = std::move(*selected);
        work_queue_.erase(selected);
        return work;
    }

    void enqueue_internal_work(WorkKind kind, std::uint64_t work_epoch, bool priority = false) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch))
                return;
            WorkItem work;
            work.kind = kind;
            work.work_epoch = work_epoch;
            if (priority)
                work_queue_.push_front(std::move(work));
            else
                work_queue_.push_back(std::move(work));
        }
        work_cv_.notify_one();
    }

    bool try_take_tool_work_locked(WorkItem& work) {
        if (!tool_response_pending_locked())
            return false;
        if (auto control = take_tool_control_work_locked()) {
            work = std::move(*control);
            return true;
        }
        if (tool_response_work_queued_ ||
            std::chrono::steady_clock::now() < tool_response_deadline_)
            return false;
        tool_response_work_queued_ = true;
        work.kind = WorkKind::kToolTimeout;
        work.work_epoch = work_epochs_.current();
        return true;
    }

    bool try_take_ready_work_locked(WorkItem& work) {
        auto ready = take_processable_work_locked();
        if (!ready)
            return false;
        work = std::move(*ready);
        return true;
    }

    bool processable_work_or_stop_locked() const {
        return stop_requested_ || has_processable_work_locked();
    }

    bool worker_should_wake_locked() const {
        return processable_work_or_stop_locked() || clock_needed_locked();
    }

    void arm_clock_locked() {
        if (clock_armed_)
            return;
        clock_armed_ = true;
        next_tick_deadline_ = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(runtime_->config.stream_tick_ms);
    }

    bool wait_for_next_work(WorkItem& work) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stop_requested_) {
            if (try_take_tool_work_locked(work) || try_take_ready_work_locked(work))
                return true;
            if (tool_response_pending_locked()) {
                work_cv_.wait_until(lock, tool_response_deadline_,
                                    [this] { return processable_work_or_stop_locked(); });
                continue;
            }
            if (clock_needed_locked()) {
                arm_clock_locked();
                if (!work_cv_.wait_until(lock, next_tick_deadline_,
                                         [this] { return processable_work_or_stop_locked(); })) {
                    work.kind = WorkKind::kTick;
                    work.work_epoch = work_epochs_.current();
                    return true;
                }
                continue;
            }
            clock_armed_ = false;
            work_cv_.wait(lock, [this] { return worker_should_wake_locked(); });
        }
        return false;
    }

    void initialize_worker() {
        initialize_host_state(work_epochs_.current());
        initialize_model_state();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_initialized_ = true;
        }
        initialized_cv_.notify_all();
    }

    void handle_worker_failure(std::exception_ptr error) {
        std::string message = "VoiceChat native worker failed";
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& exception) {
            message += ": ";
            message += exception.what();
        } catch (...) {
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_error_ = error;
            worker_initialized_ = true;
            worker_done_ = true;
            conversation_.cancel();
            (void)work_epochs_.invalidate();
            work_queue_.clear();
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kError;
            event.epoch = conversation_.epoch();
            event.sequence = conversation_.next_sequence();
            event.text = std::move(message);
            events_.clear();
            queued_output_audio_samples_ = 0;
            enqueue_event_locked(std::move(event));
        }
        initialized_cv_.notify_all();
        reset_cv_.notify_all();
        event_cv_.notify_all();
    }

    void mark_worker_done() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_done_ = true;
        }
        reset_cv_.notify_all();
        event_cv_.notify_all();
    }

    void worker_loop() noexcept {
        try {
            initialize_worker();
            while (true) {
                WorkItem work;
                if (!wait_for_next_work(work))
                    break;
                process_work(work);
                event_cv_.notify_all();
            }
        } catch (...) {
            handle_worker_failure(std::current_exception());
            return;
        }
        mark_worker_done();
    }

    void process_reset_work(const WorkItem& work) {
        if (work_is_current(work.work_epoch)) {
            initialize_host_state(work.work_epoch);
            initialize_model_state();
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kReset;
            (void)publish_current_event(std::move(event), work.work_epoch);
        }
        acknowledge_reset(work.serial);
    }

    void begin_input_buffer_if_needed(std::uint64_t work_epoch) {
        if (!is_live() || input_buffer_start_marker_.has_value() || !work_is_current(work_epoch))
            return;
        bool listening = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            listening = work_is_current(work_epoch) &&
                        conversation_.phase() == voicechat::ConversationPhase::kListening;
        }
        if (listening)
            input_buffer_start_marker_ = capture_model_marker();
    }

    void process_audio_work(const WorkItem& work) {
        if (!work.audio.empty()) {
            begin_input_buffer_if_needed(work.work_epoch);
            turn_control_.note_input();
        }
        clock_armed_ = true;
        next_tick_deadline_ =
            work.enqueued_at + std::chrono::milliseconds(runtime_->config.stream_tick_ms);
        resampler_.append(work.audio.data(), static_cast<int32_t>(work.audio.size()));
        auto native = resampler_.drain(false);
        accept_native_samples(native, false, work.work_epoch);
    }

    void process_tick_work(const WorkItem& work) {
        std::array<float, static_cast<std::size_t>(voicechat::kInputFrameSamples)> silence{};
        accept_native_samples(std::vector<float>(silence.begin(), silence.end()), false,
                              work.work_epoch);
        next_tick_deadline_ += std::chrono::milliseconds(runtime_->config.stream_tick_ms);
        const auto now = std::chrono::steady_clock::now();
        if (next_tick_deadline_ + std::chrono::milliseconds(runtime_->config.stream_tick_ms) < now)
            next_tick_deadline_ = now;
    }

    static bool is_stream_work(WorkKind kind) {
        return kind == WorkKind::kAudio || kind == WorkKind::kFinish || kind == WorkKind::kTick;
    }

    void process_stream_work(const WorkItem& work) {
        switch (work.kind) {
        case WorkKind::kAudio:
            process_audio_work(work);
            break;
        case WorkKind::kFinish:
            process_finish(work);
            break;
        case WorkKind::kTick:
            process_tick_work(work);
            break;
        default:
            break;
        }
    }

    void process_function_work(const WorkItem& work) {
        switch (work.kind) {
        case WorkKind::kFunctionStep:
            process_function_step(work.work_epoch);
            break;
        case WorkKind::kOnHoldStep:
            process_on_hold_step(work.work_epoch);
            break;
        case WorkKind::kToolResponse:
            process_tool_response(work);
            break;
        case WorkKind::kFunctionResponseStep:
            process_function_response_step(work.work_epoch);
            break;
        case WorkKind::kToolTimeout:
            process_tool_timeout(work.work_epoch);
            break;
        default:
            break;
        }
    }

    void process_control_work(const WorkItem& work) {
        try {
            if (!work_is_current(work.work_epoch))
                throw std::logic_error("VoiceChat realtime control is stale");
            switch (work.kind) {
            case WorkKind::kCommitInput:
                process_commit_input(work);
                break;
            case WorkKind::kCreateResponse:
                process_create_response(work.work_epoch);
                break;
            case WorkKind::kClearInput:
                process_clear_input(work.work_epoch);
                break;
            case WorkKind::kCancelResponse:
                process_cancel_response(work.work_epoch);
                break;
            case WorkKind::kTruncateResponse:
                process_truncate_response(work.work_epoch, work.response_epoch,
                                          work.played_output_samples);
                break;
            default:
                break;
            }
            acknowledge_control(work.serial);
        } catch (...) {
            acknowledge_control(work.serial, std::current_exception());
        }
    }

    void process_work(const WorkItem& work) {
        if (work.kind == WorkKind::kReset) {
            process_reset_work(work);
            return;
        }
        if (is_control_work(work.kind)) {
            process_control_work(work);
            return;
        }
        if (!work_is_current(work.work_epoch))
            return;
        if (is_stream_work(work.kind)) {
            process_stream_work(work);
            return;
        }
        process_function_work(work);
    }

    static bool is_user_input_event(SpeechSessionEventKind kind) {
        return kind == SpeechSessionEventKind::kUserTranscript ||
               kind == SpeechSessionEventKind::kUserSpeechStarted ||
               kind == SpeechSessionEventKind::kUserSpeechStopped;
    }

    void erase_uncommitted_input_events(std::uint64_t work_epoch) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!work_is_current(work_epoch))
            return;
        const auto epoch = conversation_.epoch();
        events_.erase(std::remove_if(events_.begin(), events_.end(),
                                     [&](const auto& event) {
                                         return event.epoch == epoch &&
                                                is_user_input_event(event.kind);
                                     }),
                      events_.end());
    }

    void reset_processed_input_frontier(std::uint64_t work_epoch) {
        resampler_.reset();
        scheduler_.clear_pending();
        mel_.reset();
        first_perception_step_ = true;
        next_mel_frame_ = 0;
        perception_cache_length_ = 0;
        std::fill(perception_channel_cache_.begin(), perception_channel_cache_.end(), 0.0F);
        std::fill(perception_time_cache_.begin(), perception_time_cache_.end(), 0.0F);
        reset_rnnt_utterance_decoder();
        turn_detector_.reset();
        deferred_audio_embeddings_.clear();
        turn_control_.reset();
        current_frame_start_marker_.reset();
        pending_response_audio_end_.reset();
        clock_armed_ = false;
        erase_uncommitted_input_events(work_epoch);
    }

    void process_clear_input(std::uint64_t work_epoch) {
        if (turn_control_.response_available())
            throw std::logic_error("VoiceChat cannot clear an already committed input turn");
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch) ||
                conversation_.phase() != voicechat::ConversationPhase::kListening)
                throw std::logic_error(
                    "VoiceChat can clear input only while the agent is listening");
        }
        if (input_buffer_start_marker_.has_value())
            restore_model_marker(*input_buffer_start_marker_);
        input_buffer_start_marker_.reset();
        reset_processed_input_frontier(work_epoch);
    }

    void flush_committed_input(std::uint64_t work_epoch) {
        auto native = resampler_.drain(true);
        resampler_.reset();
        if (!native.empty())
            scheduler_.append(native.data(), static_cast<int32_t>(native.size()));
        scheduler_.commit();
        process_scheduled_frames(false, work_epoch, true);
    }

    void finalize_committed_input(std::uint64_t work_epoch) {
        const bool had_transcript = !rnnt_tokens_.empty();
        const auto decision = turn_detector_.finalize_utterance(
            conversation_agent_speaking(work_epoch), rnnt_observation_frame_index_++);
        (void)apply_turn_decision(decision, work_epoch);
        if (!decision.speech_stopped && had_transcript) {
            emit_transcript(true, work_epoch);
            reset_rnnt_utterance_decoder();
        }
        turn_control_.commit(had_transcript || decision.speech_stopped);
    }

    void start_committed_response(bool create_response, std::uint64_t work_epoch) {
        if (!create_response) {
            suppress_native_agent_start_ = true;
            return;
        }
        suppress_native_agent_start_ = false;
        turn_control_.consume_response();
        process_model_frame(zero_audio_embedding_, work_epoch, runtime_->config.bos_token_id);
    }

    void process_commit_input(const WorkItem& work) {
        if (work.create_response && response_playback_.active())
            process_cancel_response(work.work_epoch);
        flush_committed_input(work.work_epoch);
        finalize_committed_input(work.work_epoch);
        input_buffer_start_marker_.reset();
        start_committed_response(work.create_response, work.work_epoch);
    }

    void process_create_response(std::uint64_t work_epoch) {
        if (response_playback_.active())
            process_cancel_response(work_epoch);
        suppress_native_agent_start_ = false;
        turn_control_.consume_response();
        process_model_frame(zero_audio_embedding_, work_epoch, runtime_->config.bos_token_id);
    }

    void reset_response_runtime_state() {
        function_channel_.reset();
        function_cycle_active_ = false;
        function_generation_active_ = false;
        forced_function_tokens_.clear();
        on_hold_token_queue_.clear();
        active_function_epoch_ = 0;
        agent_idle_ = true;
        agent_text_tokens_.clear();
        agent_turn_frames_ = 0;
        agent_turn_text_tokens_ = 0;
        suppress_synthesis_until_turn_started_ = true;
        suppress_native_agent_start_ = true;
    }

    void replay_cancelled_timeline(const std::vector<std::vector<float>>& audio_embeddings) {
        bool force_eos = true;
        for (const auto& audio_embedding : audio_embeddings) {
            std::pair<int32_t, int32_t> tokens;
            {
                std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
                tokens = run_thinker(previous_text_token_, runtime_->config.pad_token_id,
                                     previous_function_token_, audio_embedding, true);
            }
            timeline_replay_.push_back(thinker_replay_.size() - 1);
            previous_text_token_ = force_eos ? runtime_->config.eos_token_id : tokens.first;
            if (previous_text_token_ == runtime_->config.bos_token_id)
                previous_text_token_ = runtime_->config.pad_token_id;
            previous_function_token_ = runtime_->config.pad_token_id;
            force_eos = false;
            ++frame_index_;
        }
        if (!force_eos)
            return;
        // The checkpoint stores the last retained token as the next recurrent
        // input. Consume it once before making EOS the next input, otherwise a
        // truncate at the latest audio boundary would omit the final token from
        // the thinker's conversation state.
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            (void)run_thinker(previous_text_token_, runtime_->config.pad_token_id,
                              previous_function_token_, zero_audio_embedding_, false);
        }
        previous_text_token_ = runtime_->config.eos_token_id;
        previous_function_token_ = runtime_->config.pad_token_id;
        ++frame_index_;
    }

    void yield_truncated_response(std::uint64_t response_epoch, std::int64_t retained_samples,
                                  std::string_view reason) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!conversation_.accepts_output(response_epoch))
                throw std::invalid_argument("VoiceChat response epoch is stale");
            erase_response_output_locked(response_epoch);
            purge_response_work_locked();
            clear_pending_tools_locked();
            if (!conversation_.yield_to_user())
                throw std::logic_error("VoiceChat response is not active");
            SpeechSessionEvent yielded;
            yielded.kind = SpeechSessionEventKind::kYielded;
            yielded.epoch = conversation_.epoch();
            yielded.sequence = conversation_.next_sequence();
            yielded.frame_index = frame_index_;
            yielded.text = std::string(reason);
            enqueue_event_locked(std::move(yielded));
        }
        output_sample_cursor_ = response_start_output_sample_ + retained_samples;
        event_cv_.notify_all();
    }

    void rollback_response(std::uint64_t work_epoch, std::uint64_t response_epoch,
                           std::size_t checkpoint_index, std::string_view reason) {
        if (!work_is_current(work_epoch) || checkpoint_index >= response_checkpoints_.size())
            throw std::invalid_argument("VoiceChat response checkpoint is stale");
        const auto checkpoint = response_checkpoints_[checkpoint_index];
        validate_model_marker(checkpoint.model);
        std::vector<std::vector<float>> replay_audio;
        replay_audio.reserve(timeline_replay_.size() - checkpoint.model.timeline_steps +
                             deferred_audio_embeddings_.size());
        for (std::size_t index = checkpoint.model.timeline_steps; index < timeline_replay_.size();
             ++index) {
            replay_audio.push_back(thinker_replay_.at(timeline_replay_[index]).audio_embedding);
        }
        replay_audio.insert(replay_audio.end(),
                            std::make_move_iterator(deferred_audio_embeddings_.begin()),
                            std::make_move_iterator(deferred_audio_embeddings_.end()));
        deferred_audio_embeddings_.clear();

        restore_model_marker(checkpoint.model);
        reset_response_runtime_state();
        replay_cancelled_timeline(replay_audio);
        yield_truncated_response(response_epoch, checkpoint.response_end_sample, reason);
        turn_control_.restore_response();
        reset_response_tracking();
    }

    void process_cancel_response(std::uint64_t work_epoch) {
        if (!response_playback_.active())
            return;
        const auto response_epoch = response_playback_.epoch();
        const std::size_t checkpoint =
            function_cycle_active_ ? 0 : response_checkpoints_.size() - 1;
        rollback_response(work_epoch, response_epoch, checkpoint, "response-cancel");
    }

    void process_truncate_response(std::uint64_t work_epoch, std::uint64_t response_epoch,
                                   std::int64_t played_output_samples) {
        if (function_cycle_active_)
            throw std::logic_error("VoiceChat cannot truncate an active function cycle");
        const auto checkpoint =
            response_playback_.retained_boundary_count(response_epoch, played_output_samples);
        rollback_response(work_epoch, response_epoch, checkpoint, "response-truncate");
    }

    void process_finish(const WorkItem& work) {
        auto native = resampler_.drain(true);
        accept_native_samples(native, true, work.work_epoch);
        if (work_is_current(work.work_epoch)) {
            scheduler_.finish();
            process_scheduled_frames(true, work.work_epoch);
            finalize_input_turn(work.work_epoch);

            const int32_t tail_bound = voicechat::resolve_finish_tail_frames(
                session_config_.finish_tail_frames, runtime_->config.max_response_frames);
            for (int32_t tail = 0; tail < tail_bound && agent_reply_active(work.work_epoch);
                 ++tail) {
                std::array<float, static_cast<std::size_t>(voicechat::kInputFrameSamples)>
                    silence{};
                mel_.accept_audio(silence.data(), static_cast<int32_t>(silence.size()));
                voicechat::ScheduledInputFrame frame;
                frame.samples = silence;
                frame.epoch = work.work_epoch;
                frame.frame_index = frame_index_;
                frame.valid_input_samples = static_cast<int32_t>(silence.size());
                process_audio_frame(frame, false, work.work_epoch);
            }

            if (agent_reply_active(work.work_epoch)) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (work_is_current(work.work_epoch) &&
                    conversation_.phase() == voicechat::ConversationPhase::kAgentSpeaking) {
                    (void)conversation_.yield_to_user();
                    suppress_synthesis_until_turn_started_ = true;
                    SpeechSessionEvent yielded;
                    yielded.kind = SpeechSessionEventKind::kYielded;
                    yielded.epoch = conversation_.epoch();
                    yielded.sequence = conversation_.next_sequence();
                    yielded.text = "max-response-frames";
                    enqueue_event_locked(std::move(yielded));
                }
                event_cv_.notify_all();
            }

            publish_input_finished(work.work_epoch);
        }
    }

    void finalize_input_turn(std::uint64_t work_epoch) {
        if (!is_live()) {
            emit_transcript(true, work_epoch);
            return;
        }
        const auto final_turn = turn_detector_.finalize_utterance(
            conversation_agent_speaking(work_epoch), rnnt_observation_frame_index_++);
        const auto forced_text = apply_turn_decision(final_turn, work_epoch);
        if (forced_text.has_value())
            process_model_frame(zero_audio_embedding_, work_epoch, forced_text);
        else if (!final_turn.speech_stopped && !rnnt_tokens_.empty())
            emit_transcript(true, work_epoch);
    }

    static voicechat_audio::MelSpectrogramOptions mel_options(const voicechat::Config& config) {
        voicechat_audio::MelSpectrogramOptions options;
        options.n_fft = config.mel_n_fft;
        options.win_length = config.mel_win_length;
        options.hop_length = config.mel_hop_length;
        options.chunk_length_s = voicechat::streaming_frontend_capacity_seconds(config);
        options.sample_rate = config.input_sample_rate;
        options.symmetric_window = true;
        options.center_window_in_fft = true;
        options.preemphasis = config.mel_preemphasis;
        options.log_scale = voicechat_audio::MelLogScale::kNaturalLog;
        options.normalize_per_feature = false;
        return options;
    }

    void initialize_host_state(std::uint64_t work_epoch) {
        const auto& config = runtime_->config;
        scheduler_.reset(work_epoch);
        resampler_.reset();
        mel_.reset();
        first_perception_step_ = true;
        clock_armed_ = false;
        next_mel_frame_ = 0;
        perception_cache_length_ = 0;
        output_sample_cursor_ = 0;
        frame_index_ = 0;
        rnnt_observation_frame_index_ = 0;
        previous_text_token_ = config.pad_token_id;
        previous_function_token_ = config.pad_token_id;
        agent_idle_ = true;
        suppress_synthesis_until_turn_started_ = false;
        suppress_native_agent_start_ = false;
        turn_control_.reset();
        agent_text_tokens_.clear();
        rnnt_tokens_.clear();
        rnnt_text_.clear();
        turn_detector_.reset();
        function_channel_.reset();
        active_function_epoch_ = 0;
        function_call_serial_ = 0;
        function_cycle_active_ = false;
        function_generation_active_ = false;
        function_async_steps_ = 0;
        function_response_steps_ = 0;
        forced_function_tokens_.clear();
        on_hold_token_queue_.clear();
        deferred_audio_embeddings_.clear();
        thinker_replay_.clear();
        tts_replay_.clear();
        codec_replay_.clear();
        timeline_replay_.clear();
        response_checkpoints_.clear();
        input_buffer_start_marker_.reset();
        response_playback_.reset();
        current_frame_start_marker_.reset();
        pending_response_audio_end_.reset();
        response_start_output_sample_ = 0;
        record_replay_state_ = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            clear_pending_tools_locked();
        }
        rnnt_unk_token_id_ = -1;
        for (int32_t token = 0; token < config.rnnt_vocab_size; ++token) {
            const auto& piece = runtime_->assets.rnnt_vocabulary[static_cast<std::size_t>(token)];
            if (piece == "<unk>" || piece == "⁇") {
                rnnt_unk_token_id_ = token;
                break;
            }
        }
        const std::size_t channel_elements =
            static_cast<std::size_t>(config.perception_num_layers) *
            config.perception_att_context_left * config.perception_hidden_size;
        const std::size_t time_elements = static_cast<std::size_t>(config.perception_num_layers) *
                                          config.perception_hidden_size * 8U;
        perception_channel_cache_.assign(channel_elements, 0.0F);
        perception_time_cache_.assign(time_elements, 0.0F);
        const std::size_t rnnt_state_elements =
            static_cast<std::size_t>(config.rnnt_pred_num_layers) * config.rnnt_pred_hidden_size;
        rnnt_h_.assign(rnnt_state_elements, 0.0F);
        rnnt_c_.assign(rnnt_state_elements, 0.0F);
        zero_audio_embedding_.assign(static_cast<std::size_t>(config.hidden_size), 0.0F);
        codec_cache_.reset();
        codec_reconstruction_.reset();
    }

    void initialize_model_state() {
        std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
        record_replay_state_ = false;
        const auto& config = runtime_->config;
        if (!thinker_state_) {
            auto kv = std::make_unique<VoiceChatThinkerKvCache>(
                config.num_attention_layers, config.max_cache_length,
                config.num_key_value_heads * config.head_dim, runtime_->thinker->stream());
            std::vector<VoiceChatThinkerMambaState::TensorSpec> specs;
            specs.push_back({"conv_state", {config.conv_dim, config.mamba_d_conv}, "present_conv"});
            specs.push_back({"ssm_state",
                             {config.mamba_nheads, config.mamba_head_dim, config.mamba_d_state},
                             "present_ssm"});
            auto mamba = std::make_unique<VoiceChatThinkerMambaState>(
                config.num_mamba_layers, std::move(specs), runtime_->thinker->stream());
            thinker_state_ =
                std::make_unique<VoiceChatThinkerHybridState>(std::move(kv), std::move(mamba));
        } else {
            thinker_state_->reset();
        }
        if (!thinker_state_->ok())
            throw std::runtime_error("VoiceChat failed to allocate hybrid thinker state");
        thinker_state_->bind_to(*runtime_->thinker);

        std::fill(rnnt_h_.begin(), rnnt_h_.end(), 0.0F);
        std::fill(rnnt_c_.begin(), rnnt_c_.end(), 0.0F);
        rnnt_predictor_output_ = run_rnnt_predictor(config.rnnt_blank_id);
        tts_state_.reset_and_warmup();
        prefill_system_prompt();
        record_replay_state_ = true;
    }

    ModelStateMarker capture_model_marker() const {
        ModelStateMarker marker;
        marker.thinker_steps = thinker_replay_.size();
        marker.tts_steps = tts_replay_.size();
        marker.codec_steps = codec_replay_.size();
        marker.timeline_steps = timeline_replay_.size();
        marker.agent_text_tokens = agent_text_tokens_;
        marker.previous_text_token = previous_text_token_;
        marker.previous_function_token = previous_function_token_;
        marker.agent_turn_frames = agent_turn_frames_;
        marker.agent_turn_text_tokens = agent_turn_text_tokens_;
        marker.frame_index = frame_index_;
        marker.agent_idle = agent_idle_;
        return marker;
    }

    void validate_model_marker(const ModelStateMarker& marker) const {
        if (marker.thinker_steps > thinker_replay_.size() ||
            marker.tts_steps > tts_replay_.size() || marker.codec_steps > codec_replay_.size() ||
            marker.timeline_steps > timeline_replay_.size()) {
            throw std::logic_error("VoiceChat response checkpoint is inconsistent");
        }
    }

    void replay_model_prefix(const ModelStateMarker& marker) {
        validate_model_marker(marker);
        record_replay_state_ = false;
        try {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            thinker_state_->reset();
            thinker_state_->bind_to(*runtime_->thinker);
            tts_state_.reset_and_warmup();
            codec_cache_.reset();
            codec_reconstruction_.reset();
            prefill_system_prompt();
            for (std::size_t index = 0; index < marker.thinker_steps; ++index) {
                const auto& step = thinker_replay_[index];
                (void)run_thinker(step.text_token, step.timeline_token, step.function_token,
                                  step.audio_embedding, step.use_audio);
            }
            for (std::size_t index = 0; index < marker.tts_steps; ++index) {
                const auto& step = tts_replay_[index];
                (void)tts_state_.step(step.text_token, step.agent_idle);
            }
            for (std::size_t index = 0; index < marker.codec_steps; ++index)
                (void)decode_codec(codec_replay_[index]);
        } catch (...) {
            record_replay_state_ = true;
            throw;
        }
        record_replay_state_ = true;
    }

    void restore_model_marker(const ModelStateMarker& marker) {
        replay_model_prefix(marker);
        thinker_replay_.resize(marker.thinker_steps);
        tts_replay_.resize(marker.tts_steps);
        codec_replay_.resize(marker.codec_steps);
        timeline_replay_.resize(marker.timeline_steps);
        agent_text_tokens_ = marker.agent_text_tokens;
        previous_text_token_ = marker.previous_text_token;
        previous_function_token_ = marker.previous_function_token;
        agent_turn_frames_ = marker.agent_turn_frames;
        agent_turn_text_tokens_ = marker.agent_turn_text_tokens;
        frame_index_ = marker.frame_index;
        agent_idle_ = marker.agent_idle;
    }

    void begin_response_tracking(std::uint64_t epoch, const ModelStateMarker& start) {
        response_playback_.begin(epoch);
        response_checkpoints_.clear();
        response_checkpoints_.push_back({start, 0});
        response_start_output_sample_ = output_sample_cursor_;
    }

    void reset_response_tracking() {
        response_playback_.reset();
        response_checkpoints_.clear();
        current_frame_start_marker_.reset();
        pending_response_audio_end_.reset();
    }

    void checkpoint_published_audio() {
        if (!pending_response_audio_end_.has_value() || !response_playback_.active())
            return;
        const auto relative_end = *pending_response_audio_end_ - response_start_output_sample_;
        response_playback_.append_boundary(relative_end);
        response_checkpoints_.push_back({capture_model_marker(), relative_end});
        pending_response_audio_end_.reset();
    }

    std::string prompt_text() const {
        const std::string base =
            session_config_.system_prompt.empty()
                ? (tool_config_ ? std::string(voicechat::default_function_system_message())
                                : runtime_->config.default_system_prompt)
                : session_config_.system_prompt;
        return tool_config_ ? voicechat::render_function_system_prompt(base, function_tools_)
                            : base;
    }

    void prefill_system_prompt() {
        std::vector<int32_t> prompt_ids;
        prompt_ids.push_back(runtime_->config.bos_token_id);
        auto body = runtime_->tokenizer->encode(prompt_text());
        prompt_ids.insert(prompt_ids.end(), body.begin(), body.end());
        prompt_ids.push_back(runtime_->config.eos_token_id);
        for (const int32_t prompt_id : prompt_ids) {
            (void)run_thinker(runtime_->config.pad_token_id, prompt_id,
                              runtime_->config.pad_token_id, zero_audio_embedding_, false);
        }
        previous_text_token_ = runtime_->config.pad_token_id;
        previous_function_token_ = runtime_->config.pad_token_id;
    }

    std::vector<float> run_rnnt_predictor(int32_t token_id) {
        const auto& config = runtime_->config;
        TensorMap inputs;
        inputs["token_id"] = tensor(&token_id, {1}, DType::kInt32);
        const std::size_t stride = static_cast<std::size_t>(config.rnnt_pred_hidden_size);
        for (int32_t layer = 0; layer < config.rnnt_pred_num_layers; ++layer) {
            const auto suffix = "_" + std::to_string(layer);
            inputs["state_h" + suffix] =
                tensor(rnnt_h_.data() + static_cast<std::size_t>(layer) * stride,
                       {1, config.rnnt_pred_hidden_size}, DType::kFloat32);
            inputs["state_c" + suffix] =
                tensor(rnnt_c_.data() + static_cast<std::size_t>(layer) * stride,
                       {1, config.rnnt_pred_hidden_size}, DType::kFloat32);
        }
        auto outputs = runtime_->rnnt_predictor->forward(inputs);
        const auto prediction = outputs.find("pred_output");
        if (prediction == outputs.end())
            throw std::runtime_error("VoiceChat RNNT predictor missing pred_output");
        for (int32_t layer = 0; layer < config.rnnt_pred_num_layers; ++layer) {
            const auto suffix = "_" + std::to_string(layer);
            const auto h = outputs.find("next_h" + suffix);
            const auto c = outputs.find("next_c" + suffix);
            if (h == outputs.end() || c == outputs.end())
                throw std::runtime_error("VoiceChat RNNT predictor missing recurrent output");
            std::memcpy(rnnt_h_.data() + static_cast<std::size_t>(layer) * stride, h->second.data,
                        stride * sizeof(float));
            std::memcpy(rnnt_c_.data() + static_cast<std::size_t>(layer) * stride, c->second.data,
                        stride * sizeof(float));
        }
        const auto* first = static_cast<const float*>(prediction->second.data);
        return std::vector<float>(first, first + config.rnnt_pred_hidden_size);
    }

    TensorMap run_rnnt_joint(const float* encoder_frame) {
        const auto& config = runtime_->config;
        TensorMap inputs;
        inputs["encoder_frame"] = tensor(const_cast<float*>(encoder_frame),
                                         {1, config.perception_hidden_size}, DType::kFloat32);
        inputs["pred_output"] = tensor(rnnt_predictor_output_.data(),
                                       {1, config.rnnt_pred_hidden_size}, DType::kFloat32);
        return runtime_->rnnt_joint->forward(inputs);
    }

    RnntFrameActivity decode_rnnt_frame(const float* encoder_frame, std::uint64_t work_epoch) {
        const auto& config = runtime_->config;
        RnntFrameActivity activity;
        for (int32_t symbols = 0; symbols < config.rnnt_max_symbols_per_step; ++symbols) {
            if (!work_is_current(work_epoch))
                return activity;
            int32_t token_id = config.rnnt_blank_id;
            {
                std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
                auto outputs = run_rnnt_joint(encoder_frame);
                const auto logits = outputs.find("logits");
                if (logits == outputs.end())
                    throw std::runtime_error("VoiceChat RNNT joint missing logits");
                token_id = argmax(logits->second);
            }
            if (token_id == config.rnnt_blank_id)
                break;
            if (token_id < 0 || token_id >= config.rnnt_vocab_size)
                throw std::runtime_error("VoiceChat RNNT emitted an invalid token");
            const bool speech_token = token_id != rnnt_unk_token_id_;
            activity.emitted_speech_token = activity.emitted_speech_token || speech_token;
            rnnt_tokens_.push_back(token_id);
            {
                std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
                rnnt_predictor_output_ = run_rnnt_predictor(token_id);
            }
        }
        emit_transcript(false, work_epoch);
        return activity;
    }

    void emit_transcript(bool is_final, std::uint64_t work_epoch) {
        if (!session_config_.emit_user_transcript)
            return;
        const std::string decoded =
            normalize_rnnt_text(rnnt_tokens_, runtime_->assets.rnnt_vocabulary);
        if (!is_final && decoded == rnnt_text_)
            return;
        rnnt_text_ = decoded;
        SpeechSessionEvent event;
        event.kind = SpeechSessionEventKind::kUserTranscript;
        event.text = rnnt_text_;
        event.is_final = is_final;
        event.frame_index = frame_index_;
        (void)publish_current_event(std::move(event), work_epoch);
    }

    bool conversation_agent_speaking(std::uint64_t work_epoch) const {
        if (!work_is_current(work_epoch))
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        return work_is_current(work_epoch) &&
               conversation_.phase() == voicechat::ConversationPhase::kAgentSpeaking;
    }

    void publish_user_speech_event(SpeechSessionEventKind kind, std::int64_t frame, bool is_final,
                                   std::uint64_t work_epoch) {
        SpeechSessionEvent event;
        event.kind = kind;
        event.sample_rate = runtime_->config.input_sample_rate;
        event.frame_index = frame;
        event.media_start_sample =
            frame < 0 ? -1 : frame * runtime_->config.input_samples_per_frame;
        event.media_end_sample =
            frame < 0 ? -1 : event.media_start_sample + runtime_->config.input_samples_per_frame;
        event.text = rnnt_text_;
        event.is_final = is_final;
        (void)publish_current_event(std::move(event), work_epoch);
    }

    void reset_rnnt_utterance_decoder() {
        std::fill(rnnt_h_.begin(), rnnt_h_.end(), 0.0F);
        std::fill(rnnt_c_.begin(), rnnt_c_.end(), 0.0F);
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            rnnt_predictor_output_ = run_rnnt_predictor(runtime_->config.rnnt_blank_id);
        }
        rnnt_tokens_.clear();
        rnnt_text_.clear();
    }

    bool interrupt_agent_from_worker(std::uint64_t work_epoch) {
        std::uint64_t interrupted_epoch = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch) ||
                conversation_.phase() != voicechat::ConversationPhase::kAgentSpeaking ||
                !session_config_.enable_barge_in)
                return false;
            interrupted_epoch = conversation_.epoch();
            if (!conversation_.barge_in())
                return false;
            erase_interrupted_agent_output_locked(interrupted_epoch);
            clear_pending_tools_locked();
            deferred_audio_embeddings_.clear();
            SpeechSessionEvent yielded;
            yielded.kind = SpeechSessionEventKind::kYielded;
            yielded.epoch = conversation_.epoch();
            yielded.sequence = conversation_.next_sequence();
            yielded.frame_index = frame_index_;
            yielded.text = "barge-in";
            enqueue_event_locked(std::move(yielded));
        }
        function_channel_.reset();
        function_cycle_active_ = false;
        function_generation_active_ = false;
        forced_function_tokens_.clear();
        on_hold_token_queue_.clear();
        agent_idle_ = true;
        agent_text_tokens_.clear();
        suppress_synthesis_until_turn_started_ = true;
        suppress_native_agent_start_ = true;
        reset_response_tracking();
        event_cv_.notify_all();
        return true;
    }

    std::optional<int32_t> apply_turn_decision(const voicechat::RnntTurnDecision& decision,
                                               std::uint64_t work_epoch) {
        if (decision.speech_started) {
            publish_user_speech_event(SpeechSessionEventKind::kUserSpeechStarted,
                                      decision.speech_start_frame, false, work_epoch);
        }
        if (decision.interrupt_agent && interrupt_agent_from_worker(work_epoch))
            return runtime_->config.eos_token_id;
        if (!decision.speech_stopped)
            return std::nullopt;

        input_buffer_start_marker_.reset();
        emit_transcript(true, work_epoch);
        publish_user_speech_event(SpeechSessionEventKind::kUserSpeechStopped,
                                  decision.speech_end_frame, true, work_epoch);
        reset_rnnt_utterance_decoder();
        if (decision.start_agent) {
            suppress_native_agent_start_ = false;
            return runtime_->config.bos_token_id;
        }
        return std::nullopt;
    }

    std::pair<int32_t, int32_t> run_thinker(int32_t text_token, int32_t timeline_token,
                                            int32_t function_token,
                                            const std::vector<float>& audio_embedding,
                                            bool use_audio) {
        const auto& config = runtime_->config;
        if (audio_embedding.size() != static_cast<std::size_t>(config.hidden_size))
            throw std::runtime_error("VoiceChat thinker audio embedding width mismatch");
        float use_audio_value = use_audio ? 1.0F : 0.0F;
        TensorMap inputs;
        inputs["text_token_id"] = tensor(&text_token, {1}, DType::kInt32);
        inputs["timeline_token_id"] = tensor(&timeline_token, {1}, DType::kInt32);
        inputs["function_token_id"] = tensor(&function_token, {1}, DType::kInt32);
        inputs["audio_embed"] = tensor(const_cast<float*>(audio_embedding.data()),
                                       {1, config.hidden_size}, DType::kFloat32);
        inputs["use_audio_embed"] = tensor(&use_audio_value, {1, 1}, DType::kFloat32);
        thinker_state_->bind_to(*runtime_->thinker);
        thinker_state_->prepare_step(inputs);
        auto outputs = runtime_->thinker->forward(inputs);
        const auto text = outputs.find("logits");
        const auto function = outputs.find("function_logits");
        if (text == outputs.end() || function == outputs.end())
            throw std::runtime_error("VoiceChat thinker missing text/function logits");
        const auto result = std::make_pair(argmax(text->second), argmax(function->second));
        thinker_state_->advance();
        if (record_replay_state_) {
            thinker_replay_.push_back(
                {text_token, timeline_token, function_token, audio_embedding, use_audio});
        }
        return result;
    }

    std::string function_call_event_json(const voicechat::FunctionCall& call) const {
        nlohmann::json value = {
            {"type", "function_call"},
            {"call_id", call.call_id},
            {"name", call.name},
            {"arguments", nlohmann::json::parse(call.arguments_json)},
        };
        return value.dump();
    }

    void publish_function_calls(std::vector<voicechat::FunctionCall> calls,
                                std::uint64_t output_epoch, std::uint64_t work_epoch) {
        std::string on_hold_phrase;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch) || !conversation_.accepts_output(output_epoch))
                return;
            clear_pending_tools_locked();
            pending_tool_epoch_ = output_epoch;
            for (auto& call : calls) {
                SpeechSessionEvent event;
                event.kind = SpeechSessionEventKind::kFunctionCall;
                event.epoch = output_epoch;
                event.sequence = conversation_.next_sequence();
                event.frame_index = frame_index_;
                event.text = function_call_event_json(call);
                event.is_final = true;
                enqueue_event_locked(std::move(event));
                pending_tool_calls_.push_back({std::move(call), std::nullopt});
            }
            tool_response_deadline_ =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(runtime_->config.function_tool_timeout_ms);
            on_hold_phrase = select_on_hold_message_locked();
        }
        event_cv_.notify_all();
        work_cv_.notify_all();
        start_tool_on_hold(on_hold_phrase, output_epoch, work_epoch);
    }

    voicechat::FunctionChannelObservation
    observe_function_token(int32_t function_token, std::uint64_t work_epoch,
                           const std::optional<std::uint64_t>& output_epoch) {
        if (!tool_config_)
            return {};
        if (function_token == voicechat::kFunctionSotcTokenId) {
            if (!output_epoch.has_value())
                throw std::runtime_error("VoiceChat function call has no active agent epoch");
            active_function_epoch_ = *output_epoch;
            ++function_call_serial_;
        }
        const auto observation =
            function_channel_.observe(function_token, active_function_epoch_, function_call_serial_,
                                      function_tools_, [this](const std::vector<int32_t>& tokens) {
                                          return runtime_->tokenizer->decode(tokens);
                                      });
        if (observation.kind == voicechat::FunctionChannelObservationKind::kCallStarted) {
            if (!output_epoch.has_value())
                throw std::runtime_error("VoiceChat started a function call outside an agent turn");
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kFunctionCallStarted;
            event.is_final = false;
            event.frame_index = frame_index_;
            (void)publish_agent_event(std::move(event), work_epoch, *output_epoch);
        } else if (observation.kind == voicechat::FunctionChannelObservationKind::kCallsReady) {
            if (!output_epoch.has_value())
                throw std::runtime_error(
                    "VoiceChat completed a function call outside an agent turn");
            publish_function_calls(observation.calls, *output_epoch, work_epoch);
        } else if (observation.kind == voicechat::FunctionChannelObservationKind::kError) {
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kError;
            event.text = "VoiceChat function channel rejected output: " + observation.error;
            event.is_final = true;
            (void)publish_current_event(std::move(event), work_epoch);
        }
        return observation;
    }

    std::vector<int32_t> step_tts(int32_t text_token, bool agent_idle) {
        auto codes = tts_state_.step(text_token, agent_idle);
        if (record_replay_state_)
            tts_replay_.push_back({text_token, agent_idle});
        return codes;
    }

    void advance_tts_silently() {
        std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
        (void)step_tts(runtime_->config.pad_token_id, true);
        (void)decode_codec(runtime_->assets.tts_prompt.silence_codes);
    }

    void start_function_call_generation(std::uint64_t work_epoch, std::uint64_t output_epoch) {
        function_cycle_active_ = true;
        function_generation_active_ = true;
        function_async_steps_ = 0;
        active_function_epoch_ = output_epoch;
        enqueue_internal_work(WorkKind::kFunctionStep, work_epoch);
    }

    void process_function_step(std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch) || !function_generation_active_)
            return;
        const auto& config = runtime_->config;
        if (++function_async_steps_ > config.function_max_async_steps)
            throw std::runtime_error("VoiceChat function call exceeded its async step bound");
        std::pair<int32_t, int32_t> tokens;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            tokens = run_thinker(config.pad_token_id, config.pad_token_id, previous_function_token_,
                                 zero_audio_embedding_, false);
        }
        const int32_t function_token = tokens.second;
        const auto observation = observe_function_token(
            function_token, work_epoch, std::optional<std::uint64_t>{active_function_epoch_});
        advance_tts_silently();
        previous_text_token_ = config.pad_token_id;
        previous_function_token_ = function_token;
        ++frame_index_;
        if (observation.kind == voicechat::FunctionChannelObservationKind::kCallsReady) {
            function_generation_active_ = false;
            return;
        }
        if (observation.kind == voicechat::FunctionChannelObservationKind::kError)
            throw std::runtime_error(observation.error);
        enqueue_internal_work(WorkKind::kFunctionStep, work_epoch);
    }

    std::string pending_tool_results_json_locked() const {
        nlohmann::json results = nlohmann::json::array();
        for (const auto& pending : pending_tool_calls_) {
            if (!pending.output.has_value())
                throw std::logic_error("VoiceChat tool response set is incomplete");
            try {
                results.push_back(nlohmann::json::parse(*pending.output));
            } catch (const nlohmann::json::exception&) {
                results.push_back(*pending.output);
            }
        }
        return results.dump();
    }

    std::string pending_tool_results_json(std::uint64_t tool_epoch) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tool_epoch != pending_tool_epoch_ || pending_tool_calls_.empty())
            throw std::invalid_argument("VoiceChat tool response epoch is stale");
        return pending_tool_results_json_locked();
    }

    void process_tool_response(const WorkItem& work) {
        if (!work_is_current(work.work_epoch))
            return;
        std::vector<int32_t> forced_tokens = work.forced_function_tokens;
        if (forced_tokens.empty()) {
            const std::string results = pending_tool_results_json(work.tool_epoch);
            forced_tokens =
                voicechat::build_tool_response_tokens(results, [this](std::string_view text) {
                    return runtime_->tokenizer->encode(std::string(text));
                }).token_ids;
        }
        if (forced_tokens.size() >
            static_cast<std::size_t>(runtime_->config.function_max_response_tokens))
            throw std::invalid_argument("VoiceChat tool response exceeds its token bound");
        forced_function_tokens_.assign(forced_tokens.begin(), forced_tokens.end());
        on_hold_token_queue_.clear();
        agent_idle_ = true;
        function_response_steps_ = 0;
        enqueue_internal_work(WorkKind::kFunctionResponseStep, work.work_epoch, true);
    }

    void finish_function_response(std::uint64_t work_epoch) {
        SpeechSessionEvent completed;
        completed.kind = SpeechSessionEventKind::kFunctionResponseFinished;
        completed.is_final = true;
        completed.frame_index = frame_index_;
        (void)publish_agent_event(std::move(completed), work_epoch, active_function_epoch_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch))
                return;
            clear_pending_tools_locked();
        }
        active_function_epoch_ = 0;
        function_cycle_active_ = false;
        function_generation_active_ = false;
        forced_function_tokens_.clear();
        on_hold_token_queue_.clear();
        suppress_native_agent_start_ = false;
        auto deferred = std::move(deferred_audio_embeddings_);
        deferred_audio_embeddings_.clear();
        for (const auto& audio_embedding : deferred) {
            if (!work_is_current(work_epoch))
                return;
            process_model_frame(audio_embedding, work_epoch);
        }
        work_cv_.notify_all();
    }

    void process_function_response_step(std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch) || !function_cycle_active_)
            return;
        const auto& config = runtime_->config;
        if (++function_response_steps_ > config.function_max_async_steps)
            throw std::runtime_error("VoiceChat function response did not reach EOTR");

        if (!forced_function_tokens_.empty()) {
            const int32_t forced_token = forced_function_tokens_.front();
            forced_function_tokens_.pop_front();
            {
                std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
                (void)run_thinker(config.pad_token_id, config.pad_token_id,
                                  previous_function_token_, zero_audio_embedding_, false);
            }
            previous_text_token_ = config.pad_token_id;
            previous_function_token_ = forced_token;
            advance_tts_silently();
            ++frame_index_;
            enqueue_internal_work(WorkKind::kFunctionResponseStep, work_epoch);
            return;
        }

        std::pair<int32_t, int32_t> tokens;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            tokens = run_thinker(config.pad_token_id, config.pad_token_id, previous_function_token_,
                                 zero_audio_embedding_, false);
        }
        const auto observation =
            observe_function_token(tokens.second, work_epoch, accepted_output_epoch(work_epoch));
        previous_text_token_ = config.pad_token_id;
        previous_function_token_ = tokens.second;
        advance_tts_silently();
        ++frame_index_;
        if (observation.kind == voicechat::FunctionChannelObservationKind::kResponseFinished) {
            finish_function_response(work_epoch);
            return;
        }
        if (observation.kind == voicechat::FunctionChannelObservationKind::kError)
            throw std::runtime_error(observation.error);
        enqueue_internal_work(WorkKind::kFunctionResponseStep, work_epoch);
    }

    void process_tool_timeout(std::uint64_t work_epoch) {
        WorkItem timeout;
        timeout.kind = WorkKind::kToolResponse;
        timeout.work_epoch = work_epoch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch) || pending_tool_calls_.empty())
                return;
            timeout.tool_epoch = pending_tool_epoch_;
            for (auto& pending : pending_tool_calls_) {
                if (!pending.output.has_value()) {
                    pending.output =
                        R"({"error":"timeout","message":"The tool did not respond in time."})";
                }
            }
            const auto forced = voicechat::build_tool_response_tokens(
                pending_tool_results_json_locked(), [this](std::string_view text) {
                    return runtime_->tokenizer->encode(std::string(text));
                });
            if (forced.token_ids.size() >
                static_cast<std::size_t>(runtime_->config.function_max_response_tokens))
                throw std::runtime_error("VoiceChat timeout response exceeds its token bound");
            timeout.forced_function_tokens = forced.token_ids;
            tool_response_work_queued_ = true;
        }
        process_tool_response(timeout);
    }

    std::vector<int32_t> on_hold_tokens(std::string_view phrase) const {
        auto tokens = runtime_->tokenizer->encode(std::string(phrase));
        const auto padding = std::max<int32_t>(runtime_->config.function_on_hold_min_pad_frames,
                                               static_cast<int32_t>((phrase.size() + 1U) / 2U));
        tokens.insert(tokens.begin(), runtime_->config.bos_token_id);
        tokens.insert(tokens.end(), static_cast<std::size_t>(padding),
                      runtime_->config.pad_token_id);
        tokens.push_back(runtime_->config.eos_token_id);
        return tokens;
    }

    std::string select_on_hold_message_locked() const {
        if (pending_tool_calls_.empty())
            return {};
        const auto& call = pending_tool_calls_.front().call;
        const auto& configured = on_hold_messages_.lookup(call.name);
        if (!configured.empty())
            return configured.front();
        const auto* tool = function_tools_.find(call.name);
        if (tool != nullptr && !tool->ack_messages.empty())
            return tool->ack_messages.front();
        return {};
    }

    void start_tool_on_hold(const std::string& phrase, std::uint64_t output_epoch,
                            std::uint64_t work_epoch) {
        if (phrase.empty())
            return;
        if (session_config_.emit_agent_text) {
            SpeechSessionEvent text;
            text.kind = SpeechSessionEventKind::kAgentText;
            text.text = phrase;
            text.is_final = true;
            text.frame_index = frame_index_;
            (void)publish_agent_event(std::move(text), work_epoch, output_epoch);
        }
        agent_idle_ = false;
        const auto tokens = on_hold_tokens(phrase);
        on_hold_token_queue_.assign(tokens.begin(), tokens.end());
        on_hold_output_epoch_ = output_epoch;
        enqueue_internal_work(WorkKind::kOnHoldStep, work_epoch);
    }

    void process_on_hold_step(std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch) || !function_cycle_active_ || on_hold_token_queue_.empty())
            return;
        const int32_t token = on_hold_token_queue_.front();
        on_hold_token_queue_.pop_front();
        std::vector<int32_t> codes;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            codes = step_tts(token, false);
        }
        std::vector<float> waveform;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            waveform = decode_codec(codes);
        }
        emit_audio(std::move(waveform), work_epoch, on_hold_output_epoch_);
        if (!on_hold_token_queue_.empty())
            enqueue_internal_work(WorkKind::kOnHoldStep, work_epoch);
        else
            agent_idle_ = true;
    }

    void accept_native_samples(const std::vector<float>& samples, bool final,
                               std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch))
            return;
        if (!samples.empty()) {
            scheduler_.append(samples.data(), static_cast<int32_t>(samples.size()));
        }
        process_scheduled_frames(final, work_epoch);
    }

    void process_scheduled_frames(bool final, std::uint64_t work_epoch,
                                  bool committed_partial = false) {
        while (work_is_current(work_epoch)) {
            auto frame = scheduler_.pop();
            if (!frame)
                break;
            const int32_t mel_samples =
                committed_partial ? voicechat::kInputFrameSamples : frame->valid_input_samples;
            mel_.accept_audio(frame->samples.data(), mel_samples);
            process_audio_frame(*frame, final && frame->is_final, work_epoch);
        }
    }

    std::vector<float> make_streaming_mel(bool final) {
        const auto& config = runtime_->config;
        const auto step = voicechat::make_streaming_mel_step(
            first_perception_step_, next_mel_frame_, mel_.available_frames(), final);
        mel_.ensure_frames(next_mel_frame_ + step.valid_new_frames, final);
        std::vector<float> chunk(static_cast<std::size_t>(config.mel_num_bins) * step.engine_frames,
                                 0.0F);
        for (int32_t bin = 0; bin < config.mel_num_bins; ++bin) {
            for (int32_t column = 0; column < step.engine_frames; ++column) {
                const int32_t source = next_mel_frame_ - step.history_frames + column;
                if (source >= 0 && source < mel_.frame_count())
                    chunk[static_cast<std::size_t>(bin) * step.engine_frames + column] =
                        mel_.value(bin, source);
            }
        }
        next_mel_frame_ += step.valid_new_frames;
        return chunk;
    }

    std::vector<float> make_encoder_mask(int32_t cache_frames, int32_t key_frames) const {
        std::vector<float> encoder_mask(static_cast<std::size_t>(key_frames), -10000.0F);
        const int32_t cache_begin = cache_frames - perception_cache_length_;
        for (int32_t key = cache_begin; key < key_frames; ++key) {
            if (key >= 0)
                encoder_mask[static_cast<std::size_t>(key)] = 0.0F;
        }
        return encoder_mask;
    }

    static void require_perception_outputs(const TensorMap& outputs) {
        if (outputs.find("rnnt_encoder_output") == outputs.end() ||
            outputs.find("audio_embeddings") == outputs.end() ||
            outputs.find("cache_last_channel_next") == outputs.end() ||
            outputs.find("cache_last_time_next") == outputs.end())
            throw std::runtime_error("VoiceChat streaming perception missing required outputs");
    }

    PerceptionFrameOutputs run_perception(TensorMap& inputs) {
        const auto& config = runtime_->config;
        PerceptionFrameOutputs result;
        result.rnnt_frame.resize(static_cast<std::size_t>(config.perception_hidden_size));
        result.projected_audio.resize(static_cast<std::size_t>(config.hidden_size));

        std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
        TrtModule& perception = first_perception_step_ ? *runtime_->perception_stream_first
                                                       : *runtime_->perception_stream;
        const auto outputs = perception.forward(inputs);
        require_perception_outputs(outputs);
        const auto& rnnt = outputs.at("rnnt_encoder_output");
        const auto& audio = outputs.at("audio_embeddings");
        const auto& channel = outputs.at("cache_last_channel_next");
        const auto& time = outputs.at("cache_last_time_next");
        std::memcpy(perception_channel_cache_.data(), channel.data,
                    perception_channel_cache_.size() * sizeof(float));
        std::memcpy(perception_time_cache_.data(), time.data,
                    perception_time_cache_.size() * sizeof(float));
        std::memcpy(result.rnnt_frame.data(), rnnt.data, result.rnnt_frame.size() * sizeof(float));
        std::memcpy(result.projected_audio.data(), audio.data,
                    result.projected_audio.size() * sizeof(float));
        return result;
    }

    void process_tool_wait_audio_frame(PerceptionFrameOutputs& outputs,
                                       const voicechat::RnntTurnDecision& decision,
                                       std::uint64_t work_epoch) {
        if (decision.speech_started) {
            publish_user_speech_event(SpeechSessionEventKind::kUserSpeechStarted,
                                      decision.speech_start_frame, false, work_epoch);
        }
        if (decision.speech_stopped) {
            emit_transcript(true, work_epoch);
            publish_user_speech_event(SpeechSessionEventKind::kUserSpeechStopped,
                                      decision.speech_end_frame, true, work_epoch);
            reset_rnnt_utterance_decoder();
        }
        if (decision.interrupt_agent && interrupt_agent_from_worker(work_epoch)) {
            process_model_frame(outputs.projected_audio, work_epoch, runtime_->config.eos_token_id,
                                true);
            return;
        }
        if (deferred_audio_embeddings_.size() >=
            static_cast<std::size_t>(runtime_->config.function_max_async_steps))
            throw std::overflow_error(
                "VoiceChat tool wait exceeded its deferred audio frame bound");
        deferred_audio_embeddings_.push_back(std::move(outputs.projected_audio));
    }

    void process_audio_frame(const voicechat::ScheduledInputFrame& frame, bool final,
                             std::uint64_t work_epoch) {
        (void)frame;
        if (!work_is_current(work_epoch))
            return;
        const auto& config = runtime_->config;
        auto mel_chunk = make_streaming_mel(final);
        const int32_t mel_frames = first_perception_step_ ? 1 : 17;
        const int32_t cache_frames = config.perception_att_context_left;
        const int32_t key_frames = cache_frames + 1;
        auto encoder_mask = make_encoder_mask(cache_frames, key_frames);

        TensorMap inputs;
        inputs["mel_features"] =
            tensor(mel_chunk.data(), {config.mel_num_bins, mel_frames}, DType::kFloat32);
        inputs["cache_last_channel"] =
            tensor(perception_channel_cache_.data(),
                   {config.perception_num_layers, cache_frames, config.perception_hidden_size},
                   DType::kFloat32);
        inputs["cache_last_time"] = tensor(
            perception_time_cache_.data(),
            {config.perception_num_layers, config.perception_hidden_size, 8}, DType::kFloat32);
        inputs["encoder_mask"] = tensor(encoder_mask.data(), {1, 1, key_frames}, DType::kFloat32);

        auto outputs = run_perception(inputs);
        if (!work_is_current(work_epoch))
            return;
        perception_cache_length_ = std::min(cache_frames, perception_cache_length_ + 1);
        first_perception_step_ = false;

        const auto rnnt_activity = decode_rnnt_frame(outputs.rnnt_frame.data(), work_epoch);
        if (!work_is_current(work_epoch))
            return;
        if (!is_live()) {
            process_model_frame(outputs.projected_audio, work_epoch);
            return;
        }
        const bool waiting_for_tool = function_cycle_active_;
        const auto decision = turn_detector_.observe(rnnt_activity.emitted_speech_token,
                                                     conversation_agent_speaking(work_epoch),
                                                     rnnt_observation_frame_index_++);
        if (waiting_for_tool) {
            process_tool_wait_audio_frame(outputs, decision, work_epoch);
            return;
        }
        auto forced_text_token = apply_turn_decision(decision, work_epoch);
        process_model_frame(outputs.projected_audio, work_epoch, forced_text_token);
    }

    bool begin_agent_turn_if_needed(int32_t text_token, std::uint64_t work_epoch) {
        if (text_token != runtime_->config.bos_token_id)
            return true;
        if (!ensure_agent_turn(work_epoch))
            return false;
        input_buffer_start_marker_.reset();
        agent_idle_ = false;
        agent_text_tokens_.clear();
        agent_turn_frames_ = 0;
        agent_turn_text_tokens_ = 0;
        suppress_synthesis_until_turn_started_ = false;
        suppress_native_agent_start_ = false;
        return true;
    }

    bool should_use_silence_codes(const std::vector<int32_t>& codes, int32_t text_token) const {
        const auto& control_codes = runtime_->assets.tts_prompt.control_codes;
        const bool contains_control = std::any_of(codes.begin(), codes.end(), [&](int32_t value) {
            return std::find(control_codes.begin(), control_codes.end(), value) !=
                   control_codes.end();
        });
        return contains_control || (text_token == runtime_->config.pad_token_id && agent_idle_);
    }

    bool synthesize_model_audio(int32_t text_token, std::uint64_t work_epoch,
                                const std::optional<std::uint64_t>& output_epoch) {
        if (suppress_synthesis_until_turn_started_)
            return true;
        std::vector<int32_t> codes;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            codes = step_tts(text_token, agent_idle_);
        }
        if (!work_is_current(work_epoch))
            return false;
        auto codec_codes = codes;
        if (should_use_silence_codes(codec_codes, text_token))
            codec_codes = runtime_->assets.tts_prompt.silence_codes;
        std::vector<float> waveform;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            waveform = decode_codec(codec_codes);
        }
        if (!work_is_current(work_epoch))
            return false;
        emit_audio(std::move(waveform), work_epoch, output_epoch);
        return true;
    }

    bool is_agent_text_token(int32_t text_token) const {
        const auto& config = runtime_->config;
        return text_token != config.pad_token_id && text_token != config.bos_token_id &&
               text_token != config.eos_token_id;
    }

    bool should_force_agent_eos(int32_t text_token) const {
        const auto& config = runtime_->config;
        const int32_t projected_frames = agent_turn_frames_ + 1;
        const int32_t projected_text_tokens =
            agent_turn_text_tokens_ + (is_agent_text_token(text_token) ? 1 : 0);
        if (config.max_response_frames > 0 && projected_frames >= config.max_response_frames)
            return true;
        return config.tts_text_token_ratio_cap > 0 &&
               projected_text_tokens >= config.tts_text_token_ratio_min_tokens &&
               projected_frames >= config.tts_text_token_ratio_cap * projected_text_tokens;
    }

    void publish_model_text_token(int32_t text_token, std::uint64_t work_epoch,
                                  const std::optional<std::uint64_t>& output_epoch) {
        if (!is_agent_text_token(text_token) || !output_epoch.has_value())
            return;
        agent_text_tokens_.push_back(text_token);
        if (!session_config_.emit_agent_text)
            return;
        SpeechSessionEvent event;
        event.kind = SpeechSessionEventKind::kAgentText;
        event.text = runtime_->tokenizer->decode({text_token});
        event.is_final = false;
        event.frame_index = frame_index_;
        (void)publish_agent_event(std::move(event), work_epoch, *output_epoch);
    }

    void complete_model_frame(int32_t text_token, int32_t function_token, std::uint64_t work_epoch,
                              const std::optional<std::uint64_t>& output_epoch) {
        if (output_epoch.has_value()) {
            ++agent_turn_frames_;
            if (is_agent_text_token(text_token))
                ++agent_turn_text_tokens_;
        }
        previous_text_token_ = text_token;
        // Keep the checkpoint-owned function channel recurrent after applying
        // any host-side marker parsing or forced response injection.
        previous_function_token_ = function_token;
        if (text_token == runtime_->config.eos_token_id && output_epoch.has_value())
            finish_agent_turn(work_epoch, *output_epoch);
        ++frame_index_;
    }

    std::optional<std::pair<int32_t, int32_t>>
    infer_model_frame_tokens(const std::vector<float>& audio_embedding, std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch))
            return std::nullopt;
        std::pair<int32_t, int32_t> tokens;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            tokens = run_thinker(previous_text_token_, runtime_->config.pad_token_id,
                                 previous_function_token_, audio_embedding, true);
        }
        if (!work_is_current(work_epoch))
            return std::nullopt;
        return tokens;
    }

    void apply_model_token_overrides(int32_t& text_token, int32_t& function_token,
                                     const std::optional<int32_t>& forced_text_token,
                                     bool ignore_function_channel) const {
        const auto& config = runtime_->config;
        if (ignore_function_channel)
            function_token = config.pad_token_id;
        if (forced_text_token.has_value())
            text_token = *forced_text_token;
        else if (is_live() && suppress_native_agent_start_ && text_token == config.bos_token_id)
            text_token = config.pad_token_id;
    }

    voicechat::FunctionChannelObservation
    observe_model_function_token(int32_t function_token, std::uint64_t work_epoch,
                                 bool ignore_function_channel,
                                 std::optional<std::uint64_t>& output_epoch) {
        if (ignore_function_channel)
            return {};
        if (tool_config_ && function_token == voicechat::kFunctionSotcTokenId)
            output_epoch = ensure_agent_turn(work_epoch);
        return observe_function_token(function_token, work_epoch, output_epoch);
    }

    bool function_step_is_silent(const voicechat::FunctionChannelObservation& observation) const {
        return observation.kind == voicechat::FunctionChannelObservationKind::kCallStarted ||
               function_channel_.capturing_call() || function_channel_.awaiting_response_end();
    }

    ModelFrameDecision make_model_frame_decision(std::pair<int32_t, int32_t> tokens,
                                                 std::uint64_t work_epoch,
                                                 const std::optional<int32_t>& forced_text_token,
                                                 bool ignore_function_channel) {
        ModelFrameDecision decision;
        decision.text_token = tokens.first;
        decision.function_token = tokens.second;
        apply_model_token_overrides(decision.text_token, decision.function_token, forced_text_token,
                                    ignore_function_channel);
        decision.function_observation = observe_model_function_token(
            decision.function_token, work_epoch, ignore_function_channel, decision.output_epoch);
        decision.function_silent_step = function_step_is_silent(decision.function_observation);
        if (decision.function_silent_step)
            decision.text_token = runtime_->config.pad_token_id;
        return decision;
    }

    bool model_frame_should_force_eos(const ModelFrameDecision& decision) const {
        return is_live() && !decision.function_silent_step && decision.output_epoch.has_value() &&
               decision.text_token != runtime_->config.eos_token_id &&
               should_force_agent_eos(decision.text_token);
    }

    bool prepare_model_frame_output(ModelFrameDecision& decision, std::uint64_t work_epoch) {
        if (!begin_agent_turn_if_needed(decision.text_token, work_epoch))
            return false;
        if (!decision.output_epoch.has_value())
            decision.output_epoch = accepted_output_epoch(work_epoch);
        if (model_frame_should_force_eos(decision))
            decision.text_token = runtime_->config.eos_token_id;
        return true;
    }

    bool render_model_frame(const ModelFrameDecision& decision, std::uint64_t work_epoch) {
        if (decision.function_silent_step) {
            advance_tts_silently();
            return true;
        }
        if (!synthesize_model_audio(decision.text_token, work_epoch, decision.output_epoch))
            return false;
        publish_model_text_token(decision.text_token, work_epoch, decision.output_epoch);
        return true;
    }

    void finish_model_frame(const ModelFrameDecision& decision, std::uint64_t work_epoch) {
        complete_model_frame(decision.text_token, decision.function_token, work_epoch,
                             decision.output_epoch);
        if (decision.function_observation.kind ==
                voicechat::FunctionChannelObservationKind::kCallStarted &&
            decision.output_epoch.has_value())
            start_function_call_generation(work_epoch, *decision.output_epoch);
    }

    void process_model_frame(const std::vector<float>& audio_embedding, std::uint64_t work_epoch,
                             std::optional<int32_t> forced_text_token = std::nullopt,
                             bool ignore_function_channel = false) {
        current_frame_start_marker_ = capture_model_marker();
        pending_response_audio_end_.reset();
        auto tokens = infer_model_frame_tokens(audio_embedding, work_epoch);
        if (!tokens) {
            current_frame_start_marker_.reset();
            return;
        }
        if (record_replay_state_)
            timeline_replay_.push_back(thinker_replay_.size() - 1);
        auto decision = make_model_frame_decision(*tokens, work_epoch, forced_text_token,
                                                  ignore_function_channel);
        if (!prepare_model_frame_output(decision, work_epoch)) {
            current_frame_start_marker_.reset();
            return;
        }
        if (!render_model_frame(decision, work_epoch)) {
            current_frame_start_marker_.reset();
            return;
        }
        finish_model_frame(decision, work_epoch);
        checkpoint_published_audio();
        current_frame_start_marker_.reset();
    }

    std::optional<std::uint64_t> ensure_agent_turn(std::uint64_t work_epoch) {
        std::optional<std::uint64_t> epoch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch))
                return std::nullopt;
            if (conversation_.phase() != voicechat::ConversationPhase::kAgentSpeaking) {
                epoch = conversation_.begin_agent_turn();
                SpeechSessionEvent started;
                started.kind = SpeechSessionEventKind::kTurnStarted;
                started.epoch = *epoch;
                started.sequence = conversation_.next_sequence();
                started.frame_index = frame_index_;
                enqueue_event_locked(std::move(started));
            } else {
                epoch = conversation_.epoch();
            }
        }
        if (!response_playback_.active()) {
            if (!current_frame_start_marker_.has_value())
                throw std::logic_error("VoiceChat agent turn has no model checkpoint");
            begin_response_tracking(*epoch, *current_frame_start_marker_);
        }
        event_cv_.notify_all();
        return epoch;
    }

    void finish_agent_turn(std::uint64_t work_epoch, std::uint64_t output_epoch) {
        const std::string final_text_value = runtime_->tokenizer->decode(agent_text_tokens_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!work_is_current(work_epoch) || !conversation_.accepts_output(output_epoch))
                return;
            if (session_config_.emit_agent_text) {
                SpeechSessionEvent final_text;
                final_text.kind = SpeechSessionEventKind::kAgentText;
                final_text.epoch = output_epoch;
                final_text.sequence = conversation_.next_sequence();
                final_text.text = final_text_value;
                final_text.is_final = true;
                final_text.frame_index = frame_index_;
                enqueue_event_locked(std::move(final_text));
            }
            SpeechSessionEvent finished;
            finished.kind = SpeechSessionEventKind::kTurnFinished;
            finished.epoch = output_epoch;
            finished.sequence = conversation_.next_sequence();
            finished.frame_index = frame_index_;
            enqueue_event_locked(std::move(finished));
            (void)conversation_.finish_agent_turn();
        }
        agent_idle_ = true;
        agent_text_tokens_.clear();
        agent_turn_frames_ = 0;
        agent_turn_text_tokens_ = 0;
        suppress_native_agent_start_ = is_live();
        reset_response_tracking();
        event_cv_.notify_all();
    }

    std::vector<float> decode_codec(const std::vector<int32_t>& codes) {
        const auto& config = runtime_->config;
        if (codes.size() != static_cast<std::size_t>(config.tts_num_quantizers))
            throw std::runtime_error("VoiceChat codec code width mismatch");
        TensorMap inputs;
        inputs["codec_codes"] = tensor(const_cast<int32_t*>(codes.data()),
                                       {1, config.tts_num_quantizers}, DType::kInt32);
        const auto& bindings = voicechat::codec_cache_bindings();
        for (int32_t block = 0; block < voicechat::kCodecConvBlocks; ++block) {
            const auto& binding = bindings[static_cast<std::size_t>(block)];
            inputs[binding.input_name] =
                tensor(const_cast<float*>(codec_cache_.current_data(block)),
                       {1, binding.channels, voicechat::kCodecConvCacheWidth}, DType::kFloat32);
        }
        auto outputs = runtime_->codec->forward(inputs);
        const auto spectral = outputs.find("spectral_params");
        if (spectral == outputs.end() || spectral->second.dtype != DType::kFloat32)
            throw std::runtime_error("VoiceChat codec missing spectral_params");
        for (int32_t block = 0; block < voicechat::kCodecConvBlocks; ++block) {
            const auto& binding = bindings[static_cast<std::size_t>(block)];
            const auto output = outputs.find(binding.output_name);
            if (output == outputs.end() || output->second.dtype != DType::kFloat32 ||
                output->second.numel() != codec_cache_.element_count(block))
                throw std::runtime_error("VoiceChat codec missing a causal cache output");
            std::memcpy(codec_cache_.next_data(block), output->second.data,
                        output->second.nbytes());
        }
        codec_cache_.commit();
        const auto* first = static_cast<const float*>(spectral->second.data);
        auto waveform = codec_reconstruction_.push(first, 1);
        if (record_replay_state_)
            codec_replay_.push_back(codes);
        return waveform;
    }

    void emit_audio(std::vector<float> waveform, std::uint64_t work_epoch,
                    const std::optional<std::uint64_t>& output_epoch) {
        if (!session_config_.emit_agent_audio)
            return;
        // Batch sessions retain the complete frame-locked waveform, including
        // model silence. Live sessions with barge-in expose only audio from an
        // accepted active agent epoch.
        if (is_live() && session_config_.enable_barge_in && !output_epoch.has_value())
            return;
        waveform = resample_frame(waveform, runtime_->config.output_sample_rate,
                                  session_config_.output_sample_rate);
        SpeechSessionEvent event;
        event.kind = SpeechSessionEventKind::kAgentAudio;
        event.audio_samples = std::move(waveform);
        event.sample_rate = session_config_.output_sample_rate;
        event.media_start_sample = output_sample_cursor_;
        event.media_end_sample =
            output_sample_cursor_ + static_cast<int64_t>(event.audio_samples.size());
        // Report the shared model timeline rather than the scheduler's raw
        // input-only index.
        event.frame_index = frame_index_;
        const auto end_sample = event.media_end_sample;
        const bool published =
            output_epoch.has_value()
                ? publish_agent_event(std::move(event), work_epoch, *output_epoch)
                : publish_current_event(std::move(event), work_epoch);
        if (published) {
            output_sample_cursor_ = end_sample;
            if (output_epoch.has_value() && response_playback_.active() &&
                response_playback_.epoch() == *output_epoch) {
                pending_response_audio_end_ = end_sample;
            }
        }
    }

    std::shared_ptr<NemotronVoiceChatRuntime> runtime_;
    SpeechSessionConfig session_config_;
    SpeechSessionMode mode_{SpeechSessionMode::kLive};
    std::optional<SpeechToolSessionConfig> tool_config_;
    mutable std::mutex mutex_;
    std::mutex reset_mutex_;
    std::condition_variable work_cv_;
    std::condition_variable event_cv_;
    std::condition_variable initialized_cv_;
    std::condition_variable reset_cv_;
    voicechat::AsyncEpochGate work_epochs_;
    voicechat::ConversationState conversation_;
    std::deque<WorkItem> work_queue_;
    std::thread worker_;
    std::exception_ptr worker_error_;
    voicechat::FrameScheduler scheduler_;
    StreamingLinearResampler resampler_;
    voicechat_audio::IncrementalMelSpectrogram mel_;
    std::unique_ptr<VoiceChatThinkerHybridState> thinker_state_;
    TtsCacheState tts_state_;
    voicechat::CodecCausalCache codec_cache_;
    voicechat::CodecReconstruction codec_reconstruction_;
    voicechat::FunctionToolCatalog function_tools_;
    voicechat::FunctionOnHoldMessages on_hold_messages_;
    voicechat::FunctionChannelState function_channel_;
    voicechat::RnntTurnDetector turn_detector_;
    voicechat::ResponsePlaybackTimeline response_playback_;
    voicechat::RealtimeTurnControlState turn_control_;
    std::optional<voicechat::BoundedInputQueueBudget> input_queue_budget_;
    std::optional<voicechat::OutputEventCapacityPolicy> output_capacity_;
    std::vector<PendingToolCall> pending_tool_calls_;
    std::vector<std::vector<float>> deferred_audio_embeddings_;
    std::vector<ThinkerReplayStep> thinker_replay_;
    std::vector<TtsReplayStep> tts_replay_;
    std::vector<std::vector<int32_t>> codec_replay_;
    std::vector<std::size_t> timeline_replay_;
    std::vector<ResponseCheckpoint> response_checkpoints_;
    std::optional<ModelStateMarker> input_buffer_start_marker_;
    std::optional<ModelStateMarker> current_frame_start_marker_;
    std::optional<std::int64_t> pending_response_audio_end_;
    std::uint64_t pending_tool_epoch_{0};
    std::uint64_t active_function_epoch_{0};
    std::uint64_t function_call_serial_{0};
    std::uint64_t on_hold_output_epoch_{0};
    int32_t function_async_steps_{0};
    int32_t function_response_steps_{0};
    bool tool_response_work_queued_{false};
    bool function_cycle_active_{false};
    bool function_generation_active_{false};
    std::chrono::steady_clock::time_point tool_response_deadline_{};
    std::deque<int32_t> forced_function_tokens_;
    std::deque<int32_t> on_hold_token_queue_;
    std::vector<float> perception_channel_cache_;
    std::vector<float> perception_time_cache_;
    std::vector<float> rnnt_h_;
    std::vector<float> rnnt_c_;
    std::vector<float> rnnt_predictor_output_;
    std::vector<int32_t> rnnt_tokens_;
    std::string rnnt_text_;
    std::vector<float> zero_audio_embedding_;
    std::vector<int32_t> agent_text_tokens_;
    int32_t previous_text_token_{0};
    int32_t previous_function_token_{0};
    int32_t next_mel_frame_{0};
    int32_t perception_cache_length_{0};
    int32_t rnnt_unk_token_id_{-1};
    int32_t agent_turn_frames_{0};
    int32_t agent_turn_text_tokens_{0};
    int64_t output_sample_cursor_{0};
    int64_t response_start_output_sample_{0};
    int64_t frame_index_{0};
    int64_t rnnt_observation_frame_index_{0};
    bool first_perception_step_{true};
    bool public_input_finished_{false};
    bool worker_input_finished_{false};
    bool agent_idle_{true};
    bool suppress_synthesis_until_turn_started_{false};
    bool suppress_native_agent_start_{false};
    bool clock_armed_{false};
    std::chrono::steady_clock::time_point next_tick_deadline_{};
    bool stop_requested_{false};
    bool worker_initialized_{false};
    bool worker_done_{false};
    bool reset_in_progress_{false};
    bool record_replay_state_{false};
    std::uint64_t requested_reset_serial_{0};
    std::uint64_t completed_reset_serial_{0};
    std::uint64_t requested_control_serial_{0};
    std::uint64_t completed_control_serial_{0};
    std::uint64_t control_error_serial_{0};
    std::exception_ptr control_error_;
    std::size_t queued_output_audio_samples_{0};
    std::vector<SpeechSessionEvent> events_;
};

} // namespace

NemotronVoiceChatPipeline::NemotronVoiceChatPipeline(
    std::unique_ptr<TrtModule> thinker, std::unique_ptr<TrtModule> perception_stream_first,
    std::unique_ptr<TrtModule> perception_stream, std::unique_ptr<TrtModule> rnnt_predictor,
    std::unique_ptr<TrtModule> rnnt_joint, std::unique_ptr<TrtModule> tts,
    std::unique_ptr<TrtModule> codec, voicechat::Config config, VoiceChatAssets assets,
    std::shared_ptr<ITokenizer> tokenizer, std::string model_id)
    : runtime_(std::make_shared<NemotronVoiceChatRuntime>(
          std::move(thinker), std::move(perception_stream_first), std::move(perception_stream),
          std::move(rnnt_predictor), std::move(rnnt_joint), std::move(tts), std::move(codec),
          std::move(config), std::move(assets), std::move(tokenizer))),
      model_id_(std::move(model_id)) {}

NemotronVoiceChatPipeline::~NemotronVoiceChatPipeline() = default;

std::unique_ptr<ISpeechSession>
NemotronVoiceChatPipeline::create_speech_session(const SpeechSessionConfig& config) {
    return std::make_unique<NemotronVoiceChatSession>(runtime_, config, SpeechSessionMode::kLive);
}

std::unique_ptr<ISpeechSession>
NemotronVoiceChatPipeline::create_batch_speech_session(const SpeechSessionConfig& config) {
    return std::make_unique<NemotronVoiceChatSession>(runtime_, config, SpeechSessionMode::kBatch);
}

std::unique_ptr<ISpeechSession>
NemotronVoiceChatPipeline::create_tool_speech_session(const SpeechSessionConfig& session_config,
                                                      const SpeechToolSessionConfig& tool_config) {
    return std::make_unique<NemotronVoiceChatSession>(runtime_, session_config,
                                                      SpeechSessionMode::kLive, tool_config);
}

AudioResult NemotronVoiceChatPipeline::speak(const float* audio_in, int32_t num_samples,
                                             const GenerateConfig& config,
                                             int32_t input_sample_rate) {
    SpeechSessionConfig session_config;
    session_config.input_sample_rate =
        input_sample_rate > 0 ? input_sample_rate : runtime_->config.input_sample_rate;
    session_config.output_sample_rate = runtime_->config.output_sample_rate;
    session_config.emit_agent_text = false;
    session_config.emit_user_transcript = false;
    session_config.enable_barge_in = false;
    session_config.seed = config.seed >= 0 ? config.seed : 0;
    session_config.finish_tail_frames = 0;
    auto session = create_batch_speech_session(session_config);
    session->append_audio(audio_in, num_samples);
    const int32_t tail_frames = std::max(config.tail_frames, 0);
    std::vector<float> silence(static_cast<std::size_t>(voicechat::kInputFrameSamples), 0.0F);
    for (int32_t frame = 0; frame < tail_frames; ++frame)
        session->append_audio(silence.data(), static_cast<int32_t>(silence.size()));
    session->finish_input();

    AudioResult result;
    result.sample_rate = runtime_->config.output_sample_rate;
    bool input_completed = false;
    while (!input_completed) {
        for (auto& event : session->wait_events(-1)) {
            if (event.kind == SpeechSessionEventKind::kInputFinished) {
                input_completed = true;
                continue;
            }
            if (event.kind == SpeechSessionEventKind::kError)
                throw std::runtime_error(event.text.empty() ? "VoiceChat session failed"
                                                            : event.text);
            if (event.kind != SpeechSessionEventKind::kAgentAudio)
                continue;
            result.samples.insert(result.samples.end(), event.audio_samples.begin(),
                                  event.audio_samples.end());
        }
    }
    result.num_samples = static_cast<int32_t>(result.samples.size());
    return result;
}

TextResult NemotronVoiceChatPipeline::transcribe(const float* audio_samples, int32_t num_samples,
                                                 int32_t max_tokens, int32_t input_sample_rate) {
    (void)max_tokens;
    SpeechSessionConfig config;
    config.input_sample_rate =
        input_sample_rate > 0 ? input_sample_rate : runtime_->config.input_sample_rate;
    config.emit_agent_audio = false;
    config.emit_agent_text = false;
    config.emit_user_transcript = true;
    config.finish_tail_frames = 0;
    auto session = create_batch_speech_session(config);
    session->append_audio(audio_samples, num_samples);
    session->finish_input();
    TextResult result;
    bool input_completed = false;
    while (!input_completed) {
        for (auto& event : session->wait_events(-1)) {
            if (event.kind == SpeechSessionEventKind::kInputFinished) {
                input_completed = true;
                continue;
            }
            if (event.kind == SpeechSessionEventKind::kError)
                throw std::runtime_error(event.text.empty() ? "VoiceChat session failed"
                                                            : event.text);
            if (event.kind == SpeechSessionEventKind::kUserTranscript)
                result.text = std::move(event.text);
        }
    }
    return result;
}

} // namespace trtmc
