/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/pipeline.h"

#include "runtime/models/nemotron_voicechat/audio_helpers.h"
#include "runtime/models/nemotron_voicechat/codec_reconstruction.h"
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
#include <limits>
#include <memory>
#include <mutex>
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

bool voicechat::should_barge_in(const SpeechSessionConfig& config, bool agent_turn_active,
                                double rms) {
    return config.enable_barge_in && agent_turn_active && rms >= 0.015;
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

class NemotronVoiceChatSession final : public ISpeechSession {
  private:
    enum class WorkKind { kAudio, kFinish, kBargeIn, kReset };

    struct WorkItem {
        WorkKind kind{WorkKind::kAudio};
        std::uint64_t work_epoch{0};
        std::uint64_t serial{0};
        std::vector<float> audio;
    };

    struct PerceptionFrameOutputs {
        std::vector<float> rnnt_frame;
        std::vector<float> projected_audio;
    };

  public:
    NemotronVoiceChatSession(std::shared_ptr<NemotronVoiceChatRuntime> runtime,
                             SpeechSessionConfig session_config)
        : runtime_(std::move(runtime)), session_config_(std::move(session_config)),
          scheduler_(conversation_.epoch()),
          resampler_(session_config_.input_sample_rate, runtime_->config.input_sample_rate),
          mel_(runtime_->assets.mel_filterbank.data(), runtime_->assets.mel_freq_bins,
               runtime_->assets.mel_bins, mel_options(runtime_->config),
               runtime_->config.input_sample_rate, runtime_->assets.mel_window.data(),
               static_cast<int32_t>(runtime_->assets.mel_window.size())),
          tts_state_(*runtime_->tts, runtime_->config, runtime_->assets.tts_prompt,
                     session_config_.seed) {
        if (session_config_.input_sample_rate <= 0)
            throw std::invalid_argument("VoiceChat input_sample_rate must be positive");
        if (session_config_.output_sample_rate == 0)
            session_config_.output_sample_rate = runtime_->config.output_sample_rate;
        if (session_config_.output_sample_rate <= 0)
            throw std::invalid_argument("VoiceChat output_sample_rate must be positive");
        if (session_config_.finish_tail_frames < -1)
            throw std::invalid_argument("VoiceChat finish_tail_frames must be -1 or non-negative");

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
        std::vector<float> owned;
        owned.reserve(static_cast<std::size_t>(count));
        double energy = 0.0;
        for (int32_t index = 0; index < count; ++index) {
            const float sample = samples[index];
            owned.push_back(sample);
            energy += static_cast<double>(sample) * sample;
        }
        const double rms = count > 0 ? std::sqrt(energy / count) : 0.0;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            rethrow_worker_error_locked();
            if (!conversation_.can_accept_audio())
                throw std::logic_error("VoiceChat session is not accepting audio; call reset()");
            maybe_barge_in_locked(rms);
            WorkItem work;
            work.kind = WorkKind::kAudio;
            work.work_epoch = work_epochs_.current();
            work.audio = std::move(owned);
            work_queue_.push_back(std::move(work));
        }
        work_cv_.notify_one();
    }

    void finish_input() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            rethrow_worker_error_locked();
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
        return events;
    }

    void cancel() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto interrupted_epoch = conversation_.epoch();
            conversation_.cancel();
            (void)work_epochs_.invalidate();
            public_input_finished_ = true;
            purge_inference_work_locked();
            erase_interrupted_agent_output_locked(interrupted_epoch);
            SpeechSessionEvent event;
            event.kind = SpeechSessionEventKind::kCancelled;
            event.epoch = conversation_.epoch();
            event.sequence = conversation_.next_sequence();
            events_.push_back(std::move(event));
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
            public_input_finished_ = false;
            worker_input_finished_ = false;
            purge_inference_work_locked();
            events_.clear();
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

  private:
    void rethrow_worker_error_locked() const {
        if (worker_error_)
            std::rethrow_exception(worker_error_);
    }

    void purge_inference_work_locked() {
        work_queue_.erase(
            std::remove_if(work_queue_.begin(), work_queue_.end(),
                           [](const WorkItem& item) { return item.kind != WorkKind::kReset; }),
            work_queue_.end());
    }

    void erase_interrupted_agent_output_locked(std::uint64_t interrupted_epoch) {
        events_.erase(std::remove_if(events_.begin(), events_.end(),
                                     [&](const SpeechSessionEvent& event) {
                                         return event.epoch == interrupted_epoch &&
                                                voicechat::is_agent_output_event(event.kind);
                                     }),
                      events_.end());
    }

    void maybe_barge_in_locked(double rms) {
        const bool active = conversation_.phase() == voicechat::ConversationPhase::kAgentSpeaking;
        if (!voicechat::should_barge_in(session_config_, active, rms))
            return;
        const auto interrupted_epoch = conversation_.epoch();
        if (!conversation_.barge_in())
            return;
        const auto work_epoch = work_epochs_.invalidate();
        purge_inference_work_locked();
        erase_interrupted_agent_output_locked(interrupted_epoch);
        WorkItem barrier;
        barrier.kind = WorkKind::kBargeIn;
        barrier.work_epoch = work_epoch;
        work_queue_.push_back(std::move(barrier));
        SpeechSessionEvent yielded;
        yielded.kind = SpeechSessionEventKind::kYielded;
        yielded.epoch = conversation_.epoch();
        yielded.sequence = conversation_.next_sequence();
        yielded.text = "barge-in";
        events_.push_back(std::move(yielded));
        event_cv_.notify_all();
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
            events_.push_back(std::move(event));
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
            events_.push_back(std::move(event));
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
            events_.push_back(std::move(event));
            worker_input_finished_ = true;
        }
        event_cv_.notify_all();
    }

    void acknowledge_reset(std::uint64_t serial) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_reset_serial_ = std::max(completed_reset_serial_, serial);
        }
        reset_cv_.notify_all();
    }

    void worker_loop() noexcept {
        try {
            initialize_host_state(work_epochs_.current());
            initialize_model_state();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                worker_initialized_ = true;
            }
            initialized_cv_.notify_all();

            while (true) {
                WorkItem work;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    work_cv_.wait(lock, [this] { return stop_requested_ || !work_queue_.empty(); });
                    if (stop_requested_)
                        break;
                    work = std::move(work_queue_.front());
                    work_queue_.pop_front();
                }

                process_work(work);
                event_cv_.notify_all();
            }
        } catch (...) {
            const auto error = std::current_exception();
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
                events_.push_back(std::move(event));
            }
            initialized_cv_.notify_all();
            reset_cv_.notify_all();
            event_cv_.notify_all();
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            worker_done_ = true;
        }
        reset_cv_.notify_all();
        event_cv_.notify_all();
    }

    void process_work(const WorkItem& work) {
        if (work.kind == WorkKind::kReset) {
            if (work_is_current(work.work_epoch)) {
                initialize_host_state(work.work_epoch);
                initialize_model_state();
                SpeechSessionEvent event;
                event.kind = SpeechSessionEventKind::kReset;
                (void)publish_current_event(std::move(event), work.work_epoch);
            }
            acknowledge_reset(work.serial);
            return;
        }
        if (!work_is_current(work.work_epoch))
            return;

        switch (work.kind) {
        case WorkKind::kAudio: {
            resampler_.append(work.audio.data(), static_cast<int32_t>(work.audio.size()));
            auto native = resampler_.drain(false);
            accept_native_samples(native, false, work.work_epoch);
            break;
        }
        case WorkKind::kFinish:
            process_finish(work);
            break;
        case WorkKind::kBargeIn:
            prepare_after_barge_in(work.work_epoch);
            break;
        case WorkKind::kReset:
            break;
        }
    }

    void process_finish(const WorkItem& work) {
        auto native = resampler_.drain(true);
        accept_native_samples(native, true, work.work_epoch);
        if (work_is_current(work.work_epoch)) {
            scheduler_.finish();
            process_scheduled_frames(true, work.work_epoch);
            emit_transcript(true, work.work_epoch);

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
                    events_.push_back(std::move(yielded));
                }
                event_cv_.notify_all();
            }

            publish_input_finished(work.work_epoch);
        }
    }

    void prepare_after_barge_in(std::uint64_t work_epoch) {
        // A large append can already have populated scheduler/mel buffers when
        // the control thread interrupts it. Drop that unprocessed frontier so
        // old microphone frames cannot be replayed under the replacement
        // epoch. The thinker cache is deliberately retained as conversation
        // history; only the streaming acoustic decoder and waveform frontier
        // start a clean user turn.
        scheduler_.reset(work_epoch);
        resampler_.reset();
        mel_.reset();
        first_perception_step_ = true;
        next_mel_frame_ = 0;
        perception_cache_length_ = 0;
        std::fill(perception_channel_cache_.begin(), perception_channel_cache_.end(), 0.0F);
        std::fill(perception_time_cache_.begin(), perception_time_cache_.end(), 0.0F);
        std::fill(rnnt_h_.begin(), rnnt_h_.end(), 0.0F);
        std::fill(rnnt_c_.begin(), rnnt_c_.end(), 0.0F);
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            rnnt_predictor_output_ = run_rnnt_predictor(runtime_->config.rnnt_blank_id);
        }
        rnnt_tokens_.clear();
        rnnt_text_.clear();
        codec_cache_.reset();
        codec_reconstruction_.reset();
        agent_idle_ = true;
        agent_text_tokens_.clear();
        previous_text_token_ = runtime_->config.pad_token_id;
        previous_function_token_ = runtime_->config.pad_token_id;
        suppress_synthesis_until_turn_started_ = true;
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
        next_mel_frame_ = 0;
        perception_cache_length_ = 0;
        output_sample_cursor_ = 0;
        frame_index_ = 0;
        previous_text_token_ = config.pad_token_id;
        previous_function_token_ = config.pad_token_id;
        agent_idle_ = true;
        suppress_synthesis_until_turn_started_ = false;
        agent_text_tokens_.clear();
        rnnt_tokens_.clear();
        rnnt_text_.clear();
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
    }

    std::string prompt_text() const {
        return session_config_.system_prompt.empty() ? runtime_->config.default_system_prompt
                                                     : session_config_.system_prompt;
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

    void decode_rnnt_frame(const float* encoder_frame, std::uint64_t work_epoch) {
        const auto& config = runtime_->config;
        for (int32_t symbols = 0; symbols < config.rnnt_max_symbols_per_step; ++symbols) {
            if (!work_is_current(work_epoch))
                return;
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
            rnnt_tokens_.push_back(token_id);
            {
                std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
                rnnt_predictor_output_ = run_rnnt_predictor(token_id);
            }
        }
        emit_transcript(false, work_epoch);
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
        return result;
    }

    void accept_native_samples(const std::vector<float>& samples, bool final,
                               std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch))
            return;
        if (!samples.empty()) {
            scheduler_.append(samples.data(), static_cast<int32_t>(samples.size()));
            mel_.accept_audio(samples.data(), static_cast<int32_t>(samples.size()));
        }
        process_scheduled_frames(final, work_epoch);
    }

    void process_scheduled_frames(bool final, std::uint64_t work_epoch) {
        while (work_is_current(work_epoch)) {
            auto frame = scheduler_.pop();
            if (!frame)
                break;
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

        decode_rnnt_frame(outputs.rnnt_frame.data(), work_epoch);
        if (!work_is_current(work_epoch))
            return;
        process_model_frame(outputs.projected_audio, work_epoch);
    }

    bool begin_agent_turn_if_needed(int32_t text_token, std::uint64_t work_epoch) {
        if (text_token != runtime_->config.bos_token_id)
            return true;
        if (!ensure_agent_turn(work_epoch))
            return false;
        agent_idle_ = false;
        agent_text_tokens_.clear();
        suppress_synthesis_until_turn_started_ = false;
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
            codes = tts_state_.step(text_token, agent_idle_);
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
        previous_text_token_ = text_token;
        // Keep the checkpoint-owned channel recurrent; the host assigns it no public semantics.
        previous_function_token_ = function_token;
        if (text_token == runtime_->config.eos_token_id && output_epoch.has_value())
            finish_agent_turn(work_epoch, *output_epoch);
        ++frame_index_;
    }

    void process_model_frame(const std::vector<float>& audio_embedding, std::uint64_t work_epoch) {
        if (!work_is_current(work_epoch))
            return;
        const auto& config = runtime_->config;
        std::pair<int32_t, int32_t> tokens;
        {
            std::lock_guard<std::mutex> runtime_lock(runtime_->inference_mutex);
            tokens = run_thinker(previous_text_token_, config.pad_token_id,
                                 previous_function_token_, audio_embedding, true);
        }
        if (!work_is_current(work_epoch))
            return;
        auto [text_token, function_token] = tokens;

        if (!begin_agent_turn_if_needed(text_token, work_epoch))
            return;

        auto output_epoch = accepted_output_epoch(work_epoch);
        if (!synthesize_model_audio(text_token, work_epoch, output_epoch))
            return;
        publish_model_text_token(text_token, work_epoch, output_epoch);
        complete_model_frame(text_token, function_token, work_epoch, output_epoch);
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
                events_.push_back(std::move(started));
            } else {
                epoch = conversation_.epoch();
            }
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
                events_.push_back(std::move(final_text));
            }
            SpeechSessionEvent finished;
            finished.kind = SpeechSessionEventKind::kTurnFinished;
            finished.epoch = output_epoch;
            finished.sequence = conversation_.next_sequence();
            finished.frame_index = frame_index_;
            events_.push_back(std::move(finished));
            (void)conversation_.finish_agent_turn();
        }
        agent_idle_ = true;
        agent_text_tokens_.clear();
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
        return codec_reconstruction_.push(first, 1);
    }

    void emit_audio(std::vector<float> waveform, std::uint64_t work_epoch,
                    const std::optional<std::uint64_t>& output_epoch) {
        if (!session_config_.emit_agent_audio)
            return;
        // Offline speak explicitly disables barge-in and retains the complete
        // frame-locked waveform, including model silence. Live sessions only
        // expose audio belonging to an accepted active agent epoch.
        if (session_config_.enable_barge_in && !output_epoch.has_value())
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
        if (published)
            output_sample_cursor_ = end_sample;
    }

    std::shared_ptr<NemotronVoiceChatRuntime> runtime_;
    SpeechSessionConfig session_config_;
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
    int64_t output_sample_cursor_{0};
    int64_t frame_index_{0};
    bool first_perception_step_{true};
    bool public_input_finished_{false};
    bool worker_input_finished_{false};
    bool agent_idle_{true};
    bool suppress_synthesis_until_turn_started_{false};
    bool stop_requested_{false};
    bool worker_initialized_{false};
    bool worker_done_{false};
    std::uint64_t requested_reset_serial_{0};
    std::uint64_t completed_reset_serial_{0};
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
    return std::make_unique<NemotronVoiceChatSession>(runtime_, config);
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
    auto session = create_speech_session(session_config);
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
    auto session = create_speech_session(config);
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
