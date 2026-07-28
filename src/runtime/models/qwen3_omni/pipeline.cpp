/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/pipeline.h"

#include "runtime/models/qwen3_omni/argmax_kernel.h"
#include "runtime/models/qwen3_omni/omni_audio_plan.h"
#include "runtime/models/qwen3_omni/omni_thinker_plan.h"
#include "runtime/models/qwen3_omni/talker_runtime.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace trtmc {

namespace {

using OmniClock = std::chrono::steady_clock;

struct OmniStageTotals {
    double input_preparation_ms{0.0};
    double thinker_and_transfer_ms{0.0};
    double worker_start_ms{0.0};
    double talker_ms{0.0};
    double ipc_ms{0.0};
    double code2wav_and_transfer_ms{0.0};
    double output_materialization_ms{0.0};
};

double elapsed_ms(OmniClock::time_point start, OmniClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool omni_stage_timing_enabled() {
    const char* value = std::getenv("TRTMC_QWEN3_OMNI_STAGE_TIMING");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void report_omni_stage_timing(const OmniStageTotals& stages, double total_ms) {
    if (!omni_stage_timing_enabled())
        return;
    std::ostringstream timing;
    timing << "[trtmc.qwen3_omni_timing.json] {\"input_preparation_ms\":"
           << stages.input_preparation_ms
           << ",\"thinker_and_transfer_ms\":" << stages.thinker_and_transfer_ms
           << ",\"worker_start_ms\":" << stages.worker_start_ms
           << ",\"talker_ms\":" << stages.talker_ms << ",\"ipc_ms\":" << stages.ipc_ms
           << ",\"code2wav_and_transfer_ms\":" << stages.code2wav_and_transfer_ms
           << ",\"output_materialization_ms\":" << stages.output_materialization_ms
           << ",\"total_ms\":" << total_ms << '}';
    std::cerr << timing.str() << std::endl;
}

std::string format_omni_chat_prompt(const std::string& prompt) {
    return "<|im_start|>system\n"
           "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
           "perceiving auditory and visual inputs, as well as generating text and speech."
           "<|im_end|>\n<|im_start|>user\n" +
           prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

} // namespace

// ─── OmniPipeline (TrtModule-based) ───

OmniPipeline::OmniPipeline(std::unique_ptr<TrtModule> thinker,
                           std::unique_ptr<Qwen3OmniInferenceState> thinker_state,
                           std::unique_ptr<TrtModule> code2wav, OmniConfig config,
                           cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
                           std::string model_id_str, std::unique_ptr<TrtModule> thinker_prefill)
    : thinker_(std::move(thinker)), thinker_prefill_(std::move(thinker_prefill)),
      thinker_state_(std::move(thinker_state)), code2wav_(std::move(code2wav)),
      config_(std::make_unique<OmniConfig>(std::move(config))), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      thinker_token_id_({1}, DType::kInt32, stream) {
    if (!thinker_ || !thinker_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker module");
    if (!thinker_state_ || !thinker_state_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker cache");
    talker_runtime_ = std::make_unique<Qwen3OmniTalkerRuntime>(
        config_->hf_python, config_->talker_model_id, config_->talker_model_revision,
        config_->talker_n_codebooks, config_->code2wav_max_frames);
}

OmniPipeline::~OmniPipeline() = default;

namespace {

int32_t host_argmax(const TensorMap& outputs, int32_t& d2h_count) {
    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline thinker: no 'logits' output");
    const auto& logits = it->second;
    const auto count = static_cast<std::size_t>(logits.numel());
    if (count == 0)
        return 0;
    ++d2h_count;
    const auto* begin = static_cast<const float*>(logits.data);
    return static_cast<int32_t>(std::distance(begin, std::max_element(begin, begin + count)));
}

bool gather_prefill_kv(TrtModule& prefill, int32_t num_layers, std::vector<const void*>& present_k,
                       std::vector<const void*>& present_v) {
    present_k.resize(static_cast<std::size_t>(num_layers));
    present_v.resize(static_cast<std::size_t>(num_layers));
    for (int32_t layer = 0; layer < num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        present_k[index] = prefill.device_ptr("present_k_" + std::to_string(layer));
        present_v[index] = prefill.device_ptr("present_v_" + std::to_string(layer));
        if (present_k[index] == nullptr || present_v[index] == nullptr)
            return false;
    }
    return true;
}

Qwen3OmniKvCache* eligible_prefill_cache(TrtModule* prefill, Qwen3OmniInferenceState* state,
                                         const OmniConfig& config, int32_t sequence_length) {
    auto* cache = dynamic_cast<Qwen3OmniKvCache*>(state);
    if (prefill == nullptr || cache == nullptr || sequence_length <= 0 ||
        sequence_length > cache->max_length() || config.thinker_hidden_size <= 0 ||
        config.thinker_vocab_size <= 0 || !prefill->has_input("input_embed"))
        return nullptr;
    return cache;
}

bool can_use_device_argmax(const TrtModule& module, const DeviceTensor& token_id,
                           int32_t vocab_size) {
    return module.device_ptr("logits") != nullptr &&
           module.tensor_dtype("logits") == DType::kFloat32 && vocab_size > 0 && token_id.ok();
}

} // namespace

int32_t OmniPipeline::run_thinker_step(int32_t token_id) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    thinker_state_->prepare_step(inputs);

    ++thinker_stats_.decode_launches;
    const bool gpu_argmax =
        can_use_device_argmax(*thinker_, thinker_token_id_, config_->thinker_vocab_size);
    if (!gpu_argmax) {
        TensorMap outputs = thinker_->forward(inputs);
        thinker_state_->advance();
        return host_argmax(outputs, thinker_stats_.full_logits_d2h);
    }

    thinker_->forward_async(inputs);
    const auto* logits = static_cast<const float*>(thinker_->device_ptr("logits"));
    qwen3_omni_gpu_argmax(logits, config_->thinker_vocab_size,
                          static_cast<int32_t*>(thinker_token_id_.data()), stream_);
    thinker_state_->advance();
    cudaMemcpyAsync(&thinker_token_host_, thinker_token_id_.data(), sizeof(thinker_token_host_),
                    cudaMemcpyDeviceToHost, stream_);
    thinker_->sync();
    return thinker_token_host_;
}

bool OmniPipeline::run_thinker_prefill(const std::vector<int32_t>& input_ids, int32_t& next_token) {
    const auto sequence_length = static_cast<int32_t>(input_ids.size());
    auto* cache = eligible_prefill_cache(thinker_prefill_.get(), thinker_state_.get(), *config_,
                                         sequence_length);
    if (cache == nullptr)
        return false;

    std::vector<const void*> present_k;
    std::vector<const void*> present_v;
    if (!gather_prefill_kv(*thinker_prefill_, cache->num_layers(), present_k, present_v))
        return false;

    std::vector<float> input_embed(
        static_cast<std::size_t>(sequence_length) * config_->thinker_hidden_size, 0.0F);
    std::vector<float> use_input_embed(static_cast<std::size_t>(sequence_length), 0.0F);
    TensorMap inputs;
    inputs["token_id"] =
        Tensor{const_cast<int32_t*>(input_ids.data()), {sequence_length}, DType::kInt32};
    inputs["input_embed"] = Tensor{
        input_embed.data(), {sequence_length, config_->thinker_hidden_size}, DType::kFloat32};
    inputs["use_input_embed"] =
        Tensor{use_input_embed.data(), {sequence_length, 1}, DType::kFloat32};
    cache->bind_cache_inputs(*thinker_prefill_);
    thinker_state_->prepare_step(inputs, sequence_length);

    const bool gpu_argmax =
        can_use_device_argmax(*thinker_prefill_, thinker_token_id_, config_->thinker_vocab_size);
    ++thinker_stats_.prefill_launches;
    if (gpu_argmax) {
        thinker_prefill_->forward_async(inputs);
        const auto* logits = static_cast<const float*>(thinker_prefill_->device_ptr("logits"));
        qwen3_omni_gpu_argmax(logits, config_->thinker_vocab_size,
                              static_cast<int32_t*>(thinker_token_id_.data()), stream_);
        cache->write_prefill_kv(present_k, present_v, sequence_length);
        cudaMemcpyAsync(&thinker_token_host_, thinker_token_id_.data(), sizeof(thinker_token_host_),
                        cudaMemcpyDeviceToHost, stream_);
        thinker_prefill_->sync();
        next_token = thinker_token_host_;
        return true;
    }

    TensorMap outputs = thinker_prefill_->forward(inputs);
    cache->write_prefill_kv(present_k, present_v, sequence_length);
    next_token = host_argmax(outputs, thinker_stats_.full_logits_d2h);
    return true;
}

std::vector<int32_t> OmniPipeline::run_thinker(const std::vector<int32_t>& input_ids,
                                               int32_t max_tokens) {
    thinker_stats_ = {};
    thinker_stats_.prompt_tokens = static_cast<int32_t>(input_ids.size());
    if (input_ids.empty() || max_tokens <= 0)
        return {};

    thinker_state_->reset();
    thinker_state_->set_prompt_length(static_cast<int32_t>(input_ids.size()));
    thinker_state_->bind_to(*thinker_);

    int32_t next_token = 0;
    if (!run_thinker_prefill(input_ids, next_token)) {
        for (int32_t token : input_ids)
            next_token = run_thinker_step(token);
    }
    thinker_state_->mark_prefill_complete();

    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step) {
        const int32_t token = next_token;
        if (omni_thinker_should_stop(token, config_->thinker_eos_token_id))
            break;
        output_ids.push_back(token);
        if (step + 1 < max_tokens)
            next_token = run_thinker_step(token);
    }

    std::cerr << "[trtmc] Omni Thinker: generated " << output_ids.size() << " text tokens"
              << std::endl;
    return output_ids;
}

std::vector<int32_t> OmniPipeline::generate_thinker_ids(const std::vector<int32_t>& input_ids,
                                                        int32_t max_tokens) {
    return run_thinker(input_ids, max_tokens);
}

std::vector<float> OmniPipeline::run_code2wav(const std::vector<int32_t>& codec_tokens,
                                              int32_t n_codebooks, int32_t n_frames,
                                              double& code2wav_and_transfer_ms,
                                              double& output_materialization_ms) {
    if (!code2wav_) {
        throw std::runtime_error("OmniPipeline: required Code2Wav engine is unavailable");
    }

    const int32_t max_frames = config_->code2wav_max_frames;
    const int32_t actual_frames = std::min(n_frames, max_frames);

    std::vector<int32_t> input_codes =
        build_omni_code2wav_input_codes(codec_tokens, n_codebooks, max_frames, actual_frames);

    Tensor codes_tensor;
    codes_tensor.data = input_codes.data();
    codes_tensor.shape = {1, static_cast<int64_t>(n_codebooks), static_cast<int64_t>(max_frames)};
    codes_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["codec_tokens"] = codes_tensor;

    const auto code2wav_start = OmniClock::now();
    TensorMap outputs = code2wav_->forward(inputs);
    code2wav_and_transfer_ms = elapsed_ms(code2wav_start, OmniClock::now());

    auto it = outputs.find("waveform");
    if (it == outputs.end()) {
        std::cerr << "[trtmc] Omni Code2Wav: no 'waveform' output" << std::endl;
        return {};
    }

    const auto& wt = it->second;
    const auto total_out = wt.numel();
    const auto copy_n =
        code2wav_output_samples(*config_, actual_frames, static_cast<std::size_t>(total_out));

    const auto materialize_start = OmniClock::now();
    std::vector<float> waveform(static_cast<std::size_t>(copy_n));
    std::memcpy(waveform.data(), wt.data, copy_n * sizeof(float));
    output_materialization_ms = elapsed_ms(materialize_start, OmniClock::now());

    std::cerr << "[trtmc] Omni Code2Wav: " << actual_frames << " frames -> " << waveform.size()
              << " samples" << std::endl;
    return waveform;
}

AudioResult OmniPipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    const auto total_start = OmniClock::now();
    OmniStageTotals stages;
    const auto input_start = OmniClock::now();
    std::vector<int32_t> input_ids;
    if (tokenizer_)
        input_ids = tokenizer_->encode(format_omni_chat_prompt(prompt));
    stages.input_preparation_ms = elapsed_ms(input_start, OmniClock::now());

    int32_t max_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 768;

    AudioResult result;
    result.sample_rate = config_->sample_rate;

    std::cerr << "[trtmc] Omni: starting pipeline with " << input_ids.size() << " input tokens"
              << std::endl;

    const auto thinker_start = OmniClock::now();
    auto text_tokens = run_thinker(input_ids, max_tokens);
    stages.thinker_and_transfer_ms = elapsed_ms(thinker_start, OmniClock::now());
    if (text_tokens.empty()) {
        std::cerr << "[trtmc] Omni: Thinker produced no tokens" << std::endl;
        report_omni_stage_timing(stages, elapsed_ms(total_start, OmniClock::now()));
        return result;
    }

    if (!tokenizer_)
        throw std::runtime_error("OmniPipeline: tokenizer is required for official Talker input");
    const auto text_start = OmniClock::now();
    const std::string assistant_text = tokenizer_->decode(text_tokens);
    stages.output_materialization_ms += elapsed_ms(text_start, OmniClock::now());
    std::cerr << "[trtmc] Omni Thinker text: " << assistant_text << std::endl;

    auto talker_result = talker_runtime_->run(prompt, assistant_text);
    stages.worker_start_ms = talker_result.worker_start_ms;
    stages.talker_ms = talker_result.talker_ms;
    stages.ipc_ms = talker_result.ipc_ms;
    stages.output_materialization_ms += talker_result.output_materialization_ms;
    if (talker_result.exit_code != 0) {
        report_omni_stage_timing(stages, elapsed_ms(total_start, OmniClock::now()));
        throw std::runtime_error("OmniPipeline: official Talker failed: " +
                                 talker_result.stderr_data);
    }

    const OmniCodecPlan codec_plan =
        make_omni_codec_plan(*config_, talker_result.frame_major_codes.size());
    if (!codec_plan.should_run_codec)
        throw std::runtime_error("OmniPipeline: official Talker produced no codec frames");
    std::cerr << "[trtmc] Omni official Talker: " << codec_plan.n_frames << " frames x "
              << codec_plan.n_codebooks << " codebooks" << std::endl;
    double waveform_materialization_ms = 0.0;
    auto waveform =
        run_code2wav(talker_result.frame_major_codes, codec_plan.n_codebooks, codec_plan.n_frames,
                     stages.code2wav_and_transfer_ms, waveform_materialization_ms);
    stages.output_materialization_ms += waveform_materialization_ms;
    if (!waveform.empty()) {
        result.samples = std::move(waveform);
        result.num_samples = static_cast<int32_t>(result.samples.size());
    }

    std::cerr << "[trtmc] Omni: generated " << result.num_samples << " samples ("
              << (result.num_samples > 0
                      ? static_cast<float>(result.num_samples) / result.sample_rate
                      : 0.0F)
              << "s @ " << result.sample_rate << " Hz)" << std::endl;

    report_omni_stage_timing(stages, elapsed_ms(total_start, OmniClock::now()));
    return result;
}

} // namespace trtmc
