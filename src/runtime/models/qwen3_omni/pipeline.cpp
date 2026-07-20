/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/pipeline.h"

#include "runtime/models/qwen3_omni/omni_audio_plan.h"
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
                           std::string model_id_str)
    : thinker_(std::move(thinker)), thinker_state_(std::move(thinker_state)),
      code2wav_(std::move(code2wav)), config_(std::make_unique<OmniConfig>(std::move(config))),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)) {
    if (!thinker_ || !thinker_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker module");
    if (!thinker_state_ || !thinker_state_->ok())
        throw std::runtime_error("OmniPipeline: invalid thinker cache");
    talker_runtime_ = std::make_unique<Qwen3OmniTalkerRuntime>(
        config_->hf_python, config_->talker_model_id, config_->talker_model_revision,
        config_->talker_n_codebooks, config_->code2wav_max_frames);
}

OmniPipeline::~OmniPipeline() = default;

void OmniPipeline::run_thinker_step(int32_t token_id, std::vector<float>& logits) {
    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_tensor;
    thinker_state_->prepare_step(inputs);

    TensorMap outputs = thinker_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("OmniPipeline thinker: no 'logits' output");

    const auto& lt = it->second;
    auto n = lt.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), lt.data, n * sizeof(float));

    thinker_state_->advance();
}

static int32_t omni_argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

std::vector<int32_t> OmniPipeline::run_thinker(const std::vector<int32_t>& input_ids,
                                               int32_t max_tokens) {
    thinker_state_->reset();
    thinker_state_->bind_to(*thinker_);

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_thinker_step(input_ids[i], logits);

    if (!input_ids.empty())
        run_thinker_step(input_ids.back(), logits);

    std::vector<int32_t> output_ids;
    output_ids.reserve(static_cast<std::size_t>(max_tokens));

    for (int32_t step = 0; step < max_tokens; ++step) {
        if (logits.empty())
            break;
        int32_t token = omni_argmax(logits);
        if (token == 0 || token == config_->thinker_eos_token_id)
            break;
        output_ids.push_back(token);
        run_thinker_step(token, logits);
    }

    std::cerr << "[trtmc] Omni Thinker: generated " << output_ids.size() << " text tokens"
              << std::endl;
    return output_ids;
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
