/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/personaplex/runtime/pipeline.h"

#include "families/personaplex/runtime/decode_runtime.h"
#include "families/personaplex/runtime/resampler.h"
#include "families/personaplex/runtime/sampling_kernels.h"
#include "families/personaplex/runtime/speech_decode_stop_policy.h"
#include "families/personaplex/runtime/speech_delay_cache.h"
#include "families/personaplex/runtime/speech_depth_plan.h"
#include "families/personaplex/runtime/speech_generation_policy.h"
#include "families/personaplex/runtime/speech_mimi_decode_plan.h"
#include "families/personaplex/runtime/speech_mimi_encode_plan.h"
#include "families/personaplex/runtime/speech_performance.h"
#include "families/personaplex/runtime/speech_runtime_plan.h"
#include "families/personaplex/runtime/speech_temporal_embed_plan.h"
#include "families/personaplex/runtime/speech_waveform_postprocess.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace trtmc {

// ─── SpeechPipeline (ITrtModule-based) ───

namespace {

constexpr uint64_t kSpeechSamplingSeed = 0x5EEDC0DECAFE1234ULL;

DeviceTensor upload_float_table(const std::vector<float>& values, cudaStream_t stream,
                                const char* label) {
    if (values.empty())
        return {};
    DeviceTensor tensor({static_cast<int64_t>(values.size())}, DType::kFloat32, stream);
    if (!tensor.ok() || !tensor.copy_from_host(values.data()))
        throw std::runtime_error(std::string("SpeechPipeline: failed to upload ") + label);
    return tensor;
}

int32_t output_width(const ITrtModule& module, const char* name) {
    const auto shape = module.tensor_shape(name);
    if (shape.empty() || shape.back() <= 0 || shape.back() > INT32_MAX)
        throw std::runtime_error(std::string("SpeechPipeline: invalid output shape for ") + name);
    return static_cast<int32_t>(shape.back());
}

bool speech_is_sampling_enabled(const SpeechConfig& config) {
    return config.depth_temperature > 0.0F && config.depth_top_k > 0;
}

} // namespace

struct SpeechDeviceWorkspace {
    SpeechDeviceWorkspace(const SpeechConfig& config, cudaStream_t stream)
        : depth_projection(upload_float_table(config.depth_projection, stream, "depth projection")),
          depth_text_embedding(
              upload_float_table(config.depth_text_embedding, stream, "depth text embedding")),
          depth_audio_embeddings(
              upload_float_table(config.depth_audio_embeddings, stream, "depth audio embeddings")),
          depth_embed({std::max(config.depth_hidden_size, 1)}, DType::kFloat32, stream),
          selected_tokens({static_cast<int64_t>(std::max(config.num_codebooks + 1, 1))},
                          DType::kInt32, stream),
          dummy_token(DeviceTensor::zeros({1}, DType::kInt32, stream)),
          use_input_embed({1}, DType::kFloat32, stream), rng_state({2}, DType::kInt32, stream) {
        constexpr float one = 1.0F;
        if (!depth_embed.ok() || !selected_tokens.ok() || !dummy_token.ok() ||
            !use_input_embed.ok() || !rng_state.ok() || !use_input_embed.copy_from_host(&one) ||
            !rng_state.copy_from_host(&kSpeechSamplingSeed)) {
            throw std::runtime_error("SpeechPipeline: failed to allocate device generation state");
        }
    }

    DeviceTensor depth_projection;
    DeviceTensor depth_text_embedding;
    DeviceTensor depth_audio_embeddings;
    DeviceTensor depth_embed;
    DeviceTensor selected_tokens;
    DeviceTensor dummy_token;
    DeviceTensor use_input_embed;
    DeviceTensor rng_state;
};

SpeechPipeline::SpeechPipeline(std::unique_ptr<ITrtModule> mimi_encoder,
                               std::unique_ptr<ITrtModule> temporal,
                               std::unique_ptr<PersonaplexInferenceState> temporal_state,
                               std::vector<std::unique_ptr<ITrtModule>> depth_engines,
                               std::unique_ptr<PersonaplexInferenceState> depth_state,
                               std::unique_ptr<ITrtModule> mimi_decoder, SpeechConfig config,
                               cudaStream_t stream, std::string model_id_str)
    : temporal_(std::move(temporal)), mimi_encoder_(std::move(mimi_encoder)),
      temporal_state_(std::move(temporal_state)), depth_engines_(std::move(depth_engines)),
      depth_state_(std::move(depth_state)), mimi_decoder_(std::move(mimi_decoder)), stream_(stream),
      config_(std::move(config)), model_id_(std::move(model_id_str)) {
    if (!temporal_ || !temporal_->ok())
        throw std::runtime_error("SpeechPipeline: invalid temporal module");
    if (!temporal_state_ || !temporal_state_->ok())
        throw std::runtime_error("SpeechPipeline: invalid temporal cache");

    device_workspace_ = std::make_unique<SpeechDeviceWorkspace>(config_, stream_);
    temporal_->bind_external("token_id", device_workspace_->dummy_token.data());
    temporal_->bind_external("use_input_embed", device_workspace_->use_input_embed.data());
    for (auto& engine : depth_engines_) {
        if (!engine)
            continue;
        if (engine->stream() != stream_)
            throw std::runtime_error("SpeechPipeline: generation modules must share one stream");
        engine->bind_external("token_id", device_workspace_->dummy_token.data());
        engine->bind_external("use_input_embed", device_workspace_->use_input_embed.data());
        engine->bind_external("input_embed", device_workspace_->depth_embed.data());
    }
}

SpeechPipeline::~SpeechPipeline() = default;

// ---------------------------------------------------------------------------
// Mimi Encoder: audio waveform -> codec tokens
// ---------------------------------------------------------------------------

namespace {

struct MimiEncoderShapes {
    int32_t engine_input_samples{0};
    int32_t enc_codebooks{0};
    int32_t enc_frames{0};
};

struct MimiHostState {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

using MimiHostStates = std::unordered_map<std::string, MimiHostState>;

MimiEncoderShapes query_mimi_encoder_shapes(const ITrtModule& module) {
    MimiEncoderShapes s;
    for (const auto& info : module.input_info()) {
        if (info.name == "audio_input" && !info.shape.empty())
            s.engine_input_samples = static_cast<int32_t>(info.shape.back());
    }
    for (const auto& info : module.output_info()) {
        if (info.name == "codec_tokens" && info.shape.size() >= 2) {
            s.enc_codebooks = static_cast<int32_t>(info.shape[0]);
            s.enc_frames = static_cast<int32_t>(info.shape[1]);
        }
    }
    return s;
}

bool is_mimi_control_input(const std::string& name) {
    return name == "audio_input" || name == "mimi_position_ids" || name == "mimi_cache_indices" ||
           name == "mimi_attention_mask";
}

std::size_t mimi_state_element_count(const std::vector<int64_t>& shape) {
    std::size_t elements = 1;
    for (const auto dimension : shape)
        elements *= static_cast<std::size_t>(dimension);
    return elements;
}

bool initialize_mimi_host_states(const ITrtModule& module, MimiHostStates& states) {
    for (const auto& info : module.input_info()) {
        if (is_mimi_control_input(info.name))
            continue;
        if (info.dtype != DType::kFloat32 || info.shape.empty()) {
            std::cerr << "[speech] Mimi encoder state '" << info.name
                      << "' does not use the expected FP32 static shape" << std::endl;
            return false;
        }
        const auto elements = mimi_state_element_count(info.shape);
        states.emplace(info.name, MimiHostState{std::vector<float>(elements, 0.0F), info.shape});
    }
    return true;
}

TensorMap make_mimi_streaming_inputs(std::vector<float>& audio_chunk,
                                     const MimiEncoderShapes& shapes,
                                     MimiRingAttentionInputs& attention, MimiHostStates& states) {
    TensorMap inputs;
    inputs["audio_input"] =
        Tensor{audio_chunk.data(), {1, 1, shapes.engine_input_samples}, DType::kFloat32};
    inputs["mimi_position_ids"] =
        Tensor{attention.position_ids.data(), {kMimiFrontendTokensPerFrame}, DType::kInt32};
    inputs["mimi_cache_indices"] =
        Tensor{attention.cache_indices.data(), {kMimiFrontendTokensPerFrame, 1}, DType::kInt32};
    inputs["mimi_attention_mask"] =
        Tensor{attention.mask.data(),
               {1, 1, kMimiFrontendTokensPerFrame, kMimiAttentionContext},
               DType::kFloat32};
    for (auto& [name, state] : states)
        inputs[name] = Tensor{state.values.data(), state.shape, DType::kFloat32};
    return inputs;
}

bool consume_mimi_streaming_outputs(const TensorMap& outputs, int32_t codebooks,
                                    MimiHostStates& states, std::vector<int32_t>& tokens) {
    const auto codec = outputs.find("codec_tokens");
    if (codec == outputs.end()) {
        std::cerr << "[speech] Mimi encoder: no 'codec_tokens' output" << std::endl;
        return false;
    }
    const auto* codec_values = static_cast<const float*>(codec->second.data);
    for (int32_t codebook = 0; codebook < codebooks; ++codebook)
        tokens.push_back(static_cast<int32_t>(std::round(codec_values[codebook])));

    for (auto& [name, state] : states) {
        const auto output = outputs.find(name + "_out");
        if (output == outputs.end() ||
            output->second.nbytes() != state.values.size() * sizeof(float)) {
            std::cerr << "[speech] Mimi encoder: invalid state output for '" << name << "'"
                      << std::endl;
            return false;
        }
        std::memcpy(state.values.data(), output->second.data, output->second.nbytes());
    }
    return true;
}

void log_first_n_tokens(const char* label, const std::vector<int32_t>& tokens, int32_t n = 16) {
    std::cerr << label;
    for (int32_t i = 0; i < std::min(n, static_cast<int32_t>(tokens.size())); ++i)
        std::cerr << tokens[static_cast<std::size_t>(i)] << " ";
    std::cerr << std::endl;
}

} // anonymous namespace

std::vector<int32_t> SpeechPipeline::run_mimi_encode(const float* samples, int32_t num_samples) {
    last_encode_frames_ = 0;
    last_encode_codebooks_ = 0;

    if (!mimi_encoder_ || !mimi_encoder_->ok()) {
        std::cerr << "[speech] No Mimi TRT encoder available" << std::endl;
        return {};
    }

    const auto shapes = query_mimi_encoder_shapes(*mimi_encoder_);
    const auto max_input_samples =
        shapes.engine_input_samples * std::max(config_.mimi_max_frames, 0);
    const auto plan = build_mimi_encode_plan(num_samples, max_input_samples,
                                             std::max(config_.mimi_max_frames, 0));

    if (!plan.input_fits) {
        std::cerr << "[speech] Mimi encoder input " << num_samples
                  << " samples exceeds or cannot use the engine capacity of " << max_input_samples
                  << " samples" << std::endl;
        return {};
    }
    if (shapes.engine_input_samples <= 0 || shapes.enc_codebooks <= 0 || shapes.enc_frames != 1) {
        std::cerr << "[speech] Mimi encoder has an invalid streaming shape" << std::endl;
        return {};
    }

    MimiHostStates states;
    if (!initialize_mimi_host_states(*mimi_encoder_, states))
        return {};

    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<std::size_t>(plan.valid_frames) * shapes.enc_codebooks);
    std::vector<float> audio_chunk(static_cast<std::size_t>(shapes.engine_input_samples), 0.0F);

    std::cerr << "[speech] Mimi encoder TRT streaming: " << plan.valid_frames << " chunks x "
              << shapes.engine_input_samples << " samples, " << shapes.enc_codebooks << " codebooks"
              << std::endl;

    for (int32_t frame = 0; frame < plan.valid_frames; ++frame) {
        std::fill(audio_chunk.begin(), audio_chunk.end(), 0.0F);
        const auto source_offset = static_cast<std::size_t>(frame) * shapes.engine_input_samples;
        const auto remaining = static_cast<std::size_t>(num_samples) - source_offset;
        const auto copy_samples = std::min(audio_chunk.size(), remaining);
        std::memcpy(audio_chunk.data(), samples + source_offset, copy_samples * sizeof(float));

        auto attention = build_mimi_ring_attention_inputs(frame);
        auto inputs = make_mimi_streaming_inputs(audio_chunk, shapes, attention, states);
        TensorMap outputs = mimi_encoder_->forward(inputs);
        if (!consume_mimi_streaming_outputs(outputs, shapes.enc_codebooks, states, tokens))
            return {};
    }

    std::cerr << "[speech] Mimi encode (TRT): " << num_samples << " samples -> "
              << plan.valid_frames << " frames x " << shapes.enc_codebooks << " codebooks"
              << std::endl;

    last_encode_frames_ = plan.valid_frames;
    last_encode_codebooks_ = shapes.enc_codebooks;

    log_first_n_tokens("[speech] Encoder tokens [0:16]: ", tokens);

    return tokens;
}

// ---------------------------------------------------------------------------
// Temporal step with PersonaplexKvCache: input_embed -> logits (+ hidden_state)
// ---------------------------------------------------------------------------

SpeechTemporalDeviceOutput SpeechPipeline::run_temporal_embed_step(const float* embed_ptr,
                                                                   int32_t embed_size) {
    temporal_state_->bind_to(*temporal_);

    Tensor embed_tensor;
    embed_tensor.data = const_cast<float*>(embed_ptr);
    embed_tensor.shape = {static_cast<int64_t>(embed_size)};
    embed_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["input_embed"] = embed_tensor;
    temporal_state_->prepare_step(inputs);

    temporal_->forward_async(inputs);
    const auto* logits = static_cast<const float*>(temporal_->device_ptr("logits"));
    if (!logits || temporal_->tensor_dtype("logits") != DType::kFloat32)
        throw std::runtime_error("SpeechPipeline temporal: FP32 device logits unavailable");

    const bool sampled = speech_is_sampling_enabled(config_);
    if (!personaplex_select_token(logits, output_width(*temporal_, "logits"), sampled ? 0.7F : 0.0F,
                                  sampled ? 25 : 0, output_width(*temporal_, "logits") - 1,
                                  static_cast<uint64_t*>(device_workspace_->rng_state.data()),
                                  static_cast<int32_t*>(device_workspace_->selected_tokens.data()),
                                  stream_)) {
        throw std::runtime_error(
            "SpeechPipeline temporal: device sampling failed or is unsupported");
    }

    temporal_state_->advance();

    SpeechTemporalDeviceOutput output;
    if (temporal_->has_output("hidden_state")) {
        output.hidden = temporal_->device_ptr("hidden_state");
        output.hidden_dtype = temporal_->tensor_dtype("hidden_state");
    } else {
        output.hidden = logits;
        output.hidden_dtype = DType::kFloat32;
    }
    if (!output.hidden)
        throw std::runtime_error("SpeechPipeline temporal: device hidden state unavailable");
    return output;
}

// ---------------------------------------------------------------------------
// Depth step: generate num_codebooks tokens
// ---------------------------------------------------------------------------

namespace {

int32_t speech_clamp_token(int32_t token, int32_t vocab_size) {
    return clamp_speech_depth_token(token, vocab_size);
}

} // anonymous namespace

void SpeechPipeline::run_depth(const SpeechTemporalDeviceOutput& temporal_output,
                               int32_t text_token, bool text_token_is_forced,
                               const int32_t* forced_audio_tokens,
                               const uint8_t* forced_audio_provided) {
    if (depth_engines_.empty())
        throw std::runtime_error("SpeechPipeline: no depth engine available");

    depth_state_->reset();
    auto* selected_tokens = static_cast<int32_t*>(device_workspace_->selected_tokens.data());
    for (int32_t codebook = 0; codebook < config_.num_codebooks; ++codebook) {
        prepare_depth_input(temporal_output, codebook, text_token, text_token_is_forced,
                            forced_audio_tokens, forced_audio_provided, selected_tokens);
        enqueue_depth_step(depth_engine_for_codebook(codebook), codebook, selected_tokens);
    }
}

ITrtModule& SpeechPipeline::depth_engine_for_codebook(int32_t codebook) {
    const auto index = static_cast<std::size_t>(codebook);
    if (index >= depth_engines_.size() || !depth_engines_[index])
        throw std::runtime_error("SpeechPipeline: missing depth engine for codebook " +
                                 std::to_string(codebook));
    return *depth_engines_[index];
}

void SpeechPipeline::prepare_depth_input(const SpeechTemporalDeviceOutput& temporal_output,
                                         int32_t codebook, int32_t text_token,
                                         bool text_token_is_forced,
                                         const int32_t* forced_audio_tokens,
                                         const uint8_t* forced_audio_provided,
                                         int32_t* selected_tokens) {
    const int32_t previous_codebook = codebook - 1;
    const bool previous_is_forced = previous_codebook >= 0 && forced_audio_tokens &&
                                    forced_audio_provided &&
                                    forced_audio_provided[previous_codebook] != 0;
    const int32_t forced_previous = previous_is_forced ? forced_audio_tokens[previous_codebook] : 0;
    personaplex_prepare_depth_embedding(
        static_cast<float*>(device_workspace_->depth_embed.data()), temporal_output.hidden,
        temporal_output.hidden_dtype,
        static_cast<const float*>(device_workspace_->depth_projection.data()),
        device_workspace_->depth_projection.numel(),
        static_cast<const float*>(device_workspace_->depth_text_embedding.data()),
        device_workspace_->depth_text_embedding.numel(),
        static_cast<const float*>(device_workspace_->depth_audio_embeddings.data()),
        device_workspace_->depth_audio_embeddings.numel(), selected_tokens, codebook, text_token,
        text_token_is_forced, forced_previous, previous_is_forced, config_.depth_hidden_size,
        config_.temporal_hidden_size, config_.depth_text_vocab, config_.audio_vocab_size,
        config_.num_depformer_emb, stream_);
    if (const auto error = cudaGetLastError(); error != cudaSuccess) {
        throw std::runtime_error(std::string("SpeechPipeline depth embedding: ") +
                                 cudaGetErrorString(error));
    }
}

void SpeechPipeline::enqueue_depth_step(ITrtModule& engine, int32_t codebook,
                                        int32_t* selected_tokens) {
    depth_state_->bind_to(engine);
    TensorMap inputs;
    depth_state_->prepare_step(inputs);
    engine.forward_async(inputs);
    const auto* logits = static_cast<const float*>(engine.device_ptr("logits"));
    if (!logits || engine.tensor_dtype("logits") != DType::kFloat32)
        throw std::runtime_error("SpeechPipeline depth: FP32 device logits unavailable");
    depth_state_->advance();
    if (!personaplex_select_token(logits, output_width(engine, "logits"), config_.depth_temperature,
                                  config_.depth_top_k, config_.codebook_size - 1,
                                  static_cast<uint64_t*>(device_workspace_->rng_state.data()),
                                  selected_tokens + codebook + 1, stream_)) {
        throw std::runtime_error("SpeechPipeline depth: device sampling failed or is unsupported");
    }
}

void SpeechPipeline::download_selected_frame_tokens(std::vector<int32_t>& selected_tokens) {
    selected_tokens.resize(static_cast<std::size_t>(config_.num_codebooks + 1));
    const auto bytes = selected_tokens.size() * sizeof(int32_t);
    auto error = cudaMemcpyAsync(selected_tokens.data(), device_workspace_->selected_tokens.data(),
                                 bytes, cudaMemcpyDeviceToHost, stream_);
    if (error != cudaSuccess || cudaStreamSynchronize(stream_) != cudaSuccess)
        throw std::runtime_error("SpeechPipeline: failed to download selected frame tokens");
}

// ---------------------------------------------------------------------------
// Mimi Decoder: codec tokens -> waveform
// ---------------------------------------------------------------------------

namespace {

struct MimiDecoderShapes {
    int32_t dec_codebooks{0};
    int32_t dec_frames{0};
    std::vector<int32_t> output_dims;
};

MimiDecoderShapes query_mimi_decoder_shapes(const ITrtModule& module) {
    MimiDecoderShapes s;
    for (const auto& info : module.input_info()) {
        if (info.name == "codec_tokens" && info.shape.size() >= 2) {
            s.dec_codebooks = static_cast<int32_t>(info.shape[0]);
            s.dec_frames = static_cast<int32_t>(info.shape[1]);
        }
    }
    for (const auto& info : module.output_info()) {
        if (info.name == "audio_output") {
            s.output_dims.reserve(info.shape.size());
            for (auto d : info.shape)
                s.output_dims.push_back(static_cast<int32_t>(d));
        }
    }
    return s;
}

} // anonymous namespace

std::vector<float> SpeechPipeline::run_mimi_decode(const std::vector<int32_t>& codec_tokens,
                                                   int32_t num_frames) {
    if (num_frames <= 0)
        return {};

    int32_t actual_codebooks = 0;
    if (!codec_tokens.empty())
        actual_codebooks = static_cast<int32_t>(codec_tokens.size()) / num_frames;

    if (!mimi_decoder_ || !mimi_decoder_->ok()) {
        std::cerr << "[speech] No Mimi TRT decoder available" << std::endl;
        return {};
    }

    const auto shapes = query_mimi_decoder_shapes(*mimi_decoder_);
    const auto layout =
        build_mimi_decode_layout(shapes.dec_codebooks, shapes.dec_frames, shapes.output_dims);

    std::cerr << "[speech] Mimi decoder TRT: input [" << layout.dec_codebooks << ","
              << layout.dec_frames << "], output " << layout.total_output_elems << " samples"
              << std::endl;

    auto input_tokens = build_mimi_decoder_input(codec_tokens, num_frames, actual_codebooks,
                                                 layout.dec_frames, layout.dec_codebooks);

    // Debug: print first few input tokens
    std::cerr << "[speech] Decoder input tokens [0:16]: ";
    for (int32_t i = 0; i < std::min(16, static_cast<int32_t>(layout.input_elems)); ++i)
        std::cerr << input_tokens[static_cast<std::size_t>(i)] << " ";
    std::cerr << std::endl;

    Tensor codec_tensor;
    codec_tensor.data = input_tokens.data();
    codec_tensor.shape = {static_cast<int64_t>(shapes.dec_codebooks),
                          static_cast<int64_t>(shapes.dec_frames)};
    codec_tensor.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["codec_tokens"] = codec_tensor;

    TensorMap outputs = mimi_decoder_->forward(inputs);

    auto it = outputs.find("audio_output");
    if (it == outputs.end()) {
        std::cerr << "[speech] Mimi decoder: no 'audio_output' output" << std::endl;
        return {};
    }

    const auto& out_tensor = it->second;
    const auto total_elems = static_cast<std::size_t>(layout.total_output_elems);
    std::vector<float> waveform(total_elems);
    std::memcpy(waveform.data(), out_tensor.data, total_elems * sizeof(float));

    float rms = 0.0F;
    float mx = 0.0F;
    waveform_stats(waveform, layout.total_output_elems, rms, mx);
    std::cerr << "[speech] Mimi decode (TRT): " << layout.dec_frames << " frames -> "
              << layout.total_output_elems << " samples (RMS=" << rms << ", Max=" << mx << ")"
              << std::endl;
    return waveform;
}

// ---------------------------------------------------------------------------
// Text Prompt Injection
// ---------------------------------------------------------------------------

namespace {

void compute_text_prompt_frame_embed(const SpeechConfig& cfg, int32_t text_token_id, int32_t hidden,
                                     float* summed_embed) {
    std::fill(summed_embed, summed_embed + hidden, 0.0F);
    const int32_t text_tok = speech_clamp_token(text_token_id, cfg.temporal_text_vocab);
    const auto text_offset = static_cast<std::size_t>(text_tok) * hidden;
    add_speech_embedding_row(cfg.temporal_text_embedding, text_offset, hidden, summed_embed);

    const int32_t audio_vocab = cfg.audio_vocab_size;
    const int32_t bos = speech_clamp_token(cfg.codebook_size, audio_vocab);
    const auto emb_stride_cb = static_cast<std::size_t>(audio_vocab) * hidden;
    for (int32_t cb = 0; cb < cfg.num_codebooks; ++cb) {
        const auto emb_offset =
            static_cast<std::size_t>(cb) * emb_stride_cb + static_cast<std::size_t>(bos) * hidden;
        add_speech_embedding_row(cfg.audio_embeddings, emb_offset, hidden, summed_embed);
    }
}

} // anonymous namespace

void SpeechPipeline::run_text_prompt() {
    const auto& cfg = config_;
    const int32_t hidden = cfg.temporal_hidden_size;
    const auto& text_tokens = cfg.text_prompt_ids;
    if (text_tokens.empty())
        return;

    if (cfg.temporal_text_embedding.empty() || cfg.temporal_text_vocab <= 0 ||
        !temporal_->has_input("input_embed")) {
        std::cerr << "[speech] Cannot inject text prompt: missing embeddings" << std::endl;
        return;
    }

    std::vector<float> prompt_embeddings(text_tokens.size() * static_cast<std::size_t>(hidden));
    for (std::size_t t = 0; t < text_tokens.size(); ++t) {
        auto* embedding = prompt_embeddings.data() + t * static_cast<std::size_t>(hidden);
        compute_text_prompt_frame_embed(cfg, text_tokens[t], hidden, embedding);
        run_temporal_embed_step(embedding, hidden);
    }
    if (cudaStreamSynchronize(stream_) != cudaSuccess)
        throw std::runtime_error("SpeechPipeline: text prompt synchronization failed");

    std::cerr << "[speech] Text prompt injection complete (" << text_tokens.size()
              << " temporal steps)" << std::endl;
}

// ---------------------------------------------------------------------------
// Interleaved generation helpers (free functions, SpeechPipeline-specific)
// ---------------------------------------------------------------------------

namespace {

class CudaStageTimer {
  public:
    CudaStageTimer() {
        auto error = cudaEventCreate(&start_);
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("PersonaPlex timing event creation: ") +
                                     cudaGetErrorString(error));
        error = cudaEventCreate(&stop_);
        if (error == cudaSuccess)
            return;
        cudaEventDestroy(start_);
        start_ = nullptr;
        throw std::runtime_error(std::string("PersonaPlex timing event creation: ") +
                                 cudaGetErrorString(error));
    }

    ~CudaStageTimer() {
        if (stop_)
            cudaEventDestroy(stop_);
        if (start_)
            cudaEventDestroy(start_);
    }

    CudaStageTimer(const CudaStageTimer&) = delete;
    CudaStageTimer& operator=(const CudaStageTimer&) = delete;

    void start(cudaStream_t stream) { record(start_, stream); }
    void stop(cudaStream_t stream) { record(stop_, stream); }

    double elapsed_ms() const {
        float elapsed = 0.0F;
        const auto error = cudaEventElapsedTime(&elapsed, start_, stop_);
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("PersonaPlex timing event query: ") +
                                     cudaGetErrorString(error));
        return static_cast<double>(elapsed);
    }

  private:
    static void record(cudaEvent_t event, cudaStream_t stream) {
        const auto error = cudaEventRecord(event, stream);
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("PersonaPlex timing event record: ") +
                                     cudaGetErrorString(error));
    }

    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
};

void speech_log_depth_mode(const SpeechConfig& cfg) {
    if (speech_is_sampling_enabled(cfg)) {
        std::cerr << "[speech] Depth sampling: temperature=" << cfg.depth_temperature
                  << " top_k=" << cfg.depth_top_k << std::endl;
        return;
    }
    std::cerr << "[speech] Depth decoding: greedy (argmax)" << std::endl;
}

void speech_log_stop_configuration(const SpeechConfig& cfg, int32_t extra_tail) {
    if (cfg.text_eos_token_id >= 0) {
        std::cerr << "[speech] Text EOS early-stop enabled: eos_token_id=" << cfg.text_eos_token_id
                  << " (min_streak=" << kSpeechMinConsecutiveTextEos << ")" << std::endl;
    }
    if (extra_tail <= 0)
        return;
    std::cerr << "[speech] Text PAD fallback stop enabled after input "
                 "(pad_id="
              << cfg.text_padding_id << ", min_streak=" << kSpeechMinConsecutiveTextPadAfterInput
              << ")" << std::endl;
    std::cerr << "[speech] Post-input continuation cap: " << kSpeechMaxContinuationFramesAfterInput
              << " frames" << std::endl;
}

void speech_maybe_log_stop_decision(SpeechDecodeStopReason reason,
                                    const SpeechDecodeStopState& stop_state, int32_t offset) {
    switch (reason) {
    case SpeechDecodeStopReason::kNone:
        return;
    case SpeechDecodeStopReason::kTextEos:
        std::cerr << "[speech] Text EOS detected at offset " << offset
                  << " (streak=" << stop_state.text_eos_streak
                  << "), draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    case SpeechDecodeStopReason::kTextPadFallback:
        std::cerr << "[speech] Text PAD fallback stop at offset " << offset
                  << " (streak=" << stop_state.text_pad_streak
                  << "), draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    case SpeechDecodeStopReason::kContinuationCap:
        std::cerr << "[speech] Continuation cap reached at offset " << offset
                  << ", draining delayed frames until offset "
                  << stop_state.stop_collect_until_offset << std::endl;
        return;
    }
}

void speech_maybe_log_interleaved_debug(int32_t offset, int32_t text_input,
                                        int32_t sampled_text_token,
                                        const std::vector<int32_t>& frame_codes) {
    if (offset <= 0 || offset > 5)
        return;
    std::cerr << "[speech] Offset " << offset << " text_in=" << text_input
              << " text_out=" << sampled_text_token << " depth:";
    for (int32_t cb = 0; cb < std::min(4, static_cast<int32_t>(frame_codes.size())); ++cb)
        std::cerr << " " << frame_codes[static_cast<std::size_t>(cb)];
    std::cerr << "..." << std::endl;
}

void speech_log_output_frames_debug(const std::vector<int32_t>& output_codes,
                                    int32_t generated_frames, int32_t mimi_cb) {
    if (output_codes.empty())
        return;
    for (int32_t frame = 0; frame < generated_frames; ++frame) {
        std::cerr << "[speech] Output frame " << frame << ":";
        for (int32_t cb = 0; cb < mimi_cb; ++cb) {
            const auto idx = static_cast<std::size_t>(frame) * mimi_cb + cb;
            if (idx < output_codes.size())
                std::cerr << " " << output_codes[idx];
        }
        std::cerr << std::endl;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// speak(): Full speech-to-speech pipeline
// ---------------------------------------------------------------------------

bool SpeechPipeline::speak_validate_dual_stream() const {
    const bool has_audio_emb = !config_.audio_embeddings.empty() && config_.audio_vocab_size > 0;
    const bool has_input_embed = temporal_->has_input("input_embed");
    (void)temporal_->has_output("hidden_state");
    if (!has_audio_emb || !has_input_embed) {
        std::cerr << "[speech] ERROR: dual-stream requires audio_embeddings "
                     "and input_embed support"
                  << std::endl;
        return false;
    }
    return true;
}

void SpeechPipeline::speak_run_generation_loop(const SpeechGenerationSettings& settings,
                                               const SpeechOutputPlan& plan,
                                               DelayCacheState& delay_state,
                                               const std::vector<int32_t>& codec_tokens,
                                               std::vector<int32_t>& output_codes,
                                               int32_t& frames_collected,
                                               SpeechPerformanceTimings& timings) {
    const int32_t hidden = settings.hidden;
    SpeechDecodeStopState stop_state;
    speech_log_stop_configuration(config_, plan.extra_tail);

    std::vector<float> summed_embed(static_cast<std::size_t>(hidden));
    std::vector<int32_t> moshi_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> user_input(static_cast<std::size_t>(settings.stream_cb));
    std::vector<int32_t> target_audio_tokens(static_cast<std::size_t>(settings.num_cb));
    std::vector<uint8_t> target_audio_provided(static_cast<std::size_t>(settings.num_cb));
    std::vector<int32_t> selected_frame_tokens;
    std::vector<int32_t> frame_codes(static_cast<std::size_t>(settings.num_cb));
    CudaStageTimer temporal_timer;
    CudaStageTimer depth_timer;

    frames_collected = 0;
    for (int32_t offset = 0; offset < plan.total_iters && frames_collected < plan.output_frames;
         ++offset) {
        write_user_tokens_to_delay_cache(delay_state, codec_tokens, offset, settings.stream_cb,
                                         settings.num_frames, settings.encode_codebooks,
                                         settings.audio_bos);
        fill_initial_delay_tokens(delay_state, offset, settings.text_bos, settings.audio_bos);
        if (offset == 0) {
            seed_delay_offset_zero(delay_state, settings.text_bos, settings.audio_bos);
            continue;
        }

        const int32_t model_input_pos = offset - 1;
        const int32_t target_pos = offset;

        int32_t text_input = settings.text_pad_id;
        read_model_inputs_from_delay_cache(delay_state, model_input_pos, settings.stream_cb,
                                           text_input, moshi_input, user_input);
        compute_dual_stream_summed_embed(config_, settings.hidden, settings.stream_cb,
                                         moshi_input.data(), user_input.data(), text_input,
                                         summed_embed.data());

        temporal_timer.start(stream_);
        const auto temporal_output = run_temporal_embed_step(summed_embed.data(), settings.hidden);
        temporal_timer.stop(stream_);
        const auto text_target_idx = delay_cache_index(delay_state, 0, target_pos);
        const bool text_provided = delay_state.provided[text_target_idx] != 0;
        const int32_t forced_text_token = delay_state.cache[text_target_idx];

        build_target_audio_arrays(delay_state, target_pos, settings.num_cb, settings.audio_bos,
                                  target_audio_tokens, target_audio_provided);
        depth_timer.start(stream_);
        run_depth(temporal_output, forced_text_token, text_provided, target_audio_tokens.data(),
                  target_audio_provided.data());
        depth_timer.stop(stream_);
        download_selected_frame_tokens(selected_frame_tokens);
        timings.temporal_ms += temporal_timer.elapsed_ms();
        timings.depth_ms += depth_timer.elapsed_ms();
        const int32_t sampled_text_token = selected_frame_tokens[0];
        std::copy_n(selected_frame_tokens.begin() + 1, settings.num_cb, frame_codes.begin());

        clear_provided_flags_at_pos(delay_state, model_input_pos);
        write_generated_tokens_to_delay_cache(delay_state, target_pos, sampled_text_token,
                                              text_provided, frame_codes, settings.num_cb);
        if (collect_output_codes_from_delay_cache(delay_state, offset, delay_state.max_delay,
                                                  settings.mimi_cb, output_codes)) {
            ++frames_collected;
        }

        SpeechDecodeStopInput stop_input;
        stop_input.text_eos_token_id = config_.text_eos_token_id;
        stop_input.text_padding_id = config_.text_padding_id;
        stop_input.effective_frames = plan.effective_frames;
        stop_input.extra_tail = plan.extra_tail;
        stop_input.target_pos = target_pos;
        stop_input.sampled_text_token = sampled_text_token;
        stop_input.offset = offset;
        stop_input.max_delay = delay_state.max_delay;
        stop_input.text_provided = text_provided;
        const auto stop_decision = UpdateSpeechDecodeStopState(stop_state, stop_input);
        stop_state = stop_decision.state;
        speech_maybe_log_stop_decision(stop_decision.reason, stop_state, offset);
        speech_maybe_log_interleaved_debug(offset, text_input, sampled_text_token, frame_codes);
        if (stop_decision.should_break)
            break;
    }
}

void SpeechPipeline::speak_postprocess_waveform(std::vector<float>& waveform,
                                                int32_t generated_frames) const {
    const auto trim_result = trim_speech_waveform_to_generated_frames(
        config_.sample_rate, config_.frame_rate, generated_frames, waveform);
    if (trim_result.trimmed) {
        std::cerr << "[speech] Trimmed decoded waveform to " << trim_result.expected_samples
                  << " samples (" << generated_frames << " generated frames)" << std::endl;
    }

    const auto normalize_result = peak_normalize_speech_waveform(waveform);
    if (normalize_result.normalized) {
        std::cerr << "[speech] Peak-normalized: peak=" << normalize_result.peak
                  << " scale=" << normalize_result.scale << std::endl;
    }
}

AudioResult SpeechPipeline::speak(const float* audio_in, int32_t num_samples,
                                  const SpeechToSpeechConfig& cfg, int32_t input_sample_rate) {
    using Clock = std::chrono::steady_clock;
    const auto total_started = Clock::now();
    SpeechPerformanceTimings timings;
    AudioResult result;
    result.sample_rate = config_.sample_rate;

    const int32_t max_output_frames = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 375;

    // Resample if the input sample rate differs from the model's expected rate.
    const float* samples_ptr = audio_in;
    int32_t samples_count = num_samples;
    std::vector<float> resampled_buf;
    const int32_t target_rate = config_.sample_rate;

    if (input_sample_rate > 0 && target_rate > 0 && input_sample_rate != target_rate) {
        std::cerr << "[speech] Resampling audio from " << input_sample_rate << " Hz to "
                  << target_rate << " Hz" << std::endl;
        resampled_buf = resample_linear(audio_in, num_samples, input_sample_rate, target_rate);
        samples_ptr = resampled_buf.data();
        samples_count = static_cast<int32_t>(resampled_buf.size());
        input_sample_rate = target_rate;
    }

    std::cerr << "[speech] Starting pipeline with " << samples_count << " input samples"
              << std::endl;

    speech_log_depth_mode(config_);

    // Stage 1: Encode input audio via Mimi
    const auto encode_started = Clock::now();
    auto codec_tokens = run_mimi_encode(samples_ptr, samples_count);
    timings.codec_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - encode_started).count();

    const auto encoder_shape = resolve_encoder_shape_without_engine(
        config_, last_encode_codebooks_, last_encode_frames_, codec_tokens.size());
    const int32_t num_frames = encoder_shape.num_frames;

    std::cerr << "[speech] Encoder output: " << codec_tokens.size() << " tokens = " << num_frames
              << " frames x " << encoder_shape.encode_codebooks << " codebooks" << std::endl;

    if (num_frames <= 0) {
        std::cerr << "[speech] Encoder produced no frames" << std::endl;
        return result;
    }

    if (!speak_validate_dual_stream())
        return result;

    temporal_state_->reset();

    if (should_run_text_prompt_injection(config_))
        run_text_prompt();

    const int32_t num_cb = config_.num_codebooks;
    const int32_t hidden = config_.temporal_hidden_size;
    auto delay_state = make_delay_cache_state(config_.delays, num_cb);
    SpeechOutputPlanInput plan_input;
    plan_input.sample_rate = config_.sample_rate;
    plan_input.frame_rate = config_.frame_rate;
    plan_input.num_frames = num_frames;
    plan_input.num_input_samples = samples_count;
    plan_input.input_sample_rate = input_sample_rate;
    plan_input.tail_frames = cfg.tail_frames;
    plan_input.max_output_frames = max_output_frames;
    plan_input.max_delay = delay_state.max_delay;
    const auto plan = ComputeSpeechOutputPlan(plan_input);
    const int32_t mimi_cb = config_.mimi_decode_codebooks;
    std::vector<int32_t> output_codes;
    output_codes.reserve(static_cast<std::size_t>(mimi_cb) * plan.output_frames);

    std::cerr << "[speech] Interleaved temporal+depth with delay pattern: " << plan.output_frames
              << " output frames, " << plan.total_iters
              << " total iterations (max_delay=" << delay_state.max_delay
              << ", input_effective=" << plan.effective_frames
              << ", tail_frames=" << plan.extra_tail << ")" << std::endl;

    const SpeechGenerationSettings settings =
        make_speech_generation_settings(config_, hidden, encoder_shape);

    int32_t frames_collected = 0;
    speak_run_generation_loop(settings, plan, delay_state, codec_tokens, output_codes,
                              frames_collected, timings);

    const int32_t generated_frames = frames_collected;
    std::cerr << "[speech] Depth: generated " << generated_frames << " frames x " << num_cb
              << " codebooks (decoding first " << mimi_cb << ")" << std::endl;
    speech_log_output_frames_debug(output_codes, generated_frames, mimi_cb);

    // Stage 4: Decode output tokens to audio via Mimi decoder
    const auto decode_started = Clock::now();
    auto waveform = run_mimi_decode(output_codes, generated_frames);
    timings.codec_ms +=
        std::chrono::duration<double, std::milli>(Clock::now() - decode_started).count();
    speak_postprocess_waveform(waveform, generated_frames);

    result.samples = std::move(waveform);
    result.num_samples = static_cast<int32_t>(result.samples.size());
    std::cerr << "[speech] Generated " << result.num_samples << " samples ("
              << static_cast<float>(result.num_samples) / result.sample_rate << "s @ "
              << result.sample_rate << " Hz)" << std::endl;
    const double total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - total_started).count();
    std::cerr << "[trtmc.personaplex_timing] total_ms=" << total_ms
              << " temporal_ms=" << timings.temporal_ms << " depth_ms=" << timings.depth_ms
              << " codec_ms=" << timings.codec_ms << " host_ms=" << timings.host_ms(total_ms)
              << " frames=" << generated_frames << std::endl;
    return result;
}

} // namespace trtmc
