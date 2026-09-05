/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/pipeline.h"

#include "families/qwen3_omni/runtime/audio_plan.h"
#include "families/qwen3_omni/runtime/kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

constexpr const char* kSystemPrompt =
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech.";

std::string shape_text(const std::vector<std::int64_t>& shape) {
    std::string result{"["};
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0)
            result += ',';
        result += std::to_string(shape[index]);
    }
    result += ']';
    return result;
}

const char* dtype_text(DType dtype) {
    switch (dtype) {
    case DType::kFloat32:
        return "float32";
    case DType::kFloat16:
        return "float16";
    case DType::kBFloat16:
        return "bfloat16";
    case DType::kInt32:
        return "int32";
    case DType::kInt8:
        return "int8";
    }
    return "unknown";
}

std::string thinker_prompt(const std::string& prompt) {
    return std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
           prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string talker_prompt(const std::string& prompt, const std::string& assistant) {
    return std::string("<|im_start|>system\n") + kSystemPrompt + "<|im_end|>\n<|im_start|>user\n" +
           prompt + "<|im_end|>\n<|im_start|>assistant\n" + assistant + "<|im_end|>";
}

std::vector<float> copy_float_output(const TensorMap& outputs, const std::string& name,
                                     const std::vector<std::int64_t>& expected_shape = {}) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("Qwen3-Omni engine has no '" + name + "' output");
    const Tensor& tensor = found->second;
    if (tensor.dtype != DType::kFloat32 || tensor.data == nullptr || tensor.numel() == 0 ||
        (!expected_shape.empty() && tensor.shape != expected_shape))
        throw std::runtime_error("Qwen3-Omni engine output '" + name + "' is not float32 data");
    std::vector<float> result(tensor.numel());
    std::memcpy(result.data(), tensor.data, tensor.nbytes());
    if (!std::all_of(result.begin(), result.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::runtime_error("Qwen3-Omni engine output '" + name + "' is not finite");
    return result;
}

void collect_prefill_kv(ITrtModule& module, const TensorMap& outputs, std::int32_t layers,
                        std::int32_t sequence, std::vector<const void*>& keys,
                        std::vector<const void*>& values) {
    keys.reserve(static_cast<std::size_t>(layers));
    values.reserve(static_cast<std::size_t>(layers));
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        const std::string suffix = "_" + std::to_string(layer);
        const std::string cache_k_name = "cache_k" + suffix;
        const std::string cache_v_name = "cache_v" + suffix;
        const std::string present_k_name = "present_k" + suffix;
        const std::string present_v_name = "present_v" + suffix;
        const auto cache_k_shape = module.tensor_shape(cache_k_name);
        const auto cache_v_shape = module.tensor_shape(cache_v_name);
        const auto cache_k_dtype = module.tensor_dtype(cache_k_name);
        const auto cache_v_dtype = module.tensor_dtype(cache_v_name);
        const auto present_k_it = outputs.find(present_k_name);
        const auto present_v_it = outputs.find(present_v_name);
        if (present_k_it == outputs.end() || present_v_it == outputs.end()) {
            throw std::runtime_error(
                "Qwen3-Omni prefill did not return runtime KV tensors for layer " +
                std::to_string(layer));
        }
        const Tensor& present_k = present_k_it->second;
        const Tensor& present_v = present_v_it->second;
        const bool valid =
            cache_k_shape.size() == 2 && cache_k_shape[1] > 0 && cache_v_shape == cache_k_shape &&
            present_k.shape == std::vector<std::int64_t>{sequence, cache_k_shape[1]} &&
            present_v.shape == present_k.shape && cache_v_dtype == cache_k_dtype &&
            present_k.dtype == cache_k_dtype && present_v.dtype == cache_k_dtype;
        if (!valid) {
            throw std::runtime_error(
                "Qwen3-Omni prefill KV contract mismatch at layer " + std::to_string(layer) +
                ": sequence=" + std::to_string(sequence) +
                ", cache_k=" + shape_text(cache_k_shape) + "/" + dtype_text(cache_k_dtype) +
                ", cache_v=" + shape_text(cache_v_shape) + "/" + dtype_text(cache_v_dtype) +
                ", runtime_present_k=" + shape_text(present_k.shape) + "/" +
                dtype_text(present_k.dtype) + ", runtime_present_v=" + shape_text(present_v.shape) +
                "/" + dtype_text(present_v.dtype));
        }
        const void* key = module.device_ptr(present_k_name);
        const void* value = module.device_ptr(present_v_name);
        if (key == nullptr || value == nullptr) {
            throw std::runtime_error("Qwen3-Omni prefill is missing KV output for layer " +
                                     std::to_string(layer));
        }
        keys.push_back(key);
        values.push_back(value);
    }
}

std::int32_t argmax(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::runtime_error("Qwen3-Omni cannot select from empty logits");
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

void append_row(std::vector<float>& destination, const float* row, std::int32_t width) {
    destination.insert(destination.end(), row, row + width);
}

void add_row(std::vector<float>& destination, const float* row, std::int32_t width) {
    if (destination.size() != static_cast<std::size_t>(width))
        throw std::runtime_error("Qwen3-Omni embedding accumulator has an invalid width");
    for (std::int32_t index = 0; index < width; ++index)
        destination[static_cast<std::size_t>(index)] += row[index];
}

void round_bfloat16(std::vector<float>& values) {
    for (float& value : values)
        value = qwen3_omni_round_bfloat16(value);
}

std::string clean_assistant_text(std::string text) {
    for (const std::string marker : {"<|im_end|>", "<|endoftext|>"}) {
        const auto position = text.find(marker);
        if (position != std::string::npos)
            text.erase(position);
    }
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

} // namespace

Qwen3OmniAudioPipeline::Qwen3OmniAudioPipeline(
    std::unique_ptr<ITrtModule> thinker_prefill, std::unique_ptr<ITrtModule> thinker_decode,
    std::unique_ptr<Qwen3OmniKvCache> thinker_state, std::unique_ptr<ITrtModule> text_projection,
    std::unique_ptr<ITrtModule> talker_prefill, std::unique_ptr<ITrtModule> talker_decode,
    std::unique_ptr<Qwen3OmniKvCache> talker_state, std::unique_ptr<ITrtModule> predictor_prefill,
    std::unique_ptr<ITrtModule> predictor_decode, std::unique_ptr<Qwen3OmniKvCache> predictor_state,
    std::unique_ptr<ITrtModule> code2wav, std::vector<float> talker_codec_embedding,
    std::vector<float> predictor_codec_embeddings, Qwen3OmniRuntimeConfig config,
    std::shared_ptr<ITokenizer> tokenizer)
    : thinker_prefill_(std::move(thinker_prefill)), thinker_decode_(std::move(thinker_decode)),
      thinker_state_(std::move(thinker_state)), text_projection_(std::move(text_projection)),
      talker_prefill_(std::move(talker_prefill)), talker_decode_(std::move(talker_decode)),
      talker_state_(std::move(talker_state)), predictor_prefill_(std::move(predictor_prefill)),
      predictor_decode_(std::move(predictor_decode)), predictor_state_(std::move(predictor_state)),
      code2wav_(std::move(code2wav)), talker_codec_embedding_(std::move(talker_codec_embedding)),
      predictor_codec_embeddings_(std::move(predictor_codec_embeddings)),
      config_(std::move(config)), tokenizer_(std::move(tokenizer)) {
    if (!thinker_prefill_ || !thinker_decode_ || !thinker_state_ || !text_projection_ ||
        !talker_prefill_ || !talker_decode_ || !talker_state_ || !predictor_prefill_ ||
        !predictor_decode_ || !predictor_state_ || !code2wav_ || !tokenizer_) {
        throw std::invalid_argument("Qwen3-Omni native pipeline is missing a required component");
    }
    if (!thinker_prefill_->ok() || !thinker_decode_->ok() || !text_projection_->ok() ||
        !talker_prefill_->ok() || !talker_decode_->ok() || !predictor_prefill_->ok() ||
        !predictor_decode_->ok() || !code2wav_->ok() || !thinker_state_->ok() ||
        !talker_state_->ok() || !predictor_state_->ok()) {
        throw std::invalid_argument("Qwen3-Omni native pipeline has an invalid component");
    }
    if (config_.num_codebooks != 16 || config_.code2wav_num_quantizers != 16 ||
        config_.predictor_vocab_size != config_.codebook_size ||
        config_.talker_hidden_size != config_.predictor_hidden_size) {
        throw std::invalid_argument("Qwen3-Omni runtime dimensions do not match its codec");
    }
    if (!qwen3_omni_embedding_section_size(talker_codec_embedding_.size() * sizeof(float), 1,
                                           config_.talker_vocab_size, config_.talker_hidden_size) ||
        !qwen3_omni_embedding_section_size(predictor_codec_embeddings_.size() * sizeof(float),
                                           config_.num_codebooks - 1, config_.predictor_vocab_size,
                                           config_.predictor_hidden_size)) {
        throw std::invalid_argument("Qwen3-Omni codec embedding sections have invalid sizes");
    }
}

Qwen3OmniAudioPipeline::DecoderOutput
Qwen3OmniAudioPipeline::run_token_prefill(const std::vector<std::int32_t>& token_ids) {
    if (token_ids.empty() ||
        token_ids.size() > static_cast<std::size_t>(thinker_state_->max_length())) {
        throw std::runtime_error("Qwen3-Omni Thinker prompt exceeds its prefill profile");
    }
    Qwen3OmniKvCache* cache = thinker_state_.get();
    thinker_state_->reset();
    thinker_state_->bind_to(*thinker_decode_);
    cache->bind_cache_inputs(*thinker_prefill_);
    const auto sequence = static_cast<std::int32_t>(token_ids.size());
    TensorMap inputs;
    inputs["token_id"] =
        Tensor{const_cast<std::int32_t*>(token_ids.data()), {sequence}, DType::kInt32};
    thinker_state_->prepare_step(inputs, sequence);
    const TensorMap outputs = thinker_prefill_->forward(inputs);
    std::vector<const void*> keys;
    std::vector<const void*> values;
    collect_prefill_kv(*thinker_prefill_, outputs, thinker_state_->num_layers(), sequence, keys,
                       values);
    cache->write_prefill_kv(keys, values, sequence);
    thinker_state_->bind_to(*thinker_decode_);
    return {copy_float_output(outputs, "logits", {1, config_.thinker_vocab_size}), {}};
}

Qwen3OmniAudioPipeline::DecoderOutput Qwen3OmniAudioPipeline::run_embed_prefill(
    ITrtModule& module, Qwen3OmniKvCache& state, const std::vector<float>& embeddings,
    std::int32_t hidden_size, const std::string& logits_name, std::int32_t logits_size) {
    if (hidden_size <= 0 || embeddings.empty() ||
        embeddings.size() % static_cast<std::size_t>(hidden_size) != 0) {
        throw std::runtime_error("Qwen3-Omni embedding prefill has an invalid shape");
    }
    const auto sequence =
        static_cast<std::int32_t>(embeddings.size() / static_cast<std::size_t>(hidden_size));
    if (sequence > state.max_length())
        throw std::runtime_error("Qwen3-Omni embedding prefill exceeds its cache capacity");
    Qwen3OmniKvCache* cache = &state;
    cache->bind_cache_inputs(module);
    TensorMap inputs;
    inputs["input_embed"] =
        Tensor{const_cast<float*>(embeddings.data()), {sequence, hidden_size}, DType::kFloat32};
    state.prepare_step(inputs, sequence);
    const TensorMap outputs = module.forward(inputs);
    std::vector<const void*> keys;
    std::vector<const void*> values;
    collect_prefill_kv(module, outputs, state.num_layers(), sequence, keys, values);
    cache->write_prefill_kv(keys, values, sequence);
    return {copy_float_output(outputs, logits_name, {1, logits_size}),
            copy_float_output(outputs, "hidden_state", {1, hidden_size})};
}

Qwen3OmniAudioPipeline::DecoderOutput
Qwen3OmniAudioPipeline::run_token_step(std::int32_t token_id) {
    if (thinker_state_->position() >= thinker_state_->max_length())
        throw std::runtime_error("Qwen3-Omni Thinker exhausted its KV cache");
    TensorMap inputs;
    inputs["token_id"] = Tensor{&token_id, {1}, DType::kInt32};
    thinker_state_->prepare_step(inputs);
    const TensorMap outputs = thinker_decode_->forward(inputs);
    thinker_state_->advance();
    return {copy_float_output(outputs, "logits", {1, config_.thinker_vocab_size}), {}};
}

Qwen3OmniAudioPipeline::DecoderOutput
Qwen3OmniAudioPipeline::run_embed_step(ITrtModule& module, Qwen3OmniKvCache& state,
                                       const float* embedding, std::int32_t hidden_size,
                                       const std::string& logits_name, std::int32_t logits_size) {
    if (embedding == nullptr)
        throw std::invalid_argument("Qwen3-Omni decode embedding is null");
    if (state.position() >= state.max_length())
        throw std::runtime_error("Qwen3-Omni decoder exhausted its KV cache");
    TensorMap inputs;
    inputs["input_embed"] =
        Tensor{const_cast<float*>(embedding), {1, hidden_size}, DType::kFloat32};
    state.prepare_step(inputs);
    const TensorMap outputs = module.forward(inputs);
    state.advance();
    return {copy_float_output(outputs, logits_name, {1, logits_size}),
            copy_float_output(outputs, "hidden_state", {1, hidden_size})};
}

std::vector<std::int32_t> Qwen3OmniAudioPipeline::run_thinker(const std::string& prompt,
                                                              std::int32_t max_new_tokens) {
    const auto prompt_ids = tokenizer_->encode(thinker_prompt(prompt));
    DecoderOutput output = run_token_prefill(prompt_ids);
    const std::int32_t endoftext_token = tokenizer_->id_for_token("<|endoftext|>");
    if (endoftext_token < 0)
        throw std::runtime_error("Qwen3-Omni tokenizer has no <|endoftext|> token");
    std::vector<std::int32_t> generated;
    generated.reserve(static_cast<std::size_t>(max_new_tokens));
    for (std::int32_t step = 0; step < max_new_tokens; ++step) {
        const std::int32_t token = argmax(output.logits);
        if (token == config_.thinker_eos_token_id || token == endoftext_token)
            break;
        generated.push_back(token);
        if (step + 1 < max_new_tokens)
            output = run_token_step(token);
    }
    return generated;
}

std::vector<float>
Qwen3OmniAudioPipeline::project_tokens(const std::vector<std::int32_t>& token_ids) {
    if (token_ids.empty())
        return {};
    TensorMap inputs;
    inputs["token_id"] = Tensor{const_cast<std::int32_t*>(token_ids.data()),
                                {static_cast<std::int64_t>(token_ids.size())},
                                DType::kInt32};
    return copy_float_output(
        text_projection_->forward(inputs), "embeddings",
        {static_cast<std::int64_t>(token_ids.size()), config_.talker_hidden_size});
}

Qwen3OmniAudioPipeline::TalkerInputs
Qwen3OmniAudioPipeline::prepare_talker_inputs(const std::string& prompt,
                                              const std::string& assistant_text) {
    std::vector<std::int32_t> ids = tokenizer_->encode(talker_prompt(prompt, assistant_text));
    if (ids.size() < 2)
        throw std::runtime_error("Qwen3-Omni Talker ChatML tokenization is empty");
    if (ids.back() != config_.thinker_eos_token_id)
        throw std::runtime_error("Qwen3-Omni Talker ChatML must end with <|im_end|>");
    std::vector<std::size_t> starts;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] == config_.im_start_token_id)
            starts.push_back(index);
    }
    starts.push_back(ids.size());
    std::size_t user_start = ids.size();
    std::size_t user_end = ids.size();
    std::size_t assistant_start = ids.size();
    std::size_t assistant_end = ids.size();
    std::int32_t system_segments = 0;
    std::int32_t user_segments = 0;
    std::int32_t assistant_segments = 0;
    for (std::size_t segment = 0; segment + 1 < starts.size(); ++segment) {
        const std::size_t start = starts[segment];
        const std::size_t end = starts[segment + 1];
        if (start + 1 >= ids.size())
            continue;
        const std::int32_t role = ids[start + 1];
        if (role == config_.system_token_id) {
            ++system_segments;
        } else if (role == config_.user_token_id) {
            ++user_segments;
            user_start = start;
            user_end = end;
        } else if (role == config_.assistant_token_id && segment + 2 == starts.size()) {
            ++assistant_segments;
            assistant_start = start;
            assistant_end = end;
        }
    }
    if (starts.size() != 4 || system_segments != 1 || user_segments != 1 ||
        assistant_segments != 1 || user_start >= user_end || assistant_start >= assistant_end) {
        throw std::runtime_error(
            "Qwen3-Omni text-only Talker requires one system, user, and assistant segment");
    }

    ids.pop_back();
    user_end = std::min(user_end, ids.size());
    assistant_end = std::min(assistant_end, ids.size());
    if (assistant_end < assistant_start + 4)
        throw std::runtime_error("Qwen3-Omni Talker assistant segment is too short");
    const std::vector<float> projected = project_tokens(ids);
    const auto row = [&](std::size_t token) {
        return projected.data() + token * static_cast<std::size_t>(config_.talker_hidden_size);
    };
    const std::vector<std::int32_t> special_ids = {
        config_.tts_bos_token_id, config_.tts_eos_token_id, config_.tts_pad_token_id};
    const std::vector<float> specials = project_tokens(special_ids);
    const float* tts_bos = specials.data();
    const float* tts_eos = tts_bos + config_.talker_hidden_size;
    const float* tts_pad = tts_eos + config_.talker_hidden_size;

    TalkerInputs result;
    for (std::size_t token = user_start; token < user_end; ++token)
        append_row(result.initial, row(token), config_.talker_hidden_size);
    for (std::size_t token = assistant_start; token < assistant_start + 3; ++token)
        append_row(result.initial, row(token), config_.talker_hidden_size);
    const std::int32_t codec_special_ids[] = {
        config_.codec_nothink_id, config_.codec_think_bos_id, config_.codec_think_eos_id,
        config_.speaker_id,       config_.codec_pad_id,       config_.codec_bos_id,
    };
    for (std::size_t index = 0; index < 6; ++index) {
        const float* text = index < 4 ? tts_pad : index == 4 ? tts_bos : row(assistant_start + 3);
        const float* codec = talker_embedding_row(codec_special_ids[index]);
        for (std::int32_t column = 0; column < config_.talker_hidden_size; ++column)
            result.initial.push_back(qwen3_omni_round_bfloat16(text[column] + codec[column]));
    }
    for (std::size_t token = assistant_start + 4; token < assistant_end; ++token) {
        append_row(result.trailing, row(token), config_.talker_hidden_size);
        ++result.trailing_rows;
    }
    append_row(result.trailing, tts_eos, config_.talker_hidden_size);
    ++result.trailing_rows;
    result.pad.assign(tts_pad, tts_pad + config_.talker_hidden_size);
    return result;
}

const float* Qwen3OmniAudioPipeline::talker_embedding_row(std::int32_t token_id) const {
    if (token_id < 0 || token_id >= config_.talker_vocab_size)
        throw std::runtime_error("Qwen3-Omni Talker codec token is out of range");
    return talker_codec_embedding_.data() +
           static_cast<std::size_t>(token_id) * config_.talker_hidden_size;
}

const float* Qwen3OmniAudioPipeline::predictor_embedding_row(std::int32_t group,
                                                             std::int32_t token_id) const {
    if (group < 0 || group >= config_.num_codebooks - 1 || token_id < 0 ||
        token_id >= config_.predictor_vocab_size) {
        throw std::runtime_error("Qwen3-Omni residual codec token is out of range");
    }
    const auto table_stride =
        static_cast<std::size_t>(config_.predictor_vocab_size) * config_.predictor_hidden_size;
    return predictor_codec_embeddings_.data() + static_cast<std::size_t>(group) * table_stride +
           static_cast<std::size_t>(token_id) * config_.predictor_hidden_size;
}

std::vector<std::int32_t> Qwen3OmniAudioPipeline::run_code_predictor(
    const std::vector<float>& talker_hidden, std::int32_t coarse_code,
    std::vector<float>& next_embedding, qwen3_omni::ResidualCodeSampler& sampler) {
    if (talker_hidden.size() != static_cast<std::size_t>(config_.predictor_hidden_size))
        throw std::runtime_error("Qwen3-Omni Talker hidden state has an invalid width");
    const float* coarse_embedding = talker_embedding_row(coarse_code);
    std::vector<float> prefill = talker_hidden;
    append_row(prefill, coarse_embedding, config_.predictor_hidden_size);
    predictor_state_->reset();
    predictor_state_->bind_to(*predictor_decode_);
    DecoderOutput output =
        run_embed_prefill(*predictor_prefill_, *predictor_state_, prefill,
                          config_.predictor_hidden_size, "logits_0", config_.predictor_vocab_size);
    predictor_state_->bind_to(*predictor_decode_);

    std::vector<std::int32_t> codes;
    codes.reserve(static_cast<std::size_t>(config_.num_codebooks));
    codes.push_back(coarse_code);
    std::int32_t residual = sampler.sample(output.logits.data(), output.logits.size());
    codes.push_back(residual);
    next_embedding.assign(coarse_embedding, coarse_embedding + config_.talker_hidden_size);
    add_row(next_embedding, predictor_embedding_row(0, residual), config_.predictor_hidden_size);
    for (std::int32_t group = 1; group < config_.num_codebooks - 1; ++group) {
        output = run_embed_step(*predictor_decode_, *predictor_state_,
                                predictor_embedding_row(group - 1, residual),
                                config_.predictor_hidden_size, "logits_" + std::to_string(group),
                                config_.predictor_vocab_size);
        residual = sampler.sample(output.logits.data(), output.logits.size());
        codes.push_back(residual);
        add_row(next_embedding, predictor_embedding_row(group, residual),
                config_.predictor_hidden_size);
    }
    if (codes.size() != static_cast<std::size_t>(config_.num_codebooks))
        throw std::runtime_error("Qwen3-Omni CodePredictor returned an incomplete frame");
    round_bfloat16(next_embedding);
    return codes;
}

std::vector<std::int32_t> Qwen3OmniAudioPipeline::run_talker(
    const TalkerInputs& inputs, qwen3_omni::ResidualCodeSampler& sampler, std::int32_t max_frames) {
    talker_state_->reset();
    talker_state_->bind_to(*talker_decode_);
    DecoderOutput output =
        run_embed_prefill(*talker_prefill_, *talker_state_, inputs.initial,
                          config_.talker_hidden_size, "logits", config_.talker_vocab_size);
    talker_state_->bind_to(*talker_decode_);
    std::vector<std::int32_t> result;
    result.reserve(static_cast<std::size_t>(max_frames) * config_.num_codebooks);
    std::vector<std::int32_t> generated_coarse_codes;
    const std::int32_t frame_limit = qwen3_omni_audio_frame_limit(max_frames);
    generated_coarse_codes.reserve(static_cast<std::size_t>(frame_limit));
    for (std::int32_t frame = 0; frame < frame_limit; ++frame) {
        const std::int32_t coarse =
            qwen3_omni_talker_argmax(output.logits, config_.codebook_size,
                                     config_.codec_eos_token_id, generated_coarse_codes);
        if (coarse == config_.codec_eos_token_id)
            break;
        generated_coarse_codes.push_back(coarse);
        std::vector<float> next_embedding;
        const auto frame_codes = run_code_predictor(output.hidden, coarse, next_embedding, sampler);
        result.insert(result.end(), frame_codes.begin(), frame_codes.end());
        const float* text = frame < inputs.trailing_rows
                                ? inputs.trailing.data() +
                                      static_cast<std::size_t>(frame) * config_.talker_hidden_size
                                : inputs.pad.data();
        add_row(next_embedding, text, config_.talker_hidden_size);
        round_bfloat16(next_embedding);
        if (frame + 1 < frame_limit) {
            output =
                run_embed_step(*talker_decode_, *talker_state_, next_embedding.data(),
                               config_.talker_hidden_size, "logits", config_.talker_vocab_size);
        }
    }
    return result;
}

std::vector<float>
Qwen3OmniAudioPipeline::run_code2wav(const std::vector<std::int32_t>& frame_major_codes) {
    if (frame_major_codes.empty())
        return {};
    const auto frames = static_cast<std::int32_t>(frame_major_codes.size() /
                                                  static_cast<std::size_t>(config_.num_codebooks));
    std::vector<std::int32_t> codes = qwen3_omni_code2wav_input(
        frame_major_codes, config_.num_codebooks, config_.code2wav_max_frames);
    TensorMap inputs;
    inputs["codec_tokens"] = Tensor{
        codes.data(), {1, config_.num_codebooks, config_.code2wav_max_frames}, DType::kInt32};
    const TensorMap outputs = code2wav_->forward(inputs);
    const auto engine_samples =
        static_cast<std::int64_t>(config_.code2wav_max_frames) * config_.code2wav_upsample_factor -
        config_.code2wav_output_delay;
    std::vector<float> waveform = copy_float_output(outputs, "waveform", {1, 1, engine_samples});
    waveform.resize(qwen3_omni_output_samples(frames, config_.code2wav_upsample_factor,
                                              config_.code2wav_output_delay, waveform.size()));
    return waveform;
}

TextResult Qwen3OmniAudioPipeline::generate(const std::string& prompt,
                                            const TextGenerationConfig& config) {
    if (config.max_new_tokens <= 0 || config.temperature != 1.0F || config.top_k != 1 ||
        config.top_p != 1.0F || config.min_p != 0.0F || config.repetition_penalty != 1.0F ||
        config.use_chat_template || !config.enable_thinking || !config.lora_adapter_id.empty()) {
        throw std::invalid_argument(
            "Qwen3-Omni text generation supports only its fixed greedy decoding contract");
    }
    std::vector<std::int32_t> generated = run_thinker(prompt, config.max_new_tokens);
    if (generated.empty())
        throw std::runtime_error("Qwen3-Omni Thinker produced no text");
    std::string text = clean_assistant_text(tokenizer_->decode(generated));
    if (text.empty())
        throw std::runtime_error("Qwen3-Omni Thinker decoded to empty text");
    return TextResult{std::move(text), std::move(generated)};
}

AudioResult Qwen3OmniAudioPipeline::generate_audio(const std::string& prompt,
                                                   const AudioGenerationConfig& config) {
    const std::int32_t thinker_tokens = config.max_new_tokens > 0 ? config.max_new_tokens : 128;
    const std::int32_t talker_tokens =
        config.talker_max_new_tokens > 0 ? config.talker_max_new_tokens : config_.talker_max_frames;
    if (talker_tokens > config_.talker_max_frames)
        throw std::invalid_argument("Qwen3-Omni talker token limit exceeds the built engine");
    const std::vector<std::int32_t> generated = run_thinker(prompt, thinker_tokens);
    if (generated.empty())
        throw std::runtime_error("Qwen3-Omni Thinker produced no speakable text");
    const std::string assistant = clean_assistant_text(tokenizer_->decode(generated));
    if (assistant.empty())
        throw std::runtime_error("Qwen3-Omni Thinker decoded to empty text");
    const TalkerInputs talker_inputs = prepare_talker_inputs(prompt, assistant);
    qwen3_omni::ResidualCodeSampler sampler(
        static_cast<std::uint64_t>(config.seed >= 0 ? config.seed : 42));
    std::vector<std::int32_t> codes = run_talker(talker_inputs, sampler, talker_tokens);
    if (codes.empty())
        throw std::runtime_error("Qwen3-Omni Talker produced no codec frames");
    std::vector<float> waveform = run_code2wav(codes);
    if (waveform.empty())
        throw std::runtime_error("Qwen3-Omni Code2Wav produced no samples");
    AudioResult result;
    result.samples = std::move(waveform);
    result.num_samples = static_cast<std::int32_t>(result.samples.size());
    result.sample_rate = config_.sample_rate;
    return result;
}

} // namespace trtmc
