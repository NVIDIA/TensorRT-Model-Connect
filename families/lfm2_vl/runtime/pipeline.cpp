/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm2_vl/runtime/pipeline.h"

#include "families/lfm2_vl/runtime/chat_templates.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string decode_response(const ITokenizer& tokenizer, const std::vector<int32_t>& ids,
                            bool cleanup) {
    std::string result;
    std::vector<int32_t> text;
    for (int32_t id : ids) {
        const auto token = tokenizer.token_for_id(id);
        if (token == "<|tool_call_start|>" || token == "<|tool_call_end|>" || token == "<think>" ||
            token == "</think>") {
            result += tokenizer.decode(text);
            text.clear();
            result += token;
        } else {
            text.push_back(id);
        }
    }
    result += tokenizer.decode(text);
    if (cleanup) {
        static constexpr std::array<std::pair<const char*, const char*>, 10> replacements{
            {{" .", "."},
             {" ?", "?"},
             {" !", "!"},
             {" ,", ","},
             {" ' ", "'"},
             {" n't", "n't"},
             {" 'm", "'m"},
             {" 's", "'s"},
             {" 've", "'ve"},
             {" 're", "'re"}}};
        for (const auto& [from, to] : replacements) {
            std::size_t position = 0;
            while ((position = result.find(from, position)) != std::string::npos) {
                result.replace(position, std::strlen(from), to);
                position += std::strlen(to);
            }
        }
    }
    return result;
}

void normalize_eos_ids(Lfm2VlTextGenConfig& config) {
    if (config.id_eos_ids.empty() && config.id_eos >= 0)
        config.id_eos_ids.push_back(config.id_eos);
    std::sort(config.id_eos_ids.begin(), config.id_eos_ids.end());
    config.id_eos_ids.erase(std::unique(config.id_eos_ids.begin(), config.id_eos_ids.end()),
                            config.id_eos_ids.end());
}

} // namespace

Lfm2VlTextGenerationPipeline::Lfm2VlTextGenerationPipeline(
    std::unique_ptr<ITrtModule> decoder, std::unique_ptr<Lfm2VlInferenceState> state,
    Lfm2VlTextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id, std::unique_ptr<ITrtModule> vision, std::vector<float> positions)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(std::move(config)),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)),
      vision_(std::move(vision)), position_table_(std::move(positions)),
      zero_embedding_(2048, 0.0F) {
    normalize_eos_ids(config_);
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error("Lfm2VlTextGenerationPipeline: invalid decoder module");
    if (!state_ || !state_->ok())
        throw std::runtime_error("Lfm2VlTextGenerationPipeline: invalid inference state");
    if (config_.vocab_size <= 0)
        throw std::invalid_argument("Lfm2VlTextGenerationPipeline: invalid vocabulary size");
    if (stream_ == nullptr || decoder_->stream() != stream_)
        throw std::invalid_argument("Lfm2VlTextGenerationPipeline: stream mismatch");
    state_->bind_to(*decoder_);
}

std::vector<int32_t>
Lfm2VlTextGenerationPipeline::encode_prompt(const std::string& prompt,
                                            const TextGenerationConfig& cfg) const {
    if (!tokenizer_)
        throw std::runtime_error("Lfm2VlTextGenerationPipeline: no native tokenizer configured");
    std::string effective = prompt;
    if (cfg.use_chat_template && !config_.chat_template_format.empty()) {
        effective = lfm2_vl_apply_chat_template(config_.chat_template_format, prompt);
    }
    auto ids = tokenizer_->encode(effective);

    // The official chat template begins with BOS. If tokenizer metadata also
    // prepends BOS, keep one copy for both rendered and caller-supplied prompts.
    if (ids.size() >= 2 && config_.id_bos >= 0 && ids[0] == config_.id_bos &&
        ids[1] == config_.id_bos) {
        ids.erase(ids.begin());
    }
    if (ids.empty() && config_.id_bos >= 0)
        ids.push_back(config_.id_bos);
    return ids;
}

TextResult Lfm2VlTextGenerationPipeline::generate(const std::string& prompt,
                                                  const TextGenerationConfig& cfg) {
    const auto input_ids = encode_prompt(prompt, cfg);
    const int32_t max_new_tokens = cfg.max_new_tokens > 0 ? cfg.max_new_tokens : 128;
    const auto params = lfm2_vl_sampling_params_from_config(cfg, config_.id_eos_ids);
    auto generated = generate_from_ids(input_ids, max_new_tokens, params);

    std::vector<int32_t> new_tokens(generated.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    generated.token_ids.end());
    TextResult result{
        decode_response(*tokenizer_, new_tokens, config_.clean_up_tokenization_spaces),
        std::move(new_tokens), generated.prefill_ms, generated.decode_ms};
    result.setup_ms = generated.setup_ms;
    return result;
}

TextResult Lfm2VlTextGenerationPipeline::generate(const std::string& prompt, const float* pixels,
                                                  std::int32_t height, std::int32_t width,
                                                  const TextGenerationConfig& cfg) {
    if (pixels == nullptr || height <= 0 || width <= 0)
        throw std::invalid_argument("LFM2-VL requires a nonempty image");
    Lfm2VlRequest request;
    request.prompt = prompt.find("<image>") == std::string::npos ? "<image>" + prompt : prompt;
    request.generation = cfg;
    ImageResult image;
    image.height = height;
    image.width = width;
    image.pixels.assign(pixels, pixels + static_cast<std::size_t>(height) * width * 3);
    for (auto& value : image.pixels) {
        if (!std::isfinite(value) || value < 0 || value > 1)
            throw std::invalid_argument("LFM2-VL Float32 RGB pixels must be normalized to [0,1]");
        value = std::nearbyint(value * 255.0F);
    }
    request.images.push_back(std::move(image));
    return generate(request);
}

TextResult Lfm2VlTextGenerationPipeline::generate(const Lfm2VlRequest& request) {
    if (!vision_ || !vision_->ok())
        throw std::runtime_error("LFM2-VL vision encoder is missing");
    lfm2_vl::ImageOptions options;
    if (request.min_image_tokens >= 0)
        options.min_tokens = request.min_image_tokens;
    if (request.max_image_tokens >= 0)
        options.max_tokens = request.max_image_tokens;
    if (request.max_tiles >= 0)
        options.max_tiles = request.max_tiles;
    if (request.split_images >= 0)
        options.split = request.split_images != 0;
    if (request.split_images < -1 || request.split_images > 1)
        throw std::invalid_argument("split_images must be -1, 0, or 1");
    image_embeddings_.clear();
    std::string expanded;
    std::size_t cursor = 0;
    if (!request.integer_pixels.empty() && request.integer_pixels.size() != request.images.size())
        throw std::invalid_argument("LFM2-VL image format count mismatch");
    std::size_t image_index = 0;
    for (const auto& image : request.images) {
        options.integer_pixels =
            request.integer_pixels.empty() || request.integer_pixels[image_index];
        ++image_index;
        const auto marker = request.prompt.find("<image>", cursor);
        if (marker == std::string::npos)
            throw std::invalid_argument("LFM2-VL requires one <image> placeholder per input image");
        expanded += request.prompt.substr(cursor, marker - cursor);
        auto processed = lfm2_vl::process_image(image, options);
        expanded += processed.placeholders;
        cursor = marker + 7;
        for (const auto& tile : processed.tiles) {
            auto prepared = lfm2_vl::prepare_vision_inputs(tile, options, position_table_, 1152,
                                                           config_.precision == "bf16");
            const auto count = prepared.valid_patches;
            TensorMap inputs;
            inputs["pixel_values"] =
                Tensor{prepared.pixels.data(), {prepared.patch_count, 768}, DType::kFloat32};
            inputs["position_embeddings"] =
                Tensor{prepared.positions.data(), {prepared.patch_count, 1152}, DType::kFloat32};
            inputs["pixel_attention_mask"] =
                Tensor{prepared.mask.data(), {prepared.patch_count}, DType::kInt32};
            inputs["unshuffle_indices"] = Tensor{
                const_cast<std::int32_t*>(tile.unshuffle.data()), {count / 4, 4}, DType::kInt32};
            auto outputs = vision_->forward(inputs);
            const auto& result = outputs.at("image_embeddings");
            if (result.dtype != DType::kFloat32)
                throw std::runtime_error("LFM2-VL vision output has invalid dtype");
            const auto* data = static_cast<const float*>(result.data);
            image_embeddings_.insert(image_embeddings_.end(), data, data + count / 4 * 2048);
        }
    }
    if (request.prompt.find("<image>", cursor) != std::string::npos)
        throw std::invalid_argument("LFM2-VL image placeholder has no corresponding input image");
    expanded += request.prompt.substr(cursor);
    try {
        auto result = generate(expanded, request.generation);
        image_embeddings_.clear();
        return result;
    } catch (...) {
        image_embeddings_.clear();
        throw;
    }
}

Lfm2VlTextGenerationPipeline::GenerationResult
Lfm2VlTextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                           const TextGenerationConfig& cfg) {
    const auto params = lfm2_vl_sampling_params_from_config(cfg, config_.id_eos_ids);
    return GenerationResult{generate_from_ids(input_ids, cfg.max_new_tokens, params).token_ids};
}

Lfm2VlTextGenerationPipeline::TimedGenerationResult
Lfm2VlTextGenerationPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                int32_t max_new_tokens,
                                                const Lfm2VlSamplingParams& params) {
    if (input_ids.empty() || max_new_tokens <= 0)
        return TimedGenerationResult{input_ids};
    const auto capacity = static_cast<std::size_t>(state_->max_length());
    if (input_ids.size() > capacity ||
        static_cast<std::size_t>(max_new_tokens) > capacity - input_ids.size()) {
        throw std::runtime_error(
            "LFM2-VL prompt and generation exceed the fixed native KV cache capacity");
    }

    auto local_sampler = create_lfm2_vl_sampler(params);
    auto* active_sampler = local_sampler.get();
    active_sampler->reset();

    const auto setup_start = Clock::now();
    image_offset_ = 0;
    state_->reset();
    decoder_->reset_execution_context();
    state_->bind_to(*decoder_);
    logits_device_ptr_ = nullptr;
    const auto setup_end = Clock::now();

    std::vector<float> logits;
    const auto prefill_start = Clock::now();
    // Safe v1 path: every prompt token commits native KV and rolling
    // convolution state together. A KV-only batched-prefill shortcut would
    // leave the convolution history stale.
    for (int32_t token : input_ids)
        run_step(token, logits);
    const auto prefill_end = Clock::now();

    std::vector<int32_t> output = input_ids;
    const auto decode_start = Clock::now();
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const auto sampled = active_sampler->sample(
            logits.data(), static_cast<int32_t>(logits.size()), params, output);
        output.push_back(sampled.token_id);
        if (sampled.is_eos || step + 1 >= max_new_tokens)
            break;
        run_step(sampled.token_id, logits, true);
    }
    const auto decode_end = Clock::now();

    return TimedGenerationResult{std::move(output), milliseconds(setup_start, setup_end),
                                 milliseconds(prefill_start, prefill_end),
                                 milliseconds(decode_start, decode_end)};
}

void Lfm2VlTextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits,
                                            bool decode) {
    TensorMap inputs;
    inputs[config_.token_id_name] = Tensor{&token_id, {1}, DType::kInt32};
    std::int32_t phase = decode;
    if (decoder_->has_input("decode_phase"))
        inputs["decode_phase"] = Tensor{&phase, {1, 1}, DType::kInt32};
    state_->prepare_step(inputs);
    const std::int32_t use_image = token_id == 396 && !image_embeddings_.empty();
    const float* embedding = zero_embedding_.data();
    if (use_image) {
        if (image_offset_ + 2048 > image_embeddings_.size())
            throw std::runtime_error("LFM2-VL image placeholders exceed image features");
        embedding = image_embeddings_.data() + image_offset_;
        image_offset_ += 2048;
    }
    inputs["image_embedding"] = Tensor{const_cast<float*>(embedding), {1, 2048}, DType::kFloat32};
    inputs["use_image"] = Tensor{const_cast<std::int32_t*>(&use_image), {1, 1}, DType::kInt32};

    // Every persistent state output is device-sized. Never call forward(),
    // which materializes non-external outputs on the host; execute asynchronously
    // and transfer only the one-vocabulary logits row required for sampling.
    decoder_->forward_async(inputs);
    decoder_->sync();
    if (logits_device_ptr_ == nullptr) {
        if (!decoder_->has_output(config_.logits_output_name) ||
            decoder_->tensor_dtype(config_.logits_output_name) != DType::kFloat32) {
            throw std::runtime_error("Lfm2VlTextGenerationPipeline: FP32 logits output is missing");
        }
        const auto shape = decoder_->tensor_shape(config_.logits_output_name);
        const std::vector<int64_t> vector_shape{config_.vocab_size};
        const std::vector<int64_t> row_shape{1, config_.vocab_size};
        if (shape != vector_shape && shape != row_shape) {
            throw std::runtime_error(
                "Lfm2VlTextGenerationPipeline: logits shape must be [vocab] or [1,vocab]");
        }
        logits_device_ptr_ =
            static_cast<const float*>(decoder_->device_ptr(config_.logits_output_name));
        if (logits_device_ptr_ == nullptr)
            throw std::runtime_error(
                "Lfm2VlTextGenerationPipeline: logits device buffer is missing");
    }
    logits.resize(static_cast<std::size_t>(config_.vocab_size));
    if (cudaMemcpy(logits.data(), logits_device_ptr_, logits.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("Lfm2VlTextGenerationPipeline: logits D2H copy failed");
    }

    state_->advance();
}

} // namespace trtmc
