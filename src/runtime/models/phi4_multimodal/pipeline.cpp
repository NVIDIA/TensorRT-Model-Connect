/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/phi4_multimodal/pipeline.h"

#include "runtime/models/phi4_multimodal/image_preprocessor.h"
#include "runtime/models/phi4_multimodal/tensor_names.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

namespace {

void validate_pipeline_components(const TrtModule* text_decoder,
                                  const Phi4MultimodalInferenceState* state,
                                  const TrtModule* prefill) {
    if (text_decoder == nullptr || !text_decoder->ok())
        throw std::runtime_error("Phi4MultimodalPipeline: invalid text decoder");
    if (state == nullptr || !state->ok())
        throw std::runtime_error("Phi4MultimodalPipeline: invalid inference state");
    if (prefill == nullptr || !prefill->ok())
        throw std::runtime_error("Phi4MultimodalPipeline: invalid prefill decoder");
}

void sync_vision_config(Phi4MultimodalConfig& config,
                        const Phi4MultimodalPreprocessConfig& preprocess) {
    if (config.image_token_id < 0 && preprocess.image_token_id >= 0)
        config.image_token_id = preprocess.image_token_id;
    if (config.vision_output_dim <= 0 && preprocess.vision_output_dim > 0)
        config.vision_output_dim = preprocess.vision_output_dim;
}

} // namespace

Phi4MultimodalPipeline::Phi4MultimodalPipeline(
    std::unique_ptr<TrtModule> text_decoder, std::unique_ptr<TrtModule> vision_encoder,
    std::unique_ptr<Phi4MultimodalInferenceState> state, Phi4MultimodalConfig config,
    Phi4MultimodalPreprocessConfig vl_preprocess, cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
    std::unique_ptr<Phi4MultimodalISampler> sampler, std::unique_ptr<TrtModule> prefill)
    : text_decoder_(std::move(text_decoder)), prefill_(std::move(prefill)),
      vision_encoder_(std::move(vision_encoder)), state_(std::move(state)), config_(config),
      vl_preprocess_(std::move(vl_preprocess)), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), sampler_(std::move(sampler)) {
    validate_pipeline_components(text_decoder_.get(), state_.get(), prefill_.get());
    sync_vision_config(config_, vl_preprocess_);
}

TextResult Phi4MultimodalPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("Phi4MultimodalPipeline: no tokenizer configured");

    auto input_ids = tokenizer_->encode(prompt);
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp = phi4_multimodal_sampling_params_from_config(cfg, eos);
    auto output_ids = generate_from_ids(input_ids, max_new, sp);

    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()), output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    auto result = TextResult{std::move(text), std::move(new_tokens)};
    result.setup_ms = last_setup_ms_;
    return result;
}

namespace {

runtime::adapters::io::DecodedImage convert_float_to_decoded(const float* pixels, int32_t height,
                                                             int32_t width) {
    runtime::adapters::io::DecodedImage decoded;
    decoded.width = width;
    decoded.height = height;
    decoded.channels = 3;
    auto n = static_cast<std::size_t>(width) * height * 3;
    decoded.pixels.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        float v = std::max(0.0F, std::min(255.0F, pixels[i] * 255.0F));
        decoded.pixels[i] = static_cast<uint8_t>(v + 0.5F);
    }
    return decoded;
}

int32_t infer_feature_dim(const TrtModule& encoder, int32_t configured_dim) {
    if (configured_dim > 0)
        return configured_dim;
    for (const auto& info : encoder.output_info()) {
        if (info.name == "image_features" && info.shape.size() >= 2)
            return static_cast<int32_t>(info.shape.back());
    }
    return 0;
}

struct VlSequenceEmbeddingInputs {
    std::vector<float> input_embed;
    std::vector<float> use_input_embed;
    std::vector<std::vector<float>> deepstack_embed;
    std::vector<float> deepstack_active;
};

VlSequenceEmbeddingInputs
build_vl_sequence_embedding_inputs(const std::vector<int32_t>& input_ids, int32_t image_token_id,
                                   const std::vector<float>& image_features,
                                   const std::vector<std::vector<float>>& deepstack_features,
                                   int32_t num_features, int32_t feature_dim,
                                   int32_t feature_start = 0) {
    const auto sq = input_ids.size();
    VlSequenceEmbeddingInputs result;
    result.input_embed.assign(sq * static_cast<std::size_t>(feature_dim), 0.0F);
    result.use_input_embed.assign(sq, 0.0F);
    result.deepstack_active.assign(sq, 0.0F);
    result.deepstack_embed.resize(deepstack_features.size());
    for (auto& level : result.deepstack_embed)
        level.assign(sq * static_cast<std::size_t>(feature_dim), 0.0F);

    int32_t feature_index = feature_start;
    for (std::size_t token_index = 0; token_index < sq; ++token_index) {
        if (input_ids[token_index] != image_token_id || feature_index >= num_features)
            continue;
        const auto source_offset = static_cast<std::size_t>(feature_index) * feature_dim;
        const auto target_offset = token_index * static_cast<std::size_t>(feature_dim);
        std::copy_n(image_features.data() + source_offset, feature_dim,
                    result.input_embed.data() + target_offset);
        result.use_input_embed[token_index] = 1.0F;

        for (std::size_t level_index = 0; level_index < deepstack_features.size(); ++level_index) {
            const auto& source = deepstack_features[level_index];
            if (source_offset + static_cast<std::size_t>(feature_dim) > source.size())
                continue;
            std::copy_n(source.data() + source_offset, feature_dim,
                        result.deepstack_embed[level_index].data() + target_offset);
            result.deepstack_active[token_index] = 1.0F;
        }
        ++feature_index;
    }
    return result;
}

bool gather_vl_prefill_kv_pointers(TrtModule& prefill, const Phi4MultimodalConfig& config,
                                   std::vector<const void*>& present_k,
                                   std::vector<const void*>& present_v) {
    present_k.resize(static_cast<std::size_t>(config.num_layers));
    present_v.resize(static_cast<std::size_t>(config.num_layers));
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        present_k[index] =
            prefill.device_ptr(phi4_multimodal_expand_layer_name(config.present_k_pattern, layer));
        present_v[index] =
            prefill.device_ptr(phi4_multimodal_expand_layer_name(config.present_v_pattern, layer));
        if (present_k[index] == nullptr || present_v[index] == nullptr)
            return false;
    }
    return true;
}

Phi4MultimodalKvCache& require_vl_prefill_cache(TrtModule* prefill,
                                                Phi4MultimodalInferenceState* state,
                                                const Phi4MultimodalConfig& config,
                                                int32_t feature_dim) {
    if (prefill == nullptr)
        throw std::runtime_error("Phi-4 Multimodal native bundle is missing the prefill engine");
    if (feature_dim <= 0 || config.num_layers <= 0 || config.prefill_max_length <= 0)
        throw std::runtime_error("Phi-4 Multimodal native prefill has invalid dimensions");
    if (!prefill->has_input("input_embed"))
        throw std::runtime_error("Phi-4 Multimodal native prefill is missing input_embed");
    auto* cache = dynamic_cast<Phi4MultimodalKvCache*>(state);
    if (cache == nullptr)
        throw std::runtime_error("Phi-4 Multimodal native prefill requires its native KV cache");
    return *cache;
}

bool valid_vl_features(const std::vector<float>& image_features, int32_t num_features,
                       int32_t feature_dim) {
    if (num_features < 0)
        return false;
    const auto required =
        static_cast<std::size_t>(num_features) * static_cast<std::size_t>(feature_dim);
    return image_features.size() >= required;
}

void add_vl_prefill_base_inputs(TensorMap& inputs, Phi4MultimodalInferenceState& state,
                                const std::vector<int32_t>& input_ids,
                                VlSequenceEmbeddingInputs& embedding_inputs, int32_t feature_dim) {
    const auto sequence_length = static_cast<int32_t>(input_ids.size());
    inputs["token_id"] =
        Tensor{const_cast<int32_t*>(input_ids.data()), {sequence_length}, DType::kInt32};
    state.prepare_step(inputs, sequence_length);
    inputs["input_embed"] = Tensor{
        embedding_inputs.input_embed.data(), {sequence_length, feature_dim}, DType::kFloat32};
    inputs["use_input_embed"] =
        Tensor{embedding_inputs.use_input_embed.data(), {sequence_length, 1}, DType::kFloat32};
}

bool add_vl_prefill_deepstack_inputs(TrtModule& prefill, TensorMap& inputs,
                                     VlSequenceEmbeddingInputs& embedding_inputs,
                                     int32_t sequence_length, int32_t feature_dim) {
    if (!prefill.has_input("deepstack_active"))
        return true;
    inputs["deepstack_active"] =
        Tensor{embedding_inputs.deepstack_active.data(), {sequence_length, 1}, DType::kFloat32};
    for (std::size_t level = 0;; ++level) {
        const auto name = "deepstack_embed_" + std::to_string(level);
        if (!prefill.has_input(name))
            break;
        if (level >= embedding_inputs.deepstack_embed.size())
            return false;
        inputs[name] = Tensor{embedding_inputs.deepstack_embed[level].data(),
                              {sequence_length, feature_dim},
                              DType::kFloat32};
    }
    return true;
}

bool copy_last_prefill_logits(const TensorMap& outputs, int32_t vocab_size,
                              std::vector<float>& logits) {
    const auto logits_it = outputs.find("logits");
    if (logits_it == outputs.end())
        return false;
    const auto size = static_cast<std::size_t>(vocab_size);
    if (logits_it->second.numel() < size)
        return false;
    const auto offset = logits_it->second.numel() - size;
    logits.resize(size);
    std::memcpy(logits.data(), static_cast<const float*>(logits_it->second.data) + offset,
                size * sizeof(float));
    return true;
}

int32_t resolve_input_embed_dim(const TrtModule& decoder, int32_t configured_dim) {
    if (configured_dim > 0)
        return configured_dim;
    const auto declared_shape = decoder.tensor_shape("input_embed");
    if (!declared_shape.empty())
        return static_cast<int32_t>(declared_shape.back());
    throw std::runtime_error("Phi4MultimodalPipeline: invalid input embedding dimension");
}

std::vector<int64_t> selector_shape(const TrtModule& decoder, const std::string& name) {
    return decoder.input_rank(name) == 2 ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1};
}

void add_text_step_deepstack_inputs(TrtModule& decoder, TensorMap& inputs, int32_t embed_dim,
                                    const std::vector<const float*>& deepstack_embeds,
                                    float& deepstack_active, std::vector<float>& zero_deepstack) {
    if (!decoder.has_input("deepstack_active"))
        return;
    inputs["deepstack_active"] =
        Tensor{&deepstack_active, selector_shape(decoder, "deepstack_active"), DType::kFloat32};
    for (std::size_t index = 0;; ++index) {
        const auto name = "deepstack_embed_" + std::to_string(index);
        if (!decoder.has_input(name))
            break;
        const float* embed = index < deepstack_embeds.size() ? deepstack_embeds[index] : nullptr;
        if (embed == nullptr) {
            if (zero_deepstack.empty())
                zero_deepstack.resize(static_cast<std::size_t>(embed_dim), 0.0F);
            embed = zero_deepstack.data();
        }
        inputs[name] = Tensor{const_cast<float*>(embed), {1, embed_dim}, DType::kFloat32};
    }
}

void add_text_step_embedding_inputs(TrtModule& decoder, const Phi4MultimodalConfig& config,
                                    TensorMap& inputs, const float* input_embed,
                                    float& use_input_embed,
                                    const std::vector<const float*>& deepstack_embeds,
                                    float& deepstack_active, std::vector<float>& zero_embed,
                                    std::vector<float>& zero_deepstack) {
    if (!decoder.has_input("input_embed"))
        return;
    const int32_t embed_dim = resolve_input_embed_dim(decoder, config.vision_output_dim);
    if (input_embed == nullptr) {
        zero_embed.resize(static_cast<std::size_t>(embed_dim), 0.0F);
        input_embed = zero_embed.data();
    }
    inputs["input_embed"] =
        Tensor{const_cast<float*>(input_embed), {1, embed_dim}, DType::kFloat32};
    inputs["use_input_embed"] =
        Tensor{&use_input_embed, selector_shape(decoder, "use_input_embed"), DType::kFloat32};
    add_text_step_deepstack_inputs(decoder, inputs, embed_dim, deepstack_embeds, deepstack_active,
                                   zero_deepstack);
}

Tensor make_pixel_values_tensor(const Phi4MultimodalPreprocessedImage& preprocessed,
                                const TrtModule& encoder) {
    Tensor pixel_t;
    pixel_t.data = const_cast<float*>(preprocessed.pixel_values.data());
    for (const auto& info : encoder.input_info()) {
        if (info.name == "pixel_values") {
            pixel_t.shape = info.shape;
            break;
        }
    }
    if (pixel_t.shape.empty())
        pixel_t.shape = {static_cast<int64_t>(preprocessed.pixel_values.size())};
    pixel_t.dtype = DType::kFloat32;
    return pixel_t;
}

void add_image_grid_input(TensorMap& inputs, const Phi4MultimodalPreprocessedImage& preprocessed,
                          const TrtModule& encoder) {
    if (!encoder.has_input("image_grid_hws") || preprocessed.image_grid_hws.empty())
        return;

    Tensor grid_t;
    grid_t.data = const_cast<int32_t*>(preprocessed.image_grid_hws.data());
    grid_t.shape = {static_cast<int64_t>(preprocessed.image_grid_hws.size() / 2), 2};
    grid_t.dtype = DType::kInt32;
    inputs["image_grid_hws"] = grid_t;
}

bool copy_float_output(const TensorMap& outputs, const std::string& name,
                       std::vector<float>& values) {
    auto it = outputs.find(name);
    if (it == outputs.end())
        return false;

    auto n = it->second.numel();
    values.resize(static_cast<std::size_t>(n));
    std::memcpy(values.data(), it->second.data, n * sizeof(float));
    return true;
}

void copy_deepstack_outputs(const TensorMap& outputs,
                            std::vector<std::vector<float>>* deepstack_features) {
    if (deepstack_features == nullptr)
        return;

    deepstack_features->clear();
    for (std::size_t i = 0;; ++i) {
        const std::string name = "deepstack_features_" + std::to_string(i);
        auto ds_it = outputs.find(name);
        if (ds_it == outputs.end())
            break;
        auto ds_n = ds_it->second.numel();
        deepstack_features->emplace_back(static_cast<std::size_t>(ds_n));
        std::memcpy(deepstack_features->back().data(), ds_it->second.data, ds_n * sizeof(float));
    }
}

} // namespace

std::pair<int32_t, int32_t>
Phi4MultimodalPipeline::resolve_gen_limits(const GenerateConfig& cfg) const {
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return {max_new, eos};
}

TextResult Phi4MultimodalPipeline::generate(const std::string& prompt, const float* image_pixels,
                                            int32_t image_height, int32_t image_width,
                                            const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("Phi4MultimodalPipeline: no tokenizer configured");

    bool valid = image_pixels && image_height > 0 && image_width > 0;
    if (!valid || !vision_encoder_)
        return generate(prompt, cfg);

    // Preprocess and encode the image
    auto decoded = convert_float_to_decoded(image_pixels, image_height, image_width);
    auto preprocessed = phi4_multimodal_preprocess_decoded_image(decoded, vl_preprocess_);
    if (!preprocessed.ok)
        throw std::runtime_error("Phi4MultimodalPipeline: image preprocessing failed");

    std::vector<float> features;
    std::vector<std::vector<float>> deepstack_features;
    if (!run_vision_encoder(preprocessed, features, &deepstack_features))
        throw std::runtime_error("Phi4MultimodalPipeline: vision encoder failed");

    int32_t dim = infer_feature_dim(*vision_encoder_, config_.vision_output_dim);
    if (dim <= 0)
        throw std::runtime_error("Phi4MultimodalPipeline: cannot determine vision feature dim");
    int32_t nf = static_cast<int32_t>(features.size() / static_cast<std::size_t>(dim));

    // Format prompt, tokenize, generate with vision features
    auto input_ids = tokenizer_->encode(phi4_multimodal_format_prompt(prompt, vl_preprocess_));
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp_vl = phi4_multimodal_sampling_params_from_config(cfg, eos);
    auto out =
        generate_vl_from_ids(input_ids, features, deepstack_features, nf, dim, max_new, sp_vl);

    std::vector<int32_t> new_tokens(out.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
                                    out.end());
    auto result = TextResult{tokenizer_->decode(new_tokens), std::move(new_tokens)};
    result.setup_ms = last_setup_ms_;
    return result;
}

Phi4MultimodalPipeline::GenerationResult
Phi4MultimodalPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                     const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = phi4_multimodal_sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

void Phi4MultimodalPipeline::reset_generation_context(int32_t prompt_length) {
    const auto start = std::chrono::steady_clock::now();
    state_->reset();
    state_->set_prompt_length(prompt_length);
    text_decoder_->reset_execution_context();
    prefill_->reset_execution_context();
    state_->bind_to(*text_decoder_);
    last_setup_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::vector<int32_t>
Phi4MultimodalPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                          int32_t max_new_tokens,
                                          const Phi4MultimodalSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;
    if (max_new_tokens < 0 || input_ids.size() + static_cast<std::size_t>(max_new_tokens) >
                                  static_cast<std::size_t>(state_->max_length())) {
        throw std::runtime_error(
            "Phi4MultimodalPipeline: prompt plus requested output exceeds the model context");
    }

    // Create a per-call sampler if none was injected at construction time.
    Phi4MultimodalISampler* active_sampler = sampler_.get();
    std::unique_ptr<Phi4MultimodalISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_phi4_multimodal_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    run_native_prefill_batched(input_ids, {}, {}, 0, config_.vision_output_dim, logits);
    state_->mark_prefill_complete();

    std::vector<int32_t> output = input_ids;
    run_decode_loop(active_sampler, params, output, logits, max_new_tokens);
    return output;
}

std::vector<int32_t> Phi4MultimodalPipeline::generate_vl_from_ids(
    const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
    const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
    int32_t feature_dim, int32_t max_new_tokens, const Phi4MultimodalSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;
    if (max_new_tokens < 0 || input_ids.size() + static_cast<std::size_t>(max_new_tokens) >
                                  static_cast<std::size_t>(state_->max_length())) {
        throw std::runtime_error(
            "Phi4MultimodalPipeline: prompt plus requested output exceeds the model context");
    }

    // Create a per-call sampler if none was injected at construction time.
    Phi4MultimodalISampler* active_sampler = sampler_.get();
    std::unique_ptr<Phi4MultimodalISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_phi4_multimodal_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    run_native_prefill_batched(input_ids, image_features, deepstack_features, num_features,
                               feature_dim, logits);
    state_->mark_prefill_complete();

    std::vector<int32_t> output = input_ids;
    run_decode_loop(active_sampler, params, output, logits, max_new_tokens);
    return output;
}

void Phi4MultimodalPipeline::run_native_prefill_batched(
    const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
    const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
    int32_t feature_dim, std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    if (prefill_ != nullptr && feature_dim <= 0)
        feature_dim = resolve_input_embed_dim(*prefill_, feature_dim);
    auto& kv_cache = require_vl_prefill_cache(prefill_.get(), state_.get(), config_, feature_dim);
    if (!valid_vl_features(image_features, num_features, feature_dim))
        throw std::runtime_error("Phi4MultimodalPipeline: invalid native VL feature buffer");

    kv_cache.bind_cache_inputs(*prefill_);
    const int32_t chunk_limit = config_.prefill_max_length;
    int32_t offset = 0;
    int32_t feature_offset = 0;
    while (offset < sq) {
        const int32_t chunk_size = std::min(chunk_limit, sq - offset);
        std::vector<int32_t> chunk_ids(input_ids.begin() + offset,
                                       input_ids.begin() + offset + chunk_size);
        auto embedding_inputs = build_vl_sequence_embedding_inputs(
            chunk_ids, config_.image_token_id, image_features, deepstack_features, num_features,
            feature_dim, feature_offset);

        TensorMap inputs;
        add_vl_prefill_base_inputs(inputs, *state_, chunk_ids, embedding_inputs, feature_dim);
        if (!add_vl_prefill_deepstack_inputs(*prefill_, inputs, embedding_inputs, chunk_size,
                                             feature_dim)) {
            throw std::runtime_error(
                "Phi4MultimodalPipeline: native deepstack prefill inputs are incomplete");
        }

        auto outputs = prefill_->forward(inputs);
        if (!copy_last_prefill_logits(outputs, config_.vocab_size, logits))
            throw std::runtime_error("Phi4MultimodalPipeline: native prefill logits are invalid");

        std::vector<const void*> present_k;
        std::vector<const void*> present_v;
        if (!gather_vl_prefill_kv_pointers(*prefill_, config_, present_k, present_v)) {
            throw std::runtime_error(
                "Phi4MultimodalPipeline: native prefill KV outputs are missing");
        }
        kv_cache.commit_prefill(present_k, present_v, chunk_size);

        for (const auto token : chunk_ids) {
            if (token == config_.image_token_id && feature_offset < num_features)
                ++feature_offset;
        }
        offset += chunk_size;
    }
    state_->bind_to(*text_decoder_);
}

void Phi4MultimodalPipeline::run_decode_loop(Phi4MultimodalISampler* sampler,
                                             const Phi4MultimodalSamplingParams& params,
                                             std::vector<int32_t>& output,
                                             std::vector<float>& logits, int32_t max_new_tokens) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        Phi4MultimodalSampleResult result = sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos || step + 1 == max_new_tokens)
            break;
        run_text_step(result.token_id, logits);
    }
}

void Phi4MultimodalPipeline::run_text_step(int32_t token_id, std::vector<float>& logits) {
    run_text_step_with_embed(token_id, nullptr, 0.0F, {}, 0.0F, logits);
}

void Phi4MultimodalPipeline::run_text_step_with_embed(
    int32_t token_id, const float* input_embed, float use_input_embed,
    const std::vector<const float*>& deepstack_embeds, float deepstack_active,
    std::vector<float>& logits) {
    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["token_id"] = token_t;

    state_->prepare_step(inputs);

    std::vector<float> zero_embed;
    std::vector<float> zero_deepstack;
    add_text_step_embedding_inputs(*text_decoder_, config_, inputs, input_embed, use_input_embed,
                                   deepstack_embeds, deepstack_active, zero_embed, zero_deepstack);

    auto outputs = text_decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("Phi4MultimodalPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    state_->advance();
}

bool Phi4MultimodalPipeline::run_vision_encoder(
    const Phi4MultimodalPreprocessedImage& preprocessed, std::vector<float>& image_features,
    std::vector<std::vector<float>>* deepstack_features) {
    if (!vision_encoder_ || !vision_encoder_->ok())
        return false;

    TensorMap inputs;
    inputs["pixel_values"] = make_pixel_values_tensor(preprocessed, *vision_encoder_);
    add_image_grid_input(inputs, preprocessed, *vision_encoder_);

    auto outputs = vision_encoder_->forward(inputs);

    if (!copy_float_output(outputs, "image_features", image_features)) {
        std::cerr << "[trtmc] Vision encoder has no 'image_features' output" << std::endl;
        return false;
    }

    copy_deepstack_outputs(outputs, deepstack_features);
    return true;
}

int32_t Phi4MultimodalPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
