/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/lance/pipeline.h"

#include "runtime/models/lance/image_preprocessor.h"
#include "runtime/models/lance/tensor_names.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

namespace {

void validate_pipeline_components(const TrtModule* text_decoder, const LanceInferenceState* state,
                                  const TrtModule* prefill) {
    if (text_decoder == nullptr || !text_decoder->ok())
        throw std::runtime_error("LancePipeline: invalid text decoder");
    if (state == nullptr || !state->ok())
        throw std::runtime_error("LancePipeline: invalid inference state");
    if (prefill != nullptr && !prefill->ok())
        throw std::runtime_error("LancePipeline: invalid prefill decoder");
}

void sync_vision_config(LanceConfig& config, const LancePreprocessConfig& preprocess) {
    if (config.image_token_id < 0 && preprocess.image_token_id >= 0)
        config.image_token_id = preprocess.image_token_id;
    if (config.vision_output_dim <= 0 && preprocess.vision_output_dim > 0)
        config.vision_output_dim = preprocess.vision_output_dim;
}

} // namespace

LancePipeline::LancePipeline(std::unique_ptr<TrtModule> text_decoder,
                             std::unique_ptr<TrtModule> vision_encoder,
                             std::unique_ptr<LanceInferenceState> state, LanceConfig config,
                             LancePreprocessConfig vl_preprocess, cudaStream_t stream,
                             std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                             std::unique_ptr<LanceISampler> sampler,
                             std::unique_ptr<TrtModule> prefill)
    : text_decoder_(std::move(text_decoder)), prefill_(std::move(prefill)),
      vision_encoder_(std::move(vision_encoder)), state_(std::move(state)), config_(config),
      vl_preprocess_(std::move(vl_preprocess)), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), sampler_(std::move(sampler)) {
    validate_pipeline_components(text_decoder_.get(), state_.get(), prefill_.get());
    sync_vision_config(config_, vl_preprocess_);
}

TextResult LancePipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("LancePipeline: no tokenizer configured");

    auto input_ids = tokenizer_->encode(prompt);
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp = lance_sampling_params_from_config(cfg, eos);
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

std::vector<const float*>
select_deepstack_feature_pointers(const std::vector<std::vector<float>>& deepstack_features,
                                  int32_t feature_index, int32_t feature_dim) {
    std::vector<const float*> embeds;
    embeds.reserve(deepstack_features.size());
    for (const auto& deepstack : deepstack_features) {
        const int32_t count =
            static_cast<int32_t>(deepstack.size() / static_cast<std::size_t>(feature_dim));
        embeds.push_back(feature_index < count
                             ? deepstack.data() +
                                   static_cast<std::size_t>(feature_index) * feature_dim
                             : nullptr);
    }
    return embeds;
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
                                   int32_t num_features, int32_t feature_dim) {
    const auto sq = input_ids.size();
    VlSequenceEmbeddingInputs result;
    result.input_embed.assign(sq * static_cast<std::size_t>(feature_dim), 0.0F);
    result.use_input_embed.assign(sq, 0.0F);
    result.deepstack_active.assign(sq, 0.0F);
    result.deepstack_embed.resize(deepstack_features.size());
    for (auto& level : result.deepstack_embed)
        level.assign(sq * static_cast<std::size_t>(feature_dim), 0.0F);

    int32_t feature_index = 0;
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

bool gather_vl_prefill_kv_pointers(TrtModule& prefill, const LanceConfig& config,
                                   std::vector<const void*>& present_k,
                                   std::vector<const void*>& present_v) {
    present_k.resize(static_cast<std::size_t>(config.num_layers));
    present_v.resize(static_cast<std::size_t>(config.num_layers));
    for (int32_t layer = 0; layer < config.num_layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        present_k[index] =
            prefill.device_ptr(lance_expand_layer_name(config.present_k_pattern, layer));
        present_v[index] =
            prefill.device_ptr(lance_expand_layer_name(config.present_v_pattern, layer));
        if (present_k[index] == nullptr || present_v[index] == nullptr)
            return false;
    }
    return true;
}

LanceKvCache* eligible_vl_prefill_cache(TrtModule* prefill, LanceInferenceState* state,
                                        const LanceConfig& config, int32_t sequence_length,
                                        int32_t feature_dim) {
    if (prefill == nullptr)
        return nullptr;
    if (sequence_length <= 0 || feature_dim <= 0 || config.num_layers <= 0)
        return nullptr;
    if (!prefill->has_input("input_embed"))
        return nullptr;
    if (config.prefill_max_length > 0 && sequence_length > config.prefill_max_length)
        return nullptr;
    return dynamic_cast<LanceKvCache*>(state);
}

bool valid_vl_features(const std::vector<float>& image_features, int32_t num_features,
                       int32_t feature_dim) {
    if (num_features < 0)
        return false;
    const auto required =
        static_cast<std::size_t>(num_features) * static_cast<std::size_t>(feature_dim);
    return image_features.size() >= required;
}

std::pair<int32_t, int32_t> find_lance_vision_attention_block(const std::vector<int32_t>& input_ids,
                                                              int32_t image_token_id) {
    const auto first = std::find(input_ids.begin(), input_ids.end(), image_token_id);
    if (first == input_ids.end())
        return {-1, -1};
    auto last = first;
    while (last + 1 != input_ids.end() && *(last + 1) == image_token_id)
        ++last;

    const auto first_index = static_cast<int32_t>(std::distance(input_ids.begin(), first));
    const auto last_index = static_cast<int32_t>(std::distance(input_ids.begin(), last));
    // The official full-attention segment includes vision_start and
    // vision_end around the contiguous image placeholder run.
    if (first_index == 0 || last_index + 1 >= static_cast<int32_t>(input_ids.size()))
        return {-1, -1};
    return {first_index - 1, last_index + 2};
}

struct LanceVlMropeContext {
    bool enabled = false;
    LanceMropePositions positions;

    const LanceMropePositions* prefill_positions() const { return enabled ? &positions : nullptr; }

    const std::array<int32_t, 3>* token_position(std::size_t index) const {
        return enabled ? &positions.token_positions[index] : nullptr;
    }

    int32_t decode_position() const { return enabled ? positions.next_position : -1; }
};

LanceVlMropeContext make_lance_vl_mrope_context(const TrtModule& text_decoder,
                                                const std::vector<int32_t>& input_ids,
                                                int32_t image_token_id, int32_t num_features,
                                                const LancePreprocessConfig& preprocess) {
    LanceVlMropeContext context;
    context.enabled = text_decoder.has_input("mrope_position_ids");
    if (!context.enabled)
        return context;

    int32_t merged_grid = 0;
    if (preprocess.patch_size > 0 && preprocess.merge_size > 0) {
        merged_grid = preprocess.fixed_image_size / (preprocess.patch_size * preprocess.merge_size);
    }
    context.positions = lance_build_mrope_positions(input_ids, image_token_id, num_features,
                                                    merged_grid, merged_grid);
    return context;
}

void add_vl_prefill_base_inputs(TensorMap& inputs, LanceInferenceState& state,
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

bool add_lance_mrope_prefill_input(TrtModule& prefill, const std::vector<int32_t>& input_ids,
                                   const LanceMropePositions* mrope, int32_t sequence_length,
                                   std::vector<int32_t>& positions, TensorMap& inputs) {
    if (!prefill.has_input("mrope_position_ids"))
        return true;
    if (mrope == nullptr || mrope->token_positions.size() != input_ids.size())
        return false;
    positions.resize(static_cast<std::size_t>(3 * sequence_length));
    for (int32_t token_index = 0; token_index < sequence_length; ++token_index) {
        for (int32_t axis = 0; axis < 3; ++axis) {
            positions[static_cast<std::size_t>(axis * sequence_length + token_index)] =
                mrope->token_positions[static_cast<std::size_t>(token_index)]
                                      [static_cast<std::size_t>(axis)];
        }
    }
    inputs["mrope_position_ids"] = Tensor{positions.data(), {3, sequence_length}, DType::kInt32};
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
    throw std::runtime_error("LancePipeline: invalid input embedding dimension");
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

void add_text_step_embedding_inputs(TrtModule& decoder, const LanceConfig& config,
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

Tensor make_pixel_values_tensor(const LancePreprocessedImage& preprocessed,
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

void add_image_grid_input(TensorMap& inputs, const LancePreprocessedImage& preprocessed,
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

std::pair<int32_t, int32_t> LancePipeline::resolve_gen_limits(const GenerateConfig& cfg) const {
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return {max_new, eos};
}

TextResult LancePipeline::generate(const std::string& prompt, const float* image_pixels,
                                   int32_t image_height, int32_t image_width,
                                   const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("LancePipeline: no tokenizer configured");

    bool valid = image_pixels && image_height > 0 && image_width > 0;
    if (!valid || !vision_encoder_)
        return generate(prompt, cfg);

    // Preprocess and encode the image
    auto decoded = convert_float_to_decoded(image_pixels, image_height, image_width);
    auto preprocessed = lance_preprocess_decoded_image(decoded, vl_preprocess_);
    if (!preprocessed.ok)
        throw std::runtime_error("LancePipeline: image preprocessing failed");

    std::vector<float> features;
    std::vector<std::vector<float>> deepstack_features;
    if (!run_vision_encoder(preprocessed, features, &deepstack_features))
        throw std::runtime_error("LancePipeline: vision encoder failed");

    int32_t dim = infer_feature_dim(*vision_encoder_, config_.vision_output_dim);
    if (dim <= 0)
        throw std::runtime_error("LancePipeline: cannot determine vision feature dim");
    int32_t nf = static_cast<int32_t>(features.size() / static_cast<std::size_t>(dim));

    // Format prompt, tokenize, generate with vision features
    auto input_ids = tokenizer_->encode(lance_format_prompt(prompt, vl_preprocess_));
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp_vl = lance_sampling_params_from_config(cfg, eos);
    auto out =
        generate_vl_from_ids(input_ids, features, deepstack_features, nf, dim, max_new, sp_vl);

    std::vector<int32_t> new_tokens(out.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
                                    out.end());
    auto result = TextResult{tokenizer_->decode(new_tokens), std::move(new_tokens)};
    result.setup_ms = last_setup_ms_;
    return result;
}

LancePipeline::GenerationResult LancePipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                                            const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = lance_sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

void LancePipeline::reset_generation_context(int32_t prompt_length) {
    const auto start = std::chrono::steady_clock::now();
    state_->reset();
    state_->set_prompt_length(prompt_length);
    text_decoder_->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
    state_->bind_to(*text_decoder_);
    last_setup_ms_ =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::vector<int32_t> LancePipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                      int32_t max_new_tokens,
                                                      const LanceSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    LanceISampler* active_sampler = sampler_.get();
    std::unique_ptr<LanceISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_lance_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_text_step(input_ids[i], logits);

    run_text_step(input_ids.back(), logits);

    std::vector<int32_t> output = input_ids;
    const int32_t vocab_size = static_cast<int32_t>(logits.size());

    for (int32_t step = 0; step < max_new_tokens; ++step) {
        LanceSampleResult result = active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_text_step(result.token_id, logits);
    }

    return output;
}

std::vector<int32_t> LancePipeline::generate_vl_from_ids(
    const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
    const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
    int32_t feature_dim, int32_t max_new_tokens, const LanceSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    LanceISampler* active_sampler = sampler_.get();
    std::unique_ptr<LanceISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_lance_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    reset_generation_context(static_cast<int32_t>(input_ids.size()));

    std::vector<float> logits;
    const auto mrope = make_lance_vl_mrope_context(
        *text_decoder_, input_ids, config_.image_token_id, num_features, vl_preprocess_);
    if (!run_vl_prefill_batched(input_ids, image_features, deepstack_features, num_features,
                                feature_dim, mrope.prefill_positions(), logits)) {
        int32_t feature_index = 0;
        for (std::size_t index = 0; index < input_ids.size(); ++index) {
            const auto tid = input_ids[index];
            run_vl_prefill_token(tid, image_features, deepstack_features, num_features, feature_dim,
                                 feature_index, mrope.token_position(index), logits);
        }
    }
    state_->mark_prefill_complete();

    std::vector<int32_t> output = input_ids;
    run_vl_decode_loop(active_sampler, params, output, logits, max_new_tokens,
                       mrope.decode_position());
    return output;
}

bool LancePipeline::run_vl_prefill_batched(
    const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
    const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
    int32_t feature_dim, const LanceMropePositions* mrope, std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    auto* kv_cache =
        eligible_vl_prefill_cache(prefill_.get(), state_.get(), config_, sq, feature_dim);
    if (kv_cache == nullptr)
        return false;
    if (!valid_vl_features(image_features, num_features, feature_dim))
        return false;

    kv_cache->bind_cache_inputs(*prefill_);
    auto embedding_inputs =
        build_vl_sequence_embedding_inputs(input_ids, config_.image_token_id, image_features,
                                           deepstack_features, num_features, feature_dim);

    TensorMap inputs;
    add_vl_prefill_base_inputs(inputs, *state_, input_ids, embedding_inputs, feature_dim);
    const auto [vision_start, vision_end] =
        find_lance_vision_attention_block(input_ids, config_.image_token_id);
    if (vision_start >= 0)
        kv_cache->prepare_prefill_with_bidirectional_block(inputs, sq, vision_start, vision_end);
    std::vector<int32_t> mrope_positions;
    if (!add_lance_mrope_prefill_input(*prefill_, input_ids, mrope, sq, mrope_positions, inputs))
        return false;
    if (!add_vl_prefill_deepstack_inputs(*prefill_, inputs, embedding_inputs, sq, feature_dim))
        return false;

    auto outputs = prefill_->forward(inputs);
    if (!copy_last_prefill_logits(outputs, config_.vocab_size, logits))
        return false;

    std::vector<const void*> present_k;
    std::vector<const void*> present_v;
    if (!gather_vl_prefill_kv_pointers(*prefill_, config_, present_k, present_v))
        return false;
    kv_cache->write_prefill_kv(present_k, present_v, sq);
    return true;
}

void LancePipeline::run_vl_prefill_token(int32_t token_id, const std::vector<float>& image_features,
                                         const std::vector<std::vector<float>>& deepstack_features,
                                         int32_t num_features, int32_t feature_dim,
                                         int32_t& feature_index,
                                         const std::array<int32_t, 3>* mrope_position,
                                         std::vector<float>& logits) {
    const bool use_image_embed = token_id == config_.image_token_id && feature_index < num_features;
    if (!use_image_embed) {
        run_text_step_with_embed(token_id, nullptr, 0.0F, {}, 0.0F, mrope_position, logits);
        return;
    }

    const float* embed =
        image_features.data() + static_cast<std::size_t>(feature_index) * feature_dim;
    const auto deepstack_embeds =
        select_deepstack_feature_pointers(deepstack_features, feature_index, feature_dim);
    run_text_step_with_embed(token_id, embed, 1.0F, deepstack_embeds,
                             deepstack_embeds.empty() ? 0.0F : 1.0F, mrope_position, logits);
    ++feature_index;
}

void LancePipeline::run_vl_decode_loop(LanceISampler* sampler, const LanceSamplingParams& params,
                                       std::vector<int32_t>& output, std::vector<float>& logits,
                                       int32_t max_new_tokens, int32_t mrope_position) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        LanceSampleResult result = sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_text_step(result.token_id, logits, mrope_position);
        if (mrope_position >= 0)
            ++mrope_position;
    }
}

void LancePipeline::run_text_step(int32_t token_id, std::vector<float>& logits,
                                  int32_t mrope_position) {
    if (mrope_position < 0 || !text_decoder_->has_input("mrope_position_ids")) {
        run_text_step_with_embed(token_id, nullptr, 0.0F, {}, 0.0F, nullptr, logits);
        return;
    }
    std::array<int32_t, 3> position{};
    position.fill(mrope_position);
    run_text_step_with_embed(token_id, nullptr, 0.0F, {}, 0.0F, &position, logits);
}

void LancePipeline::run_text_step_with_embed(int32_t token_id, const float* input_embed,
                                             float use_input_embed,
                                             const std::vector<const float*>& deepstack_embeds,
                                             float deepstack_active,
                                             const std::array<int32_t, 3>* mrope_position,
                                             std::vector<float>& logits) {
    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["token_id"] = token_t;

    state_->prepare_step(inputs);

    if (mrope_position != nullptr && text_decoder_->has_input("mrope_position_ids")) {
        inputs["mrope_position_ids"] =
            Tensor{const_cast<int32_t*>(mrope_position->data()), {3}, DType::kInt32};
    }

    std::vector<float> zero_embed;
    std::vector<float> zero_deepstack;
    add_text_step_embedding_inputs(*text_decoder_, config_, inputs, input_embed, use_input_embed,
                                   deepstack_embeds, deepstack_active, zero_embed, zero_deepstack);

    auto outputs = text_decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("LancePipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    state_->advance();
}

bool LancePipeline::run_vision_encoder(const LancePreprocessedImage& preprocessed,
                                       std::vector<float>& image_features,
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

int32_t LancePipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
