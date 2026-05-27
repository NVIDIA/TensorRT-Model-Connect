#include "runtime/models/vision_language/pipeline.h"

#include "runtime/domains/multimodal/image_preprocessor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

VLPipeline::VLPipeline(std::unique_ptr<TrtModule> text_decoder,
                       std::unique_ptr<TrtModule> vision_encoder,
                       std::unique_ptr<IInferenceState> state, VLConfig config,
                       VLPreprocessConfig vl_preprocess, cudaStream_t stream,
                       std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
                       std::unique_ptr<ISampler> sampler)
    : text_decoder_(std::move(text_decoder)), vision_encoder_(std::move(vision_encoder)),
      state_(std::move(state)), config_(config), vl_preprocess_(std::move(vl_preprocess)),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)) {
    if (!text_decoder_ || !text_decoder_->ok())
        throw std::runtime_error("VLPipeline: invalid text decoder");
    if (!state_ || !state_->ok())
        throw std::runtime_error("VLPipeline: invalid inference state");

    // Sync image_token_id from VLPreprocessConfig if not set in VLConfig
    if (config_.image_token_id < 0 && vl_preprocess_.image_token_id >= 0)
        config_.image_token_id = vl_preprocess_.image_token_id;
    if (config_.vision_output_dim <= 0 && vl_preprocess_.vision_output_dim > 0)
        config_.vision_output_dim = vl_preprocess_.vision_output_dim;
}

TextResult VLPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("VLPipeline: no tokenizer configured");

    auto input_ids = tokenizer_->encode(prompt);
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp = sampling_params_from_config(cfg, eos);
    auto output_ids = generate_from_ids(input_ids, max_new, sp);

    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()), output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens)};
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

} // namespace

std::pair<int32_t, int32_t> VLPipeline::resolve_gen_limits(const GenerateConfig& cfg) const {
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return {max_new, eos};
}

TextResult VLPipeline::generate(const std::string& prompt, const float* image_pixels,
                                int32_t image_height, int32_t image_width,
                                const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("VLPipeline: no tokenizer configured");

    bool valid = image_pixels && image_height > 0 && image_width > 0;
    if (!valid || !vision_encoder_)
        return generate(prompt, cfg);

    // Preprocess and encode the image
    auto decoded = convert_float_to_decoded(image_pixels, image_height, image_width);
    auto preprocessed = preprocess_decoded_image(decoded, vl_preprocess_);
    if (!preprocessed.ok)
        throw std::runtime_error("VLPipeline: image preprocessing failed");

    std::vector<float> features;
    std::vector<std::vector<float>> deepstack_features;
    if (!run_vision_encoder(preprocessed.pixel_values.data(), preprocessed.pixel_values.size(),
                            features, &deepstack_features))
        throw std::runtime_error("VLPipeline: vision encoder failed");

    int32_t dim = infer_feature_dim(*vision_encoder_, config_.vision_output_dim);
    if (dim <= 0)
        throw std::runtime_error("VLPipeline: cannot determine vision feature dim");
    int32_t nf = static_cast<int32_t>(features.size() / static_cast<std::size_t>(dim));

    // Format prompt, tokenize, generate with vision features
    auto input_ids = tokenizer_->encode(format_vl_prompt(prompt, vl_preprocess_));
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp_vl = sampling_params_from_config(cfg, eos);
    auto out = generate_vl_from_ids(input_ids, features, deepstack_features, nf, dim, max_new,
                                    sp_vl);

    std::vector<int32_t> new_tokens(out.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
                                    out.end());
    return TextResult{tokenizer_->decode(new_tokens), std::move(new_tokens)};
}

VLPipeline::GenerationResult VLPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                                      const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

std::vector<int32_t> VLPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                                   int32_t max_new_tokens,
                                                   const SamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    ISampler* active_sampler = sampler_.get();
    std::unique_ptr<ISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*text_decoder_);

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_text_step(input_ids[i], logits);

    run_text_step(input_ids.back(), logits);

    std::vector<int32_t> output = input_ids;
    const int32_t vocab_size = static_cast<int32_t>(logits.size());

    for (int32_t step = 0; step < max_new_tokens; ++step) {
        SampleResult result = active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_text_step(result.token_id, logits);
    }

    return output;
}

std::vector<int32_t> VLPipeline::generate_vl_from_ids(const std::vector<int32_t>& input_ids,
                                                      const std::vector<float>& image_features,
                                                      const std::vector<std::vector<float>>&
                                                          deepstack_features,
                                                      int32_t num_features, int32_t feature_dim,
                                                      int32_t max_new_tokens,
                                                      const SamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    ISampler* active_sampler = sampler_.get();
    std::unique_ptr<ISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*text_decoder_);

    std::vector<float> logits;
    int32_t feature_index = 0;
    int32_t image_token = config_.image_token_id;

    // Prefill: run each token with vision embedding injection at image tokens.
    auto prefill_one = [&](int32_t tid) {
        if (tid == image_token && feature_index < num_features) {
            const int32_t used_feature_index = feature_index;
            const float* embed =
                image_features.data() + static_cast<std::size_t>(feature_index) * feature_dim;
            std::vector<const float*> deepstack_embeds;
            deepstack_embeds.reserve(deepstack_features.size());
            for (const auto& deepstack : deepstack_features) {
                const int32_t deepstack_count =
                    static_cast<int32_t>(deepstack.size() / static_cast<std::size_t>(feature_dim));
                deepstack_embeds.push_back(
                    used_feature_index < deepstack_count
                        ? deepstack.data() +
                              static_cast<std::size_t>(used_feature_index) * feature_dim
                        : nullptr);
            }
            run_text_step_with_embed(tid, embed, 1.0F, deepstack_embeds,
                                     deepstack_embeds.empty() ? 0.0F : 1.0F, logits);
            ++feature_index;
        } else {
            run_text_step_with_embed(tid, nullptr, 0.0F, {}, 0.0F, logits);
        }
    };

    for (const auto& tid : input_ids)
        prefill_one(tid);

    // Autoregressive decode
    std::vector<int32_t> output = input_ids;
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        SampleResult result = active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_text_step(result.token_id, logits);
    }

    return output;
}

void VLPipeline::run_text_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["token_id"] = token_t;

    state_->prepare_step(inputs);

    auto outputs = text_decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("VLPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    state_->advance();
}

void VLPipeline::run_text_step_with_embed(int32_t token_id, const float* input_embed,
                                          float use_input_embed,
                                          const std::vector<const float*>& deepstack_embeds,
                                          float deepstack_active,
                                          std::vector<float>& logits) {
    if (!text_decoder_->has_input("input_embed")) {
        run_text_step(token_id, logits);
        return;
    }

    TensorMap inputs;

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;
    inputs["token_id"] = token_t;

    state_->prepare_step(inputs);

    // use_input_embed scalar: 1.0 = use input_embed, 0.0 = use token embedding
    Tensor use_embed_t;
    use_embed_t.data = &use_input_embed;
    use_embed_t.shape = {1};
    use_embed_t.dtype = DType::kFloat32;
    inputs["use_input_embed"] = use_embed_t;

    // Provide the input embedding vector
    int32_t embed_dim = config_.vision_output_dim;
    std::vector<float> zero_embed;
    if (input_embed == nullptr) {
        zero_embed.resize(static_cast<std::size_t>(embed_dim), 0.0F);
        input_embed = zero_embed.data();
    }

    Tensor embed_t;
    embed_t.data = const_cast<float*>(input_embed);
    embed_t.shape = {static_cast<int64_t>(embed_dim)};
    embed_t.dtype = DType::kFloat32;
    inputs["input_embed"] = embed_t;

    // DeepStack: if the engine has deepstack inputs, provide inactive zeros.
    if (text_decoder_->has_input("deepstack_active")) {
        Tensor ds_active_t;
        ds_active_t.data = &deepstack_active;
        ds_active_t.shape = {1};
        ds_active_t.dtype = DType::kFloat32;
        inputs["deepstack_active"] = ds_active_t;

        std::vector<float> zero_deepstack;
        for (std::size_t i = 0;; ++i) {
            const std::string name = "deepstack_embed_" + std::to_string(i);
            if (!text_decoder_->has_input(name))
                break;

            const float* deepstack_embed =
                i < deepstack_embeds.size() ? deepstack_embeds[i] : nullptr;
            if (deepstack_embed == nullptr) {
                if (zero_deepstack.empty())
                    zero_deepstack.resize(static_cast<std::size_t>(embed_dim), 0.0F);
                deepstack_embed = zero_deepstack.data();
            }

            Tensor ds_embed_t;
            ds_embed_t.data = const_cast<float*>(deepstack_embed);
            ds_embed_t.shape = {1, static_cast<int64_t>(embed_dim)};
            ds_embed_t.dtype = DType::kFloat32;
            inputs[name] = ds_embed_t;
        }
    }

    auto outputs = text_decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("VLPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    state_->advance();
}

bool VLPipeline::run_vision_encoder(const float* pixel_values, std::size_t pixel_count,
                                    std::vector<float>& image_features,
                                    std::vector<std::vector<float>>* deepstack_features) {
    if (!vision_encoder_ || !vision_encoder_->ok())
        return false;

    // Build input tensor for pixel values
    Tensor pixel_t;
    pixel_t.data = const_cast<float*>(pixel_values);
    // Get the shape from the vision engine's input
    auto inputs_info = vision_encoder_->input_info();
    for (const auto& info : inputs_info) {
        if (info.name == "pixel_values") {
            pixel_t.shape = info.shape;
            break;
        }
    }
    if (pixel_t.shape.empty()) {
        // Fallback: use the pixel_count as a flat shape
        pixel_t.shape = {static_cast<int64_t>(pixel_count)};
    }
    pixel_t.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["pixel_values"] = pixel_t;

    auto outputs = vision_encoder_->forward(inputs);

    // Extract image_features output
    auto it = outputs.find("image_features");
    if (it == outputs.end()) {
        std::cerr << "[trtmc] Vision encoder has no 'image_features' output" << std::endl;
        return false;
    }

    auto n = it->second.numel();
    image_features.resize(static_cast<std::size_t>(n));
    std::memcpy(image_features.data(), it->second.data, n * sizeof(float));

    if (deepstack_features != nullptr) {
        deepstack_features->clear();
        for (std::size_t i = 0;; ++i) {
            const std::string name = "deepstack_features_" + std::to_string(i);
            auto ds_it = outputs.find(name);
            if (ds_it == outputs.end())
                break;
            auto ds_n = ds_it->second.numel();
            deepstack_features->emplace_back(static_cast<std::size_t>(ds_n));
            std::memcpy(deepstack_features->back().data(), ds_it->second.data,
                        ds_n * sizeof(float));
        }
    }

    return true;
}

int32_t VLPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
