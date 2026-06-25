#include "runtime/models/locateanything/pipeline.h"

#include "runtime/models/locateanything/image_preprocessor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtmc {

LocateAnythingPipeline::LocateAnythingPipeline(
    std::unique_ptr<TrtModule> text_decoder, std::unique_ptr<TrtModule> vision_encoder,
    std::unique_ptr<LocateanythingInferenceState> state, LocateAnythingConfig config,
    LocateAnythingPreprocessConfig vl_preprocess, cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer, std::string model_id_str,
    std::unique_ptr<LocateAnythingISampler> sampler)
    : text_decoder_(std::move(text_decoder)), vision_encoder_(std::move(vision_encoder)),
      state_(std::move(state)), config_(config), vl_preprocess_(std::move(vl_preprocess)),
      stream_(stream), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)) {
    if (!text_decoder_ || !text_decoder_->ok())
        throw std::runtime_error("LocateAnythingPipeline: invalid text decoder");
    if (!state_ || !state_->ok())
        throw std::runtime_error("LocateAnythingPipeline: invalid inference state");

    // Sync image_token_id from LocateAnythingPreprocessConfig if not set in LocateAnythingConfig
    if (config_.image_token_id < 0 && vl_preprocess_.image_token_id >= 0)
        config_.image_token_id = vl_preprocess_.image_token_id;
    if (config_.vision_output_dim <= 0 && vl_preprocess_.vision_output_dim > 0)
        config_.vision_output_dim = vl_preprocess_.vision_output_dim;
}

TextResult LocateAnythingPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("LocateAnythingPipeline: no tokenizer configured");

    auto input_ids = tokenizer_->encode(prompt);
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp = locateanything_sampling_params_from_config(cfg, eos);
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

Tensor make_pixel_values_tensor(const LocateAnythingPreprocessedImage& preprocessed,
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

void add_image_grid_input(TensorMap& inputs, const LocateAnythingPreprocessedImage& preprocessed,
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
LocateAnythingPipeline::resolve_gen_limits(const GenerateConfig& cfg) const {
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return {max_new, eos};
}

TextResult LocateAnythingPipeline::generate(const std::string& prompt, const float* image_pixels,
                                            int32_t image_height, int32_t image_width,
                                            const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("LocateAnythingPipeline: no tokenizer configured");

    bool valid = image_pixels && image_height > 0 && image_width > 0;
    if (!valid || !vision_encoder_)
        return generate(prompt, cfg);

    // Preprocess and encode the image
    auto decoded = convert_float_to_decoded(image_pixels, image_height, image_width);
    auto preprocessed = locateanything_preprocess_decoded_image(decoded, vl_preprocess_);
    if (!preprocessed.ok)
        throw std::runtime_error("LocateAnythingPipeline: image preprocessing failed");

    std::vector<float> features;
    std::vector<std::vector<float>> deepstack_features;
    if (!run_vision_encoder(preprocessed, features, &deepstack_features))
        throw std::runtime_error("LocateAnythingPipeline: vision encoder failed");

    int32_t dim = infer_feature_dim(*vision_encoder_, config_.vision_output_dim);
    if (dim <= 0)
        throw std::runtime_error("LocateAnythingPipeline: cannot determine vision feature dim");
    int32_t nf = static_cast<int32_t>(features.size() / static_cast<std::size_t>(dim));

    // Format prompt, tokenize, generate with vision features
    auto input_ids = tokenizer_->encode(locateanything_format_prompt(prompt, vl_preprocess_));
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto sp_vl = locateanything_sampling_params_from_config(cfg, eos);
    auto out =
        generate_vl_from_ids(input_ids, features, deepstack_features, nf, dim, max_new, sp_vl);

    std::vector<int32_t> new_tokens(out.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
                                    out.end());
    return TextResult{tokenizer_->decode(new_tokens), std::move(new_tokens)};
}

LocateAnythingPipeline::GenerationResult
LocateAnythingPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                     const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = locateanything_sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

std::vector<int32_t>
LocateAnythingPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                          int32_t max_new_tokens,
                                          const LocateAnythingSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    LocateAnythingISampler* active_sampler = sampler_.get();
    std::unique_ptr<LocateAnythingISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_locateanything_sampler(params);
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
        LocateAnythingSampleResult result =
            active_sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_text_step(result.token_id, logits);
    }

    return output;
}

std::vector<int32_t> LocateAnythingPipeline::generate_vl_from_ids(
    const std::vector<int32_t>& input_ids, const std::vector<float>& image_features,
    const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
    int32_t feature_dim, int32_t max_new_tokens, const LocateAnythingSamplingParams& params) {
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    // Create a per-call sampler if none was injected at construction time.
    LocateAnythingISampler* active_sampler = sampler_.get();
    std::unique_ptr<LocateAnythingISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_locateanything_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*text_decoder_);

    std::vector<float> logits;
    int32_t feature_index = 0;

    for (const auto& tid : input_ids)
        run_vl_prefill_token(tid, image_features, deepstack_features, num_features, feature_dim,
                             feature_index, logits);

    std::vector<int32_t> output = input_ids;
    run_vl_decode_loop(active_sampler, params, output, logits, max_new_tokens);
    return output;
}

void LocateAnythingPipeline::run_vl_prefill_token(
    int32_t token_id, const std::vector<float>& image_features,
    const std::vector<std::vector<float>>& deepstack_features, int32_t num_features,
    int32_t feature_dim, int32_t& feature_index, std::vector<float>& logits) {
    const bool use_image_embed = token_id == config_.image_token_id && feature_index < num_features;
    if (!use_image_embed) {
        run_text_step_with_embed(token_id, nullptr, 0.0F, {}, 0.0F, logits);
        return;
    }

    const float* embed =
        image_features.data() + static_cast<std::size_t>(feature_index) * feature_dim;
    const auto deepstack_embeds =
        select_deepstack_feature_pointers(deepstack_features, feature_index, feature_dim);
    run_text_step_with_embed(token_id, embed, 1.0F, deepstack_embeds,
                             deepstack_embeds.empty() ? 0.0F : 1.0F, logits);
    ++feature_index;
}

void LocateAnythingPipeline::run_vl_decode_loop(LocateAnythingISampler* sampler,
                                                const LocateAnythingSamplingParams& params,
                                                std::vector<int32_t>& output,
                                                std::vector<float>& logits,
                                                int32_t max_new_tokens) {
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        LocateAnythingSampleResult result = sampler->sample(logits.data(), vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        run_text_step(result.token_id, logits);
    }
}

void LocateAnythingPipeline::run_text_step(int32_t token_id, std::vector<float>& logits) {
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
        throw std::runtime_error("LocateAnythingPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    state_->advance();
}

void LocateAnythingPipeline::run_text_step_with_embed(
    int32_t token_id, const float* input_embed, float use_input_embed,
    const std::vector<const float*>& deepstack_embeds, float deepstack_active,
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
        throw std::runtime_error("LocateAnythingPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    state_->advance();
}

bool LocateAnythingPipeline::run_vision_encoder(
    const LocateAnythingPreprocessedImage& preprocessed, std::vector<float>& image_features,
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

int32_t LocateAnythingPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtmc
