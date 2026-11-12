/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/glm_ocr/runtime/model.h"

#include "families/glm_ocr/runtime/prompt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::glm_ocr {
namespace {

using internal::ConfigField;
using internal::ConfigKind;

const ConfigField kFields[] = {
    {"max_new_tokens", ConfigKind::I64, std::nullopt,
     "Cap on generated tokens (at least 1); unset generates until an end-of-sequence token or "
     "the bundle's max_sequence_length"},
    {"temperature", ConfigKind::F64, 1.0,
     "Greedy decoding only: any non-negative value with top_k 1, otherwise 0"},
    {"top_k", ConfigKind::I64, std::int64_t{1},
     "Greedy decoding only: 1, or any value with temperature 0"},
    {"top_p", ConfigKind::F64, 1.0, "In (0, 1]; no effect on greedy decoding"},
};

std::vector<char> require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error(std::string("glm_ocr bundle section is missing or empty: ") +
                                 name);
    return reader.read_section(name);
}

std::unique_ptr<ITrtModule> load_engine(const BundleReader& reader, IBackend& backend,
                                        const char* section, cudaStream_t stream) {
    const auto plan = require_section(reader, section);
    ModuleCreateOptions options{};
    options.stream = stream;
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("glm_ocr failed to load engine ") + section);
    return module;
}

double milliseconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

int positive(const nlohmann::json& json, const char* key) {
    const auto value = json.at(key).get<std::int64_t>();
    if (value <= 0 || value > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("glm_ocr runtime.json has an invalid ") + key);
    return static_cast<int>(value);
}

std::vector<float> copy_output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.dtype != DType::kFloat32 || it->second.numel() != count)
        throw std::runtime_error(std::string("glm_ocr engine output contract mismatch: ") + name);
    const auto* data = static_cast<const float*>(it->second.data);
    return std::vector<float>(data, data + count);
}

std::string layer_name(const char* prefix, int layer) {
    return prefix + std::to_string(layer);
}

} // namespace

void check_greedy_controls(double temperature, std::int64_t top_k, double top_p) {
    if (!std::isfinite(temperature) || temperature < 0.0)
        throw internal::ConfigError("glm_ocr temperature must be a finite value >= 0");
    if (top_k < 0)
        throw internal::ConfigError("glm_ocr top_k must be >= 0");
    if (!(top_p > 0.0 && top_p <= 1.0))
        throw internal::ConfigError("glm_ocr top_p must be in (0, 1]");
    if (temperature != 0.0 && top_k != 1)
        throw internal::ConfigError(
            "glm_ocr decodes greedily: sampling is not supported (use top_k 1 or temperature 0)");
}

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    if (json.at("format").get<int>() != 1)
        throw std::runtime_error("glm_ocr runtime.json has an unsupported format");
    RuntimeConfig config;
    const auto precision = json.at("precision").get<std::string>();
    if (precision == "bf16")
        config.cache_dtype = DType::kBFloat16;
    else if (precision == "fp32")
        config.cache_dtype = DType::kFloat32;
    else
        throw std::runtime_error("glm_ocr runtime.json has an unsupported precision");
    config.max_sequence_length = positive(json, "max_sequence_length");
    const auto& text = json.at("text");
    config.hidden_size = positive(text, "hidden_size");
    config.num_layers = positive(text, "num_layers");
    config.num_key_value_heads = positive(text, "num_key_value_heads");
    config.head_dim = positive(text, "head_dim");
    config.vocab_size = positive(text, "vocab_size");
    config.rope_theta = text.at("rope_theta").get<float>();
    const auto sections = text.at("mrope_section").get<std::vector<int>>();
    if (sections.size() != 3 || sections[0] + sections[1] + sections[2] != config.head_dim / 2)
        throw std::runtime_error("glm_ocr runtime.json mrope_section does not cover head_dim");
    std::copy(sections.begin(), sections.end(), config.mrope_section.begin());

    const auto& vision = json.at("vision");
    const auto& image = json.at("image");
    auto& ic = config.image;
    ic.patch_size = positive(vision, "patch_size");
    ic.temporal_patch_size = positive(vision, "temporal_patch_size");
    ic.merge_size = positive(vision, "merge_size");
    ic.head_dim = positive(vision, "head_dim");
    ic.rope_theta = vision.at("rope_theta").get<float>();
    ic.max_patches = positive(vision, "max_patches");
    config.vision_out_hidden_size = positive(vision, "out_hidden_size");
    if (positive(vision, "in_channels") != 3 || ic.head_dim % 4 != 0 ||
        config.vision_out_hidden_size != config.hidden_size)
        throw std::runtime_error("glm_ocr runtime.json vision contract is inconsistent");
    ic.min_pixels = image.at("min_pixels").get<std::int64_t>();
    ic.max_pixels = image.at("max_pixels").get<std::int64_t>();
    const auto mean = image.at("image_mean").get<std::vector<float>>();
    const auto std = image.at("image_std").get<std::vector<float>>();
    if (mean.size() != 3 || std.size() != 3 || ic.min_pixels <= 0 || ic.max_pixels <= ic.min_pixels)
        throw std::runtime_error("glm_ocr runtime.json image contract is inconsistent");
    std::copy(mean.begin(), mean.end(), ic.mean.begin());
    std::copy(std.begin(), std.end(), ic.std.begin());

    config.image_token_id = json.at("tokens").at("<|image|>").get<std::int32_t>();
    for (const auto id : json.at("eos_token_ids").get<std::vector<std::int32_t>>())
        config.eos_token_ids.insert(id);
    if (config.eos_token_ids.empty() || config.max_sequence_length < 2)
        throw std::runtime_error("glm_ocr runtime.json generation contract is inconsistent");
    return config;
}

GlmOcrModel::GlmOcrModel(const BundleReader& reader, IBackend& backend) {
    if (reader.info().task != internal::IImagesTextToText::kTask)
        throw std::runtime_error("glm_ocr bundle declares an unsupported task: " +
                                 reader.info().task);
    config_ = parse_runtime_config(require_section(reader, "runtime.json"));
    const auto tokenizer_json = require_section(reader, "tokenizer.json");
    tokenizer_ =
        std::make_unique<Tokenizer>(std::string_view(tokenizer_json.data(), tokenizer_json.size()));
    if (tokenizer_->added_token_id("<|image|>") != config_.image_token_id ||
        tokenizer_->vocab_size() > static_cast<std::size_t>(config_.vocab_size))
        throw std::runtime_error("glm_ocr tokenizer.json disagrees with runtime.json");
    embedding_ = require_section(reader, "token_embedding.bf16");
    if (embedding_.size() != static_cast<std::size_t>(config_.vocab_size) * config_.hidden_size * 2)
        throw std::runtime_error("glm_ocr token embedding has the wrong size");

    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("glm_ocr failed to create its CUDA stream");
    vision_ = load_engine(reader, backend, "vision.plan", stream_);
    prefill_ = load_engine(reader, backend, "prefill.plan", stream_);
    decode_ = load_engine(reader, backend, "decode.plan", stream_);
    if (prefill_->tensor_dtype("present_key_0") != config_.cache_dtype ||
        decode_->tensor_dtype("past_key_0") != config_.cache_dtype ||
        vision_->tensor_dtype("image_embeds") != DType::kFloat32)
        throw std::runtime_error("glm_ocr engines disagree with runtime.json precision");

    const std::vector<std::int64_t> shape{config_.max_sequence_length, config_.num_key_value_heads,
                                          config_.head_dim};
    cache_row_bytes_ = static_cast<std::size_t>(config_.num_key_value_heads) * config_.head_dim *
                       dtype_size(config_.cache_dtype);
    for (int layer = 0; layer < config_.num_layers; ++layer) {
        key_cache_.emplace_back(shape, config_.cache_dtype, stream_);
        value_cache_.emplace_back(shape, config_.cache_dtype, stream_);
        if (!key_cache_.back().ok() || !value_cache_.back().ok())
            throw std::runtime_error("glm_ocr could not allocate its key/value cache");
        // The prefill writes the prompt's rows at the start of each cache.
        prefill_->bind_external(layer_name("present_key_", layer), key_cache_.back().data());
        prefill_->bind_external(layer_name("present_value_", layer), value_cache_.back().data());
    }
}

GlmOcrModel::~GlmOcrModel() {
    decode_.reset();
    prefill_.reset();
    vision_.reset();
    key_cache_.clear();
    value_cache_.clear();
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

std::vector<internal::TaskInstance> GlmOcrModel::task_bindings() {
    return {internal::bind<internal::IImagesTextToText>(*this, Span<const ConfigField>(kFields))};
}

void GlmOcrModel::embed_row(std::int32_t token, float* out) const {
    if (token < 0 || token >= config_.vocab_size)
        throw std::out_of_range("glm_ocr token id outside the embedding table");
    const auto* row = reinterpret_cast<const unsigned char*>(embedding_.data()) +
                      static_cast<std::size_t>(token) * config_.hidden_size * 2;
    for (int i = 0; i < config_.hidden_size; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(row[2 * i]) |
                                   (static_cast<std::uint32_t>(row[2 * i + 1]) << 8);
        const std::uint32_t wide = bits << 16;
        std::memcpy(out + i, &wide, sizeof(float));
    }
}

std::int32_t GlmOcrModel::argmax(const std::vector<float>& logits) const {
    std::size_t best = 0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i]))
            throw std::runtime_error("glm_ocr decoder produced a non-finite logit");
        if (logits[i] > logits[best])
            best = i;
    }
    return static_cast<std::int32_t>(best);
}

std::vector<float> GlmOcrModel::encode_image(const VisionInput& input) {
    const auto patches = static_cast<std::int64_t>(input.num_patches());
    const auto patch_dim = static_cast<std::int64_t>(input.patches.size()) / patches;
    const auto half = static_cast<std::int64_t>(config_.image.head_dim / 2);
    TensorMap inputs;
    inputs["pixel_patches"] = {
        const_cast<float*>(input.patches.data()), {patches, patch_dim}, DType::kFloat32};
    inputs["rotary_cos"] = {
        const_cast<float*>(input.rotary_cos.data()), {patches, half}, DType::kFloat32};
    inputs["rotary_sin"] = {
        const_cast<float*>(input.rotary_sin.data()), {patches, half}, DType::kFloat32};
    const auto tokens =
        static_cast<std::size_t>(patches) / (config_.image.merge_size * config_.image.merge_size);
    return copy_output(vision_->forward(inputs), "image_embeds",
                       tokens * static_cast<std::size_t>(config_.hidden_size));
}

std::vector<float> GlmOcrModel::prefill(const std::vector<float>& embeds,
                                        const std::vector<float>& cos,
                                        const std::vector<float>& sin, std::size_t rows) {
    const auto r = static_cast<std::int64_t>(rows);
    const auto half = static_cast<std::int64_t>(config_.head_dim / 2);
    TensorMap inputs;
    inputs["inputs_embeds"] = {
        const_cast<float*>(embeds.data()), {r, config_.hidden_size}, DType::kFloat32};
    inputs["rotary_cos"] = {const_cast<float*>(cos.data()), {r, half}, DType::kFloat32};
    inputs["rotary_sin"] = {const_cast<float*>(sin.data()), {r, half}, DType::kFloat32};
    return copy_output(prefill_->forward(inputs), "logits",
                       static_cast<std::size_t>(config_.vocab_size));
}

std::vector<float> GlmOcrModel::decode_step(std::int32_t token, std::int32_t position,
                                            std::size_t filled) {
    std::vector<float> embed(config_.hidden_size);
    embed_row(token, embed.data());
    std::vector<float> cos, sin;
    text_rotary({{position, position, position}}, config_.head_dim, config_.rope_theta,
                config_.mrope_section, cos, sin);
    const std::vector<std::int64_t> past{static_cast<std::int64_t>(filled),
                                         config_.num_key_value_heads, config_.head_dim};
    for (int layer = 0; layer < config_.num_layers; ++layer) {
        auto* key = static_cast<char*>(key_cache_[layer].data());
        auto* value = static_cast<char*>(value_cache_[layer].data());
        decode_->bind_external(layer_name("past_key_", layer), key, past);
        decode_->bind_external(layer_name("past_value_", layer), value, past);
        decode_->bind_external(layer_name("present_key_", layer), key + filled * cache_row_bytes_);
        decode_->bind_external(layer_name("present_value_", layer),
                               value + filled * cache_row_bytes_);
    }
    const auto half = static_cast<std::int64_t>(config_.head_dim / 2);
    TensorMap inputs;
    inputs["inputs_embeds"] = {embed.data(), {1, config_.hidden_size}, DType::kFloat32};
    inputs["rotary_cos"] = {cos.data(), {1, half}, DType::kFloat32};
    inputs["rotary_sin"] = {sin.data(), {1, half}, DType::kFloat32};
    return copy_output(decode_->forward(inputs), "logits",
                       static_cast<std::size_t>(config_.vocab_size));
}

TextResult GlmOcrModel::run(const internal::ImagesTextToTextRequest& request,
                            internal::ConfigView supplied) {
    std::lock_guard<std::mutex> lock(mutex_);
    internal::validate_config(Span<const ConfigField>(kFields), supplied);
    const auto requested = internal::config_get<std::int64_t>(
        supplied, Span<const ConfigField>(kFields), "max_new_tokens");
    if (requested && *requested < 1)
        throw internal::ConfigError("glm_ocr max_new_tokens must be at least 1");
    const Span<const ConfigField> fields(kFields);
    check_greedy_controls(*internal::config_get<double>(supplied, fields, "temperature"),
                          *internal::config_get<std::int64_t>(supplied, fields, "top_k"),
                          *internal::config_get<double>(supplied, fields, "top_p"));

    const auto start = std::chrono::steady_clock::now();
    const auto prompt = render_prompt(request);
    std::vector<VisionInput> images;
    std::vector<ImageGrid> grids;
    for (const auto& image : prompt.images) {
        images.push_back(preprocess_image(image, config_.image));
        grids.push_back({images.back().grid_h, images.back().grid_w});
    }
    const auto layout = layout_prompt(tokenizer_->encode(prompt.text), config_.image_token_id,
                                      grids, config_.image.merge_size);
    const std::size_t rows = layout.ids.size();
    const auto capacity = static_cast<std::size_t>(config_.max_sequence_length);
    if (rows > capacity)
        throw std::invalid_argument("glm_ocr prompt of " + std::to_string(rows) +
                                    " tokens exceeds this bundle's max_sequence_length of " +
                                    std::to_string(capacity));

    const auto hidden = static_cast<std::size_t>(config_.hidden_size);
    std::vector<float> embeds(rows * hidden);
    for (std::size_t row = 0; row < rows; ++row)
        embed_row(layout.ids[row], embeds.data() + row * hidden);
    for (std::size_t k = 0; k < images.size(); ++k) {
        const auto features = encode_image(images[k]);
        std::copy(features.begin(), features.end(),
                  embeds.begin() + static_cast<std::ptrdiff_t>(layout.image_offsets[k] * hidden));
    }
    std::vector<float> cos, sin;
    text_rotary(layout.positions, config_.head_dim, config_.rope_theta, config_.mrope_section, cos,
                sin);
    const double setup_ms = milliseconds_since(start);

    auto stage = std::chrono::steady_clock::now();
    auto logits = prefill(embeds, cos, sin, rows);
    const double prefill_ms = milliseconds_since(stage);

    // The prompt fills `rows` cache rows; each later token needs one more row.
    std::size_t limit = capacity - rows + 1;
    if (requested)
        limit = std::min<std::size_t>(limit, static_cast<std::size_t>(*requested));
    stage = std::chrono::steady_clock::now();
    std::vector<std::int32_t> generated{argmax(logits)};
    std::int32_t position = layout.next_position;
    std::size_t filled = rows;
    while (config_.eos_token_ids.count(generated.back()) == 0 && generated.size() < limit) {
        logits = decode_step(generated.back(), position, filled);
        ++filled;
        ++position;
        generated.push_back(argmax(logits));
    }
    const double decode_ms = milliseconds_since(stage);

    auto text =
        tokenizer_->decode(Span<const std::int32_t>(generated.data(), generated.size()), true);
    TextResult result(std::move(text), std::move(generated), prefill_ms, decode_ms);
    result.setup_ms = setup_ms;
    return result;
}

} // namespace trtmc::glm_ocr
