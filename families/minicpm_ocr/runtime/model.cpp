/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpm_ocr/runtime/model.h"

#include "families/minicpm_ocr/runtime/decoder.h"
#include "families/minicpm_ocr/runtime/image_processor.h"
#include "families/minicpm_ocr/runtime/prompt.h"
#include "families/minicpm_ocr/runtime/tokenizer.h"
#include "families/minicpm_ocr/runtime/vision_encoder.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace trtmc::minicpm_ocr {
namespace {

namespace api = trtmc::internal;
using Json = nlohmann::json;

constexpr int32_t kChannels = 3;

struct RuntimeConfig {
    DecoderGeometry decoder;
    std::vector<int32_t> eos_token_ids;
    int32_t image_token_id{-1};
    int32_t default_max_new_tokens{0};
    bool use_image_id{true};
    ImageProcessorConfig image;
    PromptTokens tokens;
};

std::vector<char> require_section(const BundleReader& bundle, std::string_view name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("MiniCPM-OCR bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

template <typename T>
T field(const Json& object, const char* name) {
    if (!object.is_object() || !object.contains(name))
        throw std::runtime_error(std::string("MiniCPM-OCR runtime.json is missing ") + name);
    try {
        return object.at(name).get<T>();
    } catch (const Json::exception&) {
        throw std::runtime_error(std::string("MiniCPM-OCR runtime.json has an invalid ") + name);
    }
}

void expect_keys(const Json& object, std::initializer_list<const char*> names, const char* what) {
    if (!object.is_object() || object.size() != names.size())
        throw std::runtime_error(std::string("MiniCPM-OCR runtime.json has an unexpected ") + what +
                                 " field set");
    for (const auto* name : names)
        if (!object.contains(name))
            throw std::runtime_error(std::string("MiniCPM-OCR runtime.json ") + what +
                                     " is missing " + name);
}

RuntimeConfig parse_runtime(const BundleReader& bundle) {
    const auto bytes = require_section(bundle, "runtime.json");
    Json json;
    try {
        json = Json::parse(bytes.begin(), bytes.end());
    } catch (const Json::exception& error) {
        throw std::runtime_error(std::string("MiniCPM-OCR invalid runtime.json: ") + error.what());
    }
    expect_keys(json,
                {"schema_version", "precision", "hidden_size", "vocab_size", "max_cache_length",
                 "linear_layers", "attention_layers", "conv_dim", "conv_kernel", "linear_heads",
                 "linear_key_dim", "linear_value_dim", "kv_dim", "eos_token_ids", "image_token_id",
                 "default_max_new_tokens", "image", "tokens"},
                "top-level");
    if (field<int>(json, "schema_version") != 1)
        throw std::runtime_error("Unsupported MiniCPM-OCR runtime.json schema");
    RuntimeConfig c;
    const auto precision = field<std::string>(json, "precision");
    if (precision != "fp16" && precision != "fp32")
        throw std::runtime_error("MiniCPM-OCR runtime.json has an invalid precision");
    auto& d = c.decoder;
    d.dtype = precision == "fp16" ? DType::kFloat16 : DType::kFloat32;
    d.hidden_size = field<int32_t>(json, "hidden_size");
    d.vocab_size = field<int32_t>(json, "vocab_size");
    d.max_cache_length = field<int32_t>(json, "max_cache_length");
    d.linear_layers = field<int32_t>(json, "linear_layers");
    d.attention_layers = field<int32_t>(json, "attention_layers");
    d.conv_dim = field<int32_t>(json, "conv_dim");
    d.conv_kernel = field<int32_t>(json, "conv_kernel");
    d.linear_heads = field<int32_t>(json, "linear_heads");
    d.linear_key_dim = field<int32_t>(json, "linear_key_dim");
    d.linear_value_dim = field<int32_t>(json, "linear_value_dim");
    d.kv_dim = field<int32_t>(json, "kv_dim");
    c.eos_token_ids = field<std::vector<int32_t>>(json, "eos_token_ids");
    c.image_token_id = field<int32_t>(json, "image_token_id");
    c.default_max_new_tokens = field<int32_t>(json, "default_max_new_tokens");
    if (c.eos_token_ids.empty() || c.image_token_id < 0 || c.default_max_new_tokens <= 0)
        throw std::runtime_error("MiniCPM-OCR runtime.json has invalid token metadata");

    const auto& image = json.at("image");
    expect_keys(image,
                {"patch_size", "merge_size", "scale_resolution", "max_slice_nums", "slice_mode",
                 "use_image_id", "position_grid", "max_piece_patches", "image_mean", "image_std"},
                "image");
    auto& p = c.image;
    p.patch_size = field<int32_t>(image, "patch_size");
    p.merge_size = field<int32_t>(image, "merge_size");
    p.scale_resolution = field<int32_t>(image, "scale_resolution");
    p.max_slice_nums = field<int32_t>(image, "max_slice_nums");
    p.slice_mode = field<bool>(image, "slice_mode");
    c.use_image_id = field<bool>(image, "use_image_id");
    p.position_grid = field<int32_t>(image, "position_grid");
    p.max_piece_patches = field<int32_t>(image, "max_piece_patches");
    p.mean = field<std::array<float, 3>>(image, "image_mean");
    p.stddev = field<std::array<float, 3>>(image, "image_std");
    p.validate();

    const auto& tokens = json.at("tokens");
    expect_keys(tokens,
                {"image_pad", "image_start", "image_end", "slice_start", "slice_end",
                 "image_id_start", "image_id_end"},
                "tokens");
    c.tokens.image_pad = field<std::string>(tokens, "image_pad");
    c.tokens.image_start = field<std::string>(tokens, "image_start");
    c.tokens.image_end = field<std::string>(tokens, "image_end");
    c.tokens.slice_start = field<std::string>(tokens, "slice_start");
    c.tokens.slice_end = field<std::string>(tokens, "slice_end");
    c.tokens.image_id_start = field<std::string>(tokens, "image_id_start");
    c.tokens.image_id_end = field<std::string>(tokens, "image_id_end");
    return c;
}

std::shared_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle, const RuntimeConfig& c) {
    const auto data = require_section(bundle, "tokenizer.json");
    std::shared_ptr<ITokenizer> tokenizer = CreateBpeTokenizer(data.data(), data.size(), true);
    if (!tokenizer)
        throw std::runtime_error("MiniCPM-OCR tokenizer construction failed");
    if (tokenizer->id_for_token(c.tokens.image_pad) != c.image_token_id)
        throw std::runtime_error("MiniCPM-OCR tokenizer disagrees with the image token id");
    for (const auto* spelling :
         {&c.tokens.image_start, &c.tokens.image_end, &c.tokens.slice_start, &c.tokens.slice_end,
          &c.tokens.image_id_start, &c.tokens.image_id_end})
        if (tokenizer->id_for_token(*spelling) < 0)
            throw std::runtime_error("MiniCPM-OCR tokenizer lacks the special token " + *spelling);
    return tokenizer;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const BundleReader& bundle,
                                        std::string_view section, const char* label) {
    const auto plan = require_section(bundle, section);
    auto engine = backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error(std::string("MiniCPM-OCR failed to load the ") + label);
    engine->set_timing_label(label);
    return engine;
}

Span<const api::ConfigField> config_fields() {
    static const std::vector<api::ConfigField> fields{
        {"max_new_tokens", api::ConfigKind::I64, std::nullopt,
         "Generated token budget; default min(runtime default, remaining sequence capacity)"},
        {"temperature", api::ConfigKind::F64, 1.0, "Sampling temperature"},
        {"top_k", api::ConfigKind::I64, int64_t{1}, "Top-k sampling; 1 selects greedy decoding"},
        {"top_p", api::ConfigKind::F64, 1.0, "Nucleus sampling probability"},
        {"seed", api::ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 leaves it unspecified"},
        {"enable_thinking", api::ConfigKind::Bool, false,
         "Chat-template thinking switch of the checkpoint"},
    };
    return {fields.data(), fields.size()};
}

class MiniCpmOcrModel final : public api::IModel, public trtmc::internal::IImagesTextToText {
  public:
    explicit MiniCpmOcrModel(const FamilyContext& context)
        : config_(parse_runtime(context.reader)),
          tokenizer_(load_tokenizer(context.reader, config_)),
          vision_(load_engine(context.backend, context.reader, "vision.plan", "minicpm_ocr vision"),
                  config_.decoder.hidden_size, config_.decoder.dtype,
                  kChannels * config_.image.patch_size * config_.image.patch_size,
                  config_.image.max_piece_patches, config_.image.merge_size),
          decoder_(
              load_engine(context.backend, context.reader, "decoder.plan", "minicpm_ocr decoder"),
              config_.decoder, config_.eos_token_ids) {
        if (context.reader.info().task != api::IImagesTextToText::kTask)
            throw std::runtime_error("MiniCPM-OCR bundle task must be images_text_to_text");
    }

    const char* task() const noexcept override { return api::IImagesTextToText::kTask.data(); }
    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::IImagesTextToText>(*this, config_fields())};
    }

    TextResult run(const api::ImagesTextToTextRequest& request, api::ConfigView supplied) override {
        if (!request.tools.empty())
            throw std::invalid_argument("MiniCPM-OCR does not support tool definitions");
        const auto fields = config_fields();
        api::validate_config(fields, supplied);
        SamplingOptions sampling;
        sampling.temperature = *api::config_get<double>(supplied, fields, "temperature");
        sampling.top_k = *api::config_get<int64_t>(supplied, fields, "top_k");
        sampling.top_p = *api::config_get<double>(supplied, fields, "top_p");
        sampling.seed = *api::config_get<int64_t>(supplied, fields, "seed");
        if (!(sampling.temperature >= 0.0) || sampling.top_k < 0 || !(sampling.top_p >= 0.0) ||
            sampling.top_p > 1.0)
            throw std::invalid_argument("MiniCPM-OCR sampling options are out of range");
        const bool thinking = *api::config_get<bool>(supplied, fields, "enable_thinking");
        const auto requested = api::config_get<int64_t>(supplied, fields, "max_new_tokens");

        std::lock_guard<std::mutex> lock(mutex_);
        const auto setup_start = std::chrono::steady_clock::now();
        std::vector<PromptMessage> messages;
        std::vector<ProcessedImage> images;
        std::vector<ImageTokenLayout> layouts;
        for (const auto& message : request.messages) {
            PromptMessage item{role(message.role), {}};
            if (message.parts.empty())
                throw std::invalid_argument("MiniCPM-OCR messages must have content");
            for (const auto& part : message.parts) {
                std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, api::TextPartView>) {
                            item.parts.push_back({false, std::string(value.text)});
                        } else if constexpr (std::is_same_v<T, api::ImageView>) {
                            if (item.role == PromptRole::System)
                                throw std::invalid_argument(
                                    "System message cannot contain images.");
                            images.push_back(preprocess_image(config_.image, value));
                            layouts.push_back(layout(images.back()));
                            item.parts.push_back({true, {}});
                        } else {
                            throw std::invalid_argument(
                                "MiniCPM-OCR accepts text and image parts only");
                        }
                    },
                    part);
            }
            messages.push_back(std::move(item));
        }
        if (layouts.empty())
            throw std::invalid_argument("MiniCPM-OCR requests need at least one image");
        const auto text = expand_images(render_chat(messages, thinking, config_.tokens), layouts,
                                        config_.use_image_id, config_.tokens);
        DecoderPrompt prompt;
        prompt.token_ids = tokenizer_->encode(text);

        // The prompt length is known before any image is encoded: a request that cannot fit
        // the cache is refused here, without GPU work.
        const auto capacity = static_cast<int64_t>(config_.decoder.max_cache_length) + 1 -
                              static_cast<int64_t>(prompt.token_ids.size());
        const int64_t budget = requested ? *requested
                                         : std::min<int64_t>(config_.default_max_new_tokens,
                                                             std::max<int64_t>(capacity, 1));
        if (budget <= 0 || budget > INT32_MAX)
            throw std::invalid_argument("MiniCPM-OCR max_new_tokens must be positive");
        if (budget > capacity)
            throw std::invalid_argument("MiniCPM-OCR prompt (" +
                                        std::to_string(prompt.token_ids.size()) +
                                        " tokens) plus max_new_tokens (" + std::to_string(budget) +
                                        ") exceeds the bundle's sequence capacity (" +
                                        std::to_string(config_.decoder.max_cache_length) +
                                        "); rebuild with a larger max_sequence_length");

        std::size_t rows = 0;
        for (const auto& image : images)
            rows += vision_.encode(image, prompt.features);
        prompt.feature_rows.assign(prompt.token_ids.size(), -1);
        std::size_t next = 0;
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i)
            if (prompt.token_ids[i] == config_.image_token_id)
                prompt.feature_rows[i] = static_cast<int32_t>(next++);
        if (next != rows)
            throw std::runtime_error("MiniCPM-OCR image tokens and visual features disagree (" +
                                     std::to_string(next) + " vs " + std::to_string(rows) + ")");
        const double setup_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - setup_start)
                                    .count();

        auto output = decoder_.generate(prompt, sampling, static_cast<int32_t>(budget));
        std::vector<int32_t> visible = output.token_ids;
        if (output.stopped)
            visible.pop_back();
        TextResult result{python_strip(replace_invalid_utf8(tokenizer_->decode(visible))),
                          std::move(output.token_ids), output.prefill_ms, output.decode_ms};
        result.setup_ms = setup_ms;
        return result;
    }

  private:
    static PromptRole role(api::MessageRole role) {
        switch (role) {
        case api::MessageRole::System:
            return PromptRole::System;
        case api::MessageRole::User:
            return PromptRole::User;
        case api::MessageRole::Assistant:
            return PromptRole::Assistant;
        default:
            throw std::invalid_argument(
                "MiniCPM-OCR accepts system, user and assistant messages only");
        }
    }

    ImageTokenLayout layout(const ProcessedImage& image) const {
        ImageTokenLayout result;
        result.overview_tokens = image.pieces.front().tokens(config_.image.merge_size);
        result.grid_rows = image.grid.rows;
        result.grid_cols = image.grid.cols;
        for (std::size_t i = 1; i < image.pieces.size(); ++i)
            result.slice_tokens.push_back(image.pieces[i].tokens(config_.image.merge_size));
        return result;
    }

    RuntimeConfig config_;
    std::shared_ptr<ITokenizer> tokenizer_;
    VisionEncoder vision_;
    Decoder decoder_;
    std::mutex mutex_;
};

} // namespace

ITask* create_model(const FamilyContext& context) {
    return new MiniCpmOcrModel(context);
}

} // namespace trtmc::minicpm_ocr
