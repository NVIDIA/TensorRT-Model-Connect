/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/decoder.h"
#include "families/minicpmv4_6/runtime/image.h"
#include "families/minicpmv4_6/runtime/metadata.h"
#include "families/minicpmv4_6/runtime/prompt.h"
#include "families/minicpmv4_6/runtime/tokenizer.h"
#include "families/minicpmv4_6/runtime/vision_encoder.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace trtmc::minicpmv4_6 {
namespace {

namespace api = trtmc::internal;
using Json = nlohmann::json;

constexpr const char* kImagePad = "<|image_pad|>";
constexpr const char* kVideoPad = "<|video_pad|>";

std::vector<char> section(const BundleReader& bundle, std::string_view name) {
    const auto* info = bundle.find_section(name);
    if (info == nullptr || info->length == 0)
        throw std::runtime_error("MiniCPM-V 4.6 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

Json json_section(const BundleReader& bundle, std::string_view name) {
    const auto data = section(bundle, name);
    auto value = parse_json_unique(data.data(), data.size(), std::string(name));
    if (!value.is_object())
        throw std::runtime_error("MiniCPM-V 4.6 " + std::string(name) + " must be a JSON object");
    return value;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const BundleReader& bundle,
                                        std::string_view name, const ModuleCreateOptions& options) {
    const auto plan = section(bundle, name);
    auto module = backend.create_module(plan.data(), plan.size(), options);
    if (!module || !module->ok())
        throw std::runtime_error("MiniCPM-V 4.6 failed to load " + std::string(name));
    module->set_timing_label("minicpmv4_6 " + std::string(name));
    return module;
}

// Generation options of both Tasks; text continuation adds use_chat_template.
const std::vector<api::ConfigField>& common_fields() {
    static const std::vector<api::ConfigField> fields{
        {"max_new_tokens", api::ConfigKind::I64, std::int64_t{128}, "Generated token budget"},
        {"temperature", api::ConfigKind::F64, 1.0, "Sampling temperature; 0 is greedy"},
        {"top_k", api::ConfigKind::I64, std::int64_t{1},
         "Top-k sampling; 1 is greedy, 0 keeps all"},
        {"top_p", api::ConfigKind::F64, 1.0, "Nucleus sampling probability mass"},
        {"seed", api::ConfigKind::I64, std::int64_t{-1}, "Sampling seed; -1 uses a fixed default"},
        {"enable_thinking", api::ConfigKind::Bool, false, "Chat template thinking mode"},
    };
    return fields;
}

const std::vector<api::ConfigField>& text_fields() {
    static const std::vector<api::ConfigField> fields = [] {
        auto copy = common_fields();
        copy.push_back({"use_chat_template", api::ConfigKind::Bool, false,
                        "Format the prefix as one user message with the chat template"});
        return copy;
    }();
    return fields;
}

Span<const api::ConfigField> span(const std::vector<api::ConfigField>& fields) {
    return {fields.data(), fields.size()};
}

struct Options {
    std::int32_t max_new_tokens{128};
    SamplingOptions sampling;
    bool enable_thinking{false};
    bool use_chat_template{false};
};

Options read_options(api::ConfigView supplied, const std::vector<api::ConfigField>& fields) {
    const auto schema = span(fields);
    api::validate_config(schema, supplied);
    Options options;
    const auto max_new = *api::config_get<std::int64_t>(supplied, schema, "max_new_tokens");
    if (max_new <= 0 || max_new > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("MiniCPM-V 4.6 max_new_tokens must be a positive int32");
    options.max_new_tokens = static_cast<std::int32_t>(max_new);
    options.sampling = sampling_options(*api::config_get<double>(supplied, schema, "temperature"),
                                        *api::config_get<std::int64_t>(supplied, schema, "top_k"),
                                        *api::config_get<double>(supplied, schema, "top_p"),
                                        *api::config_get<std::int64_t>(supplied, schema, "seed"));
    options.enable_thinking = *api::config_get<bool>(supplied, schema, "enable_thinking");
    if (api::find_config_field(schema, "use_chat_template"))
        options.use_chat_template = *api::config_get<bool>(supplied, schema, "use_chat_template");
    return options;
}

double since_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

class Model final : public internal::IModel,
                    public internal::IImagesTextToText,
                    public internal::ITextContinuation {
  public:
    explicit Model(const FamilyContext& context) {
        const auto& bundle = context.reader;
        if (bundle.info().task != api::IImagesTextToText::kTask)
            throw std::runtime_error("MiniCPM-V 4.6 bundles have the images_text_to_text task");
        const auto runtime = json_section(bundle, "runtime.json");
        vision_ = json_section(bundle, "vision.json");
        if (json_int32(vision_, "schema_version", 1) != 1)
            throw std::runtime_error("Unsupported MiniCPM-V 4.6 vision.json schema");
        const auto precision = runtime.at("precision").get<std::string>();
        if ((precision != "fp16" && precision != "fp32") || vision_.at("precision") != precision)
            throw std::runtime_error("MiniCPM-V 4.6 bundle precision metadata disagrees");
        const DType dtype = precision == "fp16" ? DType::kFloat16 : DType::kFloat32;

        DecoderGeometry geometry;
        geometry.attention_layers = json_int32(runtime, "num_attention_layers", 1);
        geometry.deltanet_layers = json_int32(runtime, "num_mamba_layers", 1);
        geometry.kv_width =
            checked_product(json_int32(runtime, "num_key_value_heads", 1),
                            json_int32(runtime, "head_dim", 1), "num_key_value_heads x head_dim");
        geometry.conv_width = json_int32(runtime, "conv_dim", 1);
        geometry.conv_kernel = json_int32(runtime, "mamba_d_conv", 1);
        geometry.deltanet_heads = json_int32(runtime, "mamba_nheads", 1);
        geometry.deltanet_head_dim = json_int32(runtime, "mamba_head_dim", 1);
        geometry.capacity = json_int32(runtime, "max_cache_length", 2);
        geometry.cache_dtype = dtype;
        const auto hidden = json_int32(runtime, "hidden_size", 1);
        const auto vocab = json_int32(runtime, "vocab_size", 1);
        if (json_int32(vision_, "capacity", 1) != geometry.capacity ||
            json_int32(vision_, "hidden_size", 1) != hidden)
            throw std::runtime_error("MiniCPM-V 4.6 vision.json disagrees with runtime.json");
        eos_ids_ = json_token_ids(runtime, "eos_token_id", vocab);

        settings_.patch_size = json_int32(vision_, "patch_size", 1);
        settings_.positions_per_side = json_int32(vision_, "positions_per_side", 1);
        settings_.scale_resolution = json_int32(vision_, "scale_resolution", 1);
        settings_.max_slice_nums = json_int32(vision_, "max_slice_nums", 1);
        settings_.max_patches = json_int32(vision_, "max_patches", 1);
        settings_.mean = json_float3(vision_, "mean");
        settings_.stddev = json_float3(vision_, "stddev");
        settings_.validate();
        for (const char* mode : {"false", "true"})
            validate_recipe(vision_.at("prompts").at(mode));

        const auto tokenizer_json = section(bundle, "tokenizer.json");
        tokenizer_ = create_tokenizer(tokenizer_json.data(), tokenizer_json.size());
        check_vocabulary(tokenizer_->vocabulary_size(), vocab);
        const auto& ids = vision_.at("token_ids");
        if (!ids.is_object() || ids.size() != 7)
            throw std::runtime_error(
                "MiniCPM-V 4.6 vision.json token_ids must name the seven markup tokens");
        for (const auto& [name, text] :
             std::vector<std::pair<const char*, const char*>>{{"image_pad", kImagePad},
                                                              {"image_start", "<image>"},
                                                              {"image_end", "</image>"},
                                                              {"slice_start", "<slice>"},
                                                              {"slice_end", "</slice>"},
                                                              {"image_id_start", "<image_id>"},
                                                              {"image_id_end", "</image_id>"}})
            if (const auto id = json_int32(ids, name, 0);
                id >= vocab || tokenizer_->id_for_token(text) != id)
                throw std::runtime_error("MiniCPM-V 4.6 tokenizer disagrees with vision.json");
        image_pad_ = json_int32(ids, "image_pad", 0);

        auto decoder = load_engine(context.backend, bundle, "engine.plan", {});
        ModuleCreateOptions shared;
        shared.stream = decoder->stream();
        auto prefill = load_engine(context.backend, bundle, "prefill.plan", shared);
        decoder_ = std::make_unique<Decoder>(std::move(decoder), std::move(prefill), geometry,
                                             vocab, hidden);
        vision_encoder_ = std::make_unique<VisionEncoder>(
            load_engine(context.backend, bundle, "vision.plan", {}), hidden, dtype,
            checked_product(3, checked_product(settings_.patch_size, settings_.patch_size, "patch"),
                            "patch width"),
            settings_.max_patches);
    }

    const char* task() const noexcept override { return api::IImagesTextToText::kTask.data(); }

    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::IImagesTextToText>(*this, span(common_fields())),
                api::bind<api::ITextContinuation>(*this, span(text_fields()))};
    }

    TextResult run(const api::ImagesTextToTextRequest& request, api::ConfigView config) override {
        const auto options = read_options(config, common_fields());
        if (!request.tools.empty())
            throw std::invalid_argument("MiniCPM-V 4.6 does not map tool definitions");
        if (request.messages.empty())
            throw std::invalid_argument("MiniCPM-V 4.6 requires at least one message");
        std::lock_guard<std::mutex> lock(mutex_);
        const auto start = std::chrono::steady_clock::now();
        std::vector<PromptMessage> messages;
        std::vector<PreparedImage> images;
        const std::int64_t budget = prompt_budget(options);
        // Every message renders at least its start marker, and the prompt's text may not exceed
        // what the budget's ids can stand for: both are checked before anything is copied.
        if (request.messages.size() > static_cast<std::size_t>(budget))
            prompt_too_long();
        const std::size_t max_points =
            tokenizer_->max_code_points(static_cast<std::size_t>(budget));
        std::size_t points = 0;
        std::int64_t image_rows = 0;
        for (const auto& message : request.messages) {
            PromptMessage item;
            item.role = role(message.role);
            if (message.parts.empty())
                throw std::invalid_argument("MiniCPM-V 4.6 messages must have content");
            bool first = true;
            for (const auto& part : message.parts) {
                // The provider template joins content parts with newlines and
                // renders each image as one placeholder.
                if (!first) {
                    if (++points > max_points)
                        prompt_too_long();
                    item.content += "\n";
                }
                first = false;
                std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, api::TextPartView>) {
                            points += code_points(value.text);
                            if (points > max_points)
                                prompt_too_long();
                            item.content += checked_text(value.text);
                        } else if constexpr (std::is_same_v<T, api::ImageView>) {
                            if (item.role == Role::System)
                                throw std::invalid_argument(
                                    "MiniCPM-V 4.6 system messages cannot contain images");
                            // Count the image's rows from its slice plan before any slice
                            // is allocated; stop once the images alone exceed the cache.
                            image_rows += planned_rows(settings_, plan_image(settings_, value));
                            if (image_rows > budget)
                                throw std::invalid_argument(
                                    "MiniCPM-V 4.6 images need more rows than the bundle's " +
                                    std::to_string(decoder_->capacity()) +
                                    "-token capacity leaves after max_new_tokens");
                            images.push_back(prepare_image(settings_, value));
                            item.content += kImagePad;
                        } else {
                            throw std::invalid_argument(
                                "MiniCPM-V 4.6 maps text and image parts only");
                        }
                    },
                    part);
            }
            messages.push_back(std::move(item));
        }
        auto text = format_conversation(recipe(options), messages);
        text = expand_images(text, images);
        PromptRows prompt;
        // The token budget holds before any vision engine launch.
        auto ids = tokenizer_->encode(text, static_cast<std::size_t>(budget));
        if (!ids)
            prompt_too_long();
        prompt.token_ids = std::move(*ids);
        prompt.feature_rows.assign(prompt.token_ids.size(), -1);
        std::int32_t rows = 0;
        for (const auto& image : images)
            for (const auto& slice : image.slices) {
                const auto features = vision_encoder_->encode(slice);
                prompt.features.insert(prompt.features.end(), features.begin(), features.end());
                rows += slice.tokens();
            }
        std::int32_t next = 0;
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i)
            if (prompt.token_ids[i] == image_pad_)
                prompt.feature_rows[i] = next++;
        if (next != rows)
            throw std::invalid_argument(
                "MiniCPM-V 4.6 image placeholder tokens do not match the image features");
        const double setup_ms = since_ms(start);
        auto result = generate(prompt, options);
        result.setup_ms = setup_ms;
        return result;
    }

    TextResult run(const api::TextContinuationRequest& request, api::ConfigView config) override {
        const auto options = read_options(config, text_fields());
        std::lock_guard<std::mutex> lock(mutex_);
        const auto budget = static_cast<std::size_t>(prompt_budget(options));
        PromptRows prompt;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix)) {
            if (code_points(*text) > tokenizer_->max_code_points(budget))
                prompt_too_long();
            std::string formatted(*text);
            if (options.use_chat_template)
                formatted =
                    format_conversation(recipe(options), {{Role::User, checked_text(*text)}});
            auto ids = tokenizer_->encode(formatted, budget);
            if (!ids)
                prompt_too_long();
            prompt.token_ids = std::move(*ids);
        } else {
            const auto& tokens = std::get<Span<const std::int32_t>>(request.prefix);
            if (options.use_chat_template)
                throw std::invalid_argument(
                    "MiniCPM-V 4.6 applies the chat template to text prefixes only");
            if (tokens.size() > budget)
                prompt_too_long();
            prompt.token_ids.assign(tokens.begin(), tokens.end());
        }
        prompt.feature_rows.assign(prompt.token_ids.size(), -1);
        return generate(prompt, options);
    }

  private:
    // Prompt rows left after max_new_tokens; a prompt needs at least one.
    std::int64_t prompt_budget(const Options& options) const {
        const std::int64_t budget =
            static_cast<std::int64_t>(decoder_->capacity()) - options.max_new_tokens;
        if (budget < 1)
            throw std::invalid_argument("MiniCPM-V 4.6 max_new_tokens leaves no room for a prompt "
                                        "in the bundle's " +
                                        std::to_string(decoder_->capacity()) + "-token capacity");
        return budget;
    }

    [[noreturn]] void prompt_too_long() const {
        throw std::invalid_argument(
            "MiniCPM-V 4.6 prompt plus max_new_tokens exceeds the bundle's " +
            std::to_string(decoder_->capacity()) + "-token capacity");
    }

    static Role role(api::MessageRole value) {
        switch (value) {
        case api::MessageRole::System:
            return Role::System;
        case api::MessageRole::User:
            return Role::User;
        case api::MessageRole::Assistant:
            return Role::Assistant;
        default:
            throw std::invalid_argument(
                "MiniCPM-V 4.6 accepts system, user and assistant messages");
        }
    }

    // Placeholder spellings in text would be expanded as images by the provider processor.
    static std::string checked_text(std::string_view text) {
        std::string value(text);
        if (value.find(kImagePad) != std::string::npos ||
            value.find(kVideoPad) != std::string::npos)
            throw std::invalid_argument(
                "MiniCPM-V 4.6 text parts cannot contain media placeholders");
        return value;
    }

    const Json& recipe(const Options& options) const {
        return vision_.at("prompts").at(options.enable_thinking ? "true" : "false");
    }

    // Replace each rendered image placeholder, in order, by the image's slice markup.
    static std::string expand_images(const std::string& text,
                                     const std::vector<PreparedImage>& images) {
        std::string result;
        std::size_t position = 0, index = 0;
        const std::string marker = kImagePad;
        for (std::size_t found = text.find(marker); found != std::string::npos;
             found = text.find(marker, position)) {
            if (index == images.size())
                throw std::logic_error(
                    "MiniCPM-V 4.6 rendered more image placeholders than images");
            result.append(text, position, found - position);
            result += image_placeholder(images[index], static_cast<std::int32_t>(index));
            ++index;
            position = found + marker.size();
        }
        if (index != images.size())
            throw std::logic_error("MiniCPM-V 4.6 rendered fewer image placeholders than images");
        result.append(text, position, std::string::npos);
        return result;
    }

    TextResult generate(const PromptRows& prompt, const Options& options) {
        Sampler sampler(options.sampling);
        const auto generation =
            decoder_->generate(prompt, options.max_new_tokens, eos_ids_, sampler);
        TextResult result(tokenizer_->decode(generation.token_ids), generation.token_ids,
                          generation.prefill_ms, generation.decode_ms);
        return result;
    }

    Json vision_;
    ImageSettings settings_;
    std::vector<std::int32_t> eos_ids_;
    std::int32_t image_pad_{-1};
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<Decoder> decoder_;
    std::unique_ptr<VisionEncoder> vision_encoder_;
    std::mutex mutex_;
};

} // namespace
} // namespace trtmc::minicpmv4_6

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("minicpmv4_6 does not support --kv-cache-size");
    return new trtmc::minicpmv4_6::Model(context);
}
