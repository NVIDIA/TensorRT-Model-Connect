/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/native_media.h"

#include "families/qwen3_8/runtime/native_visual.h"
#include "families/qwen3_8/runtime/plugin_helpers.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>

namespace trtmc::qwen3_8 {
namespace {
namespace api = trtmc::internal;
using Json = nlohmann::json;

// Python/Jinja strip() whitespace, including Unicode White_Space and the four
// ASCII information separators. Preserve all interior text and non-English data.
std::string trim(std::string text) {
    static const std::vector<std::string> spaces{" ",
                                                 "\t",
                                                 "\n",
                                                 "\r",
                                                 "\v",
                                                 "\f",
                                                 "\x1c",
                                                 "\x1d",
                                                 "\x1e",
                                                 "\x1f",
                                                 "\xc2\x85",
                                                 "\xc2\xa0",
                                                 "\xe1\x9a\x80",
                                                 "\xe2\x80\x80",
                                                 "\xe2\x80\x81",
                                                 "\xe2\x80\x82",
                                                 "\xe2\x80\x83",
                                                 "\xe2\x80\x84",
                                                 "\xe2\x80\x85",
                                                 "\xe2\x80\x86",
                                                 "\xe2\x80\x87",
                                                 "\xe2\x80\x88",
                                                 "\xe2\x80\x89",
                                                 "\xe2\x80\x8a",
                                                 "\xe2\x80\xa8",
                                                 "\xe2\x80\xa9",
                                                 "\xe2\x80\xaf",
                                                 "\xe2\x81\x9f",
                                                 "\xe3\x80\x80"};
    std::size_t begin = 0, end = text.size();
    bool found = true;
    while (found && begin < end) {
        found = false;
        for (const auto& space : spaces)
            if (space.size() <= end - begin && text.compare(begin, space.size(), space) == 0) {
                begin += space.size();
                found = true;
                break;
            }
    }
    found = true;
    while (found && begin < end) {
        found = false;
        for (const auto& space : spaces)
            if (space.size() <= end - begin &&
                text.compare(end - space.size(), space.size(), space) == 0) {
                end -= space.size();
                found = true;
                break;
            }
    }
    return text.substr(begin, end - begin);
}

std::string value(const Json& object, const char* name) {
    return object.at(name).get<std::string>();
}
struct Message {
    api::MessageRole role;
    std::string text;
};

std::string format_prompt(const Json& recipe, const std::vector<Message>& messages) {
    if (messages.empty())
        throw std::invalid_argument("Qwen native media requires messages");
    int64_t last_user = -1;
    std::vector<std::string> content;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        content.push_back(trim(messages[i].text));
        const auto& text = content.back();
        const bool tool_response = text.rfind("<tool_response>", 0) == 0 && text.size() >= 16 &&
                                   text.compare(text.size() - 16, 16, "</tool_response>") == 0;
        if (messages[i].role == api::MessageRole::User && !tool_response)
            last_user = i;
    }
    if (last_user < 0)
        throw std::invalid_argument("Qwen native media requires a user query");
    std::string output;
    if (messages[0].role != api::MessageRole::System)
        output = value(recipe, "default_prefix");
    else if (content[0].empty())
        output = value(recipe, "empty_system_prefix");
    else
        output = value(recipe.at("system"), "prefix") + content[0] +
                 value(recipe.at("system"), "suffix");
    for (std::size_t i = 0; i < messages.size(); ++i) {
        if (messages[i].role == api::MessageRole::System) {
            if (i)
                throw std::invalid_argument("Qwen native system message must be first");
            continue;
        }
        if (messages[i].role == api::MessageRole::User) {
            output += value(recipe.at("user"), "prefix") + content[i] +
                      value(recipe.at("user"), "suffix");
            continue;
        }
        if (messages[i].role != api::MessageRole::Assistant)
            throw std::invalid_argument(
                "Qwen native media accepts system, user and assistant roles");
        std::string body = content[i], reasoning;
        const auto close = body.find("</think>");
        if (recipe.at("parse_assistant_thinking").get<bool>() && close != std::string::npos) {
            reasoning = body.substr(0, close);
            while (!reasoning.empty() && reasoning.back() == '\n')
                reasoning.pop_back();
            const auto open = reasoning.rfind("<think>");
            if (open != std::string::npos)
                reasoning.erase(0, open + 7);
            reasoning = trim(reasoning);
            body.erase(0, body.rfind("</think>") + 8);
            body.erase(0, body.find_first_not_of('\n'));
        }
        const auto& form = recipe.at(static_cast<int64_t>(i) > last_user ? "assistant_recent"
                                                                         : "assistant_history");
        output += value(form, "prefix");
        if (form.at("include_reasoning").get<bool>())
            output += reasoning + value(form, "reasoning_suffix");
        output += body + value(form, "suffix");
    }
    return output + value(recipe, "generation");
}

NativeVisionConfig vision_config(const Json& meta) {
    NativeVisionConfig c;
    c.patch = meta.at("patch");
    c.temporal = meta.at("temporal");
    c.merge = meta.at("merge");
    c.positions = meta.at("positions");
    c.head_dim = meta.at("head_dim");
    c.hidden = meta.at("hidden");
    c.theta = meta.at("theta");
    c.min_tokens = meta.at("min_tokens");
    c.max_tokens = meta.at("max_tokens");
    c.mean = meta.at("mean").get<std::array<float, 3>>();
    c.stddev = meta.at("stddev").get<std::array<float, 3>>();
    c.validate();
    return c;
}

Span<const api::ConfigField> fields(bool media) {
    static const std::vector<api::ConfigField> text{
        {"max_new_tokens", api::ConfigKind::I64, std::nullopt, "Generated token budget"},
        {"temperature", api::ConfigKind::F64, 1.0, "Sampling temperature"},
        {"top_k", api::ConfigKind::I64, int64_t{1}, "Top-k sampling"},
        {"top_p", api::ConfigKind::F64, 1.0, "Nucleus sampling"},
        {"seed", api::ConfigKind::I64, int64_t{-1}, "Sampling seed; -1 leaves it unspecified"},
        {"use_chat_template", api::ConfigKind::Bool, false, "Apply checkpoint provider template"},
        {"enable_thinking", api::ConfigKind::Bool, true, "Provider template thinking option"},
    };
    static const auto multimodal = [] {
        auto copy = text;
        for (auto& item : copy)
            if (item.name == "use_chat_template")
                item.default_value = true;
        return copy;
    }();
    const auto& selected = media ? multimodal : text;
    return {selected.data(), selected.size()};
}

TextGenerationConfig generation_config(api::ConfigView supplied, bool media, int32_t length) {
    const auto schema = fields(media);
    api::validate_config(schema, supplied);
    auto integer = [&](const char* name, int32_t fallback) {
        const auto number = api::config_get<int64_t>(supplied, schema, name).value_or(fallback);
        if (number < INT32_MIN || number > INT32_MAX)
            throw std::invalid_argument(std::string("Qwen native option outside int32 range: ") +
                                        name);
        return static_cast<int32_t>(number);
    };
    TextGenerationConfig cfg;
    cfg.max_new_tokens = integer("max_new_tokens", length);
    if (cfg.max_new_tokens <= 0)
        throw std::invalid_argument("Qwen native max_new_tokens must be positive");
    cfg.top_k = integer("top_k", 1);
    cfg.seed = integer("seed", -1);
    cfg.temperature = *api::config_get<double>(supplied, schema, "temperature");
    cfg.top_p = *api::config_get<double>(supplied, schema, "top_p");
    cfg.use_chat_template = *api::config_get<bool>(supplied, schema, "use_chat_template");
    cfg.enable_thinking = *api::config_get<bool>(supplied, schema, "enable_thinking");
    if (media && !cfg.use_chat_template)
        throw std::invalid_argument("Qwen native media requires its provider chat template");
    return cfg;
}

class NativeMediaTask final : public ITextGeneration,
                              public IVisionLanguageGeneration,
                              public api::IModel,
                              public api::ITextContinuation,
                              public api::IImagesTextToText,
                              public api::IVideoTextToText,
                              public api::IImageVideoTextToText {
  public:
    NativeMediaTask(const BundleReader& bundle, std::unique_ptr<RecurrentPipeline> decoder,
                    std::unique_ptr<ITrtModule> visual, std::shared_ptr<ITokenizer> tokenizer)
        : meta_(Json::parse(require_text_section(bundle, "native_visual.json"))),
          decoder_(std::move(decoder)), visual_(std::move(visual), vision_config(meta_)),
          tokenizer_(std::move(tokenizer)), task_(bundle.info().task),
          capacity_(meta_.at("capacity")), input_limit_(meta_.at("input_limit")),
          hidden_(meta_.at("hidden")) {
        const auto runtime = Json::parse(require_text_section(bundle, "runtime.json"));
        if (meta_.at("schema_version") != 1 ||
            (task_ != "vision_language_generation" && task_ != "images_text_to_text") ||
            capacity_ < 2 || input_limit_ < 1 || input_limit_ > capacity_ || !tokenizer_ ||
            !decoder_ || runtime.at("max_cache_length") != capacity_ ||
            runtime.at("hidden_size") != hidden_ ||
            runtime.at("precision") != meta_.at("precision"))
            throw std::runtime_error("Qwen native media metadata disagrees with decoder bundle");
        const auto precision = value(meta_, "precision");
        if (precision != "fp16" && precision != "fp32" && precision != "bf16")
            throw std::runtime_error("Invalid Qwen native media precision");
        dtype_ = precision == "fp16"   ? DType::kFloat16
                 : precision == "bf16" ? DType::kBFloat16
                                       : DType::kFloat32;
        const auto& tokens = meta_.at("token_ids");
        image_token_ = tokens.at("image_token");
        video_token_ = tokens.at("video_token");
        start_token_ = tokens.at("vision_start");
        end_token_ = tokens.at("vision_end");
        for (const auto& [id, symbol] :
             std::vector<std::pair<int32_t, const char*>>{{image_token_, "<|image_pad|>"},
                                                          {video_token_, "<|video_pad|>"},
                                                          {start_token_, "<|vision_start|>"},
                                                          {end_token_, "<|vision_end|>"}})
            if (id < 0 || tokenizer_->id_for_token(symbol) != id)
                throw std::runtime_error("Qwen native media tokenizer metadata disagrees");
        for (const char* mode : {"false", "true"}) {
            const auto& recipe = meta_.at("prompts").at(mode);
            for (const char* field : {"default_prefix", "empty_system_prefix", "generation"})
                (void)value(recipe, field);
            (void)recipe.at("parse_assistant_thinking").get<bool>();
            for (const char* kind : {"user", "system", "assistant_recent", "assistant_history"}) {
                (void)value(recipe.at(kind), "prefix");
                (void)value(recipe.at(kind), "suffix");
                if (std::string(kind).rfind("assistant", 0) == 0) {
                    (void)recipe.at(kind).at("include_reasoning").get<bool>();
                    (void)value(recipe.at(kind), "reasoning_suffix");
                }
            }
        }
    }

    const char* task() const noexcept override { return task_.c_str(); }
    int32_t default_max_new_tokens() const override { return std::min(128, capacity_ - 1); }
    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::ITextContinuation>(*this, fields(false)),
                api::bind<api::IImagesTextToText>(*this, fields(true)),
                api::bind<api::IVideoTextToText>(*this, fields(true)),
                api::bind<api::IImageVideoTextToText>(*this, fields(true))};
    }
    TextResult run(const api::TextContinuationRequest& request, api::ConfigView config) override {
        const auto* text = std::get_if<std::string_view>(&request.prefix);
        if (!text)
            throw std::invalid_argument("Qwen native accepts UTF-8 text prefixes only");
        return generate(std::string(*text),
                        generation_config(config, false, default_max_new_tokens()));
    }
    TextResult run(const api::ImagesTextToTextRequest& input, api::ConfigView cfg) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Qwen native media does not map tools");
        return execute_messages(input.messages,
                                generation_config(cfg, true, default_max_new_tokens()));
    }
    TextResult run(const api::VideoTextToTextRequest& input, api::ConfigView cfg) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Qwen native media does not map tools");
        return execute_messages(input.messages,
                                generation_config(cfg, true, default_max_new_tokens()));
    }
    TextResult run(const api::ImageVideoTextToTextRequest& input, api::ConfigView cfg) override {
        return execute_messages(input.messages,
                                generation_config(cfg, true, default_max_new_tokens()));
    }
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::string formatted =
            config.use_chat_template
                ? format_prompt(meta_.at("prompts").at(config.enable_thinking ? "true" : "false"),
                                {{api::MessageRole::User, prompt}})
                : prompt;
        return execute(formatted, {}, config);
    }
    TextResult generate(const std::string& prompt, const float* pixels, int32_t height,
                        int32_t width, const TextGenerationConfig& cfg) override {
        if (!pixels) {
            if (height || width)
                throw std::invalid_argument("Qwen native image dimensions require pixels");
            return generate(prompt, cfg);
        }
        if (height <= 0 || width <= 0 || !cfg.use_chat_template ||
            static_cast<uint64_t>(height) * width >
                std::numeric_limits<std::size_t>::max() / 3 / sizeof(float))
            throw std::invalid_argument(
                "Qwen native image requires positive dimensions and provider chat template");
        std::lock_guard<std::mutex> lock(mutex_);
        auto groups = visual_.encode(
            api::ImageView{pixels, static_cast<std::size_t>(height) * width * 3 * sizeof(float),
                           static_cast<uint32_t>(height), static_cast<uint32_t>(width), 3,
                           api::ImageFormat::Float32});
        const auto content = native_visual_text(groups) + prompt;
        return execute(format_prompt(meta_.at("prompts").at(cfg.enable_thinking ? "true" : "false"),
                                     {{api::MessageRole::User, content}}),
                       groups, cfg);
    }

  private:
    template <typename Messages>
    TextResult execute_messages(Messages messages, const TextGenerationConfig& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (messages.empty())
            throw std::invalid_argument("Qwen native media requires messages");
        std::vector<Message> prompt;
        std::vector<NativeVisualGroup> visual_groups;
        std::size_t visual_tokens = 0;
        for (const auto& message : messages) {
            if ((message.role != api::MessageRole::User &&
                 message.role != api::MessageRole::Assistant &&
                 message.role != api::MessageRole::System) ||
                message.parts.empty())
                throw std::invalid_argument(
                    "Qwen native media requires nonempty system/user/assistant messages");
            if (message.role == api::MessageRole::System && !prompt.empty())
                throw std::invalid_argument("Qwen native system message must be first");
            Message item{message.role, {}};
            for (const auto& part : message.parts) {
                std::visit(
                    [&](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, api::TextPartView>)
                            item.text += value.text;
                        else if constexpr (std::is_same_v<T, api::ImageView> ||
                                           std::is_same_v<T, api::VideoView>) {
                            if (message.role == api::MessageRole::System)
                                throw std::invalid_argument(
                                    "Qwen native system message cannot contain media");
                            auto groups = visual_.encode(value);
                            for (const auto& group : groups)
                                visual_tokens +=
                                    static_cast<std::size_t>(group.height) * group.width;
                            if (visual_tokens > static_cast<std::size_t>(input_limit_))
                                throw std::invalid_argument(
                                    "Qwen native media exceeds visual input capacity");
                            item.text += native_visual_text(groups);
                            for (auto& group : groups)
                                visual_groups.push_back(std::move(group));
                        } else
                            throw std::invalid_argument(
                                "Qwen native media does not map tool or reasoning parts");
                    },
                    part);
            }
            prompt.push_back(std::move(item));
        }
        return execute(
            format_prompt(meta_.at("prompts").at(cfg.enable_thinking ? "true" : "false"), prompt),
            visual_groups, cfg);
    }
    TextResult execute(const std::string& prompt, const std::vector<NativeVisualGroup>& groups,
                       TextGenerationConfig cfg) {
        if (cfg.max_new_tokens <= 0)
            cfg.max_new_tokens = default_max_new_tokens();
        const auto ids = tokenizer_->encode(prompt);
        if (ids.size() > static_cast<std::size_t>(input_limit_))
            throw std::invalid_argument("Qwen native prompt exceeds input capacity");
        const auto prepared = native_media_prompt(ids, groups, hidden_, dtype_, image_token_,
                                                  video_token_, start_token_, end_token_);
        const auto output = decoder_->generate_media_ids(ids, prepared, cfg);
        std::vector<int32_t> generated(output.token_ids.begin() +
                                           static_cast<std::ptrdiff_t>(ids.size()),
                                       output.token_ids.end());
        auto text = tokenizer_->decode(generated);
        return {std::move(text), std::move(generated)};
    }
    Json meta_;
    std::unique_ptr<RecurrentPipeline> decoder_;
    NativeVisualEncoder visual_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string task_;
    int32_t capacity_, input_limit_, hidden_, image_token_, video_token_, start_token_, end_token_;
    DType dtype_;
    std::mutex mutex_;
};
} // namespace

ITask* create_native_media_task(const BundleReader& bundle,
                                std::unique_ptr<RecurrentPipeline> decoder,
                                std::unique_ptr<ITrtModule> visual,
                                std::shared_ptr<ITokenizer> tokenizer) {
    return new NativeMediaTask(bundle, std::move(decoder), std::move(visual), std::move(tokenizer));
}
} // namespace trtmc::qwen3_8
