/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "native_task.h"

#include "lexical_table.h"
#include "native_decoder.h"
#include "native_visual.h"
#include "sampler.h"
#include "tokenizer.h"
#include "trtmc/internal/language.h"
#include "trtmc/internal/model.h"

#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <type_traits>

namespace trtmc::qwen3_8_flash_next {
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
        throw std::invalid_argument("Flash-Next native media requires messages");
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
        throw std::invalid_argument("Flash-Next native media requires a user query");
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
                throw std::invalid_argument("Flash-Next native system message must be first");
            continue;
        }
        if (messages[i].role == api::MessageRole::User) {
            output += value(recipe.at("user"), "prefix") + content[i] +
                      value(recipe.at("user"), "suffix");
            continue;
        }
        if (messages[i].role != api::MessageRole::Assistant)
            throw std::invalid_argument(
                "Flash-Next native media accepts system, user and assistant roles");
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
            throw std::invalid_argument(
                std::string("Flash-Next native option outside int32 range: ") + name);
        return static_cast<int32_t>(number);
    };
    TextGenerationConfig cfg;
    cfg.max_new_tokens = integer("max_new_tokens", length);
    if (cfg.max_new_tokens <= 0)
        throw std::invalid_argument("Flash-Next native max_new_tokens must be positive");
    cfg.top_k = integer("top_k", 1);
    cfg.seed = integer("seed", -1);
    cfg.temperature = *api::config_get<double>(supplied, schema, "temperature");
    cfg.top_p = *api::config_get<double>(supplied, schema, "top_p");
    cfg.use_chat_template = *api::config_get<bool>(supplied, schema, "use_chat_template");
    cfg.enable_thinking = *api::config_get<bool>(supplied, schema, "enable_thinking");
    if (media && !cfg.use_chat_template)
        throw std::invalid_argument("Flash-Next native media requires its provider chat template");
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
    explicit NativeMediaTask(const FamilyContext& c)
        : reader_(c.reader), meta_(Json::parse(text("native.json"))), table_(reader_),
          decoder_(
              c.backend, [this](const auto& n) { return reader_.read_section(n); },
              [this](const auto& ids) { return table_.rows(ids); }),
          tokenizer_(text("tokenizer.json")), task_(c.reader.info().task),
          input_limit_(meta_.at("max_input_length")),
          max_visual_tokens_(std::min(input_limit_, 1024)) {
        if (meta_.at("format") != 1 || meta_.at("task") != task_ ||
            meta_.at("weight_format") != "prepared_flash_next_nvfp4" ||
            meta_.at("capacity") != decoder_.capacity() || input_limit_ <= 0 ||
            input_limit_ > decoder_.capacity())
            throw std::runtime_error("Invalid Flash-Next native bundle metadata");
        if (task_ == IVisionLanguageGeneration::kTask) {
            const auto& m = meta_.at("visual");
            NativeVisionConfig v;
            v.hidden = m.at("hidden");
            v.patch = m.at("patch");
            v.temporal = m.at("temporal");
            v.merge = m.at("merge");
            v.positions = m.at("positions");
            v.head_dim = m.at("head_dim");
            v.theta = m.at("theta");
            v.max_tokens = m.at("max_tokens");
            v.image_min_pixels = m.at("image_min_pixels");
            v.image_max_pixels = m.at("image_max_pixels");
            v.video_min_pixels = m.at("video_min_pixels");
            v.video_max_pixels = m.at("video_max_pixels");
            const auto plan = reader_.read_section("visual.plan");
            visual_ = std::make_unique<NativeVisualEncoder>(
                c.backend.create_module(plan.data(), plan.size(), {}), v);
        } else if (task_ != ITextGeneration::kTask)
            throw std::runtime_error("Unrecognized Flash-Next task");
    }
    const char* task() const noexcept override { return task_.c_str(); }
    int32_t default_max_new_tokens() const override {
        return std::min(128, decoder_.capacity() - 1);
    }
    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::ITextContinuation>(*this, fields(false)),
                api::bind<api::IImagesTextToText>(*this, fields(true)),
                api::bind<api::IVideoTextToText>(*this, fields(true)),
                api::bind<api::IImageVideoTextToText>(*this, fields(true))};
    }
    TextResult run(const api::TextContinuationRequest& request, api::ConfigView config) override {
        const auto* text = std::get_if<std::string_view>(&request.prefix);
        if (!text)
            throw std::invalid_argument("Flash-Next native accepts UTF-8 text prefixes only");
        return generate(std::string(*text),
                        generation_config(config, false, default_max_new_tokens()));
    }
    TextResult run(const api::ImagesTextToTextRequest& input, api::ConfigView cfg) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Flash-Next native media does not map tools");
        return execute_messages(input.messages,
                                generation_config(cfg, true, default_max_new_tokens()));
    }
    TextResult run(const api::VideoTextToTextRequest& input, api::ConfigView cfg) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Flash-Next native media does not map tools");
        return execute_messages(input.messages,
                                generation_config(cfg, true, default_max_new_tokens()));
    }
    TextResult run(const api::ImageVideoTextToTextRequest& input, api::ConfigView cfg) override {
        return execute_messages(input.messages,
                                generation_config(cfg, true, default_max_new_tokens()));
    }
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!config.system_prompt.empty() && !config.use_chat_template)
            throw std::invalid_argument("Flash-Next system prompts need a provider template");
        std::vector<Message> messages;
        if (!config.system_prompt.empty())
            messages.push_back({api::MessageRole::System, config.system_prompt});
        messages.push_back({api::MessageRole::User, prompt});
        const std::string formatted =
            config.use_chat_template
                ? format_prompt(meta_.at("prompts").at(config.enable_thinking ? "true" : "false"),
                                messages)
                : prompt;
        return execute(formatted, {}, config);
    }
    TextResult generate(const std::string& prompt, const float* pixels, int32_t height,
                        int32_t width, const TextGenerationConfig& cfg) override {
        if (!pixels) {
            if (height || width)
                throw std::invalid_argument("Flash-Next native image dimensions require pixels");
            return generate(prompt, cfg);
        }
        if (height <= 0 || width <= 0 || !cfg.use_chat_template ||
            static_cast<uint64_t>(height) * width >
                std::numeric_limits<std::size_t>::max() / 3 / sizeof(float))
            throw std::invalid_argument(
                "Flash-Next native image requires positive dimensions and provider chat template");
        std::lock_guard<std::mutex> lock(mutex_);
        auto groups = visual_->encode(
            api::ImageView{pixels, static_cast<std::size_t>(height) * width * 3 * sizeof(float),
                           static_cast<uint32_t>(height), static_cast<uint32_t>(width), 3,
                           api::ImageFormat::Float32});
        const auto content = native_visual_text(groups) + prompt;
        std::vector<Message> messages;
        if (!cfg.system_prompt.empty())
            messages.push_back({api::MessageRole::System, cfg.system_prompt});
        messages.push_back({api::MessageRole::User, content});
        return execute(
            format_prompt(meta_.at("prompts").at(cfg.enable_thinking ? "true" : "false"), messages),
            groups, cfg);
    }

  private:
    template <typename Messages>
    TextResult execute_messages(Messages messages, const TextGenerationConfig& cfg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (messages.empty())
            throw std::invalid_argument("Flash-Next native media requires messages");
        std::vector<Message> prompt;
        if (!cfg.system_prompt.empty())
            prompt.push_back({api::MessageRole::System, cfg.system_prompt});
        std::vector<NativeVisualGroup> visual_groups;
        std::size_t visual_tokens = 0;
        for (const auto& message : messages) {
            if ((message.role != api::MessageRole::User &&
                 message.role != api::MessageRole::Assistant &&
                 message.role != api::MessageRole::System) ||
                message.parts.empty())
                throw std::invalid_argument(
                    "Flash-Next native media requires nonempty system/user/assistant messages");
            if (message.role == api::MessageRole::System && !prompt.empty())
                throw std::invalid_argument("Flash-Next native system message must be first");
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
                                    "Flash-Next native system message cannot contain media");
                            auto groups = visual_->encode(value);
                            for (const auto& group : groups)
                                visual_tokens +=
                                    static_cast<std::size_t>(group.height) * group.width;
                            if (visual_tokens > static_cast<std::size_t>(
                                                    std::min(input_limit_, max_visual_tokens_)))
                                throw std::invalid_argument(
                                    "Flash-Next native media exceeds visual input capacity");
                            item.text += native_visual_text(groups);
                            for (auto& group : groups)
                                visual_groups.push_back(std::move(group));
                        } else
                            throw std::invalid_argument(
                                "Flash-Next native media does not map tool or reasoning parts");
                    },
                    part);
            }
            prompt.push_back(std::move(item));
        }
        return execute(
            format_prompt(meta_.at("prompts").at(cfg.enable_thinking ? "true" : "false"), prompt),
            visual_groups, cfg);
    }
    std::string text(const std::string& name) const {
        const auto raw = reader_.read_section(name);
        return {raw.begin(), raw.end()};
    }
    TextResult execute(const std::string& prompt, const std::vector<NativeVisualGroup>& groups,
                       TextGenerationConfig c) {
        if (!std::isfinite(c.temperature) || c.temperature < 0 || !std::isfinite(c.top_p) ||
            c.top_p <= 0 || c.top_p > 1 || c.top_k < 0 || c.seed < -1 || !std::isfinite(c.min_p) ||
            c.min_p < 0 || c.min_p > 1 || c.eos_token_id < -1 || c.eos_token_id >= 248320 ||
            c.repetition_penalty != 1 || !c.lora_adapter_id.empty() || c.stop_on_boxed_answer ||
            (c.text_generation_mode != "auto" && c.text_generation_mode != "autoregressive") ||
            c.source_language_token_id != -1 || c.forced_bos_token_id != -1 ||
            c.guidance_scale != -1 || c.cfg_scale != -1 || c.num_steps != -1 || c.sde_gamma != -1 ||
            !c.initial_latents.empty() || !c.condition_latents.empty() ||
            !c.condition_mask.empty() || !c.sampling_steps.empty() || !c.sde_noises.empty() ||
            c.block_length || c.confidence_threshold != -1)
            throw std::invalid_argument("Unsupported Flash-Next native generation control");
        auto ids = tokenizer_.encode(prompt);
        const int budget = c.max_new_tokens > 0 ? c.max_new_tokens : default_max_new_tokens();
        if (ids.empty() || ids.size() > static_cast<std::size_t>(input_limit_) ||
            budget > decoder_.capacity() - static_cast<int>(ids.size()))
            throw std::invalid_argument("Flash-Next prompt and generation exceed bundle capacity");
        std::vector<uint8_t> types(ids.size(), 0);
        std::vector<Grid> images, videos;
        std::size_t group = 0;
        int image = meta_.at("image_token_id"), video = meta_.at("video_token_id"),
            start = meta_.at("vision_start_token_id"), end = meta_.at("vision_end_token_id");
        for (std::size_t i = 0; i < ids.size();) {
            if (ids[i] != image && ids[i] != video) {
                ++i;
                continue;
            }
            const bool is_video = ids[i] == video;
            const auto begin = i;
            while (i < ids.size() && ids[i] == (is_video ? video : image))
                ++i;
            if (group >= groups.size() || !begin || ids[begin - 1] != start || i == ids.size() ||
                ids[i] != end)
                throw std::invalid_argument(
                    "Flash-Next visual placeholder lacks matching media or delimiters");
            const auto& v = groups[group++];
            if (v.video != is_video || v.height <= 0 || v.width <= 0 ||
                static_cast<int64_t>(v.height) * v.width != static_cast<int64_t>(i - begin) ||
                v.features.size() != (i - begin) * 2560)
                throw std::invalid_argument("Flash-Next visual features do not match prompt span");
            std::fill(types.begin() + begin, types.begin() + i, is_video ? 2 : 1);
            (is_video ? videos : images).push_back({1, v.height * 2, v.width * 2});
        }
        if (group != groups.size())
            throw std::invalid_argument("Unused Flash-Next media feature group");
        const auto pos = prompt_positions(types, images, videos, 2, input_limit_);
        std::vector<std::uint16_t> logits;
        decoder_.reset();
        std::size_t gi = 0, row = 0;
        for (std::size_t off = 0; off < ids.size();) {
            const auto last = std::min(ids.size(), off + decoder_.max_chunk());
            std::vector<uint8_t> mask(last - off, 0);
            std::vector<uint16_t> pixels;
            for (std::size_t j = off; j < last; ++j) {
                if (!types[j])
                    continue;
                mask[j - off] = 1;
                const auto& v = groups[gi].features;
                pixels.insert(pixels.end(), v.begin() + row, v.begin() + row + 2560);
                row += 2560;
                if (row == v.size()) {
                    row = 0;
                    ++gi;
                }
            }
            logits = decoder_.forward({ids.begin() + off, ids.begin() + last},
                                      {pos.prompt.begin() + off, pos.prompt.begin() + last}, mask,
                                      pixels, last == ids.size());
            off = last;
        }
        const auto params = flash_next_sampling_params_from_config(
            c, -1, meta_.at("eos").get<std::vector<int32_t>>());
        auto sampler = create_flash_next_sampler(params);
        std::vector<int32_t> answer;
        std::vector<float> scores(logits.size());
        for (int step = 0; step < budget; ++step) {
            for (std::size_t k = 0; k < logits.size(); ++k) {
                __half_raw raw{};
                raw.x = logits[k];
                scores[k] = __half2float(static_cast<__half>(raw));
                if (!std::isfinite(scores[k]))
                    throw std::runtime_error("Flash-Next returned non-finite logits");
            }
            const auto token =
                sampler->sample(scores.data(), static_cast<int32_t>(scores.size()), params);
            answer.push_back(token.token_id);
            if (token.is_eos || step + 1 == budget)
                break;
            const auto next = pos.next + step;
            logits = decoder_.forward({token.token_id}, {{next, next, next}}, {}, {}, true);
        }
        return {tokenizer_.decode(answer), std::move(answer)};
    }
    BundleReader reader_;
    Json meta_;
    LexicalTable table_;
    NativeDecoder decoder_;
    Tokenizer tokenizer_;
    std::unique_ptr<NativeVisualEncoder> visual_;
    std::string task_;
    int input_limit_, max_visual_tokens_;
    std::mutex mutex_;
};

class TextOnly final : public ITextGeneration, public api::IModel, public api::ITextContinuation {
  public:
    explicit TextOnly(const FamilyContext& c) : state_(c) {}
    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::ITextContinuation>(*this, fields(false))};
    }
    int32_t default_max_new_tokens() const override { return state_.default_max_new_tokens(); }
    TextResult run(const api::TextContinuationRequest& req, api::ConfigView cfg) override {
        return state_.run(req, cfg);
    }
    TextResult generate(const std::string& prompt, const TextGenerationConfig& c) override {
        return state_.generate(prompt, c);
    }

  private:
    NativeMediaTask state_;
};
} // namespace
ITask* create_native(const FamilyContext& c) {
    if (c.reader.info().task == ITextGeneration::kTask)
        return new TextOnly(c);
    auto* vision = new NativeMediaTask(c);
    return static_cast<IVisionLanguageGeneration*>(vision);
}
} // namespace trtmc::qwen3_8_flash_next
