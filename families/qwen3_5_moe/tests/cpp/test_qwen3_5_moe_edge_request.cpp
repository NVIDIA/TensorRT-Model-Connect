/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_5_moe/runtime/edge_llm/media.h"

#include <edgellm/cpp/chatTemplate/chatTemplate.h>
#include <functional>
#include <iostream>
namespace edge = trtmc::qwen3_5_moe::edge_llm;
int main() {
    int failures = 0;
    const auto check = [&](bool value) {
        if (!value)
            ++failures;
    };
    const auto text_defaults = edge::generation_config({}, false, 128);
    const auto image_defaults = edge::generation_config({}, true, 128);
    check(!text_defaults.use_chat_template && text_defaults.seed == -1);
    check(image_defaults.use_chat_template && image_defaults.seed == -1);
    check(text_defaults.max_new_tokens == 128 && image_defaults.max_new_tokens == 128);
    trtmc::TextGenerationConfig config;
    config.temperature = 0;
    config.top_k = 1;
    config.top_p = 1;
    for (const int budget : {0, -1, 10, 1024}) {
        config.max_new_tokens = budget;
        for (const bool chat : {false, true}) {
            config.use_chat_template = chat;
            config.enable_thinking = chat;
            auto raw = edge::make_request("raw", config, false);
            check(raw.applyChatTemplate == chat && raw.enableThinking == chat);
            check(raw.requests.front().messages.size() == 1);
            check(raw.requests.front().messages[0].contents[0].content == "raw");
            check(raw.maxGenerateLength == (budget > 0 ? budget : 128));
            auto image = edge::make_request("image", config, true);
            check(image.applyChatTemplate == chat && image.enableThinking == chat);
            check(image.requests.front().messages[0].role == "user");
            check(image.requests.front().messages.size() == 1);
            check(image.requests.front().messages[0].contents[0].type == "image");
            check(image.requests.front().messages[0].contents[1].content == "image");
            check(image.temperature == 0 && image.topK == 1 && image.topP == 1);
        }
    }
    const nlohmann::json tokenizer{{"post_processor", {{"type", "ByteLevel"}}},
                                   {"added_tokens", {{{"id", 248046}, {"content", "<|im_end|>"}}}}};
    const nlohmann::json tokenizer_config{{"eos_token", "<|im_end|>"}};
    const nlohmann::json engine{{"eos_token_id", {248044, 248046}}};
    edge::validate_raw_tokenizer(tokenizer, tokenizer_config, engine);
    for (const int fault : {0, 1, 2, 3, 4}) {
        auto tok = tokenizer, config = tokenizer_config, model = engine;
        if (fault == 0)
            tok["post_processor"] = {{"type", "TemplateProcessing"}};
        if (fault == 1)
            config["add_bos_token"] = true;
        if (fault == 2)
            config["add_eos_token"] = true;
        if (fault == 3)
            tok["added_tokens"][0]["id"] = 200020;
        if (fault == 4)
            model["eos_token_id"] = {248046, 200020};
        try {
            edge::validate_raw_tokenizer(tok, config, model);
            ++failures;
        } catch (const std::exception&) {
        }
    }
    trt_edgellm::chat_template::ChatTemplate formatter;
    check(formatter.load(QWEN35_MOE_EDGE_TEMPLATE));
    for (const std::string prompt : {"", " ", "  describe  "}) {
        for (const bool chat : {false, true})
            for (const bool thinking : {false, true}) {
                config.use_chat_template = chat;
                config.enable_thinking = thinking;
                for (const bool has_image : {false, true}) {
                    const auto request = edge::make_request(prompt, config, has_image);
                    trt_edgellm::rt::LLMGenerationRequest::FormattedRequest formatted;
                    auto options = trt_edgellm::chat_template::ChatTemplate::optionsFrom(request);
                    check(formatter.apply(request.requests.front(), formatted, options));
                    const std::string body =
                        (has_image ? "<|vision_start|><|image_pad|><|vision_end|>" : "") + prompt;
                    const std::string expected =
                        chat ? "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n" +
                                   (thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
                             : body;
                    check(formatted.formattedCompleteRequest == expected);
                }
            }
    }
    namespace api = trtmc::internal;
    const std::uint8_t first[] = {255, 0, 0}, second[] = {0, 255, 0};
    std::vector<api::ImageView> frames{{first, 3, 1, 1, 3, api::ImageFormat::UInt8},
                                       {second, 3, 1, 1, 3, api::ImageFormat::UInt8}};
    std::vector<double> times{1.0, 2.5};
    api::VideoView video{{frames.data(), frames.size()}, {times.data(), times.size()}};
    const auto packed = edge::video_buffer(video);
    check(packed.isVideo && packed.frames == 2 && packed.timestamps == times);
    check(packed.data()[0] == 255 && packed.data()[4] == 255);
    const auto single = edge::video_buffer({{frames.data(), 1}, {}});
    check(single.isVideo && single.frames == 1 && single.fps == 1.0);
    std::vector<api::VideoTextPart> parts{video, api::TextPartView{"Describe the video."}};
    api::VideoTextMessage message{api::MessageRole::User, {parts.data(), parts.size()}};
    api::VideoTextToTextRequest input{{&message, 1}, {}};
    const auto mapped = edge::media_request(input, image_defaults, 128);
    check(mapped.requests.front().messages.front().contents.front().type == "video");
    check(mapped.requests.front().imageBuffers.front().timestamps == times);
    for (int fault : {0, 1, 2, 3, 4}) {
        auto bad = video;
        auto bad_frames = frames;
        auto bad_times = times;
        if (fault == 0)
            bad.frames = {};
        if (fault == 1)
            bad.timestamps_seconds = {times.data(), 1};
        if (fault == 2 || fault == 3) {
            bad_times[1] = fault == 2 ? 0.0 : std::numeric_limits<double>::infinity();
            bad.timestamps_seconds = {bad_times.data(), bad_times.size()};
        }
        if (fault == 4) {
            bad_frames[1].width = 2;
            bad.frames = {bad_frames.data(), bad_frames.size()};
        }
        try {
            edge::video_buffer(bad);
            ++failures;
        } catch (const std::invalid_argument&) {
        }
    }
    trt_edgellm::rt::LLMGenerationResponse response{};
    response.outputIds = {{42}};
    response.outputTexts = {"answer"};
    response.inputTokenCounts = {300};
    response.finishReasons = {trt_edgellm::rt::FinishReason::kEndId};
    edge::validate_response(response, 1024, 1024, 10);
    const auto rejects = [&](const auto& invalid, int budget) {
        try {
            edge::validate_response(invalid, 1024, 1024, budget);
            ++failures;
        } catch (const std::exception&) {
        }
    };
    rejects(response, 800); // Early EOS does not make an overcapacity request valid.
    auto missing = response;
    missing.inputTokenCounts.clear();
    rejects(missing, 10);
    auto clipped = response;
    clipped.finishReasons = {trt_edgellm::rt::FinishReason::kLength};
    rejects(clipped, 10);
    edge::validate_response(clipped, 1024, 1024, 1);
    auto failed = response;
    failed.finishReasons = {trt_edgellm::rt::FinishReason::kError};
    rejects(failed, 10);
    if (failures)
        std::cerr << "Qwen3.5-MoE Edge request mapping failed\n";
    return failures ? 1 : 0;
}
