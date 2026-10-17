/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Family runtime regressions: recurrent outputs and optional Edge media requests.

#include "families/qwen3_5/runtime/recurrent_output_initializers.h"

#ifdef TRTMC_HAS_EDGE_LLM
#include "families/qwen3_5/runtime/edge_llm/media.h"
#endif

#include <array>
#include <iostream>
#include <limits>
#include <vector>

namespace {

namespace under_test = trtmc::qwen3_5_recurrent;

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_initialize_rwkv_outputs() {
    std::vector<float> logits;
    std::vector<std::vector<float>> attn;
    std::vector<std::vector<float>> ff;
    std::vector<std::vector<float>> num;
    std::vector<std::vector<float>> den;
    std::vector<std::vector<float>> maxv;

    under_test::initialize_rwkv_outputs(3, 11, 5, logits, attn, ff, num, den, maxv);

    check(logits.size() == 11, "rwkv outputs allocate logits");
    check(attn.size() == 3 && attn[0].size() == 5, "rwkv outputs allocate attn");
    check(ff.size() == 3 && ff[1].size() == 5, "rwkv outputs allocate ff");
    check(num.size() == 3 && num[2].size() == 5, "rwkv outputs allocate num");
    check(den.size() == 3 && den[0].size() == 5, "rwkv outputs allocate den");
    check(maxv.size() == 3 && maxv[1].size() == 5, "rwkv outputs allocate max");
}

void test_initialize_mamba_outputs() {
    std::vector<float> logits;
    std::vector<std::vector<float>> conv;
    std::vector<std::vector<float>> ssm;

    under_test::initialize_mamba_outputs(2, 13, 6, 7, logits, conv, ssm);

    check(logits.size() == 13, "mamba outputs allocate logits");
    check(conv.size() == 2 && conv[0].size() == 6, "mamba outputs allocate conv");
    check(ssm.size() == 2 && ssm[1].size() == 7, "mamba outputs allocate ssm");
}

void test_qwen3_5_recurrent_contracts() {
    std::vector<std::vector<float>> a(2, std::vector<float>(4, 1.0F));
    std::vector<std::vector<float>> b(2, std::vector<float>(4, 2.0F));
    std::vector<std::vector<float>> bad_layers(1, std::vector<float>(4, 0.0F));
    std::vector<std::vector<float>> bad_sizes = a;
    bad_sizes[1].resize(3);

    const auto ok_states = std::array<const std::vector<std::vector<float>>*, 2>{&a, &b};
    const auto bad_layer_states =
        std::array<const std::vector<std::vector<float>>*, 2>{&a, &bad_layers};

    check(under_test::validate_state_layer_count(ok_states, 2),
          "qwen3.5 contract accepts matching layer count");
    check(!under_test::validate_state_layer_count(bad_layer_states, 2),
          "qwen3.5 contract rejects layer count mismatch");

    const auto ok_specs = std::array<under_test::StateTensorView, 2>{
        under_test::StateTensorView{&a, 4}, under_test::StateTensorView{&b, 4}};
    const auto bad_specs = std::array<under_test::StateTensorView, 2>{
        under_test::StateTensorView{&a, 4}, under_test::StateTensorView{&bad_sizes, 4}};

    check(under_test::validate_state_tensor_sizes(ok_specs, 2),
          "qwen3.5 contract accepts matching tensor sizes");
    check(!under_test::validate_state_tensor_sizes(bad_specs, 2),
          "qwen3.5 contract rejects tensor size mismatch");

    std::vector<std::vector<float>> outputs;
    under_test::initialize_layer_outputs(3, 2, outputs);
    check(outputs.size() == 3, "qwen3.5 contract allocates layer outputs");
    check(outputs[0] == std::vector<float>({0.0F, 0.0F}),
          "qwen3.5 contract initializes outputs to zero");
}

#ifdef TRTMC_HAS_EDGE_LLM
void test_edge_video_requests() {
    namespace edge = trtmc::qwen3_5::edge_llm;
    namespace api = trtmc::internal;
    std::vector<std::uint8_t> red{255, 0, 0}, blue{0, 0, 255};
    std::vector<api::ImageView> frames{
        {red.data(), red.size(), 1, 1, 3, api::ImageFormat::UInt8},
        {blue.data(), blue.size(), 1, 1, 3, api::ImageFormat::UInt8}};
    std::vector<double> times{0.25, 1.75};
    api::VideoView video{{frames.data(), frames.size()}, {times.data(), times.size()}};
    const auto packed = edge::video_buffer(video);
    check(packed.isVideo && packed.frames == 2 && packed.timestamps == times,
          "edge video preserves frame count and timestamps");
    check(packed.data()[0] == 255 && packed.data()[3] == 0 && packed.data()[5] == 255,
          "edge video preserves frame order and RGB bytes");
    const auto untimed = edge::video_buffer({{frames.data(), 1}, {}});
    check(untimed.isVideo && untimed.frames == 1 && untimed.fps == 1.0 &&
              untimed.timestamps.empty(),
          "edge untimed video uses documented one FPS");
    std::vector<api::VideoTextPart> parts{api::TextPartView{"before"}, video,
                                          api::TextPartView{"after"}};
    api::VideoTextMessage message{api::MessageRole::User, {parts.data(), parts.size()}};
    api::VideoTextToTextRequest input{{&message, 1}, {}};
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = 8;
    config.use_chat_template = true;
    config.enable_thinking = false;
    const auto request = edge::media_request(input, config, 128);
    const auto& item = request.requests.front();
    check(item.imageBuffers.size() == 1 && item.imageBuffers[0].isVideo,
          "edge video remains one video rather than independent images");
    check(item.messages[0].contents.size() == 3 && item.messages[0].contents[1].type == "video" &&
              item.messages[0].contents[0].content == "before" &&
              item.messages[0].contents[2].content == "after",
          "edge video preserves mixed content ordering");
    check(request.applyChatTemplate && !request.enableThinking && request.maxGenerateLength == 8,
          "edge video preserves generation settings");
    std::vector<api::ImageVideoTextPart> mixed{frames.front(), video, api::TextPartView{"mixed"}};
    api::ImageVideoTextMessage mixed_message{api::MessageRole::User, {mixed.data(), mixed.size()}};
    api::ImageVideoTextToTextRequest mixed_input{{&mixed_message, 1}};
    const auto mapped = edge::media_request(mixed_input, config, 128);
    check(mapped.requests.front().imageBuffers.size() == 2 &&
              !mapped.requests.front().imageBuffers[0].isVideo &&
              mapped.requests.front().imageBuffers[1].isVideo,
          "edge mixed image/video preserves the media types");
    for (int variant = 0; variant < 7; ++variant) {
        auto invalid = video;
        auto invalid_frames = frames;
        auto invalid_times = times;
        switch (variant) {
        case 0:
            invalid.frames = {};
            break;
        case 1:
            invalid.timestamps_seconds = {times.data(), 1};
            break;
        case 2:
            invalid_times[1] = std::numeric_limits<double>::quiet_NaN();
            invalid.timestamps_seconds = {invalid_times.data(), invalid_times.size()};
            break;
        case 3:
            invalid_times[1] = -1;
            invalid.timestamps_seconds = {invalid_times.data(), invalid_times.size()};
            break;
        case 4:
            invalid_frames[1].width = 2;
            invalid.frames = {invalid_frames.data(), invalid_frames.size()};
            break;
        case 5:
            invalid_frames[1].byte_size = 1;
            invalid.frames = {invalid_frames.data(), invalid_frames.size()};
            break;
        case 6:
            invalid_frames[1].data = nullptr;
            invalid.frames = {invalid_frames.data(), invalid_frames.size()};
            break;
        }
        bool rejected = false;
        try {
            (void)edge::video_buffer(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        check(rejected, "edge video rejects invalid frames or timestamps");
    }
}
#endif

} // namespace

int main() {
    test_initialize_rwkv_outputs();
    test_initialize_mamba_outputs();
    test_qwen3_5_recurrent_contracts();
#ifdef TRTMC_HAS_EDGE_LLM
    test_edge_video_requests();
#endif

    if (g_failures != 0) {
        std::cerr << g_failures << " recurrent output initializer test(s) failed\n";
        return 1;
    }
    return 0;
}
