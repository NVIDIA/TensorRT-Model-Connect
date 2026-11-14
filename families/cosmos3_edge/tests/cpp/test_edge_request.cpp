/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cosmos3_edge/runtime/edge_llm/request.h"

#include <iostream>

using namespace trtmc;
using namespace trtmc::internal;
using namespace trtmc::cosmos3_edge::edge_llm;

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("Cosmos3-Edge request contract regression");
}
template <class F>
void rejected(F fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid Cosmos3-Edge request was accepted");
}
int main() {
    try {
        uint8_t pixels[]{0, 128, 255, 255, 128, 0};
        ImageView frames[]{{pixels, 3, 1, 1, 3, ImageFormat::UInt8},
                           {pixels + 3, 3, 1, 1, 3, ImageFormat::UInt8}};
        double stamps[]{0.25, 2.75};
        VideoView video{{frames, 2}, {stamps, 2}};
        std::vector<ImageVideoTextPart> parts{video, TextPartView{"Describe the action."}};
        ImageVideoTextMessage message{MessageRole::User, {parts.data(), parts.size()}};
        TextGenerationConfig config;
        config.use_chat_template = true;
        config.max_new_tokens = 16;
        config.seed = 42;
        auto request = make_request({&message, 1}, config, 128);
        require(request.samplingSeed == 42);
        require(request.maxGenerateLength == 16);
        require(request.requests[0].messages[0].contents[0].type == "video");
        const auto& data = request.requests[0].imageBuffers[0];
        require(data.isVideo && data.frames == 2 &&
                data.timestamps == std::vector<double>({0.25, 2.75}));
        require(std::memcmp(data.data(), pixels, sizeof(pixels)) == 0);
        stamps[1] = 0.1;
        rejected([&] { make_request({&message, 1}, config, 128); });
        stamps[1] = 2.75;
        frames[1].byte_size = 2;
        rejected([&] { make_request({&message, 1}, config, 128); });
        frames[1].byte_size = 3;
        config.min_p = 0.1;
        rejected([&] { make_request({&message, 1}, config, 128); });
        config.min_p = 0;
        config.max_new_tokens = 128;
        rejected([&] { make_request({&message, 1}, config, 128); });
        require(safe_artifact_path("edge_llm/checkpoint/transformer/part.safetensors"));
        require(!safe_artifact_path("edge_llm/checkpoint/../secret"));
        require(!safe_artifact_path("/edge_llm/engine/llm.engine"));
        trt_edgellm::rt::LLMGenerationResponse response{};
        response.outputIds = {{1}};
        response.outputTexts = {"x"};
        response.inputTokenCounts = {120};
        response.finishReasons = {trt_edgellm::rt::FinishReason::kEndId};
        rejected([&] { validate_response(response, 128, 128, 16); });
        response.inputTokenCounts = {10};
        validate_response(response, 128, 128, 16);
        std::cout << "Cosmos3-Edge media/request contracts passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
