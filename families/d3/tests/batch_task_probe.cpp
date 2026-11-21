/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
using Json = nlohmann::json;
Json serialize(const trtmc::StructuredDecisionResult& result) {
    Json scores = Json::array();
    for (const auto& score : result.scores)
        scores.push_back({{"question_id", score.question_id},
                          {"option_ids", score.option_ids},
                          {"logits", score.logits},
                          {"probabilities", score.probabilities}});
    return {{"response", Json::parse(result.document)}, {"scores", scores}};
}
int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument(
                "usage: d3_batch_task_probe BUNDLE RUNTIME_ROOT RECORD RGB8");
        trtmc::BundleReader reader(argv[1]);
        auto task = trtmc::load_task(reader, argv[2]);
        auto* single = dynamic_cast<trtmc::IStructuredDecision*>(task.get());
        auto* batch = dynamic_cast<trtmc::IBatchStructuredDecision*>(task.get());
        if (!single || !batch)
            throw std::runtime_error("missing structured decision interfaces");
        std::ifstream file(argv[3]);
        if (!file)
            throw std::invalid_argument("cannot read D3 record");
        Json record;
        file >> record;
        std::ifstream rgb(argv[4], std::ios::binary);
        if (!rgb)
            throw std::invalid_argument("cannot read D3 image");
        const std::vector<unsigned char> pixels{std::istreambuf_iterator<char>(rgb), {}};
        if (pixels.size() != 256 * 256 * 3)
            throw std::invalid_argument("expected 256x256 RGB8 fixture");
        trtmc::ImageResult image;
        image.height = image.width = 256;
        image.pixels.assign(pixels.begin(), pixels.end());
        std::vector<trtmc::StructuredDecisionRequest> requests;
        Json out = {
            {"inputs", Json::array()}, {"single", Json::array()}, {"batches", Json::array()}};
        const std::vector<std::string> keys{"color", "red", "brightness"};
        for (int i = 0; i < 9; ++i) {
            Json item = record;
            const auto key = keys[i % keys.size()];
            item["questions"] = Json{{key, record.at("questions").at(key)}};
            item["state"] = record.at("state").get<std::string>() + std::string(i * 7, ' ');
            trtmc::StructuredDecisionRequest request;
            request.document = item.dump();
            const int channel = (i / 2) % 3;
            if (i % 2) {
                auto colored = image;
                for (std::size_t p = 0; p < colored.pixels.size(); p += 3) {
                    const auto red = image.pixels[p];
                    colored.pixels[p] = colored.pixels[p + 1] = colored.pixels[p + 2] = 0;
                    colored.pixels[p + channel] = red;
                }
                request.images = {std::move(colored)};
            }
            requests.push_back(request);
            out["inputs"].push_back({{"document", request.document},
                                     {"image", i % 2 != 0},
                                     {"image_channel", channel}});
            out["single"].push_back(serialize(single->decide(request)));
        }
        if (!batch->decide_batch({}).empty())
            throw std::runtime_error("empty batch returned results");
        for (const int count : {1, 2, 4, 8, 9}) {
            std::vector<trtmc::StructuredDecisionRequest> values(requests.begin(),
                                                                 requests.begin() + count);
            Json results = Json::array();
            for (const auto& result : batch->decide_batch(values))
                results.push_back(serialize(result));
            out["batches"].push_back({{"count", count}, {"results", results}});
        }
        // Same prompt geometry with different image content must use a real
        // compatible GPU batch; length bucketing must not hide cross-row bugs.
        auto red_request = requests.at(5);
        red_request.images = {image};
        const auto blue_request = requests.at(5);
        const auto red_single = single->decide(red_request);
        const auto blue_single = single->decide(blue_request);
        if (red_single.input_tokens != blue_single.input_tokens)
            throw std::runtime_error("same-prompt image pair has different token counts");
        Json pair = {
            {"inputs", Json::array()}, {"single", Json::array()}, {"batch", Json::array()}};
        pair["inputs"].push_back(
            {{"document", red_request.document}, {"image", true}, {"image_channel", 0}});
        pair["inputs"].push_back(
            {{"document", blue_request.document}, {"image", true}, {"image_channel", 2}});
        pair["single"].push_back(serialize(red_single));
        pair["single"].push_back(serialize(blue_single));
        for (const auto& result : batch->decide_batch({red_request, blue_request}))
            pair["batch"].push_back(serialize(result));
        out["same_length"] = std::move(pair);
        std::reverse(requests.begin(), requests.end());
        out["reversed"] = Json::array();
        for (const auto& result : batch->decide_batch(requests))
            out["reversed"].push_back(serialize(result));
        auto invalid = requests;
        invalid.back().document = "{}";
        bool rejected = false;
        try {
            batch->decide_batch(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid batch accepted");
        out["after_invalid"] = serialize(single->decide(requests.back()));
        std::cout << out.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
