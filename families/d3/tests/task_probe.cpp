/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
    try {
        if (argc != 5)
            throw std::invalid_argument("usage: d3_task_probe BUNDLE RUNTIME_ROOT RECORD RGB8");
        trtmc::BundleReader reader(argv[1]);
        auto loaded = trtmc::load_task(reader, argv[2]);
        auto* model = dynamic_cast<trtmc::IStructuredDecision*>(loaded.get());
        if (!model)
            throw std::runtime_error("bundle lacks structured_decision");
        std::ifstream file(argv[3]);
        if (!file)
            throw std::invalid_argument("cannot read D3 record");
        trtmc::StructuredDecisionRequest request;
        request.document = std::string(std::istreambuf_iterator<char>(file), {});
        std::ifstream rgb(argv[4], std::ios::binary);
        if (!rgb)
            throw std::invalid_argument("cannot read D3 image");
        const std::vector<unsigned char> pixels{std::istreambuf_iterator<char>(rgb), {}};
        trtmc::ImageResult image;
        image.height = image.width = 256;
        image.pixels.assign(pixels.begin(), pixels.end());
        nlohmann::json results = nlohmann::json::array();
        // Reuse one loaded Task for text -> image -> text. The last response
        // must exactly match the first; this catches request-state leakage.
        for (int i = 0; i < 3; ++i) {
            request.images =
                i == 1 ? std::vector<trtmc::ImageResult>{image} : std::vector<trtmc::ImageResult>{};
            auto result = model->decide(request);
            nlohmann::json scores = nlohmann::json::array();
            for (const auto& score : result.scores)
                scores.push_back({{"question_id", score.question_id},
                                  {"option_ids", score.option_ids},
                                  {"logits", score.logits},
                                  {"probabilities", score.probabilities}});
            results.push_back(
                {{"response", nlohmann::json::parse(result.document)}, {"scores", scores}});
        }
        std::cout << results.dump() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
