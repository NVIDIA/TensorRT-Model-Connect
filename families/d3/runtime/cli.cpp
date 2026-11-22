/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/cli.h"

#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
extern "C" int trtmc_family_cli_v1(const char* handler, const char* values_json,
                                   const char* default_runtime_root, void* context,
                                   trtmc_cli_write_v1 output, trtmc_cli_write_v1 error) {
    try {
        if (std::string(handler) != "decide")
            throw std::invalid_argument("unknown D3 command");
        const auto values = nlohmann::json::parse(values_json);
        trtmc::BundleReader reader(values.at("bundle").get<std::string>());
        if (reader.info().family != "d3")
            throw std::invalid_argument("D3 command requires a d3 bundle");
        auto loaded = trtmc::load_task(
            reader, values.value("runtime_root", std::string(default_runtime_root)));
        auto* model = dynamic_cast<trtmc::IStructuredDecision*>(loaded.get());
        if (!model)
            throw std::runtime_error("D3 bundle lacks structured_decision");
        std::ifstream file(values.at("record").get<std::string>());
        if (!file)
            throw std::invalid_argument("cannot read D3 record");
        trtmc::StructuredDecisionRequest request;
        request.document = std::string(std::istreambuf_iterator<char>(file), {});
        // A raw RGB8 file avoids codec-dependent behavior in the correctness probe.
        // The abstract Task accepts decoded RGB tensors directly.
        if (values.contains("image_rgb")) {
            const auto bytes = reader.read_section("runtime.json");
            const auto config = nlohmann::json::parse(bytes.begin(), bytes.end());
            trtmc::ImageResult image;
            image.height = config.at("image_height");
            image.width = config.at("image_width");
            std::ifstream rgb(values.at("image_rgb").get<std::string>(), std::ios::binary);
            if (!rgb)
                throw std::invalid_argument("cannot read D3 RGB8 file");
            const std::vector<unsigned char> pixels{std::istreambuf_iterator<char>(rgb), {}};
            if (pixels.size() != static_cast<std::size_t>(image.height) * image.width * 3)
                throw std::invalid_argument("RGB8 file size differs from bundle image dimensions");
            image.pixels.assign(pixels.begin(), pixels.end());
            request.images.push_back(std::move(image));
        }
        const auto result = model->decide(request);
        nlohmann::json scores = nlohmann::json::array();
        for (const auto& score : result.scores)
            scores.push_back({{"question_id", score.question_id},
                              {"option_ids", score.option_ids},
                              {"logits", score.logits},
                              {"probabilities", score.probabilities}});
        const auto response =
            nlohmann::json{{"response", nlohmann::json::parse(result.document)}, {"scores", scores}}
                .dump() +
            '\n';
        output(context, response.data(), response.size());
        return 0;
    } catch (const std::exception& exception) {
        const auto message = std::string("Error: ") + exception.what() + '\n';
        error(context, message.data(), message.size());
        return 1;
    }
}
