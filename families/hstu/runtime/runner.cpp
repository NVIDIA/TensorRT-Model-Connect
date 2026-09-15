/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void check_keys(const Json& object, const std::set<std::string>& allowed) {
    if (!object.is_object())
        throw std::invalid_argument("hstu request entries must be JSON objects");
    for (const auto& entry : object.items()) {
        if (allowed.count(entry.key()) == 0)
            throw std::invalid_argument("hstu unknown request field " + entry.key());
    }
}

std::vector<std::int64_t> read_ids(const Json& object, const char* key, bool required = false) {
    if (!object.contains(key) && !required)
        return {};
    const auto& values = object.at(key);
    if (!values.is_array())
        throw std::invalid_argument(std::string("hstu requires an integer array for ") + key);
    std::vector<std::int64_t> ids;
    for (const auto& value : values) {
        if (!value.is_number_integer())
            throw std::invalid_argument(std::string("hstu requires integer values for ") + key);
        if (value.is_number_unsigned() &&
            value.get<std::uint64_t>() >
                static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::invalid_argument("hstu identifier exceeds int64");
        ids.push_back(value.get<std::int64_t>());
    }
    return ids;
}

trtmc::RecommendationSequence read_sequence(const Json& source) {
    check_keys(source, {"history_item_ids", "history_action_ids", "contextual_features",
                        "candidate_item_ids", "token_timestamps"});
    trtmc::RecommendationSequence sequence;
    sequence.history_item_ids = read_ids(source, "history_item_ids", true);
    sequence.history_action_ids = read_ids(source, "history_action_ids");
    sequence.candidate_item_ids = read_ids(source, "candidate_item_ids", true);
    sequence.token_timestamps = read_ids(source, "token_timestamps");
    if (source.contains("contextual_features")) {
        if (!source.at("contextual_features").is_array())
            throw std::invalid_argument("hstu contextual_features must be an array");
        for (const auto& feature : source.at("contextual_features")) {
            check_keys(feature, {"name", "ids"});
            sequence.contextual_features.push_back(
                {feature.at("name").get<std::string>(), read_ids(feature, "ids", true)});
        }
    }
    return sequence;
}

trtmc::RecommendationRequest read_request(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("hstu cannot open request " + path);
    Json source;
    input >> source;
    check_keys(source, {"sequences"});
    if (!source.at("sequences").is_array())
        throw std::invalid_argument("hstu sequences must be an array");
    trtmc::RecommendationRequest request;
    for (const auto& sequence : source.at("sequences"))
        request.sequences.push_back(read_sequence(sequence));
    return request;
}

Json output_json(const trtmc::RecommendationResult& result) {
    Json sequences = Json::array();
    for (const auto& sequence : result.sequences) {
        sequences.push_back({{"candidate_item_ids", sequence.candidate_item_ids},
                             {"num_candidates", sequence.num_candidates},
                             {"embedding_dim", sequence.embedding_dim},
                             {"output_dim", sequence.output_dim},
                             {"logits", sequence.logits},
                             {"scores", sequence.scores},
                             {"embeddings", sequence.embeddings},
                             {"sequence_embeddings", sequence.sequence_embeddings},
                             {"sequence_length", sequence.sequence_length}});
    }
    return {{"sequences", sequences}};
}

std::map<std::string, std::string> options(int argc, char** argv) {
    const std::set<std::string> allowed = {"--bundle", "--runtime-root", "--input-json",
                                           "--output-json"};
    std::map<std::string, std::string> result;
    for (int index = 1; index < argc; index += 2) {
        const std::string key = argv[index];
        if (allowed.count(key) == 0 || index + 1 >= argc ||
            !result.emplace(key, argv[index + 1]).second)
            throw std::invalid_argument("usage: trtmc-hstu --bundle FILE --runtime-root DIR "
                                        "--input-json FILE --output-json FILE");
    }
    if (result.size() != allowed.size())
        throw std::invalid_argument(
            "trtmc-hstu requires --bundle, --runtime-root, --input-json, and --output-json");
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = options(argc, argv);
        const auto request = read_request(args.at("--input-json"));
        const auto load_start = Clock::now();
        auto task = trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"));
        auto* recommendation = dynamic_cast<trtmc::IRecommendation*>(task.get());
        if (recommendation == nullptr)
            throw std::runtime_error("hstu bundle does not implement IRecommendation");
        const auto inference_start = Clock::now();
        const auto result = recommendation->recommend(request);
        const auto completed = Clock::now();
        auto json = output_json(result);
        json["load_ms"] =
            std::chrono::duration<double, std::milli>(inference_start - load_start).count();
        json["inference_ms"] =
            std::chrono::duration<double, std::milli>(completed - inference_start).count();
        std::ofstream output(args.at("--output-json"));
        if (!output || !(output << json.dump(2) << '\n'))
            throw std::runtime_error("hstu cannot write output JSON");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "trtmc-hstu: " << error.what() << '\n';
        return 1;
    }
}
