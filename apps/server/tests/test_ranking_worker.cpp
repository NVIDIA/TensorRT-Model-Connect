/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/native_worker.h"
#include "trtmc/task.h"

#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;

class Ranker final : public trtmc::IReranking {
  public:
    const char* task() const noexcept override { return kTask; }
    float rerank(const std::string&, const std::string&) override {
        throw std::runtime_error("expected a document-list Task call");
    }
    std::vector<float> rerank_batch(const std::string& query,
                                    const std::vector<std::string>& documents) override {
        if (query == "nonfinite")
            return {std::numeric_limits<float>::infinity(), 0};
        if (query != "Exact query: caf\xc3\xa9" ||
            documents != std::vector<std::string>{"first", "second"})
            throw std::runtime_error("fixture received changed inputs");
        return {-2.0F, 3.0F};
    }
};

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

std::vector<Json> messages(const std::string& output) {
    std::vector<Json> result;
    std::istringstream lines(output);
    for (std::string line; std::getline(lines, line);)
        result.push_back(Json::parse(line));
    return result;
}

void test_scores_identity_and_recovery() {
    Ranker ranker;
    const std::vector<Json> requests{
        {{"id", "wrong"}, {"op", "generate"}, {"prompt", "x"}},
        {{"id", "empty"}, {"op", "rerank"}, {"query", "q"}, {"documents", Json::array()}},
        {{"id", "scores"},
         {"op", "rerank"},
         {"query", "Exact query: caf\xc3\xa9"},
         {"documents", {"first", "second"}}},
        {{"id", "stop"}, {"op", "shutdown"}},
    };
    std::stringstream input;
    for (const auto& request : requests)
        input << request.dump() << '\n';
    std::ostringstream output;
    require(trtmc::server::run_task_worker(ranker, input, output) == 0, "worker completed");
    const auto replies = messages(output.str());
    require(replies.size() == 5 && replies[0].at("capabilities") == Json::array({"reranking"}),
            "reranker advertises the correct capability");
    for (const auto index : {1, 2})
        require(replies[index].at("error").at("type") == "invalid_request_error",
                "invalid requests are recoverable");
    const auto& scores = replies[3].at("result");
    require(replies[3].at("id") == "scores" && scores.at("scores") == Json::array({-2.0, 3.0}),
            "real scores retain input-document order");
    require(scores.at("model_call_ms").get<double>() >= 0 &&
                scores.at("timing_scope") == "public_task_call_wall",
            "Task timing is explicit");
}

void test_nonfinite_is_fatal_and_redacted() {
    Ranker ranker;
    std::istringstream input("{\"id\":\"bad\",\"op\":\"rerank\",\"query\":\"nonfinite\","
                             "\"documents\":[\"first\",\"second\"]}\n");
    std::ostringstream output;
    require(trtmc::server::run_task_worker(ranker, input, output) == 1,
            "invalid native output retires worker");
    const auto replies = messages(output.str());
    require(replies.size() == 2 && replies[1].at("ok") == false &&
                replies[1].at("error").at("message") == "native worker operation failed",
            "native errors are redacted");
}
} // namespace

int main() {
    try {
        test_scores_identity_and_recovery();
        test_nonfinite_is_fatal_and_redacted();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
