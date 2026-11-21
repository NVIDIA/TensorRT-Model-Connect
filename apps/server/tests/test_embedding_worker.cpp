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

class Encoder final : public trtmc::IEncoding {
  public:
    const char* task() const noexcept override { return kTask; }
    trtmc::EmbeddingResult encode(const std::string& text) override {
        if (text == "nonfinite")
            return {{std::numeric_limits<float>::infinity()}, 1};
        if (text == "wrong_dimension")
            return {{0.25F, -2.0F}, 3};
        if (text != "Exact input: caf\xc3\xa9")
            throw std::runtime_error("fixture received a changed input");
        std::vector<float> values(384, 0.25F);
        values.back() = -2.0F;
        return {std::move(values), 384};
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

void test_complete_vector_and_recovery() {
    Encoder encoder;
    const std::vector<Json> requests{
        {{"id", "wrong"}, {"op", "generate"}, {"prompt", "x"}},
        {{"id", "empty"}, {"op", "embed"}, {"input", ""}},
        {{"id", "vector"}, {"op", "embed"}, {"input", "Exact input: caf\xc3\xa9"}},
        {{"id", "stop"}, {"op", "shutdown"}},
    };
    std::stringstream input;
    for (const auto& request : requests)
        input << request.dump() << '\n';
    std::ostringstream output;
    require(trtmc::server::run_task_worker(encoder, input, output) == 0, "worker completed");
    const auto replies = messages(output.str());
    require(replies.size() == 5, "every request has a response");
    require(replies[0].at("capabilities") == Json::array({"embeddings"}),
            "encoding advertises embeddings only");
    for (const auto index : {1, 2})
        require(replies[index].at("error").at("type") == "invalid_request_error",
                "invalid request is recoverable");
    const auto& vector = replies[3].at("result");
    require(replies[3].at("id") == "vector" && vector.at("dim") == 384 &&
                vector.at("values").size() == 384 && vector.at("values")[383] == -2.0,
            "the full unmodified vector survives JSON transport");
    require(vector.at("model_call_ms").get<double>() >= 0 &&
                vector.at("timing_scope") == "public_task_call_wall",
            "native timing is explicit");
}

void test_invalid_output_is_fatal_and_redacted() {
    for (const auto* text : {"nonfinite", "wrong_dimension"}) {
        Encoder encoder;
        std::istringstream input(Json{{"id", "bad"}, {"op", "embed"}, {"input", text}}.dump() +
                                 "\n");
        std::ostringstream output;
        require(trtmc::server::run_task_worker(encoder, input, output) == 1,
                "invalid native output retires worker");
        const auto replies = messages(output.str());
        require(replies.size() == 2 && replies[1].at("ok") == false &&
                    replies[1].at("error").at("message") == "native worker operation failed",
                "native output errors are redacted");
    }
}

} // namespace

int main() {
    try {
        test_complete_vector_and_recovery();
        test_invalid_output_is_fatal_and_redacted();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
