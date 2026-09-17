/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/native_worker.h"
#include "trtmc/task.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class FakeText final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 17; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        prompt_ = prompt;
        config_ = config;
        trtmc::TextResult result{"reply:" + prompt, {1, 2, 3}, 2.0, 3.0};
        result.setup_ms = 1.0;
        return result;
    }

    std::string prompt_;
    trtmc::TextGenerationConfig config_;
};

class ThrowingText final : public trtmc::ITextGeneration {
  public:
    std::int32_t default_max_new_tokens() const override { return 8; }
    trtmc::TextResult generate(const std::string&, const trtmc::TextGenerationConfig&) override {
        throw std::runtime_error("private model detail");
    }
};

std::vector<Json> records(const std::string& text) {
    std::vector<Json> result;
    std::istringstream input(text);
    for (std::string line; std::getline(input, line);)
        result.push_back(Json::parse(line));
    return result;
}

} // namespace

int main() {
    FakeText task;
    std::istringstream input("{\"id\":\"one\",\"op\":\"generate\",\"prompt\":\"hello\","
                             "\"config\":{\"max_new_tokens\":9,\"temperature\":0.25,\"top_k\":0,"
                             "\"top_p\":0.8,\"min_p\":0.1,\"seed\":7,\"system_prompt\":\"brief\","
                             "\"use_chat_template\":true,\"enable_thinking\":false}}\n"
                             "{\"id\":\"bad\",\"op\":\"generate\",\"prompt\":\"x\","
                             "\"config\":{\"unknown\":1}}\n"
                             "{\"id\":\"fractional\",\"op\":\"generate\",\"prompt\":\"x\","
                             "\"config\":{\"max_new_tokens\":1.5}}\n"
                             "{\"id\":\"overflow\",\"op\":\"generate\",\"prompt\":\"x\","
                             "\"config\":{\"seed\":4294967296}}\n"
                             "{\"id\":\"stop\",\"op\":\"shutdown\"}\n");
    std::ostringstream output;

    check(trtmc::server::run_text_worker(task, input, output) == 0, "worker exits cleanly");
    const auto messages = records(output.str());
    check(messages.size() == 6, "worker emits ready and five responses");
    if (messages.size() != 6)
        return 1;
    check(messages.at(0)["event"] == "ready", "worker advertises readiness");
    check(messages.at(0)["protocol_version"] == 1, "worker protocol is versioned");
    check(messages.at(1)["id"] == "one" && messages.at(1)["ok"] == true,
          "generate response preserves id");
    check(messages.at(1)["result"]["text"] == "reply:hello", "generate returns text");
    check(messages.at(1)["result"]["completion_tokens"] == 3, "generate returns token count");
    check(task.prompt_ == "hello", "worker forwards prompt");
    check(task.config_.max_new_tokens == 9, "worker forwards token limit");
    check(task.config_.top_k == 0 && task.config_.seed == 7, "worker forwards sampling");
    check(task.config_.system_prompt == "brief" && task.config_.use_chat_template,
          "worker leaves chat handling with family");
    check(messages.at(2)["id"] == "bad" && messages.at(2)["ok"] == false,
          "invalid config is rejected");
    check(messages.at(2)["error"]["type"] == "invalid_request_error",
          "protocol error remains client visible");
    check(messages.at(3)["id"] == "fractional" && messages.at(3)["ok"] == false,
          "fractional integral config is rejected");
    check(messages.at(3)["error"]["message"] == "config.max_new_tokens must be an integer",
          "fractional integral error is explicit");
    check(messages.at(4)["id"] == "overflow" && messages.at(4)["ok"] == false,
          "out-of-range integral config is rejected");
    check(messages.at(4)["error"]["message"] == "config.seed is out of range",
          "integral range error is explicit");
    check(messages.at(5)["result"]["status"] == "shutting_down", "shutdown is acknowledged");

    ThrowingText throwing;
    std::istringstream failing_input("{\"id\":\"failure\",\"op\":\"generate\",\"prompt\":\"x\"}\n");
    std::ostringstream failing_output;
    check(trtmc::server::run_text_worker(throwing, failing_input, failing_output) == 1,
          "runtime error retires worker");
    const auto failing_messages = records(failing_output.str());
    check(failing_messages.size() == 2, "failing worker emits ready and one error");
    if (failing_messages.size() != 2)
        return 1;
    check(failing_messages.at(1)["error"]["message"] == "native worker operation failed",
          "runtime detail is redacted");
    return failures == 0 ? 0 : 1;
}
