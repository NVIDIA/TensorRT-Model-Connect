/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"
#include "server/native_worker.h"
#include "trtmc/task.h"

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

class Transcriber final : public trtmc::ITranscription {
  public:
    bool fail{false};
    int calls{0};
    trtmc::TextResult transcribe(const float* samples, std::int32_t count,
                                 const trtmc::TranscriptionConfig& config) override {
        require(count == 4 && samples[0] == -1.0F && samples[2] == 0.5F,
                "decoded mono samples must reach the Task unchanged");
        require(config.input_sample_rate == 8000 && config.max_output_tokens == 7,
                "actual rate and explicit output limit must reach the Task");
        ++calls;
        if (fail)
            throw std::runtime_error("private failure path");
        return {"Exact transcript: caf\xc3\xa9.", {}};
    }
};

std::vector<Json> messages(const std::string& output) {
    std::vector<Json> result;
    std::istringstream lines(output);
    for (std::string line; std::getline(lines, line);)
        result.push_back(Json::parse(line));
    return result;
}

void verify(const std::string& path) {
    Transcriber task;
    std::stringstream input;
    input << Json{{"id", "missing"}, {"op", "transcribe"}, {"audio_path", path + ".missing"}}.dump()
          << '\n';
    input << Json{{"id", "config"},
                  {"op", "transcribe"},
                  {"audio_path", path},
                  {"config", {{"max_output_tokens", 0}}}}
                 .dump()
          << '\n';
    input << Json{{"id", "clip"},
                  {"op", "transcribe"},
                  {"audio_path", path},
                  {"config", {{"max_output_tokens", 7}}}}
                 .dump()
          << '\n';
    input << Json{{"id", "stop"}, {"op", "shutdown"}}.dump() << '\n';
    std::ostringstream output;
    require(trtmc::server::run_task_worker(task, input, output) == 0, "worker completes");
    const auto replies = messages(output.str());
    require(replies.size() == 5 && task.calls == 1, "invalid requests never invoke the Task");
    require(replies[0].at("capabilities") == Json::array({"audio_transcription"}),
            "only transcription is advertised");
    require(replies[1].at("error").at("type") == "invalid_request_error" &&
                replies[2].at("error").at("type") == "invalid_request_error",
            "input errors recover");
    const auto& result = replies[3].at("result");
    require(replies[3].at("id") == "clip" && result.at("text") == "Exact transcript: caf\xc3\xa9.",
            "complete transcript and identity survive transport");
    require(result.at("audio_sample_count") == 4 && result.at("audio_sample_rate") == 8000 &&
                result.at("model_call_ms").get<double>() >= 0 &&
                result.at("timing_scope") == "public_task_call_wall",
            "audio identity and timing remain explicit");

    task.fail = true;
    std::istringstream bad(Json{{"id", "bad"},
                                {"op", "transcribe"},
                                {"audio_path", path},
                                {"config", {{"max_output_tokens", 7}}}}
                               .dump() +
                           "\n");
    std::ostringstream failure;
    require(trtmc::server::run_task_worker(task, bad, failure) == 1,
            "native failure retires the worker");
    require(messages(failure.str())[1].at("error").at("message") ==
                "native worker operation failed",
            "native details are redacted");
}
} // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("trtmc-asr-" + std::to_string(std::random_device{}()) + ".wav");
    try {
        const std::vector<float> values{-1.0F, 0.0F, 0.5F, 1.0F};
        trtmc::cli::io::write_wav_interleaved({values.data(), values.size()}, 8000, 1,
                                              path.string());
        verify(path.string());
        std::filesystem::remove(path);
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
