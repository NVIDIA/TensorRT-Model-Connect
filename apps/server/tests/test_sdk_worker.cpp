/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/native_worker.h"
#include "trtmc/core.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::filesystem::path write_bundle(const std::filesystem::path& root, const std::string& task,
                                   const std::string& family = "api_fixture") {
    const auto path = root / ("server-sdk-" + family + "-" + task + ".bundle");
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", task},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream file(path, std::ios::binary);
    file.write("BUNDLE\x01\x00", 8);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (int shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((size >> shift) & 0xffU));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write("PLAN", 4);
    require(static_cast<bool>(file), "write SDK worker fixture bundle");
    return path;
}

Json generate(const char* id, Json config = Json::object()) {
    return {{"id", id}, {"op", "generate"}, {"prompt", "hello"}, {"config", std::move(config)}};
}

std::vector<Json> records(const std::string& output) {
    std::istringstream input(output);
    std::vector<Json> result;
    for (std::string line; std::getline(input, line);)
        result.push_back(Json::parse(line));
    return result;
}

struct Run {
    int status;
    std::vector<Json> messages;
};

Run invoke(const std::filesystem::path& bundle, const trtmc::LoadOptions& options,
           const std::vector<Json>& requests, bool direct_model = false) {
    std::ostringstream encoded;
    for (const auto& request : requests)
        encoded << request.dump() << '\n';
    std::istringstream input(encoded.str());
    std::ostringstream output;
    int status;
    if (direct_model) {
        const auto model = trtmc::Model::load(bundle.string(), options);
        status = trtmc::server::run_text_worker(model, input, output);
    } else {
        status = trtmc::server::run_bundle_worker(bundle.string(), options, input, output);
    }
    return {status, records(output.str())};
}

void test_defaults_and_config(const std::filesystem::path& root,
                              const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "text_continuation");
    const auto defaults = invoke(bundle, options, {generate("defaults")}, true);
    require(defaults.status == 0 && defaults.messages.size() == 2, "SDK model overload succeeds");
    const auto& ready = defaults.messages[0];
    require(ready.at("event") == "ready" && ready.at("protocol_version") == 1 &&
                ready.at("capabilities") == Json::array({"text_generation"}) &&
                ready.at("default_max_new_tokens") == 4,
            "ready transport preserves the family token default");
    const auto& result = defaults.messages[1].at("result");
    require(result.at("text") == "hello!|eos" && result.at("completion_tokens") == 2 &&
                result.at("setup_ms") == 0 && result.at("prefill_ms") == 0.75 &&
                result.at("decode_ms") == 4,
            "missing Config values remain family-owned");

    const Json supplied{{"max_new_tokens", 0},
                        {"temperature", 0.25},
                        {"emit_eos", false},
                        {"suffix", std::string("tail\0end", 8)},
                        {"token_biases", {7, -3}},
                        {"schedule", {1.25, 2.5}},
                        {"labels", {"first", "second"}},
                        {"context_limit", 5}};
    const auto explicit_values = invoke(bundle, options, {generate("configured", supplied)});
    require(explicit_values.status == 0 && explicit_values.messages.size() == 2,
            "all seven Config kinds pass through the bundle entry point");
    const auto& configured = explicit_values.messages[1].at("result");
    require(configured.at("text") == std::string("hellotail\0end", 13) + "|first|second|7" &&
                configured.at("setup_ms") == 1.25 && configured.at("prefill_ms") == 0.25 &&
                configured.at("decode_ms") == 0,
            "explicit zero, false, length-delimited strings and arrays are not defaulted");
}

void test_invalid_config_is_nonfatal(const std::filesystem::path& root,
                                     const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "text_continuation");
    const std::vector<Json> bad_configs{
        {{"unknown", 1}},
        {{"max_new_tokens", 1.5}},
        {{"max_new_tokens", true}},
        {{"max_new_tokens", std::uint64_t{18446744073709551615ULL}}},
        {{"max_new_tokens", -1}},
        {{"temperature", 3.0}},
        {{"context_limit", 1}},
        {{"suffix", 7}},
        {{"emit_eos", 1}},
        {{"token_biases", {1.5}}},
        {{"schedule", {"bad"}}},
        {{"labels", {1}}},
        Json::array(),
        nullptr,
    };
    std::vector<Json> requests;
    for (const auto& config : bad_configs)
        requests.push_back(generate("bad", config));
    requests.push_back(generate("after-errors"));
    requests.push_back({{"id", "stop"}, {"op", "shutdown"}});
    const auto run = invoke(bundle, options, requests);
    require(run.status == 0 && run.messages.size() == requests.size() + 1,
            "invalid SDK requests do not retire the worker");
    for (std::size_t i = 0; i < bad_configs.size(); ++i) {
        const auto& error = run.messages[i + 1];
        require(error.at("id") == "bad" && error.at("ok") == false &&
                    error.at("error").at("type") == "invalid_request_error",
                "type, range and unknown-field errors are client errors");
    }
    require(run.messages[bad_configs.size() + 1].at("result").at("text") == "hello!|eos",
            "a valid request still succeeds after invalid Config");
    require(run.messages.back().at("result").at("status") == "shutting_down",
            "SDK worker acknowledges shutdown");
}

void test_task_selection_and_failures(const std::filesystem::path& root,
                                      const trtmc::LoadOptions& options) {
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
             {"conditional_text_generation", "conditional:hello!"},
             {"text_translation", "translation:fixed-src->en:hello!"}}) {
        const auto run =
            invoke(write_bundle(root, item.first, "text_fixture"), options, {generate("selected")});
        require(run.status == 0 && run.messages.size() == 2 &&
                    run.messages[0].at("default_max_new_tokens") == 128 &&
                    run.messages[1].at("result").at("text") == item.second,
                "primary Task selection does not inject undeclared defaults or languages");
    }
    const auto failure = invoke(write_bundle(root, "must_not_run"), options,
                                {generate("failure"), generate("must-not-retry")});
    require(failure.status == 1 && failure.messages.size() == 2 &&
                failure.messages[1].at("error").at("type") == "runtime_error" &&
                failure.messages[1].at("error").at("message") == "native worker operation failed",
            "SDK runtime error is redacted and retires the worker without retry");
    std::istringstream input;
    std::ostringstream output;
    try {
        trtmc::server::run_bundle_worker(write_bundle(root, "disabled").string(), options, input,
                                         output);
        throw std::runtime_error("a model with no enabled text Task was accepted");
    } catch (const trtmc::Error& error) {
        require(error.code() == TRTMC_UNSUPPORTED && output.str().empty(),
                "unavailable SDK Task fails before ready, with no legacy retry");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        test_defaults_and_config(root, options);
        test_invalid_config_is_nonfatal(root, options);
        test_task_selection_and_failures(root, options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
