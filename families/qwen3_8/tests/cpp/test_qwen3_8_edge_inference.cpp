/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace {
/// Apply only explicitly named public configuration fields; unknown test inputs are errors.
trtmc::TextGenerationConfig generation_config(const nlohmann::json& values) {
    trtmc::TextGenerationConfig result;
    if (!values.is_object())
        throw std::invalid_argument("config must be an object");
    for (const auto& [key, value] : values.items()) {
        if (key == "max_new_tokens")
            result.max_new_tokens = value.get<int32_t>();
        else if (key == "temperature")
            result.temperature = value.get<float>();
        else if (key == "top_k")
            result.top_k = value.get<int32_t>();
        else if (key == "top_p")
            result.top_p = value.get<float>();
        else if (key == "min_p")
            result.min_p = value.get<float>();
        else if (key == "seed")
            result.seed = value.get<int32_t>();
        else if (key == "eos_token_id")
            result.eos_token_id = value.get<int32_t>();
        else if (key == "use_chat_template")
            result.use_chat_template = value.get<bool>();
        else if (key == "enable_thinking")
            result.enable_thinking = value.get<bool>();
        else if (key == "text_generation_mode")
            result.text_generation_mode = value.get<std::string>();
        else if (key == "stop_on_boxed_answer")
            result.stop_on_boxed_answer = value.get<bool>();
        else if (key == "lora_adapter_id")
            result.lora_adapter_id = value.get<std::string>();
        else if (key == "repetition_penalty")
            result.repetition_penalty = value.get<float>();
        else if (key == "block_length")
            result.block_length = value.get<int32_t>();
        else if (key == "confidence_threshold")
            result.confidence_threshold = value.get<float>();
        else if (key == "forced_bos_token_id")
            result.forced_bos_token_id = value.get<int32_t>();
        else if (key == "source_language_token_id")
            result.source_language_token_id = value.get<int32_t>();
        else
            throw std::invalid_argument("Unknown public generation field: " + key);
    }
    return result;
}

/// Parse required qualification inputs without an environment or runtime-root fallback.
std::map<std::string, std::string> arguments(int argc, char** argv) {
    std::map<std::string, std::string> result;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc || !result.emplace(argv[i], argv[i + 1]).second)
            throw std::invalid_argument("Expected unique --option VALUE pairs");
    }
    for (const auto* name : {"--bundle", "--runtime-root", "--requests", "--output"})
        if (!result.count(name))
            throw std::invalid_argument(std::string("Missing ") + name);
    if (result.size() != 4)
        throw std::invalid_argument("Unknown qualification option");
    return result;
}
} // namespace

/// Exercise the public load/generate path once, retaining one task across the complete case list.
int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        const trtmc::BundleReader bundle(args.at("--bundle"));
        if (bundle.info().family != "qwen3_8" || !bundle.find_section("edge_llm.json"))
            throw std::runtime_error("Expected a qwen3_8 Edge bundle, not native fallback");
        nlohmann::json report{{"backend", "native"}, {"results", nlohmann::json::array()}};
        if (bundle.find_section("edge_llm.json")) {
            const auto bytes = bundle.read_section("edge_llm.json");
            const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
            report["backend"] = "edge_llm";
            report["edge_revision"] = marker.at("edge_revision");
        }
        std::ifstream request_file(args.at("--requests"));
        const auto requests = nlohmann::json::parse(request_file);
        if (!requests.is_array() || requests.empty())
            throw std::invalid_argument("Requests must be a nonempty JSON array");
        auto task = trtmc::load_task(args.at("--bundle"), args.at("--runtime-root"));
        auto* generator = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        if (!generator)
            throw std::runtime_error("Bundle does not expose ITextGeneration");
        bool passed = true;
        std::map<std::string, nlohmann::json> previous;
        for (const auto& request : requests) {
            nlohmann::json output{{"id", request.at("id")}};
            try {
                const auto config =
                    generation_config(request.value("config", nlohmann::json::object()));
                const auto result =
                    generator->generate(request.at("prompt").get<std::string>(), config);
                output["text"] = result.text;
                output["token_ids"] = result.token_ids;
            } catch (const std::exception& error) {
                output["error"] = error.what();
            }
            if (output.contains("error") != request.value("expect_error", false))
                passed = false;
            if (request.contains("equal_to")) {
                const auto earlier = previous.find(request.at("equal_to").get<std::string>());
                if (earlier == previous.end() || output.contains("error") ||
                    earlier->second.value("text", "") != output.value("text", "") ||
                    earlier->second.value("token_ids", nlohmann::json::array()) !=
                        output.value("token_ids", nlohmann::json::array()))
                    passed = false;
            }
            if (!output.contains("error") && output.at("token_ids").empty())
                passed = false;
            previous[request.at("id").get<std::string>()] = output;
            report["results"].push_back(std::move(output));
        }
        report["passed"] = passed;
        task.reset();
        std::ofstream output(args.at("--output"));
        output << report.dump(2) << '\n';
        output.close();
        if (!output)
            throw std::runtime_error("Cannot write qualification output");
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
