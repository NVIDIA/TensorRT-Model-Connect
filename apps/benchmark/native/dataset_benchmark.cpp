/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dataset_answer.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

struct Sample {
    std::string id;
    std::string prompt;
    std::string answer;
    std::optional<std::int32_t> seed_index;
};

struct Options {
    std::string bundle;
    std::string dataset;
    std::string output;
    std::string runtime_root;
    std::int32_t max_new_tokens{12000};
    float temperature{1.0F};
    std::int32_t top_k{1};
    float top_p{1.0F};
    float min_p{0.0F};
    std::int32_t seed{-1};
    bool use_chat_template{false};
    bool enable_thinking{true};
    bool stop_on_answer{false};
    std::int32_t stop_check_interval{16};
};

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char character) {
        return std::isspace(character) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char character) {
                          return std::isspace(character) != 0;
                      }).base();
    return first < last ? std::string(first, last) : std::string();
}

void usage() {
    std::cerr << "Usage: trtmc_dataset_benchmark BUNDLE DATASET.jsonl OUTPUT.jsonl "
                 "--runtime-root PATH [--max-new-tokens N] [--temperature F] [--top-k N] "
                 "[--top-p F] [--min-p F] [--seed N] [--chat-template] [--no-thinking] "
                 "[--stop-on-answer] [--stop-check-interval N]\n";
}

Options parse_options(int argc, char** argv) {
    if (argc < 4) {
        usage();
        throw std::invalid_argument("bundle, dataset, and output are required");
    }
    Options options;
    options.bundle = argv[1];
    options.dataset = argv[2];
    options.output = argv[3];
    for (int index = 4; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc)
                throw std::invalid_argument(argument + " requires a value");
            return argv[index];
        };
        if (argument == "--runtime-root")
            options.runtime_root = value();
        else if (argument == "--max-new-tokens")
            options.max_new_tokens = std::stoi(value());
        else if (argument == "--temperature")
            options.temperature = std::stof(value());
        else if (argument == "--top-k")
            options.top_k = std::stoi(value());
        else if (argument == "--top-p")
            options.top_p = std::stof(value());
        else if (argument == "--min-p")
            options.min_p = std::stof(value());
        else if (argument == "--seed")
            options.seed = std::stoi(value());
        else if (argument == "--chat-template")
            options.use_chat_template = true;
        else if (argument == "--no-thinking")
            options.enable_thinking = false;
        else if (argument == "--stop-on-answer")
            options.stop_on_answer = true;
        else if (argument == "--stop-check-interval")
            options.stop_check_interval = std::stoi(value());
        else
            throw std::invalid_argument("unknown argument: " + argument);
    }
    if (options.runtime_root.empty())
        throw std::invalid_argument("--runtime-root is required");
    if (options.max_new_tokens < 1 || options.top_k < 1 || options.stop_check_interval < 1) {
        throw std::invalid_argument("integer generation limits must be positive");
    }
    return options;
}

std::vector<Sample> load_samples(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open dataset " + path);
    std::vector<Sample> samples;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (trim(line).empty())
            continue;
        Json value;
        try {
            value = Json::parse(line);
        } catch (const Json::exception& error) {
            throw std::runtime_error("invalid JSON on dataset line " + std::to_string(line_number) +
                                     ": " + error.what());
        }
        if (!value.is_object() || !value.contains("prompt") || !value.at("prompt").is_string()) {
            throw std::runtime_error("dataset line " + std::to_string(line_number) +
                                     " must contain a string prompt");
        }
        Sample sample;
        if (value.contains("sample_id")) {
            sample.id = value.at("sample_id").is_string() ? value.at("sample_id").get<std::string>()
                                                          : value.at("sample_id").dump();
        } else {
            sample.id = std::to_string(line_number);
        }
        sample.prompt = value.at("prompt").get<std::string>();
        if (sample.prompt.empty())
            throw std::runtime_error("dataset prompt must be non-empty");
        if (value.contains("answer"))
            sample.answer = value.at("answer").is_string() ? value.at("answer").get<std::string>()
                                                           : value.at("answer").dump();
        if (value.contains("seed_index"))
            sample.seed_index = value.at("seed_index").get<std::int32_t>();
        samples.push_back(std::move(sample));
    }
    if (samples.empty())
        throw std::runtime_error("dataset contains no samples");
    return samples;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        auto task = trtmc::load_task(options.bundle, options.runtime_root);
        auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        if (text == nullptr)
            throw std::runtime_error("bundle task does not implement ITextGeneration");

        const auto samples = load_samples(options.dataset);
        std::ofstream output(options.output);
        if (!output)
            throw std::runtime_error("cannot write " + options.output);

        trtmc::TextGenerationConfig config;
        config.max_new_tokens = options.max_new_tokens;
        config.temperature = options.temperature;
        config.top_k = options.top_k;
        config.top_p = options.top_p;
        config.min_p = options.min_p;
        config.seed = options.seed;
        config.use_chat_template = options.use_chat_template;
        config.enable_thinking = options.enable_thinking;
        config.stop_on_boxed_answer = options.stop_on_answer;
        config.stop_check_interval = options.stop_check_interval;

        for (std::size_t index = 0; index < samples.size(); ++index) {
            const Sample& sample = samples[index];
            if (options.seed >= 0) {
                config.seed =
                    options.seed + sample.seed_index.value_or(static_cast<std::int32_t>(index));
            }
            const auto started = std::chrono::steady_clock::now();
            const trtmc::TextResult result = text->generate(sample.prompt, config);
            const double wall_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
            const double tokens_per_second =
                result.decode_ms > 0.0
                    ? static_cast<double>(result.token_ids.size()) / (result.decode_ms / 1000.0)
                    : 0.0;
            output << Json{{"sample_id", sample.id},
                           {"gold_answer", sample.answer},
                           {"pred_answer",
                            trtmc::examples::dataset_benchmark::extract_answer(result.text)
                                .value_or("")},
                           {"generated_tokens", result.token_ids.size()},
                           {"generated_token_ids", result.token_ids},
                           {"setup_ms", result.setup_ms},
                           {"prefill_ms", result.prefill_ms},
                           {"decode_ms", result.decode_ms},
                           {"wall_ms", wall_ms},
                           {"tokens_per_sec", tokens_per_second},
                           {"text", result.text}}
                          .dump()
                   << '\n';
            output.flush();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "trtmc_dataset_benchmark: " << error.what() << '\n';
        return 1;
    }
}
