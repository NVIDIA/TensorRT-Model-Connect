/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"
#include "dataset_answer.h"
#include "task_runtime.h"
#include "trtmc/control.hpp"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"
#include "trtmc/text.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

struct Sample {
    std::string id;
    std::string prompt;
    std::string answer;
    std::optional<std::int64_t> seed_index;
};

struct Options {
    std::string bundle;
    std::string dataset;
    std::string output;
    std::string runtime_root;
    std::int64_t max_new_tokens{12000};
    double temperature{1.0};
    std::int64_t top_k{1};
    double top_p{1.0};
    double min_p{0.0};
    std::int64_t seed{-1};
    bool use_chat_template{false};
    bool enable_thinking{true};
    bool stop_on_answer{false};
    std::int64_t stop_check_interval{16};
    std::set<std::string> explicit_config;
    std::vector<std::pair<std::string, std::string>> config_entries;
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
                 "[--runtime-root PATH] [--max-new-tokens N] [--temperature F] [--top-k N] "
                 "[--top-p F] [--min-p F] [--seed N] [--chat-template] [--no-thinking] "
                 "[--stop-on-answer] [--stop-check-interval N] [--set NAME=VALUE]\n";
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
        auto typed = [&](trtmc::ConfigKind kind) {
            return trtmc::app::parse_config_value(value(), {argument, kind, std::nullopt, ""});
        };
        if (argument == "--runtime-root")
            options.runtime_root = value();
        else if (argument == "--max-new-tokens")
            options.max_new_tokens = typed(trtmc::ConfigKind::I64).get<std::int64_t>();
        else if (argument == "--temperature")
            options.temperature = typed(trtmc::ConfigKind::F64).get<double>();
        else if (argument == "--top-k")
            options.top_k = typed(trtmc::ConfigKind::I64).get<std::int64_t>();
        else if (argument == "--top-p")
            options.top_p = typed(trtmc::ConfigKind::F64).get<double>();
        else if (argument == "--min-p")
            options.min_p = typed(trtmc::ConfigKind::F64).get<double>();
        else if (argument == "--seed")
            options.seed = typed(trtmc::ConfigKind::I64).get<std::int64_t>();
        else if (argument == "--chat-template")
            options.use_chat_template = true;
        else if (argument == "--no-thinking")
            options.enable_thinking = false;
        else if (argument == "--stop-on-answer")
            options.stop_on_answer = true;
        else if (argument == "--stop-check-interval")
            options.stop_check_interval = typed(trtmc::ConfigKind::I64).get<std::int64_t>();
        else if (argument == "--set") {
            const auto entry = value();
            const auto equals = entry.find('=');
            if (equals == 0 || equals == std::string::npos)
                throw std::invalid_argument("--set requires NAME=VALUE");
            options.config_entries.emplace_back(entry.substr(0, equals), entry.substr(equals + 1));
        } else
            throw std::invalid_argument("unknown argument: " + argument);
        if (argument != "--runtime-root" && argument != "--set") {
            std::string name = argument.substr(2);
            std::replace(name.begin(), name.end(), '-', '_');
            if (name == "chat_template")
                name = "use_chat_template";
            if (name == "no_thinking")
                name = "enable_thinking";
            if (name == "stop_on_answer")
                name = "stop_on_boxed_answer";
            if (!options.explicit_config.insert(name).second)
                throw std::invalid_argument("duplicate option: " + argument);
        }
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
        if (value.contains("seed_index")) {
            const auto& offset = value.at("seed_index");
            if (!offset.is_number_integer() ||
                (offset.is_number_unsigned() &&
                 offset.get<std::uint64_t>() >
                     static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
                throw std::invalid_argument("seed_index must be an int64 integer");
            sample.seed_index = offset.get<std::int64_t>();
        }
        samples.push_back(std::move(sample));
    }
    if (samples.empty())
        throw std::runtime_error("dataset contains no samples");
    return samples;
}

std::int64_t sample_seed(std::int64_t seed, const Sample& sample, std::size_t index) {
    if (seed < 0)
        return seed;
    if (index > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("sample index does not fit int64");
    const std::int64_t offset =
        sample.seed_index ? *sample.seed_index : static_cast<std::int64_t>(index);
    if (offset > 0 && seed > std::numeric_limits<std::int64_t>::max() - offset)
        throw std::invalid_argument("per-sample seed exceeds int64");
    return seed + offset; // A nonnegative seed plus a negative offset cannot underflow.
}

std::int32_t legacy_integer(std::int64_t value) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("value exceeds the existing int32 generation contract");
    return static_cast<std::int32_t>(value);
}

float legacy_float(double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::invalid_argument("value exceeds the existing float32 generation contract");
    return static_cast<float>(value);
}

void write_sample(std::ostream& output, const Sample& sample, std::string_view text,
                  trtmc::Span<const std::int32_t> tokens, double setup_ms, double prefill_ms,
                  double decode_ms, double wall_ms, Json receipt = Json::object()) {
    const std::string owned_text(text);
    Json ids = Json::array();
    for (auto token : tokens)
        ids.push_back(token);
    receipt.update(
        Json{{"sample_id", sample.id},
             {"gold_answer", sample.answer},
             {"pred_answer",
              trtmc::examples::dataset_benchmark::extract_answer(owned_text).value_or("")},
             {"generated_tokens", tokens.size()},
             {"generated_token_ids", std::move(ids)},
             {"setup_ms", setup_ms},
             {"prefill_ms", prefill_ms},
             {"decode_ms", decode_ms},
             {"wall_ms", wall_ms},
             {"tokens_per_sec",
              decode_ms > 0.0 ? static_cast<double>(tokens.size()) / (decode_ms / 1000.0) : 0.0},
             {"text", owned_text}});
    output << receipt.dump() << '\n';
    output.flush();
    if (!output)
        throw std::runtime_error("failed to write dataset result");
}

trtmc::Config benchmark_config(const Options& options,
                               const std::vector<trtmc::ConfigField>& fields, const Sample& sample,
                               std::size_t index) {
    // These are the existing benchmark workload, not shared runtime defaults.
    // Supply implicit workload controls only where this loaded Task declares
    // them. Explicit unsupported controls always fail; nothing is silently ignored.
    const trtmc::Config workload{{"max_new_tokens", options.max_new_tokens},
                                 {"temperature", double{options.temperature}},
                                 {"top_k", options.top_k},
                                 {"top_p", double{options.top_p}},
                                 {"min_p", double{options.min_p}},
                                 {"seed", options.seed},
                                 {"use_chat_template", options.use_chat_template},
                                 {"enable_thinking", options.enable_thinking},
                                 {"stop_on_boxed_answer", options.stop_on_answer},
                                 {"stop_check_interval", options.stop_check_interval}};
    auto field_for = [&](std::string_view name) {
        return std::find_if(fields.begin(), fields.end(),
                            [&](const auto& field) { return field.name == name; });
    };
    auto with_seed = [&](const std::string& name, trtmc::ConfigValue value) {
        if (name == "seed")
            return trtmc::ConfigValue(sample_seed(value.get<std::int64_t>(), sample, index));
        return value;
    };
    trtmc::Config result;
    for (const auto& entry : workload.entries()) {
        const bool explicitly_set = options.explicit_config.count(entry.name) != 0;
        const bool raw_override =
            std::any_of(options.config_entries.begin(), options.config_entries.end(),
                        [&](const auto& item) { return item.first == entry.name; });
        if (raw_override && !explicitly_set)
            continue;
        const auto field = field_for(entry.name);
        if (field == fields.end()) {
            if (explicitly_set)
                throw std::invalid_argument("selected Task does not declare config '" + entry.name +
                                            "'");
            continue;
        }
        if (field->kind != entry.value.kind())
            throw std::invalid_argument("benchmark config type mismatch: " + entry.name);
        result.add(entry.name, with_seed(entry.name, entry.value));
    }
    for (const auto& [name, text] : options.config_entries) {
        const auto field = field_for(name);
        if (field == fields.end())
            throw std::invalid_argument("selected Task does not declare config '" + name + "'");
        result.add(name, with_seed(name, trtmc::app::parse_config_value(text, *field)));
    }
    return result; // Explicit duplicates remain for the family to reject.
}

template <class Task, class Request>
void run_sdk(const Task& task, Request make_request, const Options& options,
             const std::vector<Sample>& samples, std::ostream& output) {
    const auto fields = task.config_fields();
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const auto& sample = samples[index];
        const auto config = benchmark_config(options, fields, sample, index);
        const auto request = make_request(sample.prompt);
        const auto started = std::chrono::steady_clock::now();
        const auto result = task.run(request, config);
        const double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
                .count();
        // Answer extraction and JSON/result serialization remain outside timing.
        Json submitted = Json::object();
        for (const auto& entry : config.entries())
            submitted[entry.name] = trtmc::app::config_value_json(entry.value);
        write_sample(
            output, sample, result.text(), result.token_ids(), result.setup_ms(),
            result.prefill_ms(), result.decode_ms(), wall_ms,
            {{"task", std::string(Task::kTask)}, {"submitted_config", std::move(submitted)}});
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto primary = trtmc::Bundle::open(options.bundle).info().task;
        const auto samples = load_samples(options.dataset);
        if (!trtmc::app::uses_existing_task_runtime(primary)) {
            trtmc::LoadOptions load;
            load.runtime_root = options.runtime_root;
            const auto model = trtmc::Model::load(options.bundle, load);
            std::ofstream output(options.output);
            if (!output)
                throw std::runtime_error("cannot write " + options.output);
            if (primary == trtmc::TextContinuation::kTask)
                run_sdk(
                    model.task<trtmc::TextContinuation>(),
                    [](const std::string& prompt) {
                        return trtmc::TextContinuationRequest{prompt};
                    },
                    options, samples, output);
            else if (primary == trtmc::ConditionalTextGeneration::kTask)
                run_sdk(
                    model.task<trtmc::ConditionalTextGeneration>(),
                    [](const std::string& prompt) {
                        return trtmc::ConditionalTextGenerationRequest{prompt};
                    },
                    options, samples, output);
            else if (primary == trtmc::CorruptedTextReconstruction::kTask)
                run_sdk(
                    model.task<trtmc::CorruptedTextReconstruction>(),
                    [](const std::string& prompt) {
                        return trtmc::CorruptedTextReconstructionRequest{prompt};
                    },
                    options, samples, output);
            else if (primary == trtmc::TextSummarization::kTask)
                run_sdk(
                    model.task<trtmc::TextSummarization>(),
                    [](const std::string& prompt) {
                        return trtmc::TextSummarizationRequest{prompt};
                    },
                    options, samples, output);
            else
                throw std::invalid_argument("dataset prompt is not a complete input for Task '" +
                                            primary + "'");
            return 0;
        }
        if (options.runtime_root.empty())
            throw std::invalid_argument("--runtime-root is required for an existing bundle mode");
        if (!options.config_entries.empty())
            throw std::invalid_argument("--set requires a semantic Task bundle");
        if (options.max_new_tokens < 1 || options.top_k < 1 || options.stop_check_interval < 1)
            throw std::invalid_argument("existing integer generation limits must be positive");
        auto task = trtmc::load_task(options.bundle, options.runtime_root);
        auto* text = dynamic_cast<trtmc::ITextGeneration*>(task.get());
        if (text == nullptr)
            throw std::runtime_error("bundle task does not implement ITextGeneration");

        std::ofstream output(options.output);
        if (!output)
            throw std::runtime_error("cannot write " + options.output);

        trtmc::TextGenerationConfig config;
        config.max_new_tokens = legacy_integer(options.max_new_tokens);
        config.temperature = legacy_float(options.temperature);
        config.top_k = legacy_integer(options.top_k);
        config.top_p = legacy_float(options.top_p);
        config.min_p = legacy_float(options.min_p);
        config.seed = legacy_integer(options.seed);
        config.use_chat_template = options.use_chat_template;
        config.enable_thinking = options.enable_thinking;
        config.stop_on_boxed_answer = options.stop_on_answer;
        config.stop_check_interval = legacy_integer(options.stop_check_interval);

        for (std::size_t index = 0; index < samples.size(); ++index) {
            const Sample& sample = samples[index];
            if (options.seed >= 0) {
                const auto seed = sample_seed(options.seed, sample, index);
                if (seed < std::numeric_limits<std::int32_t>::min() ||
                    seed > std::numeric_limits<std::int32_t>::max())
                    throw std::invalid_argument(
                        "per-sample seed exceeds the existing int32 contract");
                config.seed = static_cast<std::int32_t>(seed);
            }
            const auto started = std::chrono::steady_clock::now();
            const trtmc::TextResult result = text->generate(sample.prompt, config);
            const double wall_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
            write_sample(output, sample, result.text,
                         {result.token_ids.data(), result.token_ids.size()}, result.setup_ms,
                         result.prefill_ms, result.decode_ms, wall_ms);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "trtmc_dataset_benchmark: " << error.what() << '\n';
        return 1;
    }
}
