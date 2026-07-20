/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <trtmc/pipeline.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMinimumBundleCount = 2;

struct Options {
    fs::path runtime_cache;
    fs::path output;
    std::string prompt;
    int32_t max_new_tokens{8};
    std::vector<fs::path> bundles;
};

struct LoadedPipeline {
    fs::path bundle;
    std::string model_id;
    std::string pipeline_type;
    std::unique_ptr<trtmc::IPipeline> pipeline;
};

struct ResultEntry {
    fs::path bundle;
    std::string model_id;
    std::string pipeline_type;
    trtmc::TextResult result;
};

std::string require_value(int argc, char** argv, int& index, std::string_view option) {
    if (++index >= argc)
        throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--runtime-cache") {
            options.runtime_cache = require_value(argc, argv, index, argument);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, index, argument);
        } else if (argument == "--prompt") {
            options.prompt = require_value(argc, argv, index, argument);
        } else if (argument == "--max-new-tokens") {
            const std::string raw = require_value(argc, argv, index, argument);
            std::size_t consumed = 0;
            const long value = std::stol(raw, &consumed);
            if (consumed != raw.size() || value < 1 || value > 1024)
                throw std::invalid_argument("--max-new-tokens must be in [1, 1024]");
            options.max_new_tokens = static_cast<int32_t>(value);
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::invalid_argument("unknown option: " + argument);
        } else {
            options.bundles.emplace_back(argument);
        }
    }
    if (options.runtime_cache.empty())
        throw std::invalid_argument("--runtime-cache is required");
    if (options.output.empty())
        throw std::invalid_argument("--output is required");
    if (options.prompt.empty())
        throw std::invalid_argument("--prompt is required");
    if (options.bundles.size() < kMinimumBundleCount)
        throw std::invalid_argument("at least two bundle paths are required");

    std::set<fs::path> unique;
    for (auto& bundle : options.bundles) {
        bundle = fs::canonical(bundle);
        if (!fs::is_regular_file(bundle))
            throw std::invalid_argument("bundle is not a regular file: " + bundle.string());
        if (!unique.insert(bundle).second)
            throw std::invalid_argument("bundle paths must be distinct");
    }
    options.runtime_cache = fs::absolute(options.runtime_cache).lexically_normal();
    options.output = fs::absolute(options.output).lexically_normal();
    return options;
}

void write_json_string(std::ostream& output, std::string_view value) {
    constexpr char kHex[] = "0123456789abcdef";
    output.put('"');
    for (const unsigned char byte : value) {
        switch (byte) {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (std::iscntrl(byte) != 0) {
                output << "\\u00" << kHex[(byte >> 4U) & 0x0FU] << kHex[byte & 0x0FU];
            } else {
                output.put(static_cast<char>(byte));
            }
        }
    }
    output.put('"');
}

void write_string_array(std::ostream& output, const std::vector<fs::path>& paths) {
    output.put('[');
    for (std::size_t index = 0; index < paths.size(); ++index) {
        if (index != 0)
            output.put(',');
        write_json_string(output, paths[index].string());
    }
    output.put(']');
}

void write_result_array(std::ostream& output, const std::vector<ResultEntry>& entries) {
    output.put('[');
    for (std::size_t index = 0; index < entries.size(); ++index) {
        if (index != 0)
            output.put(',');
        const ResultEntry& entry = entries[index];
        output << "{\"bundle\":";
        write_json_string(output, entry.bundle.string());
        output << ",\"model_id\":";
        write_json_string(output, entry.model_id);
        output << ",\"pipeline_type\":";
        write_json_string(output, entry.pipeline_type);
        output << ",\"generated\":";
        write_json_string(output, entry.result.text);
        output << ",\"token_ids\":[";
        for (std::size_t token_index = 0; token_index < entry.result.token_ids.size();
             ++token_index) {
            if (token_index != 0)
                output.put(',');
            output << entry.result.token_ids[token_index];
        }
        output << "]}";
    }
    output.put(']');
}

ResultEntry generate(LoadedPipeline& loaded, const std::string& prompt,
                     const trtmc::GenerateConfig& config) {
    auto result = loaded.pipeline->generate(prompt, config);
    if (result.text.empty())
        throw std::runtime_error("generation returned empty text for " + loaded.bundle.string());
    if (result.token_ids.empty())
        throw std::runtime_error("generation returned no token IDs for " + loaded.bundle.string());
    return ResultEntry{loaded.bundle, loaded.model_id, loaded.pipeline_type, std::move(result)};
}

void require_deterministic(const std::vector<ResultEntry>& forward,
                           const std::vector<ResultEntry>& reverse) {
    if (forward.size() != reverse.size())
        throw std::runtime_error("forward and reverse result counts differ");
    for (const auto& reference : forward) {
        const auto match = std::find_if(reverse.begin(), reverse.end(), [&](const auto& candidate) {
            return candidate.bundle == reference.bundle;
        });
        if (match == reverse.end())
            throw std::runtime_error("reverse pass omitted " + reference.bundle.string());
        if (match->model_id != reference.model_id ||
            match->pipeline_type != reference.pipeline_type ||
            match->result.text != reference.result.text ||
            match->result.token_ids != reference.result.token_ids)
            throw std::runtime_error("non-deterministic result for " + reference.bundle.string());
    }
}

void write_output(const Options& options, const std::vector<ResultEntry>& forward,
                  const std::vector<ResultEntry>& reverse,
                  const std::vector<ResultEntry>& concurrent) {
    if (!options.output.parent_path().empty())
        fs::create_directories(options.output.parent_path());
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("unable to open output: " + options.output.string());
    output << "{\"schema\":\"trtmc.qwen.edgellm.coexistence.v1\",\"load_order\":";
    write_string_array(output, options.bundles);
    output << ",\"forward\":";
    write_result_array(output, forward);
    output << ",\"reverse\":";
    write_result_array(output, reverse);
    output << ",\"concurrent\":";
    write_result_array(output, concurrent);
    output << "}\n";
    if (!output)
        throw std::runtime_error("unable to write output: " + options.output.string());
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        fs::create_directories(options.runtime_cache);

        trtmc::LoadOptions load_options;
        load_options.runtime_cache_path = options.runtime_cache.string();
        std::vector<LoadedPipeline> loaded;
        loaded.reserve(options.bundles.size());
        for (const auto& bundle : options.bundles) {
            auto pipeline = trtmc::load(bundle.string(), load_options);
            if (!pipeline)
                throw std::runtime_error("trtmc::load returned null for " + bundle.string());
            const std::string model_id = pipeline->model_id();
            const std::string pipeline_type = pipeline->pipeline_type();
            if (model_id.empty() || pipeline_type.empty())
                throw std::runtime_error("loaded pipeline metadata is empty for " +
                                         bundle.string());
            loaded.push_back(LoadedPipeline{bundle, model_id, pipeline_type, std::move(pipeline)});
        }

        trtmc::GenerateConfig generation;
        generation.max_new_tokens = options.max_new_tokens;
        generation.temperature = 0.0F;
        generation.top_p = 1.0F;
        generation.top_k = 1;
        generation.use_chat_template = false;
        generation.enable_thinking = false;

        std::vector<ResultEntry> forward;
        forward.reserve(loaded.size());
        for (auto& pipeline : loaded)
            forward.push_back(generate(pipeline, options.prompt, generation));

        std::vector<ResultEntry> reverse;
        reverse.reserve(loaded.size());
        for (auto iterator = loaded.rbegin(); iterator != loaded.rend(); ++iterator)
            reverse.push_back(generate(*iterator, options.prompt, generation));

        require_deterministic(forward, reverse);

        std::vector<std::future<ResultEntry>> pending;
        pending.reserve(loaded.size());
        for (std::size_t index = 0; index < loaded.size(); ++index) {
            pending.push_back(std::async(std::launch::async, [&, index] {
                return generate(loaded[index], options.prompt, generation);
            }));
        }
        std::vector<ResultEntry> concurrent;
        concurrent.reserve(pending.size());
        for (auto& result : pending)
            concurrent.push_back(result.get());
        require_deterministic(forward, concurrent);

        write_output(options, forward, reverse, concurrent);
        std::cout << "coexistence-ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen EdgeLLM coexistence runner failed: " << error.what() << '\n';
        return 1;
    }
}
