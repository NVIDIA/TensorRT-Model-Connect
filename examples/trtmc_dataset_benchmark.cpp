/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/jsonl_io.h"
#include "cli/args.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string trim(std::string value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return value.substr(start, end - start);
}

std::vector<trtmc::cli::DatasetSample> load_samples(const std::string& dataset_path) {
    std::ifstream input(dataset_path);
    if (!input)
        throw std::runtime_error("Failed to open dataset file: " + dataset_path);

    std::vector<trtmc::cli::DatasetSample> samples;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (trim(line).empty())
            continue;
        samples.push_back(trtmc::cli::parse_dataset_line(line, line_no));
    }
    return samples;
}

std::optional<std::string> normalize_answer_value(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), ','), value.end());
    static const std::regex int_re("-?\\d+");
    std::smatch match;
    if (!std::regex_search(value, match, int_re))
        return std::nullopt;
    return match.str(0);
}

std::optional<std::string> extract_answer_from_text(const std::string& text) {
    static const std::regex boxed_re(R"(\\boxed\{([^}]*)\})");
    static const std::regex final_re(R"(Final answer:\s*([^\n\r]+))", std::regex_constants::icase);
    static const std::regex answer_re(R"((?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*(-?\d+))",
                                      std::regex_constants::icase);
    static const std::regex discourse_quantity_re(
        R"((?:therefore|thus|hence|so),?\s+(?:the\s+)?(?:answer|area|sum|difference|product|remainder|probability|count|number|value|total)[^\n\r]{0,64}?(?:is|=)\s*(-?\d+))",
        std::regex_constants::icase);
    static const std::regex mn_re(R"(m\s*\+\s*n\s*=\s*(-?\d+))", std::regex_constants::icase);
    static const std::regex int_re("-?\\d+");

    std::smatch match;
    if (std::regex_search(text, match, boxed_re)) {
        if (auto norm = normalize_answer_value(match.str(1)))
            return norm;
    }
    if (std::regex_search(text, match, final_re)) {
        if (auto norm = normalize_answer_value(match.str(1)))
            return norm;
    }

    std::optional<std::string> last_phrase_match;
    const std::regex* phrase_res[] = {&answer_re, &discourse_quantity_re, &mn_re};
    for (const std::regex* re : phrase_res) {
        for (std::sregex_iterator it(text.begin(), text.end(), *re), end; it != end; ++it) {
            if (auto norm = normalize_answer_value((*it).str(1)))
                last_phrase_match = norm;
        }
    }
    if (last_phrase_match)
        return last_phrase_match;

    std::optional<std::string> last_int;
    for (std::sregex_iterator it(text.begin(), text.end(), int_re), end; it != end; ++it)
        last_int = (*it).str(0);
    return last_int;
}

void usage() {
    std::cerr << "Usage: trtmc_dataset_benchmark <bundle.bundle> <dataset.jsonl> <output.jsonl> "
                 "[--max-new-tokens N] [--hf-python PATH] [--kv-cache-size SIZE] "
                 "[--backend-dir PATH] [--model-plugin-dir PATH] "
                 "[--temperature F] [--top-k N] [--top-p F] [--min-p F] [--seed N] "
                 "[--chat-template] [--no-thinking] [--stop-on-answer] "
                 "[--stop-check-interval N]\n";
}

std::uint64_t parse_size_bytes(const std::string& text) {
    auto parsed = trtmc::cli::parse_byte_size(text);
    if (!parsed.has_value())
        throw std::runtime_error("Invalid kv-cache-size: " + text);
    return *parsed;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }

    std::string bundle_path = argv[1];
    std::string dataset_path = argv[2];
    std::string output_path = argv[3];
    int32_t max_new_tokens = 12000;
    trtmc::LoadOptions load_options;
    float temperature = 1.0F;
    int32_t top_k = 1;
    float top_p = 1.0F;
    float min_p = 0.0F;
    int32_t seed = -1;
    bool use_chat_template = false;
    bool enable_thinking = true;
    bool stop_on_answer = false;
    int32_t stop_check_interval = 16;
    std::string config_path;
    std::vector<std::string> set_tokens;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };
        if (arg == "--config") {
            config_path = need_value(arg);
        } else if (arg == "--set") {
            set_tokens.emplace_back(need_value(arg));
        } else if (arg == "--max-new-tokens") {
            max_new_tokens = std::stoi(need_value(arg));
        } else if (arg == "--hf-python") {
            load_options.hf_python = need_value(arg);
        } else if (arg == "--kv-cache-size") {
            load_options.kv_cache_size_bytes = parse_size_bytes(need_value(arg));
        } else if (arg == "--backend-dir") {
            load_options.backend_search_paths.emplace_back(need_value(arg));
        } else if (arg == "--model-plugin-dir") {
            load_options.model_plugin_search_paths.emplace_back(need_value(arg));
        } else if (arg == "--temperature") {
            temperature = std::stof(need_value(arg));
        } else if (arg == "--top-k") {
            top_k = std::stoi(need_value(arg));
        } else if (arg == "--top-p") {
            top_p = std::stof(need_value(arg));
        } else if (arg == "--min-p") {
            min_p = std::stof(need_value(arg));
        } else if (arg == "--seed") {
            seed = std::stoi(need_value(arg));
        } else if (arg == "--chat-template") {
            use_chat_template = true;
        } else if (arg == "--no-thinking") {
            enable_thinking = false;
        } else if (arg == "--stop-on-answer") {
            stop_on_answer = true;
        } else if (arg == "--stop-check-interval") {
            stop_check_interval = std::stoi(need_value(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    // Generic config surface (no per-knob flags). Forward the inputs to
    // LoadOptions so pipeline_factory actually resolves them into the
    // ConfigBundle attached to PipelineContext. (Without this, plugins
    // only see schema defaults and the --set values silently no-op.)
    load_options.config_path = config_path;
    load_options.set_tokens = set_tokens;

    auto samples = load_samples(dataset_path);
    auto pipeline = trtmc::load(bundle_path, load_options);
    if (!pipeline)
        throw std::runtime_error("Failed to load bundle: " + bundle_path);

    trtmc::GenerateConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = temperature;
    cfg.top_k = top_k;
    cfg.top_p = top_p;
    cfg.min_p = min_p;
    cfg.seed = seed;
    cfg.use_chat_template = use_chat_template;
    cfg.enable_thinking = enable_thinking;
    cfg.stop_on_boxed_answer = stop_on_answer;
    cfg.stop_check_interval = stop_check_interval;

    std::ofstream output(output_path);
    if (!output)
        throw std::runtime_error("Failed to open output file: " + output_path);

    for (std::size_t sample_idx = 0; sample_idx < samples.size(); ++sample_idx) {
        const auto& sample = samples[sample_idx];
        if (seed >= 0) {
            const int32_t seed_index = sample.seed_index.value_or(static_cast<int32_t>(sample_idx));
            cfg.seed = seed + seed_index;
        }
        auto wall_start = std::chrono::steady_clock::now();
        trtmc::TextResult result = pipeline->generate(sample.prompt, cfg);
        auto wall_end = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        const std::size_t generated_tokens = result.token_ids.size();
        const double tok_per_sec =
            (result.decode_ms > 0.0 && generated_tokens > 0)
                ? (static_cast<double>(generated_tokens) / (result.decode_ms / 1000.0))
                : 0.0;
        const std::string pred_answer = extract_answer_from_text(result.text).value_or("");

        nlohmann::json record;
        record["sample_id"] = sample.sample_id;
        record["gold_answer"] = sample.answer;
        record["pred_answer"] = pred_answer;
        record["generated_tokens"] = generated_tokens;
        record["generated_token_ids"] = result.token_ids;
        record["setup_ms"] = result.setup_ms;
        record["prefill_ms"] = result.prefill_ms;
        record["decode_ms"] = result.decode_ms;
        record["wall_ms"] = wall_ms;
        record["tokens_per_sec"] = tok_per_sec;
        record["text"] = result.text;
        output << record.dump() << '\n';
        output.flush();

        std::cerr << "[trtmc.dataset_benchmark] sample=" << sample.sample_id
                  << " generated_tokens=" << generated_tokens << " decode_ms=" << result.decode_ms
                  << " tok/s=" << tok_per_sec << '\n';
    }

    return 0;
}
