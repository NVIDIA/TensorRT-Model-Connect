/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/config/cli_support.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Sample {
    std::string sample_id;
    std::string answer;
    std::string prompt;
    std::optional<int32_t> seed_index;
};

std::string trim(std::string value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return value.substr(start, end - start);
}

std::string unescape_json_string(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        char ch = raw[i];
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (i + 1 >= raw.size())
            throw std::runtime_error("Invalid trailing escape in JSON string");
        char esc = raw[++i];
        switch (esc) {
        case '\\':
            out.push_back('\\');
            break;
        case '"':
            out.push_back('"');
            break;
        case 'n':
            out.push_back('\n');
            break;
        case 'r':
            out.push_back('\r');
            break;
        case 't':
            out.push_back('\t');
            break;
        default:
            throw std::runtime_error(std::string("Unsupported JSON escape: \\") + esc);
        }
    }
    return out;
}

bool extract_json_field(const std::string& line, const std::string& key, std::string& value) {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = line.find(needle);
    if (pos == std::string::npos)
        return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos)
        throw std::runtime_error("Malformed JSON line: missing ':' for key " + key);
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
        ++pos;
    if (pos >= line.size() || line[pos] != '"')
        throw std::runtime_error("Malformed JSON line: expected string value for key " + key);
    ++pos;
    std::string raw;
    bool escaped = false;
    for (; pos < line.size(); ++pos) {
        char ch = line[pos];
        if (escaped) {
            raw.push_back(ch);
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            raw.push_back(ch);
            escaped = true;
            continue;
        }
        if (ch == '"') {
            value = unescape_json_string(raw);
            return true;
        }
        raw.push_back(ch);
    }
    throw std::runtime_error("Malformed JSON line: unterminated string for key " + key);
}

bool extract_json_int_field(const std::string& line, const std::string& key, int32_t& value) {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = line.find(needle);
    if (pos == std::string::npos)
        return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos)
        throw std::runtime_error("Malformed JSON line: missing ':' for key " + key);
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
        ++pos;
    if (pos >= line.size())
        throw std::runtime_error("Malformed JSON line: missing integer value for key " + key);

    std::size_t end = pos;
    if (line[end] == '-')
        ++end;
    while (end < line.size() && std::isdigit(static_cast<unsigned char>(line[end])))
        ++end;
    if (end == pos || (line[pos] == '-' && end == pos + 1))
        throw std::runtime_error("Malformed JSON line: expected integer value for key " + key);

    value = std::stoi(line.substr(pos, end - pos));
    return true;
}

std::vector<Sample> load_samples(const std::string& dataset_path) {
    std::ifstream input(dataset_path);
    if (!input)
        throw std::runtime_error("Failed to open dataset file: " + dataset_path);

    std::vector<Sample> samples;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (trim(line).empty())
            continue;
        Sample sample;
        if (!extract_json_field(line, "sample_id", sample.sample_id) ||
            !extract_json_field(line, "answer", sample.answer) ||
            !extract_json_field(line, "prompt", sample.prompt)) {
            throw std::runtime_error("Dataset line missing required fields at line " +
                                     std::to_string(line_no));
        }
        int32_t seed_index = 0;
        if (extract_json_int_field(line, "seed_index", seed_index))
            sample.seed_index = seed_index;
        samples.push_back(std::move(sample));
    }
    return samples;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (char ch : text) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
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
    std::cerr << "Usage: trtmc_dataset_benchmark <bundle.trtfb> <dataset.jsonl> <output.jsonl> "
                 "[--max-new-tokens N] [--hf-python PATH] [--kv-cache-size SIZE] "
                 "[--backend-dir PATH] "
                 "[--temperature F] [--top-k N] [--top-p F] [--min-p F] [--seed N] "
                 "[--chat-template] [--no-thinking] [--stop-on-answer] "
                 "[--stop-check-interval N]\n";
}

std::uint64_t parse_size_bytes(const std::string& text) {
    if (text.empty())
        throw std::runtime_error("Empty kv-cache-size");
    std::size_t idx = 0;
    const double value = std::stod(text, &idx);
    std::string suffix = text.substr(idx);
    for (char& ch : suffix)
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    double multiplier = 1.0;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1.0;
    } else if (suffix == "K" || suffix == "KB" || suffix == "KIB") {
        multiplier = 1024.0;
    } else if (suffix == "M" || suffix == "MB" || suffix == "MIB") {
        multiplier = 1024.0 * 1024.0;
    } else if (suffix == "G" || suffix == "GB" || suffix == "GIB") {
        multiplier = 1024.0 * 1024.0 * 1024.0;
    } else if (suffix == "T" || suffix == "TB" || suffix == "TIB") {
        multiplier = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    } else {
        throw std::runtime_error("Unsupported kv-cache-size suffix: " + suffix);
    }
    return static_cast<std::uint64_t>(value * multiplier);
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

        output << "{\"sample_id\":\"" << json_escape(sample.sample_id) << "\""
               << ",\"gold_answer\":\"" << json_escape(sample.answer) << "\""
               << ",\"pred_answer\":\"" << json_escape(pred_answer) << "\""
               << ",\"generated_tokens\":" << generated_tokens << ",\"generated_token_ids\":[";
        for (std::size_t token_idx = 0; token_idx < result.token_ids.size(); ++token_idx) {
            if (token_idx > 0)
                output << ',';
            output << result.token_ids[token_idx];
        }
        output << "]"
               << ",\"prefill_ms\":" << std::fixed << std::setprecision(6) << result.prefill_ms
               << ",\"decode_ms\":" << std::fixed << std::setprecision(6) << result.decode_ms
               << ",\"wall_ms\":" << std::fixed << std::setprecision(6) << wall_ms
               << ",\"tokens_per_sec\":" << std::fixed << std::setprecision(6) << tok_per_sec
               << ",\"text\":\"" << json_escape(result.text) << "\"}\n";
        output.flush();

        std::cerr << "[trtmc.dataset_benchmark] sample=" << sample.sample_id
                  << " generated_tokens=" << generated_tokens << " decode_ms=" << result.decode_ms
                  << " tok/s=" << tok_per_sec << '\n';
    }

    return 0;
}
