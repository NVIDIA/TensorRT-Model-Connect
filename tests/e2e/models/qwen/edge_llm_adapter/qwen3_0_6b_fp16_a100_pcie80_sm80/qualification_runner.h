// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen_edge_qualification {

namespace fs = std::filesystem;
using Json = nlohmann::json;

inline constexpr char kSchema[] = "trtmc.edgellm.long-lived.v1";
inline constexpr int32_t kWarmups = 5;
inline constexpr int32_t kMeasuredRequests = 30;

struct CliPaths {
    fs::path request;
    fs::path output;
};

struct Configuration {
    Json runtime;
    std::string prompt;
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    bool use_chat_template;
    bool enable_thinking;
};

struct Sample {
    std::string generated;
    std::vector<int32_t> token_ids;
};

struct Iteration {
    double latency_ms;
    Sample sample;
};

struct Measurements {
    double elapsed_ms;
    std::vector<Iteration> iterations;
};

struct RuntimeVersions {
    int32_t tensorrt_major;
    int32_t tensorrt_minor;
    int32_t tensorrt_patch;
    int32_t tensorrt_build;
    int32_t cuda_runtime;
};

inline std::string tensorrt_version_string(const RuntimeVersions& versions) {
    return std::to_string(versions.tensorrt_major) + "." + std::to_string(versions.tensorrt_minor) +
           "." + std::to_string(versions.tensorrt_patch) + "." +
           std::to_string(versions.tensorrt_build);
}

inline CliPaths parse_cli(int argc, char** argv) {
    CliPaths paths;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if ((argument == "--request" || argument == "--output") && index + 1 < argc) {
            fs::path& destination = argument == "--request" ? paths.request : paths.output;
            if (!destination.empty())
                throw std::invalid_argument("duplicate runner argument: " + argument);
            destination = argv[++index];
        } else {
            throw std::invalid_argument("usage: runner --request FILE --output FILE");
        }
    }
    if (paths.request.empty() || paths.output.empty())
        throw std::invalid_argument("usage: runner --request FILE --output FILE");
    return paths;
}

inline Json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("unable to open JSON input: " + path.string());
    Json value;
    input >> value;
    return value;
}

inline void require_exact_keys(const Json& object, std::initializer_list<const char*> expected,
                               const std::string& context) {
    if (!object.is_object())
        throw std::invalid_argument(context + " must be an object");
    std::set<std::string> actual;
    for (auto item = object.begin(); item != object.end(); ++item)
        actual.insert(item.key());
    std::set<std::string> required;
    for (const char* key : expected)
        required.emplace(key);
    if (actual != required)
        throw std::invalid_argument(context + " has unexpected or missing fields");
}

template <typename Value>
inline Value require_value(const Json& object, const char* key, const std::string& context) {
    try {
        return object.at(key).get<Value>();
    } catch (const Json::exception&) {
        throw std::invalid_argument(context + "." + key + " has the wrong type");
    }
}

inline Configuration read_configuration(const fs::path& path, const char* expected_runtime) {
    const Json root = read_json(path);
    require_exact_keys(root,
                       {"schema", "runtime", "prompt", "generation", "warmups_per_repetition",
                        "measured_requests_per_repetition", "require_native_token_ids",
                        "synchronize_each_request"},
                       "request");
    if (require_value<std::string>(root, "schema", "request") != kSchema)
        throw std::invalid_argument("request.schema is unsupported");
    if (require_value<int32_t>(root, "warmups_per_repetition", "request") != kWarmups ||
        require_value<int32_t>(root, "measured_requests_per_repetition", "request") !=
            kMeasuredRequests ||
        !require_value<bool>(root, "require_native_token_ids", "request") ||
        !require_value<bool>(root, "synchronize_each_request", "request"))
        throw std::invalid_argument("request does not describe the qualified measurement");

    const Json runtime = root.at("runtime");
    if (!runtime.is_object() ||
        require_value<std::string>(runtime, "kind", "request.runtime") != expected_runtime)
        throw std::invalid_argument("request.runtime.kind does not match this runner");
    const Json& generation = root.at("generation");
    require_exact_keys(
        generation,
        {"max_new_tokens", "temperature", "top_p", "top_k", "use_chat_template", "enable_thinking"},
        "request.generation");

    Configuration configuration{
        runtime,
        require_value<std::string>(root, "prompt", "request"),
        require_value<int32_t>(generation, "max_new_tokens", "request.generation"),
        require_value<float>(generation, "temperature", "request.generation"),
        require_value<float>(generation, "top_p", "request.generation"),
        require_value<int32_t>(generation, "top_k", "request.generation"),
        require_value<bool>(generation, "use_chat_template", "request.generation"),
        require_value<bool>(generation, "enable_thinking", "request.generation"),
    };
    if (configuration.prompt.empty() || configuration.max_new_tokens != 32 ||
        configuration.temperature != 0.0F || configuration.top_p != 1.0F ||
        configuration.top_k != 1 || configuration.use_chat_template ||
        configuration.enable_thinking)
        throw std::invalid_argument("request generation settings are outside qualification");
    return configuration;
}

inline fs::path require_path(const Json& runtime, const char* key) {
    fs::path path = require_value<std::string>(runtime, key, "request.runtime");
    if (path.empty())
        throw std::invalid_argument(std::string("request.runtime.") + key + " is empty");
    return path;
}

inline void validate_sample(const Sample& sample) {
    if (sample.generated.empty() || sample.token_ids.empty())
        throw std::runtime_error("runtime returned an empty qualification response");
    if (std::any_of(sample.token_ids.begin(), sample.token_ids.end(),
                    [](int32_t token) { return token < 0; }))
        throw std::runtime_error("runtime returned an invalid token ID");
}

template <typename Generate, typename Synchronize>
Measurements measure(Generate&& generate, Synchronize&& synchronize) {
    for (int32_t index = 0; index < kWarmups; ++index) {
        validate_sample(generate());
        synchronize();
    }

    using Clock = std::chrono::steady_clock;
    std::vector<Iteration> iterations;
    iterations.reserve(kMeasuredRequests);
    synchronize();
    const auto measured_start = Clock::now();
    for (int32_t index = 0; index < kMeasuredRequests; ++index) {
        synchronize();
        const auto start = Clock::now();
        Sample sample = generate();
        synchronize();
        const auto end = Clock::now();
        validate_sample(sample);
        const double latency = std::chrono::duration<double, std::milli>(end - start).count();
        iterations.push_back({latency, std::move(sample)});
    }
    const auto measured_end = Clock::now();
    return {std::chrono::duration<double, std::milli>(measured_end - measured_start).count(),
            std::move(iterations)};
}

inline void write_result(const fs::path& path, const char* runtime_kind,
                         const RuntimeVersions& versions, const Measurements& measurements) {
    Json iterations = Json::array();
    for (const Iteration& iteration : measurements.iterations) {
        iterations.push_back({{"latency_ms", iteration.latency_ms},
                              {"generated", iteration.sample.generated},
                              {"token_ids", iteration.sample.token_ids}});
    }
    const Json result = {{"schema", kSchema},
                         {"runtime_kind", runtime_kind},
                         {"runtime_initializations", 1},
                         {"decoding_cuda_graph_captured", true},
                         {"observed_tensorrt_version", tensorrt_version_string(versions)},
                         {"observed_cuda_runtime_version", versions.cuda_runtime},
                         {"native_token_ids", true},
                         {"synchronized_each_request", true},
                         {"warmups_completed", kWarmups},
                         {"measured_elapsed_ms", measurements.elapsed_ms},
                         {"iterations", std::move(iterations)}};
    std::ofstream output(path, std::ios::trunc);
    if (!output)
        throw std::runtime_error("unable to open JSON output: " + path.string());
    output << result.dump(2) << '\n';
    if (!output)
        throw std::runtime_error("unable to write JSON output: " + path.string());
}

} // namespace qwen_edge_qualification
