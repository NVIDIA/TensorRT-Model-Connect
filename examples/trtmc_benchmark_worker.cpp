/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/pipeline.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Arguments {
    std::string request_path;
    std::string output_path;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: trtmc_benchmark_worker --request REQUEST.json --output RESULT.json\n";
            std::exit(0);
        }
        if (argument != "--request" && argument != "--output") {
            throw std::runtime_error("unknown argument: " + argument);
        }
        if (++index >= argc) {
            throw std::runtime_error(argument + " requires a path");
        }
        std::string& destination =
            argument == "--request" ? arguments.request_path : arguments.output_path;
        destination = argv[index];
    }
    if (arguments.request_path.empty() || arguments.output_path.empty()) {
        throw std::runtime_error("--request and --output are required");
    }
    return arguments;
}

Json read_json(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open request: " + path);
    }
    Json value;
    stream >> value;
    if (!value.is_object()) {
        throw std::runtime_error("request must be a JSON object");
    }
    return value;
}

void write_json(const std::string& path, const Json& value) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open output: " + path);
    }
    stream << value.dump(2) << '\n';
}

template <typename Value>
Value optional_value(const Json& object, const std::string& key, Value fallback) {
    if (!object.contains(key)) {
        return fallback;
    }
    return object.at(key).get<Value>();
}

trtmc::LoadOptions load_options(const Json& runtime) {
    trtmc::LoadOptions options;
    options.hf_python = optional_value<std::string>(runtime, "hf_python", "");
    options.runtime_cache_path = optional_value<std::string>(runtime, "runtime_cache_path", "");
    options.cuda_graphs = optional_value<bool>(runtime, "cuda_graphs", false);
    options.kv_cache_size_bytes = optional_value<std::uint64_t>(runtime, "kv_cache_size_bytes", 0);
    options.config_path = optional_value<std::string>(runtime, "config_path", "");
    options.set_tokens = optional_value<std::vector<std::string>>(runtime, "set_tokens", {});
    options.backend_search_paths =
        optional_value<std::vector<std::string>>(runtime, "backend_search_paths", {});
    options.model_plugin_search_paths =
        optional_value<std::vector<std::string>>(runtime, "model_plugin_search_paths", {});
    return options;
}

trtmc::GenerateConfig generate_config(const Json& request) {
    trtmc::GenerateConfig config;
    config.max_new_tokens = optional_value<int32_t>(request, "max_new_tokens", 20);
    config.temperature = optional_value<float>(request, "temperature", 0.0F);
    config.top_k = optional_value<int32_t>(request, "top_k", 1);
    config.top_p = optional_value<float>(request, "top_p", 1.0F);
    config.min_p = optional_value<float>(request, "min_p", 0.0F);
    config.seed = optional_value<int32_t>(request, "seed", 42);
    config.guidance_scale = optional_value<float>(request, "guidance_scale", -1.0F);
    config.cfg_scale = optional_value<float>(request, "cfg_scale", -1.0F);
    config.num_steps = optional_value<int32_t>(request, "num_inference_steps", -1);
    config.negative_prompt = optional_value<std::string>(request, "negative_prompt", "");
    config.height = optional_value<int32_t>(request, "height", 0);
    config.width = optional_value<int32_t>(request, "width", 0);
    return config;
}

double elapsed_milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

double finite_sum(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0, [](double total, float value) {
        return std::isfinite(value) ? total + value : total;
    });
}

Json run_generate(trtmc::IPipeline& pipeline, const Json& request, int warmup, int iterations) {
    const std::string prompt = request.at("prompt").get<std::string>();
    const trtmc::GenerateConfig config = generate_config(request);
    trtmc::TextResult last;
    for (int index = 0; index < warmup; ++index) {
        last = pipeline.generate(prompt, config);
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        last = pipeline.generate(prompt, config);
        observations.push_back({
            {"iteration", index},
            {"runtime_e2e_wall_ms", elapsed_milliseconds(start)},
            {"output_tokens", last.token_ids.size()},
            {"prefill_ms", last.prefill_ms},
            {"decode_ms", last.decode_ms},
        });
    }
    const std::size_t text_limit = 4096;
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"text", last.text.substr(0, text_limit)},
             {"text_truncated", last.text.size() > text_limit},
             {"token_ids", last.token_ids},
         }},
    };
}

struct TranscriptionOutcome {
    trtmc::TextResult result;
    double first_partial_ms{-1.0};
    int chunks{1};
};

trtmc::TranscriptionStreamConfig transcription_stream_config(const trtmc::AudioResult& audio,
                                                             const Json& request,
                                                             const Json& streaming) {
    trtmc::TranscriptionStreamConfig config;
    config.input_sample_rate = audio.sample_rate;
    config.max_new_tokens = optional_value<int>(request, "max_new_tokens", 224);
    config.language = optional_value<std::string>(request, "language", "");
    if (streaming.contains("att_context_size")) {
        const auto context = streaming.at("att_context_size").get<std::vector<int>>();
        if (context.size() != 2) {
            throw std::runtime_error("streaming att_context_size must contain [left, right]");
        }
        config.att_context_left = context[0];
        config.att_context_right = context[1];
    }
    return config;
}

TranscriptionOutcome transcribe_streaming(trtmc::IPipeline& pipeline,
                                          const trtmc::AudioResult& audio, const Json& request,
                                          const Json& streaming) {
    const auto config = transcription_stream_config(audio, request, streaming);
    auto stream = pipeline.create_transcription_stream(config);
    const int chunk_ms = optional_value<int>(streaming, "chunk_ms", 160);
    if (chunk_ms <= 0) {
        throw std::runtime_error("streaming chunk_ms must be positive");
    }
    const int samples_per_chunk = std::max<int>(
        1, static_cast<int>(static_cast<std::int64_t>(audio.sample_rate) * chunk_ms / 1000));
    const auto stream_start = Clock::now();
    trtmc::TranscriptionStreamResult final;
    double first_partial_ms = -1.0;
    int chunks = 0;
    for (std::size_t offset = 0; offset < audio.samples.size();) {
        const auto remaining = audio.samples.size() - offset;
        const auto take =
            std::min<std::size_t>(remaining, static_cast<std::size_t>(samples_per_chunk));
        const bool is_final = offset + take >= audio.samples.size();
        final =
            stream->accept_audio(audio.samples.data() + offset, static_cast<int>(take), is_final);
        ++chunks;
        if (first_partial_ms < 0.0 && (!final.text.empty() || !final.token_ids.empty())) {
            first_partial_ms = elapsed_milliseconds(stream_start);
        }
        offset += take;
    }
    trtmc::TextResult result;
    result.text = std::move(final.text);
    result.token_ids = std::move(final.token_ids);
    return {std::move(result), first_partial_ms, chunks};
}

TranscriptionOutcome transcribe_once(trtmc::IPipeline& pipeline, const trtmc::AudioResult& audio,
                                     const Json& request) {
    const Json streaming = request.value("streaming", Json::object());
    if (optional_value<bool>(streaming, "enabled", false)) {
        return transcribe_streaming(pipeline, audio, request, streaming);
    }
    trtmc::TranscriptionConfig config;
    config.max_output_tokens = optional_value<int>(request, "max_new_tokens", 224);
    config.input_sample_rate = audio.sample_rate;
    const std::string language = optional_value<std::string>(request, "language", "");
    if (!language.empty()) {
        config.source_language = language;
    }
    return {pipeline.transcribe(audio.samples.data(), audio.num_samples, config), -1.0, 1};
}

Json run_transcribe(trtmc::IPipeline& pipeline, const Json& request, int warmup, int iterations) {
    const auto audio = trtmc::io::read_wav(request.at("audio_path").get<std::string>());
    if (audio.samples.empty() || audio.sample_rate <= 0) {
        throw std::runtime_error("transcribe audio input must contain samples and a sample rate");
    }
    TranscriptionOutcome last;
    for (int index = 0; index < warmup; ++index) {
        last = transcribe_once(pipeline, audio, request);
    }
    const double input_audio_seconds =
        static_cast<double>(audio.samples.size()) / static_cast<double>(audio.sample_rate);
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        last = transcribe_once(pipeline, audio, request);
        Json observation = {
            {"iteration", index},
            {"runtime_e2e_wall_ms", elapsed_milliseconds(start)},
            {"input_audio_seconds", input_audio_seconds},
            {"output_tokens", last.result.token_ids.size()},
            {"audio_chunks", last.chunks},
        };
        if (last.first_partial_ms >= 0.0) {
            observation["first_partial_ms"] = last.first_partial_ms;
        }
        observations.push_back(std::move(observation));
    }
    const std::size_t text_limit = 4096;
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"text", last.result.text.substr(0, text_limit)},
             {"text_truncated", last.result.text.size() > text_limit},
             {"token_ids", last.result.token_ids},
             {"input_samples", audio.samples.size()},
             {"input_sample_rate", audio.sample_rate},
             {"input_audio_seconds", input_audio_seconds},
         }},
    };
}

Json run_generate_image(trtmc::IPipeline& pipeline, const Json& request, int warmup,
                        int iterations) {
    const std::string prompt = request.at("prompt").get<std::string>();
    const trtmc::GenerateConfig config = generate_config(request);
    const int batch_size = optional_value<int>(request, "batch_size", 1);
    const std::vector<std::string> prompts =
        request.contains("prompts")
            ? request.at("prompts").get<std::vector<std::string>>()
            : std::vector<std::string>(static_cast<std::size_t>(batch_size), prompt);
    if (prompts.size() != static_cast<std::size_t>(batch_size)) {
        throw std::runtime_error("prompts must match batch_size");
    }
    std::vector<std::uint32_t> seeds;
    if (request.contains("seeds")) {
        seeds = request.at("seeds").get<std::vector<std::uint32_t>>();
        if (seeds.size() != prompts.size()) {
            throw std::runtime_error("seeds must match prompts");
        }
    } else {
        seeds.reserve(static_cast<std::size_t>(batch_size));
        for (int index = 0; index < batch_size; ++index) {
            seeds.push_back(static_cast<std::uint32_t>(config.seed + index));
        }
    }
    trtmc::io::LoadedImage input_image;
    if (request.contains("image_path")) {
        if (batch_size != 1) {
            throw std::runtime_error("image-conditioned generation supports batch_size=1 only");
        }
        input_image = trtmc::io::read_image(request.at("image_path").get<std::string>());
        if (input_image.empty()) {
            throw std::runtime_error("cannot decode image-conditioned generation input");
        }
    }
    std::vector<trtmc::ImageResult> last;
    const auto generate = [&]() {
        if (!input_image.empty()) {
            return std::vector<trtmc::ImageResult>{
                pipeline.generate_image(prompts.front(), input_image.pixels.data(),
                                        input_image.height, input_image.width, config)};
        }
        return pipeline.generate_image_batch(prompts, seeds, config);
    };
    for (int index = 0; index < warmup; ++index) {
        last = generate();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        last = generate();
        const std::size_t generated_pixels =
            std::accumulate(last.begin(), last.end(), std::size_t{0},
                            [](std::size_t count, const trtmc::ImageResult& image) {
                                return count + image.pixels.size();
                            });
        const std::size_t generated_frames = std::accumulate(
            last.begin(), last.end(), std::size_t{0},
            [](std::size_t count, const trtmc::ImageResult& image) {
                return count + static_cast<std::size_t>(std::max<int32_t>(image.num_frames, 1));
            });
        observations.push_back({
            {"iteration", index},
            {"runtime_e2e_wall_ms", elapsed_milliseconds(start)},
            {"generated_images", last.size()},
            {"generated_frames", generated_frames},
            {"generated_pixels", generated_pixels},
        });
    }
    if (last.empty()) {
        throw std::runtime_error("generate_image_batch returned no images");
    }
    const auto& first = last.front();
    double output_sum = 0.0;
    std::size_t element_count = 0;
    for (const auto& image : last) {
        output_sum += finite_sum(image.pixels);
        element_count += image.pixels.size();
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"batch_size", last.size()},
             {"height", first.height},
             {"width", first.width},
             {"channels", first.channels},
             {"num_frames", first.num_frames},
             {"element_count", element_count},
             {"finite_sum", output_sum},
         }},
    };
}

Json run_embedding(trtmc::IPipeline& pipeline, const Json& request, int warmup, int iterations,
                   bool pooled) {
    const std::string prompt = request.at("prompt").get<std::string>();
    trtmc::EmbeddingResult last;
    const auto encode = [&]() { return pooled ? pipeline.embed(prompt) : pipeline.encode(prompt); };
    for (int index = 0; index < warmup; ++index) {
        last = encode();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        last = encode();
        observations.push_back({
            {"iteration", index},
            {"runtime_e2e_wall_ms", elapsed_milliseconds(start)},
            {"embedding_vectors", 1},
            {"embedding_elements", last.data.size()},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"dim", last.dim},
             {"element_count", last.data.size()},
             {"finite_sum", finite_sum(last.data)},
         }},
    };
}

Json run_encode(trtmc::IPipeline& pipeline, const Json& request, int warmup, int iterations) {
    return run_embedding(pipeline, request, warmup, iterations, false);
}

Json run_embed(trtmc::IPipeline& pipeline, const Json& request, int warmup, int iterations) {
    return run_embedding(pipeline, request, warmup, iterations, true);
}

std::vector<float> float_array(const Json& request, const std::string& key) {
    if (!request.contains(key)) {
        return {};
    }
    return request.at(key).get<std::vector<float>>();
}

Json run_solve(trtmc::IPipeline& pipeline, const Json& request, int warmup, int iterations) {
    std::vector<float> branch = float_array(request, "branch_input");
    if (branch.empty()) {
        branch = float_array(request, "field_input");
    }
    const std::vector<float> trunk = float_array(request, "trunk_input");
    trtmc::EmbeddingResult last;
    const auto solve = [&]() {
        return pipeline.solve(
            branch.empty() ? nullptr : branch.data(), static_cast<int32_t>(branch.size()),
            trunk.empty() ? nullptr : trunk.data(), static_cast<int32_t>(trunk.size()));
    };
    for (int index = 0; index < warmup; ++index) {
        last = solve();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const auto start = Clock::now();
        last = solve();
        observations.push_back({
            {"iteration", index},
            {"runtime_e2e_wall_ms", elapsed_milliseconds(start)},
            {"windows", 1},
            {"forecast_elements", last.data.size()},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"dim", last.dim},
             {"element_count", last.data.size()},
             {"finite_sum", finite_sum(last.data)},
         }},
    };
}

Json execute(const Json& request) {
    if (request.value("schema_version", 0) != 1) {
        throw std::runtime_error("unsupported request schema_version");
    }
    const std::string bundle = request.at("bundle").get<std::string>();
    const std::string operation = request.at("operation").get<std::string>();
    const Json runtime = request.value("runtime", Json::object());
    const Json measurement = request.at("measurement");
    const int warmup = measurement.at("warmup").get<int>();
    const int iterations = measurement.at("iterations").get<int>();
    if (warmup < 0 || iterations <= 0) {
        throw std::runtime_error("warmup must be non-negative and iterations must be positive");
    }

    const auto load_start = Clock::now();
    auto pipeline = trtmc::load(bundle, load_options(runtime));
    const double load_ms = elapsed_milliseconds(load_start);

    using OperationRunner = Json (*)(trtmc::IPipeline&, const Json&, int, int);
    static const std::unordered_map<std::string, OperationRunner> runners = {
        {"generate", run_generate}, {"generate_image", run_generate_image},
        {"encode", run_encode},     {"embed", run_embed},
        {"solve", run_solve},       {"transcribe", run_transcribe},
    };
    const auto runner = runners.find(operation);
    if (runner == runners.end()) {
        throw std::runtime_error("unsupported operation: " + operation);
    }
    Json operation_result = runner->second(*pipeline, request.at("request"), warmup, iterations);

    return {
        {"schema_version", "trtmc.benchmark-worker-result/v1"},
        {"status", "completed"},
        {"case_name", request.at("case_name")},
        {"case_digest", request.at("case_digest")},
        {"model_id", pipeline->model_id()},
        {"pipeline_type", pipeline->pipeline_type()},
        {"operation", operation},
        {"timing_scope", "public_pipeline_call_wall"},
        {"load_ms", load_ms},
        {"warmup", warmup},
        {"iterations", iterations},
        {"observations", std::move(operation_result.at("observations"))},
        {"output_summary", std::move(operation_result.at("output_summary"))},
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string output_path;
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        output_path = arguments.output_path;
        write_json(output_path, execute(read_json(arguments.request_path)));
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "trtmc_benchmark_worker: " << exception.what() << '\n';
        if (!output_path.empty()) {
            try {
                write_json(output_path, {
                                            {"schema_version", "trtmc.benchmark-worker-result/v1"},
                                            {"status", "failed"},
                                            {"error", exception.what()},
                                        });
            } catch (const std::exception&) {
            }
        }
        return 1;
    }
}
