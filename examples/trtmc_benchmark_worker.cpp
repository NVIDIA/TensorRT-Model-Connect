/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/pipeline.h"
#include "trtmc/runtime/measurement.h"
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

#ifndef TRTMC_BUILD_CONFIGURATION
#define TRTMC_BUILD_CONFIGURATION "unknown"
#endif

#ifndef TRTMC_SOURCE_REVISION
#define TRTMC_SOURCE_REVISION "unknown"
#endif

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Arguments {
    bool metadata{false};
    std::string request_path;
    std::string output_path;
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: trtmc_benchmark_worker --metadata\n"
                << "       trtmc_benchmark_worker --request REQUEST.json --output RESULT.json\n";
            std::exit(0);
        }
        if (argument == "--metadata") {
            arguments.metadata = true;
            continue;
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
    if (arguments.metadata) {
        if (!arguments.request_path.empty() || !arguments.output_path.empty()) {
            throw std::runtime_error("--metadata cannot be combined with request arguments");
        }
        return arguments;
    }
    if (arguments.request_path.empty() || arguments.output_path.empty()) {
        throw std::runtime_error("--request and --output are required");
    }
    return arguments;
}

Json worker_metadata() {
    return {
        {"schema_version", "trtmc.benchmark-worker-metadata/v1"},
        {"build",
         {
             {"configuration", TRTMC_BUILD_CONFIGURATION},
             {"source_revision", TRTMC_SOURCE_REVISION},
         }},
    };
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

std::vector<float> read_float32_values(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open float32 request input: " + path);
    }
    const std::streamoff byte_count = stream.tellg();
    if (byte_count < 0 || byte_count % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("float32 request input has an invalid byte size: " + path);
    }
    std::vector<float> values(static_cast<std::size_t>(byte_count) / sizeof(float));
    stream.seekg(0);
    if (!values.empty()) {
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(byte_count));
    }
    if (!stream) {
        throw std::runtime_error("cannot read float32 request input: " + path);
    }
    return values;
}

std::vector<float> optional_float32_values(const Json& request, const std::string& value_key,
                                           const std::string& path_key) {
    if (request.contains(value_key) && request.contains(path_key)) {
        throw std::runtime_error("request cannot set both " + value_key + " and " + path_key);
    }
    if (request.contains(path_key)) {
        return read_float32_values(request.at(path_key).get<std::string>());
    }
    return optional_value<std::vector<float>>(request, value_key, {});
}

trtmc::LoadOptions load_options(const Json& runtime) {
    trtmc::LoadOptions options;
    options.hf_python = optional_value<std::string>(runtime, "hf_python", "");
    options.runtime_cache_path = optional_value<std::string>(runtime, "runtime_cache_path", "");
    options.cuda_graphs = optional_value<bool>(runtime, "cuda_graphs", false);
    options.kv_cache_size_bytes = optional_value<std::uint64_t>(runtime, "kv_cache_size_bytes", 0);
    options.config_path = optional_value<std::string>(runtime, "config_path", "");
    options.set_tokens = optional_value<std::vector<std::string>>(runtime, "set_tokens", {});
    if (runtime.contains("config")) {
        const Json& config = runtime.at("config");
        if (!config.is_object()) {
            throw std::runtime_error("runtime.config must be an object");
        }
        for (const auto& [name, value] : config.items()) {
            const std::string encoded = value.is_string() ? value.get<std::string>() : value.dump();
            options.set_tokens.push_back(name + "=" + encoded);
        }
    }
    options.backend_search_paths =
        optional_value<std::vector<std::string>>(runtime, "backend_search_paths", {});
    options.model_plugin_search_paths =
        optional_value<std::vector<std::string>>(runtime, "model_plugin_search_paths", {});
    return options;
}

trtmc::GenerateConfig generate_config(const Json& request) {
    trtmc::GenerateConfig config;
    config.max_new_tokens = optional_value<int32_t>(request, "max_new_tokens", 20);
    config.num_samples = optional_value<int32_t>(request, "num_samples", 1);
    config.temperature = optional_value<float>(request, "temperature", 0.0F);
    config.top_k = optional_value<int32_t>(request, "top_k", 1);
    config.top_p = optional_value<float>(request, "top_p", 1.0F);
    config.min_p = optional_value<float>(request, "min_p", 0.0F);
    config.seed = optional_value<int32_t>(request, "seed", -1);
    config.guidance_scale = optional_value<float>(request, "guidance_scale", -1.0F);
    config.cfg_scale = optional_value<float>(request, "cfg_scale", -1.0F);
    config.num_steps = optional_value<int32_t>(request, "num_inference_steps", -1);
    config.sde_gamma = optional_value<float>(request, "sde_gamma", -1.0F);
    config.initial_latents =
        optional_float32_values(request, "initial_latents", "initial_latents_path");
    config.condition_latents =
        optional_float32_values(request, "condition_latents", "condition_latents_path");
    config.condition_mask =
        optional_float32_values(request, "condition_mask", "condition_mask_path");
    config.sampling_steps =
        optional_float32_values(request, "sampling_steps", "sampling_steps_path");
    config.sde_noises = optional_float32_values(request, "sde_noises", "sde_noises_path");
    config.negative_prompt = optional_value<std::string>(request, "negative_prompt", "");
    config.height = optional_value<int32_t>(request, "height", 0);
    config.width = optional_value<int32_t>(request, "width", 0);
    config.eos_token_id = optional_value<int32_t>(request, "eos_token_id", -1);
    config.text_generation_mode = optional_value<std::string>(request, "generation_mode", "auto");
    config.block_length = optional_value<int32_t>(request, "block_length", 0);
    config.confidence_threshold = optional_value<float>(request, "threshold", -1.0F);
    config.tail_frames = optional_value<int32_t>(request, "tail_frames", 0);
    config.use_chat_template = optional_value<bool>(request, "use_chat_template", false);
    config.enable_thinking = optional_value<bool>(request, "enable_thinking", true);
    return config;
}

double elapsed_milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct TimingConfig {
    int warmup{0};
    int iterations{1};
    std::string scope{"public_pipeline_call_wall"};
    bool asset_loading_included{false};
};

class IterationTimer {
  public:
    explicit IterationTimer(const std::string& scope) : scope_(scope), start_(Clock::now()) {
        if (scope_ == "model_call_wall") {
            trtmc::runtime_measurement::reset_model_call();
        }
    }

    double elapsed_ms() const {
        if (scope_ == "public_pipeline_call_wall") {
            return elapsed_milliseconds(start_);
        }
        const double elapsed = trtmc::runtime_measurement::model_call_wall_ms();
        if (elapsed < 0.0) {
            throw std::runtime_error(
                "model_call_wall requested but the operation invoked no TensorRT module");
        }
        return elapsed;
    }

  private:
    std::string scope_;
    Clock::time_point start_;
};

double finite_sum(const std::vector<float>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0, [](double total, float value) {
        return std::isfinite(value) ? total + value : total;
    });
}

Json run_generate(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const std::string prompt = request.at("prompt").get<std::string>();
    const trtmc::GenerateConfig config = generate_config(request);
    const auto load_input_image = [&]() {
        auto image = trtmc::io::read_image(request.at("image_path").get<std::string>());
        if (image.empty()) {
            throw std::runtime_error("cannot decode generate image input");
        }
        return image;
    };
    trtmc::io::LoadedImage input_image;
    if (request.contains("image_path") && !timing.asset_loading_included) {
        input_image = load_input_image();
    }
    const auto generate = [&]() {
        if (request.contains("image_path") && timing.asset_loading_included) {
            input_image = load_input_image();
        }
        if (!input_image.empty()) {
            return pipeline.generate(prompt, input_image.pixels.data(), input_image.height,
                                     input_image.width, config);
        }
        return pipeline.generate(prompt, config);
    };
    trtmc::TextResult last;
    for (int index = 0; index < warmup; ++index) {
        last = generate();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = generate();
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
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

Json run_transcribe(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const auto load_audio = [&]() {
        auto value = trtmc::io::read_wav(request.at("audio_path").get<std::string>());
        if (value.samples.empty() || value.sample_rate <= 0) {
            throw std::runtime_error(
                "transcribe audio input must contain samples and a sample rate");
        }
        return value;
    };
    trtmc::AudioResult audio;
    if (!timing.asset_loading_included) {
        audio = load_audio();
    }
    const auto transcribe = [&]() {
        if (timing.asset_loading_included) {
            audio = load_audio();
        }
        return transcribe_once(pipeline, audio, request);
    };
    TranscriptionOutcome last;
    for (int index = 0; index < warmup; ++index) {
        last = transcribe();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = transcribe();
        const double measured_ms = timer.elapsed_ms();
        const double input_audio_seconds =
            static_cast<double>(audio.samples.size()) / static_cast<double>(audio.sample_rate);
        Json observation = {
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
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
    const double input_audio_seconds =
        static_cast<double>(audio.samples.size()) / static_cast<double>(audio.sample_rate);
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

std::vector<std::string> image_prompts(const Json& request, int batch_size) {
    std::vector<std::string> prompts;
    if (request.contains("prompts")) {
        prompts = request.at("prompts").get<std::vector<std::string>>();
    } else {
        prompts = std::vector<std::string>(static_cast<std::size_t>(batch_size),
                                           request.at("prompt").get<std::string>());
    }
    if (prompts.size() != static_cast<std::size_t>(batch_size)) {
        throw std::runtime_error("prompts must match batch_size");
    }
    return prompts;
}

std::vector<std::uint32_t> image_seeds(const Json& request, const trtmc::GenerateConfig& config,
                                       std::size_t prompt_count) {
    std::vector<std::uint32_t> seeds;
    if (request.contains("seeds")) {
        seeds = request.at("seeds").get<std::vector<std::uint32_t>>();
        if (seeds.size() != prompt_count) {
            throw std::runtime_error("seeds must match prompts");
        }
    } else {
        seeds.reserve(prompt_count);
        for (std::size_t index = 0; index < prompt_count; ++index) {
            seeds.push_back(static_cast<std::uint32_t>(config.seed + index));
        }
    }
    return seeds;
}

trtmc::io::LoadedImage image_condition(const Json& request, int batch_size) {
    trtmc::io::LoadedImage input_image;
    if (!request.contains("image_path")) {
        return input_image;
    }
    if (batch_size != 1) {
        throw std::runtime_error("image-conditioned generation supports batch_size=1 only");
    }
    input_image = trtmc::io::read_image(request.at("image_path").get<std::string>());
    if (input_image.empty()) {
        throw std::runtime_error("cannot decode image-conditioned generation input");
    }
    return input_image;
}

std::vector<trtmc::ImageResult> generate_images(trtmc::IPipeline& pipeline,
                                                const std::vector<std::string>& prompts,
                                                const std::vector<std::uint32_t>& seeds,
                                                const trtmc::io::LoadedImage& input_image,
                                                const trtmc::GenerateConfig& config) {
    if (!input_image.empty()) {
        return {pipeline.generate_image(prompts.front(), input_image.pixels.data(),
                                        input_image.height, input_image.width, config)};
    }
    return pipeline.generate_image_batch(prompts, seeds, config);
}

Json run_generate_image(trtmc::IPipeline& pipeline, const Json& request,
                        const TimingConfig& timing) {
    const trtmc::GenerateConfig config = generate_config(request);
    const int batch_size = optional_value<int>(request, "batch_size", 1);
    const std::vector<std::string> prompts = image_prompts(request, batch_size);
    const std::vector<std::uint32_t> seeds = image_seeds(request, config, prompts.size());
    const trtmc::io::LoadedImage input_image = image_condition(request, batch_size);
    std::vector<trtmc::ImageResult> last;
    const auto generate = [&]() {
        return generate_images(pipeline, prompts, seeds, input_image, config);
    };
    for (int index = 0; index < timing.warmup; ++index) {
        last = generate();
    }
    Json observations = Json::array();
    for (int index = 0; index < timing.iterations; ++index) {
        const IterationTimer timer(timing.scope);
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
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
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

std::size_t audio_sample_count(const trtmc::AudioResult& audio) {
    if (!audio.samples.empty()) {
        return audio.samples.size();
    }
    return static_cast<std::size_t>(std::max<int32_t>(audio.num_samples, 0));
}

double audio_seconds(const trtmc::AudioResult& audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output must have a positive sample rate");
    }
    return static_cast<double>(audio_sample_count(audio)) / static_cast<double>(audio.sample_rate);
}

Json audio_summary(const trtmc::AudioResult& audio) {
    return {
        {"num_samples", audio_sample_count(audio)},
        {"sample_rate", audio.sample_rate},
        {"audio_seconds", audio_seconds(audio)},
        {"finite_sum", finite_sum(audio.samples)},
    };
}

Json run_generate_audio(trtmc::IPipeline& pipeline, const Json& request,
                        const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const std::string prompt = request.at("prompt").get<std::string>();
    const trtmc::GenerateConfig config = generate_config(request);
    trtmc::AudioResult last;
    for (int index = 0; index < warmup; ++index) {
        last = pipeline.generate_audio(prompt, config);
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = pipeline.generate_audio(prompt, config);
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"output_samples", audio_sample_count(last)},
            {"output_audio_seconds", audio_seconds(last)},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary", audio_summary(last)},
    };
}

Json run_speak(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const auto audio = trtmc::io::read_wav(request.at("audio_path").get<std::string>());
    if (audio.samples.empty() || audio.sample_rate <= 0) {
        throw std::runtime_error("speak audio input must contain samples and a sample rate");
    }
    const trtmc::GenerateConfig config = generate_config(request);
    const double input_seconds =
        static_cast<double>(audio.samples.size()) / static_cast<double>(audio.sample_rate);
    trtmc::AudioResult last;
    for (int index = 0; index < warmup; ++index) {
        last = pipeline.speak(audio.samples.data(), static_cast<int32_t>(audio.samples.size()),
                              config, audio.sample_rate);
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = pipeline.speak(audio.samples.data(), static_cast<int32_t>(audio.samples.size()),
                              config, audio.sample_rate);
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"input_audio_seconds", input_seconds},
            {"output_audio_seconds", audio_seconds(last)},
        });
    }
    Json summary = audio_summary(last);
    summary["input_samples"] = audio.samples.size();
    summary["input_sample_rate"] = audio.sample_rate;
    summary["input_audio_seconds"] = input_seconds;
    return {
        {"observations", std::move(observations)},
        {"output_summary", std::move(summary)},
    };
}

trtmc::io::LoadedImage load_request_image(const Json& request, const std::string& operation) {
    auto image = trtmc::io::read_image(request.at("image_path").get<std::string>());
    if (image.empty()) {
        throw std::runtime_error("cannot decode " + operation + " image input");
    }
    return image;
}

Json run_segment(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const auto image = load_request_image(request, "segment");
    trtmc::SegmentResult last;
    for (int index = 0; index < warmup; ++index) {
        last = pipeline.segment(image.pixels.data(), image.height, image.width);
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = pipeline.segment(image.pixels.data(), image.height, image.width);
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"segmented_images", 1},
            {"mask_pixels", last.mask.size()},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"height", last.height},
             {"width", last.width},
             {"mask_pixels", last.mask.size()},
         }},
    };
}

Json run_segment_prompted(trtmc::IPipeline& pipeline, const Json& request,
                          const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const auto image = load_request_image(request, "segment_prompted");
    const auto segment = [&]() {
        if (request.contains("prompt")) {
            return pipeline.segment_prompted_text(image.pixels.data(), image.height, image.width,
                                                  request.at("prompt").get<std::string>());
        }
        return pipeline.segment_prompted(image.pixels.data(), image.height, image.width,
                                         optional_value<float>(request, "point_x", 0.5F),
                                         optional_value<float>(request, "point_y", 0.5F),
                                         optional_value<bool>(request, "is_foreground", true));
    };
    trtmc::PromptedSegmentationResult last;
    for (int index = 0; index < warmup; ++index) {
        last = segment();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = segment();
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"segmented_images", 1},
            {"generated_masks", std::max<int32_t>(last.num_masks, 0)},
            {"mask_pixels", last.masks.size()},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"height", last.height},
             {"width", last.width},
             {"num_masks", last.num_masks},
             {"mask_elements", last.masks.size()},
             {"mask_finite_sum", finite_sum(last.masks)},
         }},
    };
}

Json run_classify(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const auto image = load_request_image(request, "classify");
    trtmc::ClassificationResult last;
    for (int index = 0; index < warmup; ++index) {
        last = pipeline.classify(image.pixels.data(), image.height, image.width);
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = pipeline.classify(image.pixels.data(), image.height, image.width);
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"classified_images", 1},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"top_class", last.top_class},
             {"top_score", last.top_score},
             {"num_classes", last.logits.size()},
             {"logits_finite_sum", finite_sum(last.logits)},
         }},
    };
}

std::size_t detection_count(const std::string& payload) {
    const Json parsed = Json::parse(payload, nullptr, false);
    if (parsed.is_array()) {
        return parsed.size();
    }
    if (parsed.is_object() && parsed.contains("detections") && parsed.at("detections").is_array()) {
        return parsed.at("detections").size();
    }
    return 0;
}

Json run_detect(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const auto image = load_request_image(request, "detect");
    const float threshold = optional_value<float>(request, "score_threshold", 0.3F);
    std::string last;
    for (int index = 0; index < warmup; ++index) {
        last = pipeline.detect(image.pixels.data(), image.height, image.width, threshold);
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = pipeline.detect(image.pixels.data(), image.height, image.width, threshold);
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"detected_images", 1},
            {"detections", detection_count(last)},
        });
    }
    const std::size_t output_limit = 4096;
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"detections", detection_count(last)},
             {"json", last.substr(0, output_limit)},
             {"json_truncated", last.size() > output_limit},
         }},
    };
}

Json run_rerank(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const std::string query = request.at("query").get<std::string>();
    const auto documents = request.at("documents").get<std::vector<std::string>>();
    if (documents.empty()) {
        throw std::runtime_error("rerank documents cannot be empty");
    }
    std::vector<float> last;
    const auto rerank = [&]() {
        std::vector<float> scores;
        scores.reserve(documents.size());
        for (const auto& document : documents) {
            scores.push_back(pipeline.rerank(query, document));
        }
        return scores;
    };
    for (int index = 0; index < warmup; ++index) {
        last = rerank();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = rerank();
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
            {"documents", documents.size()},
        });
    }
    return {
        {"observations", std::move(observations)},
        {"output_summary",
         {
             {"documents", documents.size()},
             {"scores", last},
             {"scores_finite_sum", finite_sum(last)},
         }},
    };
}

Json run_embedding(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing,
                   bool pooled) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
    const std::string prompt = request.at("prompt").get<std::string>();
    trtmc::EmbeddingResult last;
    const auto encode = [&]() { return pooled ? pipeline.embed(prompt) : pipeline.encode(prompt); };
    for (int index = 0; index < warmup; ++index) {
        last = encode();
    }
    Json observations = Json::array();
    for (int index = 0; index < iterations; ++index) {
        const IterationTimer timer(timing.scope);
        last = encode();
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
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

Json run_encode(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    return run_embedding(pipeline, request, timing, false);
}

Json run_embed(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    return run_embedding(pipeline, request, timing, true);
}

std::vector<float> float_array(const Json& request, const std::string& key) {
    if (!request.contains(key)) {
        return {};
    }
    return request.at(key).get<std::vector<float>>();
}

Json run_solve(trtmc::IPipeline& pipeline, const Json& request, const TimingConfig& timing) {
    const int warmup = timing.warmup;
    const int iterations = timing.iterations;
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
        const IterationTimer timer(timing.scope);
        last = solve();
        const double measured_ms = timer.elapsed_ms();
        observations.push_back({
            {"iteration", index},
            {"measured_wall_ms", measured_ms},
            {"runtime_e2e_wall_ms", measured_ms},
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

bool supported_timing_scope(const std::string& scope) {
    return scope == "public_pipeline_call_wall" || scope == "model_call_wall";
}

bool supported_timed_asset(const std::string& operation, const Json& operation_request) {
    return (operation == "generate" && operation_request.contains("image_path")) ||
           (operation == "transcribe" && operation_request.contains("audio_path"));
}

TimingConfig timing_config(const Json& measurement, const std::string& operation,
                           const Json& operation_request) {
    TimingConfig timing;
    timing.warmup = measurement.at("warmup").get<int>();
    timing.iterations = measurement.at("iterations").get<int>();
    timing.scope =
        optional_value<std::string>(measurement, "timing_scope", "public_pipeline_call_wall");
    timing.asset_loading_included =
        optional_value<bool>(measurement, "asset_loading_included", false);
    if (timing.warmup < 0 || timing.iterations <= 0) {
        throw std::runtime_error("warmup must be non-negative and iterations must be positive");
    }
    if (!supported_timing_scope(timing.scope)) {
        throw std::runtime_error("unsupported measurement timing_scope: " + timing.scope);
    }
    if (timing.scope == "model_call_wall" && timing.asset_loading_included) {
        throw std::runtime_error("model_call_wall cannot include asset loading");
    }
    if (timing.asset_loading_included && !supported_timed_asset(operation, operation_request)) {
        throw std::runtime_error("asset_loading_included requires generate.image_path or "
                                 "transcribe.audio_path");
    }
    return timing;
}

Json execute(const Json& request) {
    if (request.value("schema_version", 0) != 1) {
        throw std::runtime_error("unsupported request schema_version");
    }
    const std::string bundle = request.at("bundle").get<std::string>();
    const std::string operation = request.at("operation").get<std::string>();
    const Json runtime = request.value("runtime", Json::object());
    const Json& operation_request = request.at("request");
    const TimingConfig timing =
        timing_config(request.at("measurement"), operation, operation_request);

    const auto load_start = Clock::now();
    auto pipeline = trtmc::load(bundle, load_options(runtime));
    const double load_ms = elapsed_milliseconds(load_start);

    using OperationRunner = Json (*)(trtmc::IPipeline&, const Json&, const TimingConfig&);
    static const std::unordered_map<std::string, OperationRunner> runners = {
        {"generate", run_generate},
        {"generate_image", run_generate_image},
        {"generate_audio", run_generate_audio},
        {"speak", run_speak},
        {"segment", run_segment},
        {"segment_prompted", run_segment_prompted},
        {"classify", run_classify},
        {"detect", run_detect},
        {"rerank", run_rerank},
        {"encode", run_encode},
        {"embed", run_embed},
        {"solve", run_solve},
        {"transcribe", run_transcribe},
    };
    const auto runner = runners.find(operation);
    if (runner == runners.end()) {
        throw std::runtime_error("unsupported operation: " + operation);
    }
    Json operation_result = runner->second(*pipeline, operation_request, timing);

    return {
        {"schema_version", "trtmc.benchmark-worker-result/v1"},
        {"status", "completed"},
        {"case_name", request.at("case_name")},
        {"case_digest", request.at("case_digest")},
        {"model_id", pipeline->model_id()},
        {"pipeline_type", pipeline->pipeline_type()},
        {"operation", operation},
        {"timing_scope", timing.scope},
        {"asset_loading_included", timing.asset_loading_included},
        {"load_ms", load_ms},
        {"warmup", timing.warmup},
        {"iterations", timing.iterations},
        {"observations", std::move(operation_result.at("observations"))},
        {"output_summary", std::move(operation_result.at("output_summary"))},
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string output_path;
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        if (arguments.metadata) {
            std::cout << worker_metadata().dump(2) << '\n';
            return 0;
        }
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
