/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"
#include "config.h"
#include "task_runtime.h"
#include "trtmc/action.hpp"
#include "trtmc/audio.hpp"
#include "trtmc/control.hpp"
#include "trtmc/features.hpp"
#include "trtmc/language.hpp"
#include "trtmc/numeric.hpp"
#include "trtmc/perception.hpp"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/speech.hpp"
#include "trtmc/structure.hpp"
#include "trtmc/task.h"
#include "trtmc/text.hpp"
#include "trtmc/tracking.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Arguments {
    std::string request_path;
    std::string output_path;
};

struct Timing {
    int warmup{0};
    int iterations{1};
    bool asset_loading_included{false};
};

using Image = trtmc::cli::io::LoadedImage;
using Audio = trtmc::AudioResult; // Existing mono-only interfaces.
using trtmc::cli::io::read_wav;

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            std::cout << "trtmc_benchmark_worker --request REQUEST.json --output RESULT.json\n";
            std::exit(0);
        }
        if (argument != "--request" && argument != "--output")
            throw std::invalid_argument("unknown argument: " + argument);
        if (++index >= argc)
            throw std::invalid_argument(argument + " requires a path");
        (argument == "--request" ? result.request_path : result.output_path) = argv[index];
    }
    if (result.request_path.empty() || result.output_path.empty())
        throw std::invalid_argument("--request and --output are required");
    return result;
}

Json read_json(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open " + path);
    Json result;
    input >> result;
    return result;
}

void write_json(const std::string& path, const Json& value) {
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("cannot write " + path);
    output << value.dump(2) << '\n';
    output.close();
    if (!output)
        throw std::runtime_error("failed to write " + path);
}

template <typename T>
T optional_value(const Json& value, const char* name, T default_value) {
    return value.contains(name) ? value.at(name).get<T>() : default_value;
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Timing parse_timing(const Json& value) {
    Timing result;
    result.warmup = value.at("warmup").get<int>();
    result.iterations = value.at("iterations").get<int>();
    result.asset_loading_included = optional_value<bool>(value, "asset_loading_included", false);
    if (result.warmup < 0 || result.iterations < 1)
        throw std::invalid_argument("warmup must be non-negative and iterations positive");
    if (optional_value<std::string>(value, "timing_scope", "public_task_call_wall") !=
        "public_task_call_wall") {
        throw std::invalid_argument("only public_task_call_wall is supported");
    }
    return result;
}

template <typename Interface>
Interface& require_interface(trtmc::ITask& task, const char* name) {
    auto* value = dynamic_cast<Interface*>(&task);
    if (value == nullptr)
        throw std::runtime_error(std::string("loaded task does not implement ") + name);
    return *value;
}

Image read_image(const std::string& path) {
    auto image = trtmc::cli::io::read_image(path);
    if (image.empty())
        throw std::runtime_error("cannot read image " + path);
    return image;
}

std::vector<float> read_float32(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("cannot read float32 input " + path);
    const auto end = input.tellg();
    if (end <= 0 || end % static_cast<std::streamoff>(sizeof(float)) != 0)
        throw std::runtime_error("invalid float32 input " + path);
    const auto bytes = static_cast<std::uint64_t>(end);
    std::vector<float> values(static_cast<std::size_t>(bytes / sizeof(float)));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!input)
        throw std::runtime_error("truncated float32 input " + path);
    return values;
}

trtmc::TextGenerationConfig text_config(const Json& request) {
    trtmc::TextGenerationConfig config;
    config.max_new_tokens = optional_value<std::int32_t>(request, "max_new_tokens", 128);
    config.temperature = optional_value<float>(request, "temperature", 1.0F);
    config.top_k = optional_value<std::int32_t>(request, "top_k", 1);
    config.top_p = optional_value<float>(request, "top_p", 1.0F);
    config.min_p = optional_value<float>(request, "min_p", 0.0F);
    config.seed = optional_value<std::int32_t>(request, "seed", -1);
    config.guidance_scale = optional_value<float>(request, "guidance_scale", -1.0F);
    config.cfg_scale = optional_value<float>(request, "cfg_scale", -1.0F);
    config.num_steps = optional_value<std::int32_t>(request, "num_steps", -1);
    config.text_generation_mode =
        optional_value<std::string>(request, "text_generation_mode", "auto");
    config.block_length = optional_value<std::int32_t>(request, "block_length", 0);
    config.confidence_threshold = optional_value<float>(request, "confidence_threshold", -1.0F);
    config.use_chat_template = optional_value<bool>(request, "use_chat_template", false);
    config.enable_thinking = optional_value<bool>(request, "enable_thinking", true);
    config.repetition_penalty = optional_value<float>(request, "repetition_penalty", 1.0F);
    return config;
}

trtmc::ImageGenerationConfig image_config(const Json& request) {
    trtmc::ImageGenerationConfig config;
    config.num_samples = optional_value<std::int32_t>(request, "batch_size", 1);
    config.seed = optional_value<std::int32_t>(request, "seed", -1);
    config.guidance_scale = optional_value<float>(request, "guidance_scale", -1.0F);
    config.cfg_scale = optional_value<float>(request, "cfg_scale", -1.0F);
    config.num_steps = optional_value<std::int32_t>(request, "num_steps", -1);
    config.negative_prompt = optional_value<std::string>(request, "negative_prompt", "");
    config.height = optional_value<std::int32_t>(request, "height", 0);
    config.width = optional_value<std::int32_t>(request, "width", 0);
    return config;
}

template <typename Invoke, typename Observe>
Json measure(const Timing& timing, Invoke&& invoke, Observe&& observe) {
    using Result = decltype(invoke());
    std::optional<Result> last;
    for (int index = 0; index < timing.warmup; ++index)
        last = invoke();
    Json observations = Json::array();
    for (int index = 0; index < timing.iterations; ++index) {
        last.reset(); // Previous-result destruction is not part of this call.
        const auto started = Clock::now();
        auto result = invoke();
        const auto wall_ms = elapsed_ms(started);
        last.emplace(std::move(result));
        Json observation = observe(*last);
        observation["runtime_e2e_wall_ms"] = wall_ms;
        observations.push_back(std::move(observation));
    }
    return {{"observations", std::move(observations)},
            {"output_summary", last ? observe(*last) : Json::object()}};
}

Json run_generate(trtmc::ITask& task, const Json& request, const Timing& timing) {
    const std::string prompt = request.at("prompt").get<std::string>();
    const auto config = text_config(request);
    if (!request.contains("image_path")) {
        auto& interface = require_interface<trtmc::ITextGeneration>(task, "ITextGeneration");
        return measure(
            timing, [&]() { return interface.generate(prompt, config); },
            [](const trtmc::TextResult& result) {
                return Json{{"output_tokens", result.token_ids.size()},
                            {"token_ids", result.token_ids},
                            {"prefill_ms", result.prefill_ms},
                            {"decode_ms", result.decode_ms},
                            {"text", result.text}};
            });
    }
    auto& interface =
        require_interface<trtmc::IVisionLanguageGeneration>(task, "IVisionLanguageGeneration");
    const std::string path = request.at("image_path").get<std::string>();
    std::optional<Image> cached;
    if (!timing.asset_loading_included)
        cached = read_image(path);
    return measure(
        timing,
        [&]() {
            if (cached) {
                return interface.generate(prompt, cached->pixels.data(), cached->height,
                                          cached->width, config);
            }
            const Image image = read_image(path);
            return interface.generate(prompt, image.pixels.data(), image.height, image.width,
                                      config);
        },
        [](const trtmc::TextResult& result) {
            return Json{{"output_tokens", result.token_ids.size()},
                        {"token_ids", result.token_ids},
                        {"prefill_ms", result.prefill_ms},
                        {"decode_ms", result.decode_ms},
                        {"text", result.text}};
        });
}

Json image_observation(const std::vector<trtmc::ImageResult>& results) {
    std::size_t frames = 0;
    std::size_t pixels = 0;
    for (const auto& result : results) {
        frames += static_cast<std::size_t>(std::max(result.num_frames, 1));
        pixels += result.pixels.size();
    }
    Json value = {{"generated_images", results.size()},
                  {"batch_size", results.size()},
                  {"generated_frames", frames},
                  {"output_elements", pixels}};
    if (!results.empty()) {
        value["height"] = results.front().height;
        value["width"] = results.front().width;
        value["channels"] = results.front().channels;
        value["num_frames"] = results.front().num_frames;
        value["media_type"] = results.front().num_frames > 1 ? "video" : "image";
    }
    return value;
}

Json run_generate_image(trtmc::ITask& task, const Json& request, const Timing& timing) {
    const auto config = image_config(request);
    const std::string prompt =
        request.at("prompt").is_array() ? "" : request.at("prompt").get<std::string>();
    std::function<std::vector<trtmc::ImageResult>()> invoke;
    std::optional<Image> cached;

    if (auto* batch = dynamic_cast<trtmc::IImageBatchGeneration*>(&task)) {
        const auto prompts = request.at("prompt").get<std::vector<std::string>>();
        auto seeds = optional_value<std::vector<std::uint32_t>>(request, "seeds", {});
        if (seeds.empty())
            seeds.assign(prompts.size(), static_cast<std::uint32_t>(std::max(config.seed, 0)));
        invoke = [batch, prompts, seeds, config]() {
            return batch->generate_image_batch(prompts, seeds, config);
        };
    } else if (auto* edit = dynamic_cast<trtmc::IImageEditing*>(&task)) {
        const std::string path = request.at("image_path").get<std::string>();
        if (!timing.asset_loading_included)
            cached = read_image(path);
        invoke = [edit, prompt, path, config, &cached]() {
            if (cached) {
                return std::vector<trtmc::ImageResult>{edit->generate_image(
                    prompt, cached->pixels.data(), cached->height, cached->width, config)};
            }
            const Image image = read_image(path);
            return std::vector<trtmc::ImageResult>{edit->generate_image(
                prompt, image.pixels.data(), image.height, image.width, config)};
        };
    } else if (auto* world = dynamic_cast<trtmc::IWorldModelGeneration*>(&task)) {
        const std::string path = request.at("image_path").get<std::string>();
        if (!timing.asset_loading_included)
            cached = read_image(path);
        invoke = [world, prompt, path, config, request, &cached]() {
            std::optional<Image> loaded;
            if (!cached)
                loaded = read_image(path);
            const Image& image = cached ? *cached : *loaded;
            trtmc::WorldModelRequest value;
            value.prompt = prompt;
            value.image = image.pixels;
            value.image_height = image.height;
            value.image_width = image.width;
            value.action = optional_value<std::string>(request, "action", "");
            value.camera_intrinsics =
                optional_value<std::vector<float>>(request, "camera_intrinsics", {});
            value.num_frames = optional_value<std::int32_t>(request, "num_frames", 0);
            value.generation = config;
            return std::vector<trtmc::ImageResult>{world->generate_world(value)};
        };
    } else {
        auto& image = require_interface<trtmc::IImageGeneration>(task, "IImageGeneration");
        invoke = [&image, prompt, config]() {
            return std::vector<trtmc::ImageResult>{image.generate_image(prompt, config)};
        };
    }
    return measure(timing, invoke, image_observation);
}

Json run_generate_audio(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::IAudioGeneration>(task, "IAudioGeneration");
    trtmc::AudioGenerationConfig config;
    config.max_new_tokens = optional_value<std::int32_t>(request, "max_new_tokens", 128);
    config.talker_max_new_tokens =
        optional_value<std::int32_t>(request, "talker_max_new_tokens", 0);
    config.seed = optional_value<std::int32_t>(request, "seed", -1);
    const std::string prompt = request.at("prompt").get<std::string>();
    return measure(
        timing, [&]() { return interface.generate_audio(prompt, config); },
        [](const trtmc::AudioResult& result) {
            const double seconds =
                result.sample_rate > 0
                    ? static_cast<double>(result.samples.size()) / result.sample_rate
                    : 0.0;
            return Json{{"output_samples", result.samples.size()},
                        {"num_samples", result.samples.size()},
                        {"output_audio_seconds", seconds},
                        {"sample_rate", result.sample_rate}};
        });
}

Json run_speak(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::ISpeechToSpeech>(task, "ISpeechToSpeech");
    const std::string path = request.at("audio_path").get<std::string>();
    std::optional<Audio> cached;
    if (!timing.asset_loading_included)
        cached = read_wav(path);
    trtmc::SpeechToSpeechConfig config;
    config.max_new_tokens = optional_value<std::int32_t>(request, "max_new_tokens", 50);
    config.seed = optional_value<std::int32_t>(request, "seed", -1);
    config.tail_frames = optional_value<std::int32_t>(request, "tail_frames", 0);
    return measure(
        timing,
        [&]() {
            std::optional<Audio> loaded;
            if (!cached)
                loaded = read_wav(path);
            const Audio& audio = cached ? *cached : *loaded;
            return std::pair<trtmc::AudioResult, double>{
                interface.speak(audio.samples.data(),
                                static_cast<std::int32_t>(audio.samples.size()), config,
                                audio.sample_rate),
                static_cast<double>(audio.samples.size()) / audio.sample_rate};
        },
        [](const auto& value) {
            const auto& result = value.first;
            return Json{{"input_audio_seconds", value.second},
                        {"output_audio_seconds",
                         result.sample_rate > 0
                             ? static_cast<double>(result.samples.size()) / result.sample_rate
                             : 0.0},
                        {"output_samples", result.samples.size()},
                        {"num_samples", result.samples.size()},
                        {"sample_rate", result.sample_rate}};
        });
}

Json run_transcribe(trtmc::ITask& task, const Json& request, const Timing& timing) {
    const std::string path = request.at("audio_path").get<std::string>();
    std::optional<Audio> cached;
    if (!timing.asset_loading_included)
        cached = read_wav(path);
    const bool streaming = optional_value<bool>(request, "streaming", false);
    if (!streaming) {
        auto& interface = require_interface<trtmc::ITranscription>(task, "ITranscription");
        trtmc::TranscriptionConfig config;
        config.max_output_tokens = optional_value<std::int32_t>(request, "max_new_tokens", 224);
        config.source_language = optional_value<std::string>(request, "language", "en");
        return measure(
            timing,
            [&]() {
                std::optional<Audio> loaded;
                if (!cached)
                    loaded = read_wav(path);
                const Audio& audio = cached ? *cached : *loaded;
                config.input_sample_rate = audio.sample_rate;
                return std::pair<trtmc::TextResult, double>{
                    interface.transcribe(audio.samples.data(),
                                         static_cast<std::int32_t>(audio.samples.size()), config),
                    static_cast<double>(audio.samples.size()) / audio.sample_rate};
            },
            [](const auto& value) {
                return Json{{"input_audio_seconds", value.second},
                            {"output_tokens", value.first.token_ids.size()},
                            {"text", value.first.text}};
            });
    }

    auto& interface =
        require_interface<trtmc::IStreamingTranscription>(task, "IStreamingTranscription");
    return measure(
        timing,
        [&]() {
            std::optional<Audio> loaded;
            if (!cached)
                loaded = read_wav(path);
            const Audio& audio = cached ? *cached : *loaded;
            trtmc::TranscriptionStreamConfig config;
            config.input_sample_rate = audio.sample_rate;
            config.max_new_tokens = optional_value<std::int32_t>(request, "max_new_tokens", 224);
            config.language = optional_value<std::string>(request, "language", "");
            auto stream = interface.create_transcription_stream(config);
            const int chunk_ms = optional_value<int>(request, "chunk_ms", 160);
            const std::size_t chunk = std::max<std::size_t>(
                1, static_cast<std::size_t>(audio.sample_rate) * chunk_ms / 1000U);
            trtmc::TranscriptionStreamResult result;
            double first_partial = 0.0;
            const auto started = Clock::now();
            for (std::size_t offset = 0; offset < audio.samples.size(); offset += chunk) {
                const std::size_t count = std::min(chunk, audio.samples.size() - offset);
                result = stream->accept_audio(audio.samples.data() + offset,
                                              static_cast<std::int32_t>(count),
                                              offset + count == audio.samples.size());
                if (first_partial == 0.0 && !result.text.empty())
                    first_partial = elapsed_ms(started);
            }
            if (!result.is_final)
                result = stream->finish();
            return std::tuple<trtmc::TranscriptionStreamResult, double, double>{
                result, static_cast<double>(audio.samples.size()) / audio.sample_rate,
                first_partial};
        },
        [](const auto& value) {
            return Json{{"input_audio_seconds", std::get<1>(value)},
                        {"output_tokens", std::get<0>(value).token_ids.size()},
                        {"first_partial_ms", std::get<2>(value)},
                        {"text", std::get<0>(value).text}};
        });
}

Json run_segment(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::ISegmentation>(task, "ISegmentation");
    const Image image = read_image(request.at("image_path").get<std::string>());
    return measure(
        timing, [&]() { return interface.segment(image.pixels.data(), image.height, image.width); },
        [](const trtmc::SegmentResult& result) {
            return Json{{"segmented_images", 1},
                        {"num_masks", 1},
                        {"height", result.height},
                        {"width", result.width},
                        {"mask_pixels", result.mask.size()}};
        });
}

Json run_segment_prompted(trtmc::ITask& task, const Json& request, const Timing& timing) {
    const Image image = read_image(request.at("image_path").get<std::string>());
    std::function<trtmc::PromptedSegmentationResult()> invoke;
    if (request.contains("prompt")) {
        auto& interface =
            require_interface<trtmc::ITextPromptedSegmentation>(task, "ITextPromptedSegmentation");
        const std::string prompt = request.at("prompt").get<std::string>();
        invoke = [&interface, &image, prompt]() {
            return interface.segment_prompted_text(image.pixels.data(), image.height, image.width,
                                                   prompt);
        };
    } else {
        auto& interface = require_interface<trtmc::IPointPromptedSegmentation>(
            task, "IPointPromptedSegmentation");
        const float x = optional_value<float>(request, "point_x", 0.5F);
        const float y = optional_value<float>(request, "point_y", 0.5F);
        const bool foreground = optional_value<bool>(request, "is_foreground", true);
        invoke = [&interface, &image, x, y, foreground]() {
            return interface.segment_prompted(image.pixels.data(), image.height, image.width, x, y,
                                              foreground);
        };
    }
    return measure(timing, invoke, [](const trtmc::PromptedSegmentationResult& result) {
        return Json{{"segmented_images", 1},         {"generated_masks", result.num_masks},
                    {"num_masks", result.num_masks}, {"height", result.height},
                    {"width", result.width},         {"mask_pixels", result.masks.size()}};
    });
}

Json run_classify(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::IImageClassification>(task, "IImageClassification");
    const Image image = read_image(request.at("image_path").get<std::string>());
    return measure(
        timing,
        [&]() { return interface.classify(image.pixels.data(), image.height, image.width); },
        [](const trtmc::ClassificationResult& result) {
            return Json{{"classified_images", 1},
                        {"top_class", result.top_class},
                        {"top_score", result.top_score}};
        });
}
Json run_detect(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::IObjectDetection>(task, "IObjectDetection");
    const Image image = read_image(request.at("image_path").get<std::string>());
    return measure(
        timing, [&]() { return interface.detect(image.pixels.data(), image.height, image.width); },
        [](const trtmc::ObjectDetectionResult& result) {
            return Json{{"detected_images", 1},
                        {"detections", result.boxes.size()},
                        {"image_height", result.image_height},
                        {"image_width", result.image_width}};
        });
}

Json run_extract_features(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface =
        require_interface<trtmc::IImageFeatureExtractor>(task, "IImageFeatureExtractor");
    const Image image = read_image(request.at("image_path").get<std::string>());
    return measure(
        timing,
        [&]() {
            return interface.extract_image_features(image.pixels.data(), image.height, image.width);
        },
        [](const trtmc::ImageFeaturesResult& result) {
            return Json{{"processed_images", 1},
                        {"last_hidden_state_shape", result.last_hidden_state_shape},
                        {"pooler_output_shape", result.pooler_output_shape},
                        {"feature_elements",
                         result.last_hidden_state.size() + result.pooler_output.size()}};
        });
}

Json run_disparity(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::IStereoDisparity>(task, "IStereoDisparity");
    const Image left = read_image(request.at("left_image_path").get<std::string>());
    const Image right = read_image(request.at("right_image_path").get<std::string>());
    if (left.height != right.height || left.width != right.width)
        throw std::invalid_argument("stereo images must have identical dimensions");
    auto invoke = [&]() {
        return interface.estimate_disparity(left.pixels.data(), right.pixels.data(), left.height,
                                            left.width);
    };
    trtmc::StereoDisparityResult last;
    for (int index = 0; index < timing.warmup; ++index)
        last = invoke();
    Json observations = Json::array();
    for (int index = 0; index < timing.iterations; ++index) {
        last = {};
        const auto started = Clock::now();
        auto result = invoke();
        const auto wall_ms = elapsed_ms(started);
        last = std::move(result);
        observations.push_back({{"runtime_e2e_wall_ms", wall_ms},
                                {"stereo_pairs", 1},
                                {"disparity_pixels", last.disparity.size()}});
    }
    const std::string artifact = request.at("_artifact_path").get<std::string>();
    std::ofstream output(artifact, std::ios::binary);
    output.write(reinterpret_cast<const char*>(last.disparity.data()),
                 static_cast<std::streamsize>(last.disparity.size() * sizeof(float)));
    output.close();
    if (!output)
        throw std::runtime_error("cannot write disparity artifact " + artifact);
    return {{"observations", std::move(observations)},
            {"output_summary",
             {{"stereo_pairs", 1},
              {"disparity_pixels", last.disparity.size()},
              {"element_count", last.disparity.size()},
              {"height", last.height},
              {"width", last.width},
              {"disparity_artifact", artifact}}}};
}

Json run_rerank(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::IReranking>(task, "IReranking");
    const std::string query = request.at("query").get<std::string>();
    const auto documents = request.at("documents").get<std::vector<std::string>>();
    return measure(
        timing, [&]() { return interface.rerank_batch(query, documents); },
        [documents](const std::vector<float>& result) {
            return Json{{"documents", documents.size()}, {"scores", result}};
        });
}

Json run_embedding(trtmc::ITask& task, const Json& request, const Timing& timing, bool pooled) {
    const std::string prompt = request.at("prompt").get<std::string>();
    std::function<trtmc::EmbeddingResult()> invoke;
    if (pooled) {
        auto& interface = require_interface<trtmc::IEmbedding>(task, "IEmbedding");
        invoke = [&interface, prompt]() { return interface.embed(prompt); };
    } else {
        auto& interface = require_interface<trtmc::IEncoding>(task, "IEncoding");
        invoke = [&interface, prompt]() { return interface.encode(prompt); };
    }
    return measure(timing, invoke, [](const trtmc::EmbeddingResult& result) {
        return Json{{"embedding_vectors", 1},
                    {"embedding_elements", result.data.size()},
                    {"dim", result.dim}};
    });
}

Json run_encode(trtmc::ITask& task, const Json& request, const Timing& timing) {
    return run_embedding(task, request, timing, false);
}

Json run_embed(trtmc::ITask& task, const Json& request, const Timing& timing) {
    return run_embedding(task, request, timing, true);
}

Json run_solve(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::ITimeSeriesForecast>(task, "ITimeSeriesForecast");
    const auto values = request.at("past_values").get<std::vector<float>>();
    auto mask = optional_value<std::vector<float>>(request, "observed_mask", {});
    if (mask.empty())
        mask.assign(values.size(), 1.0F);
    if (mask.size() != values.size())
        throw std::invalid_argument("observed_mask length must match past_values");
    const auto frequency = optional_value<std::int32_t>(request, "frequency", 0);
    return measure(
        timing,
        [&]() {
            return interface.forecast({trtmc::Span<const float>(values.data(), values.size()),
                                       trtmc::Span<const float>(mask.data(), mask.size()),
                                       frequency});
        },
        [](const trtmc::ForecastResult& result) {
            return Json{{"windows", 1},
                        {"forecast_elements", result.values.size()},
                        {"shape", result.shape}};
        });
}

Json run_control(trtmc::ITask& task, const Json& request, const Timing& timing) {
    auto& interface = require_interface<trtmc::IRobotControl>(task, "IRobotControl");
    const std::string image_path = request.at("image_path").get<std::string>();
    const std::string state_path = request.at("state_path").get<std::string>();
    std::optional<Image> cached_image;
    std::optional<std::vector<float>> cached_state;
    if (!timing.asset_loading_included) {
        cached_image = read_image(image_path);
        cached_state = read_float32(state_path);
    }
    return measure(
        timing,
        [&]() {
            std::optional<Image> loaded_image;
            std::optional<std::vector<float>> loaded_state;
            if (!cached_image)
                loaded_image = read_image(image_path);
            if (!cached_state)
                loaded_state = read_float32(state_path);
            const Image& image = cached_image ? *cached_image : *loaded_image;
            const auto& state = cached_state ? *cached_state : *loaded_state;
            return interface.predict_action_chunk({{image.pixels.data(), image.pixels.size()},
                                                   image.height,
                                                   image.width,
                                                   3,
                                                   {state.data(), state.size()}});
        },
        [](const trtmc::RobotActionChunk& result) {
            return Json{{"action_steps", result.num_actions},
                        {"action_dim", result.action_dim},
                        {"action_values", result.actions.size()},
                        {"within_training_bounds", result.within_training_bounds},
                        {"inference_ms", result.inference_ms}};
        });
}

trtmc::Config sdk_config(const Json& request, const std::vector<trtmc::ConfigField>& fields,
                         std::initializer_list<std::string_view> input_names) {
    trtmc::Config config;
    auto add = [&](const std::string& name, const Json& value) {
        const auto field = std::find_if(fields.begin(), fields.end(),
                                        [&](const auto& field) { return field.name == name; });
        if (field == fields.end())
            throw std::invalid_argument("selected Task does not declare config '" + name + "'");
        if (field->kind == trtmc::ConfigKind::String && !value.is_string())
            throw std::invalid_argument("config '" + name + "' must be a string");
        config.add(name, trtmc::app::parse_config_value(
                             value.is_string() && field->kind == trtmc::ConfigKind::String
                                 ? value.get<std::string>()
                                 : value.dump(),
                             *field));
    };
    for (const auto& [name, value] : request.items()) {
        if (std::find(input_names.begin(), input_names.end(), name) != input_names.end())
            continue;
        if (name == "config") {
            if (!value.is_object())
                throw std::invalid_argument("config must be an object");
            for (const auto& [key, item] : value.items())
                add(key, item);
        } else
            add(name, value);
    }
    return config; // No model defaults; conflicting flat/nested keys stay duplicated.
}

Json transcript_observation(const trtmc::TextResultView& result) {
    Json tokens = Json::array(), segments = Json::array();
    for (auto id : result.token_ids)
        tokens.push_back(id);
    for (const auto& segment : result.segments) {
        Json ids = Json::array();
        for (auto id : segment.token_ids)
            ids.push_back(id);
        segments.push_back({{"start_seconds", segment.start_seconds},
                            {"end_seconds", segment.end_seconds},
                            {"text", std::string(segment.text)},
                            {"token_ids", std::move(ids)}});
    }
    return {{"output_tokens", result.token_ids.size()},
            {"token_ids", std::move(tokens)},
            {"segments", std::move(segments)},
            {"text", std::string(result.text)},
            {"setup_ms", result.setup_ms},
            {"prefill_ms", result.prefill_ms},
            {"decode_ms", result.decode_ms}};
}

Json text_observation(const trtmc::TextContinuationResult& result) {
    return transcript_observation({result.text(), result.token_ids(), result.setup_ms(),
                                   result.prefill_ms(), result.decode_ms(), result.segments()});
}

std::optional<std::string> language_input(const Json& request, const char* name) {
    if (!request.contains(name))
        return {};
    const auto& value = request.at(name);
    if (!value.is_string() || value.get_ref<const std::string&>().empty())
        throw std::invalid_argument(std::string(name) + " must be a non-empty string");
    return value.get<std::string>();
}

void check_streaming_input(const Json& request, bool expected) {
    if (request.contains("streaming") &&
        (!request.at("streaming").is_boolean() || request.at("streaming").get<bool>() != expected))
        throw std::invalid_argument("streaming must agree with the selected semantic Task");
}

trtmc::AudioView audio_view(const trtmc::cli::io::LoadedAudio& audio) {
    return {{audio.samples.data(), audio.samples.size()},
            static_cast<std::uint32_t>(audio.sample_rate),
            static_cast<std::uint32_t>(audio.channels)};
}

struct AudioInputSummary {
    std::size_t samples;
    std::uint32_t sample_rate, channels;
    void add_to(Json& value) const {
        value["input_samples"] = samples;
        value["input_frames"] = samples / channels;
        value["input_channels"] = channels;
        value["input_sample_rate"] = sample_rate;
        value["input_audio_seconds"] = static_cast<double>(samples) / channels / sample_rate;
    }
};
AudioInputSummary input_summary(const trtmc::cli::io::LoadedAudio& audio) {
    return {audio.samples.size(), static_cast<std::uint32_t>(audio.sample_rate),
            static_cast<std::uint32_t>(audio.channels)};
}

Json run_streaming_transcribe(const trtmc::Model& model, const Json& request,
                              const Timing& timing) {
    check_streaming_input(request, true);
    if (request.contains("target_language"))
        throw std::invalid_argument("streaming speech translation is not implemented");
    const auto language = language_input(request, "language");
    const auto task = model.task<trtmc::StreamingSpeechTranscription>();
    const auto config = sdk_config(request, task.config_fields(),
                                   {"audio_path", "language", "streaming", "chunk_ms"});
    const auto path = request.at("audio_path").get<std::string>();
    std::uint64_t chunk_ms = 160;
    if (request.contains("chunk_ms")) {
        const auto& value = request.at("chunk_ms");
        if (!value.is_number_integer() ||
            (!value.is_number_unsigned() && value.get<std::int64_t>() <= 0) ||
            value.get<std::uint64_t>() == 0)
            throw std::invalid_argument("chunk_ms must be a positive integer");
        chunk_ms = value.get<std::uint64_t>();
    }
    std::optional<trtmc::cli::io::LoadedAudio> cached;
    if (!timing.asset_loading_included)
        cached = trtmc::cli::io::read_wav_interleaved(path);
    return measure(
        timing,
        [&]() {
            std::optional<trtmc::cli::io::LoadedAudio> loaded;
            if (!cached)
                loaded = trtmc::cli::io::read_wav_interleaved(path);
            const auto& audio = cached ? *cached : *loaded;
            const auto rate = static_cast<std::uint32_t>(audio.sample_rate);
            const auto channels = static_cast<std::uint32_t>(audio.channels);
            if (chunk_ms > std::numeric_limits<std::uint64_t>::max() / rate)
                throw std::invalid_argument("chunk_ms overflows frame count");
            const auto frames = std::max<std::uint64_t>(1, rate * chunk_ms / 1000);
            if (frames > std::numeric_limits<std::size_t>::max() / channels)
                throw std::invalid_argument("chunk_ms overflows interleaved sample count");
            const auto chunk = static_cast<std::size_t>(frames * channels);
            auto stream = task.create({{rate, channels}, language}, config);
            std::optional<trtmc::SpeechTranscriptUpdate> result;
            std::optional<double> first_partial;
            const auto started = Clock::now(); // Same boundary as the existing stream benchmark.
            for (std::size_t offset = 0; offset < audio.samples.size();) {
                const auto count = std::min(chunk, audio.samples.size() - offset);
                result = stream.accept_audio({audio.samples.data() + offset, count},
                                             offset + count == audio.samples.size());
                offset += count;
                if (!first_partial && !result->transcript().text.empty())
                    first_partial = elapsed_ms(started);
            }
            if (!result || !result->is_final())
                result = stream.finish();
            if (!result->is_final())
                throw std::runtime_error("transcription stream did not finish its input epoch");
            // A fresh stream per invocation; its release is inside the measured call.
            stream.close();
            return std::make_tuple(std::move(*result), input_summary(audio),
                                   first_partial.value_or(0));
        },
        [](const auto& value) {
            const auto& update = std::get<0>(value);
            Json observation = transcript_observation(update.transcript());
            std::get<1>(value).add_to(observation);
            observation["first_partial_ms"] = std::get<2>(value);
            observation["is_final"] = update.is_final();
            observation["chunk_index"] = update.chunk_index();
            observation["accepted_samples"] = update.accepted_samples();
            return observation;
        });
}

Json run_batch_transcribe(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (!request.is_object() || request.size() != 1 || !request.contains("items") ||
        !request.at("items").is_array() || request.at("items").empty())
        throw std::invalid_argument("batch speech requires only a nonempty items array");
    const auto primary = model.info().bundle_task;
    const auto& items = request.at("items");
    struct Item {
        std::string path;
        std::optional<std::string> source, target;
        bool translation;
    };
    std::vector<Item> prepared;
    for (const auto& item : items) {
        bool translation = primary == trtmc::BatchSpeechTranslation::kTask;
        if (primary == trtmc::MixedBatchSpeechToText::kTask) {
            const auto kind = item.at("kind").get<std::string>();
            if (kind != "transcription" && kind != "translation")
                throw std::invalid_argument(
                    "mixed speech kind must be transcription or translation");
            translation = kind == "translation";
        } else if (item.contains("kind"))
            throw std::invalid_argument("kind is only accepted by mixed batch speech");
        if (!translation && item.contains("target_language"))
            throw std::invalid_argument("target_language requires a translation item");
        prepared.push_back({item.at("audio_path").get<std::string>(),
                            language_input(item, "source_language"),
                            language_input(item, "target_language"), translation});
    }
    auto read = [&]() {
        std::vector<trtmc::cli::io::LoadedAudio> loaded;
        loaded.reserve(prepared.size());
        for (const auto& item : prepared)
            loaded.push_back(trtmc::cli::io::read_wav_interleaved(item.path));
        return loaded;
    };
    auto batch = [&](const auto& task, auto prototype, auto make_input) {
        const auto fields = task.config_fields();
        std::vector<trtmc::Config> configs;
        for (const auto& item : items)
            configs.push_back(sdk_config(
                item, fields, {"audio_path", "source_language", "target_language", "kind"}));
        std::optional<std::vector<trtmc::cli::io::LoadedAudio>> cached;
        if (!timing.asset_loading_included)
            cached = read();
        return measure(
            timing,
            [&]() {
                std::optional<std::vector<trtmc::cli::io::LoadedAudio>> loaded;
                if (!cached)
                    loaded = read();
                const auto& audio = cached ? *cached : *loaded;
                auto input = prototype;
                std::vector<AudioInputSummary> summaries;
                for (std::size_t i = 0; i < audio.size(); ++i) {
                    input.items.push_back(
                        {make_input(audio_view(audio[i]), prepared[i]), configs[i]});
                    summaries.push_back(input_summary(audio[i]));
                }
                return std::make_pair(task.run(input), std::move(summaries));
            },
            [&](const auto& value) {
                if (value.first.size() != value.second.size())
                    throw std::runtime_error("batch speech result count differs from input count");
                Json outputs = Json::array();
                double seconds = 0;
                std::uint64_t tokens = 0;
                for (std::size_t i = 0; i < value.first.size(); ++i) {
                    auto item = transcript_observation(value.first[i]);
                    value.second[i].add_to(item);
                    item["kind"] = prepared[i].translation ? "translation" : "transcription";
                    seconds += item.at("input_audio_seconds").template get<double>();
                    tokens += item.at("output_tokens").template get<std::uint64_t>();
                    outputs.push_back(std::move(item));
                }
                return Json{{"transcribed_items", outputs.size()},
                            {"input_audio_seconds", seconds},
                            {"output_tokens", tokens},
                            {"items", std::move(outputs)}};
            });
    };
    if (primary == trtmc::BatchSpeechTranscription::kTask)
        return batch(model.task<trtmc::BatchSpeechTranscription>(),
                     trtmc::BatchSpeechTranscriptionRequest{}, [](auto audio, const auto& item) {
                         return trtmc::SpeechTranscriptionRequest{audio, item.source};
                     });
    if (primary == trtmc::BatchSpeechTranslation::kTask)
        return batch(model.task<trtmc::BatchSpeechTranslation>(),
                     trtmc::BatchSpeechTranslationRequest{}, [](auto audio, const auto& item) {
                         return trtmc::SpeechTranslationRequest{audio, item.target, item.source};
                     });
    return batch(model.task<trtmc::MixedBatchSpeechToText>(),
                 trtmc::MixedBatchSpeechToTextRequest{},
                 [](auto audio, const auto& item) -> trtmc::MixedSpeechTextRequest {
                     if (item.translation)
                         return trtmc::SpeechTranslationRequest{audio, item.target, item.source};
                     return trtmc::SpeechTranscriptionRequest{audio, item.source};
                 });
}

Json run_transcribe(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    if (primary == trtmc::BatchSpeechTranscription::kTask ||
        primary == trtmc::BatchSpeechTranslation::kTask ||
        primary == trtmc::MixedBatchSpeechToText::kTask)
        return run_batch_transcribe(model, request, timing);
    if (primary == trtmc::StreamingSpeechTranscription::kTask)
        return run_streaming_transcribe(model, request, timing);
    if (primary != trtmc::SpeechTranscription::kTask && primary != trtmc::SpeechTranslation::kTask)
        throw std::invalid_argument("transcribe requires a transcription or translation Task");
    check_streaming_input(request, false);
    const auto language = language_input(request, "language");
    if (primary == trtmc::SpeechTranscription::kTask && request.contains("target_language"))
        throw std::invalid_argument("target_language requires SpeechTranslation");
    const auto target = language_input(request, "target_language");
    const auto path = request.at("audio_path").get<std::string>();
    std::optional<trtmc::cli::io::LoadedAudio> cached;
    if (!timing.asset_loading_included)
        cached = trtmc::cli::io::read_wav_interleaved(path);
    auto run = [&](const auto& task, auto make_input) {
        const auto fields = task.config_fields();
        auto config = sdk_config(
            request, fields,
            {"audio_path", "language", "target_language", "streaming", "max_new_tokens"});
        // Preserve the existing worker spelling, without overwriting a nested key.
        if (request.contains("max_new_tokens")) {
            const auto limit =
                sdk_config({{"max_output_tokens", request.at("max_new_tokens")}}, fields, {});
            for (const auto& entry : limit.entries())
                config.add(entry.name, entry.value);
        }
        return measure(
            timing,
            [&]() {
                std::optional<trtmc::cli::io::LoadedAudio> loaded;
                if (!cached)
                    loaded = trtmc::cli::io::read_wav_interleaved(path);
                const auto& audio = cached ? *cached : *loaded;
                return std::make_pair(task.run(make_input(audio_view(audio)), config),
                                      input_summary(audio));
            },
            [](const auto& value) {
                const auto& result = value.first;
                Json observation = transcript_observation({result.text(), result.token_ids(),
                                                           result.setup_ms(), result.prefill_ms(),
                                                           result.decode_ms(), result.segments()});
                value.second.add_to(observation);
                return observation;
            });
    };
    if (primary == trtmc::SpeechTranscription::kTask)
        return run(model.task<trtmc::SpeechTranscription>(),
                   [&](auto audio) { return trtmc::SpeechTranscriptionRequest{audio, language}; });
    return run(model.task<trtmc::SpeechTranslation>(), [&](auto audio) {
        return trtmc::SpeechTranslationRequest{audio, target, language};
    });
}

Json audio_observation(const trtmc::AudioGenerationResult& result) {
    return {
        {"output_samples", result.samples().size()},
        {"num_samples", result.samples().size()},
        {"output_frames", result.frame_count()},
        {"channels", result.channels()},
        {"sample_rate", result.sample_rate()},
        {"output_audio_seconds", static_cast<double>(result.frame_count()) / result.sample_rate()},
        {"setup_ms", result.setup_ms()},
        {"inference_ms", result.inference_ms()}};
}

Json run_generate_audio(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    const auto prompt = request.at("prompt").get<std::string>();
    const bool streaming = primary == trtmc::StreamingTextToSpeech::kTask;
    check_streaming_input(request, streaming);
    if (primary == trtmc::TextToAudio::kTask) {
        if (request.contains("language"))
            throw std::invalid_argument("typed synthesis language requires TextToSpeech");
        const auto task = model.task<trtmc::TextToAudio>();
        const auto config = sdk_config(request, task.config_fields(), {"prompt", "streaming"});
        return measure(timing, [&]() { return task.run({prompt}, config); }, audio_observation);
    }
    const auto language = language_input(request, "language");
    if (primary == trtmc::TextToSpeech::kTask) {
        const auto task = model.task<trtmc::TextToSpeech>();
        const auto config =
            sdk_config(request, task.config_fields(), {"prompt", "language", "streaming"});
        return measure(
            timing, [&]() { return task.run({prompt, language}, config); }, audio_observation);
    }
    if (!streaming)
        throw std::invalid_argument("generate_audio requires an audio generation Task");
    const auto task = model.task<trtmc::StreamingTextToSpeech>();
    const auto config =
        sdk_config(request, task.config_fields(), {"prompt", "language", "streaming"});
    return measure(
        timing,
        [&]() {
            // The SDK already validates delivery counts and format against every
            // callback. A benchmark sink need not duplicate that accounting or copy PCM.
            const auto summary =
                task.run({prompt, language}, [](const trtmc::AudioView&) {}, config);
            if (summary.outcome != trtmc::AudioDeliveryOutcome::Complete)
                throw std::runtime_error("streaming TTS did not complete");
            return summary;
        },
        [](const trtmc::StreamingAudioSummary& result) {
            return Json{{"output_samples", result.emitted_sample_count},
                        {"num_samples", result.emitted_sample_count},
                        {"output_frames", result.emitted_frame_count},
                        {"channels", result.output.channels},
                        {"sample_rate", result.output.sample_rate},
                        {"output_audio_seconds", static_cast<double>(result.emitted_frame_count) /
                                                     result.output.sample_rate},
                        {"setup_ms", result.setup_ms},
                        {"inference_ms", result.inference_ms}};
        });
}

template <class T>
Json json_values(trtmc::Span<const T> values);

Json speech_event_observation(const trtmc::SpeechDialogueEventView& event) {
    static constexpr const char* kinds[] = {"agent_audio",
                                            "agent_text",
                                            "user_transcript",
                                            "turn_started",
                                            "turn_finished",
                                            "yielded",
                                            "cancelled",
                                            "reset",
                                            "error",
                                            "input_finished",
                                            "user_speech_started",
                                            "user_speech_stopped",
                                            "function_call",
                                            "function_call_started",
                                            "function_response_finished",
                                            "input_cleared"};
    const auto kind = static_cast<std::uint32_t>(event.kind);
    if (kind == 0 || kind > sizeof(kinds) / sizeof(*kinds))
        throw std::runtime_error("unknown speech event kind");
    Json value{
        {"kind", kinds[kind - 1]},
        {"epoch", event.epoch},
        {"sequence", event.sequence},
        {"frame_index", event.frame_index},
        {"media_start_sample", event.media_start_sample},
        {"media_end_sample", event.media_end_sample},
        {"text", std::string(event.text)},
        {"is_final", event.is_final},
        {"audio_samples", event.audio.samples.size()},
        {"audio", json_values(event.audio.samples)},
        {"channels", event.audio.channels},
        {"sample_rate", event.audio.sample_rate ? Json(*event.audio.sample_rate) : Json(nullptr)}};
    if (event.tool_call) {
        static constexpr const char* states[] = {"unknown", "complete", "incomplete", "malformed"};
        const auto state = static_cast<std::uint32_t>(event.tool_call->state);
        if (state >= sizeof(states) / sizeof(*states))
            throw std::runtime_error("unknown speech tool-call state");
        value["tool_call"] = {{"call_id", event.tool_call->call_id},
                              {"name", event.tool_call->name},
                              {"arguments_json", event.tool_call->arguments_json},
                              {"state", states[state]}};
    }
    return value;
}

Json run_speech_dialogue(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    const bool tools_enabled = primary == trtmc::ToolSpeechDialogue::kTask;
    if (!tools_enabled && primary != trtmc::DuplexSpeechDialogue::kTask &&
        primary != trtmc::OfflineSpeechDialogue::kTask)
        throw std::invalid_argument("speech_dialogue requires a typed dialogue Task");
    const auto path = request.at("audio_path").get<std::string>();
    std::optional<std::string> system_prompt;
    if (request.contains("system_prompt"))
        system_prompt = request.at("system_prompt").get<std::string>();
    std::int64_t timeout_ms = 30000;
    if (request.contains("timeout_ms")) {
        const auto& value = request.at("timeout_ms");
        if (!value.is_number_integer() || value < 0 ||
            value > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("timeout_ms must be a nonnegative int32");
        timeout_ms = value.get<std::int64_t>();
    }
    std::optional<std::uint64_t> chunk_frames;
    if (request.contains("chunk_frames")) {
        const auto& value = request.at("chunk_frames");
        if (!value.is_number_integer() || value <= 0)
            throw std::invalid_argument("chunk_frames must be a positive integer");
        chunk_frames = value.get<std::uint64_t>();
    }
    struct Reply {
        std::string name, content;
        bool is_error;
    };
    std::vector<Reply> replies;
    trtmc::ToolSpeechDialogueRequest tool_request;
    if (tools_enabled) {
        const auto& definitions = request.at("tools");
        const auto& supplied_replies = request.at("tool_replies");
        if (!definitions.is_array() || definitions.empty() || !supplied_replies.is_array())
            throw std::invalid_argument(
                "tool dialogue requires nonempty tools and an explicit tool_replies array");
        for (const auto& tool : definitions)
            tool_request.tools.push_back({tool.at("name").get<std::string>(),
                                          tool.value("description", std::string{}),
                                          tool.at("parameters_schema_json").get<std::string>()});
        for (const auto& reply : supplied_replies)
            replies.push_back({reply.at("name").get<std::string>(),
                               reply.at("content_text").get<std::string>(),
                               reply.value("is_error", false)});
        if (request.contains("acknowledgements")) {
            if (!request.at("acknowledgements").is_array())
                throw std::invalid_argument("acknowledgements must be an array");
            for (const auto& ack : request.at("acknowledgements"))
                tool_request.acknowledgements.push_back(
                    {ack.at("tool_name").get<std::string>(),
                     ack.at("messages").get<std::vector<std::string>>()});
        }
        if (request.contains("default_acknowledgements"))
            tool_request.default_acknowledgements =
                request.at("default_acknowledgements").get<std::vector<std::string>>();
    } else {
        for (const auto* name :
             {"tools", "tool_replies", "acknowledgements", "default_acknowledgements"})
            if (request.contains(name))
                throw std::invalid_argument("preset tool interaction requires ToolSpeechDialogue");
    }
    struct Result {
        trtmc::SpeechSessionInfo info;
        AudioInputSummary input;
        std::vector<trtmc::SpeechEvents> batches;
        std::size_t replies{0}, input_chunks{0}, append_attempts{0};
    };
    std::optional<trtmc::cli::io::LoadedAudio> cached;
    if (!timing.asset_loading_included)
        cached = trtmc::cli::io::read_wav_interleaved(path);
    auto perform = [&](auto create) {
        return measure(
            timing,
            [&]() {
                std::optional<trtmc::cli::io::LoadedAudio> loaded;
                if (!cached)
                    loaded = trtmc::cli::io::read_wav_interleaved(path);
                const auto& audio = cached ? *cached : *loaded;
                if (audio.samples.empty())
                    throw std::invalid_argument("speech dialogue requires nonempty input audio");
                const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
                const trtmc::SpeechDialogueRequest input{
                    {static_cast<std::uint32_t>(audio.sample_rate),
                     static_cast<std::uint32_t>(audio.channels)},
                    system_prompt};
                auto session = create(input);
                Result result{session.info(), input_summary(audio), {}};
                if (result.info.input.sample_rate !=
                        static_cast<std::uint32_t>(audio.sample_rate) ||
                    result.info.input.channels != static_cast<std::uint32_t>(audio.channels))
                    throw std::runtime_error(
                        "speech session input format differs from supplied audio");
                bool input_finished = false, ended = false;
                auto consume = [&](trtmc::SpeechReadResult read) {
                    bool epoch_ended = read.status == trtmc::SpeechPollStatus::EpochEnd;
                    if (read.events) {
                        if (read.events->state() == trtmc::SpeechReadState::Failed)
                            throw std::runtime_error("speech dialogue failed");
                        epoch_ended = read.events->state() == trtmc::SpeechReadState::EpochEnded;
                        for (std::size_t i = 0; i < read.events->size(); ++i) {
                            const auto event = read.events->at(i);
                            if (event.kind == trtmc::SpeechEventKind::Cancelled)
                                throw std::runtime_error(
                                    "speech dialogue cancelled before normal completion");
                            if (event.tool_call) {
                                if (!tools_enabled || result.replies == replies.size())
                                    throw std::invalid_argument(
                                        "no preset reply for the emitted tool call");
                                const auto& reply = replies[result.replies];
                                if (reply.name != event.tool_call->name)
                                    throw std::invalid_argument(
                                        "preset reply does not match the emitted tool name");
                                // Reply text is input data. No tool, command or network action is
                                // executed.
                                session.tools().submit_tool_result(
                                    event.epoch,
                                    {event.tool_call->call_id, reply.content, reply.is_error});
                                ++result.replies;
                            }
                        }
                        result.batches.push_back(std::move(*read.events));
                    }
                    if (epoch_ended) {
                        if (!input_finished)
                            throw std::runtime_error(
                                "speech epoch ended before input was finished");
                        ended = true;
                    }
                };
                auto read = [&]() {
                    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                               deadline - Clock::now())
                                               .count();
                    consume(session.wait_events(
                        std::max<std::int64_t>(0, std::min<std::int64_t>(remaining, 500))));
                    if (!ended && Clock::now() >= deadline)
                        throw std::runtime_error("speech dialogue timed out; input chunking and "
                                                 "preset tool replies must match the workload");
                };
                const auto frames = audio.samples.size() / static_cast<std::size_t>(audio.channels);
                const auto chunk = static_cast<std::size_t>(std::min<std::uint64_t>(
                                       chunk_frames.value_or(frames), frames)) *
                                   audio.channels;
                for (std::size_t offset = 0; offset < audio.samples.size(); offset += chunk) {
                    const trtmc::Span<const float> samples{
                        audio.samples.data() + offset,
                        std::min(chunk, audio.samples.size() - offset)};
                    ++result.append_attempts;
                    while (!session.append_audio(samples)) {
                        read(); // Backpressure accepts nothing: retry the same unchanged chunk.
                        ++result.append_attempts;
                    }
                    ++result.input_chunks;
                }
                if (tools_enabled) {
                    session.realtime().commit_input_turn();
                    consume(session.take_events());
                    while (result.replies < replies.size())
                        read();
                }
                session.finish_input();
                input_finished = true;
                while (!ended)
                    read();
                session.close(); // Create, execution, completion and release are all measured.
                return result;
            },
            [&](const Result& result) {
                Json events = Json::array(), states = Json::array();
                double output_seconds = 0;
                for (const auto& batch : result.batches) {
                    states.push_back(static_cast<std::uint32_t>(batch.state()));
                    for (std::size_t i = 0; i < batch.size(); ++i) {
                        const auto event = batch.at(i);
                        if (event.kind == trtmc::SpeechEventKind::AgentAudio &&
                            !event.audio.samples.empty()) {
                            if (!event.audio.sample_rate || !*event.audio.sample_rate ||
                                !event.audio.channels)
                                throw std::runtime_error(
                                    "agent audio omitted its sample rate or channels");
                            output_seconds += static_cast<double>(event.audio.samples.size()) /
                                              event.audio.channels / *event.audio.sample_rate;
                        }
                        events.push_back(speech_event_observation(event));
                    }
                }
                Json output{
                    {"events", std::move(events)},
                    {"read_states", std::move(states)},
                    {"output_audio_seconds", output_seconds},
                    {"submitted_tool_replies", result.replies},
                    {"input_chunks", result.input_chunks},
                    {"append_attempts", result.append_attempts},
                    {"system_prompt",
                     result.info.system_prompt ? Json(*result.info.system_prompt) : Json(nullptr)},
                    {"source_language", result.info.source_language
                                            ? Json(*result.info.source_language)
                                            : Json(nullptr)},
                    {"output_format", result.info.output
                                          ? Json{{"sample_rate", result.info.output->sample_rate},
                                                 {"channels", result.info.output->channels}}
                                          : Json(nullptr)},
                    {"lifecycle_scope",
                     tools_enabled ? "fresh_create_append_commit_preset_replies_finish_drain_close"
                                   : "fresh_create_append_finish_drain_close"}};
                result.input.add_to(output);
                return output;
            });
    };
    auto config_for = [&](const auto& task) {
        return sdk_config(request, task.config_fields(),
                          {"audio_path", "system_prompt", "chunk_frames", "timeout_ms", "tools",
                           "tool_replies", "acknowledgements", "default_acknowledgements"});
    };
    if (tools_enabled) {
        const auto task = model.task<trtmc::ToolSpeechDialogue>();
        const auto config = config_for(task);
        return perform([&](const auto& input) {
            tool_request.dialogue = input;
            return task.create(tool_request, config);
        });
    }
    if (primary == trtmc::OfflineSpeechDialogue::kTask) {
        const auto task = model.task<trtmc::OfflineSpeechDialogue>();
        const auto config = config_for(task);
        return perform([&](const auto& input) { return task.create(input, config); });
    }
    const auto task = model.task<trtmc::DuplexSpeechDialogue>();
    const auto config = config_for(task);
    return perform([&](const auto& input) { return task.create(input, config); });
}

Json run_speak(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (model.info().bundle_task != trtmc::SpeechToSpeechResponse::kTask)
        throw std::invalid_argument("speak requires SpeechToSpeechResponse");
    const auto task = model.task<trtmc::SpeechToSpeechResponse>();
    const auto config = sdk_config(request, task.config_fields(), {"audio_path"});
    const auto path = request.at("audio_path").get<std::string>();
    std::optional<trtmc::cli::io::LoadedAudio> cached;
    if (!timing.asset_loading_included)
        cached = trtmc::cli::io::read_wav_interleaved(path);
    return measure(
        timing,
        [&]() {
            std::optional<trtmc::cli::io::LoadedAudio> loaded;
            if (!cached)
                loaded = trtmc::cli::io::read_wav_interleaved(path);
            const auto& audio = cached ? *cached : *loaded;
            return std::make_pair(task.run({audio_view(audio)}, config), input_summary(audio));
        },
        [](const auto& value) {
            auto observation = audio_observation(value.first);
            value.second.add_to(observation);
            return observation;
        });
}

trtmc::ImageInput sdk_image_view(const Image& image) {
    return {{image.pixels.data(), image.pixels.size()},
            static_cast<std::uint32_t>(image.height),
            static_cast<std::uint32_t>(image.width)};
}

template <class T>
Json json_values(const T* data, std::size_t count) {
    auto result = Json::array();
    for (std::size_t i = 0; i < count; ++i) {
        if constexpr (std::is_floating_point_v<T>)
            if (!std::isfinite(data[i]))
                throw std::runtime_error("benchmark output contains a non-finite value");
        result.push_back(data[i]);
    }
    return result;
}
template <class T>
Json json_values(trtmc::Span<const T> values) {
    return json_values(values.data(), values.size());
}
Json json_strings(trtmc_strings_view strings) {
    auto output = Json::array();
    for (std::uint64_t i = 0; i < strings.size; ++i)
        output.push_back(std::string(trtmc::detail::string_view(strings.data[i])));
    return output;
}
const char* score_kind(std::uint32_t kind) {
    switch (kind) {
    case TRTMC_SCORE_LOGIT:
        return "logit";
    case TRTMC_SCORE_PROBABILITY:
        return "probability";
    case TRTMC_SCORE_UNBOUNDED:
        return "unbounded";
    default:
        throw std::runtime_error("unknown score kind");
    }
}
void check_batch_size(const Json& request, std::size_t count) {
    if (!request.contains("batch_size"))
        return;
    const auto& value = request.at("batch_size");
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0) ||
        value.get<std::uint64_t>() != count)
        throw std::invalid_argument("batch_size must equal the actual request count");
}

Json run_classify(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (model.info().bundle_task != trtmc::ImageToClassScores::kTask)
        throw std::invalid_argument("classify requires ImageToClassScores");
    check_batch_size(request, 1);
    const auto task = model.task<trtmc::ImageToClassScores>();
    const auto config = sdk_config(request, task.config_fields(), {"image_path", "batch_size"});
    const auto path = request.at("image_path").get<std::string>();
    std::optional<Image> cached;
    if (!timing.asset_loading_included)
        cached = read_image(path);
    return measure(
        timing,
        [&]() {
            std::optional<Image> loaded;
            if (!cached)
                loaded = read_image(path);
            return task.run({sdk_image_view(cached ? *cached : *loaded)}, config);
        },
        [](const trtmc::LabelScoresResult& result) {
            const auto scores = result.scores();
            auto labels = Json::array();
            for (auto label : result.labels())
                labels.push_back(std::string(label));
            Json output{{"classified_images", 1},
                        {"scores", json_values(scores)},
                        {"score_kind", score_kind(result.kind())},
                        {"labels", std::move(labels)},
                        {"vocabulary_id", std::string(result.vocabulary_id())},
                        {"top_class", -1},
                        {"top_score", nullptr}};
            if (!scores.empty()) {
                const auto best = std::max_element(scores.begin(), scores.end());
                output["top_class"] = best - scores.begin();
                output["top_score"] = *best;
            }
            return output;
        });
}

template <class Result>
Json image_token_observation(const Result& result) {
    const auto matrix = result.features();
    auto tokens = Json::array();
    for (const auto& token : result.tokens()) {
        Json item;
        if (token.role == TRTMC_IMAGE_TOKEN_CLASS)
            item["role"] = "class";
        else if (token.role == TRTMC_IMAGE_TOKEN_REGISTER)
            item["role"] = "register";
        else if (token.role == TRTMC_IMAGE_TOKEN_PATCH)
            item = {
                {"role", "patch"},
                {"grid_row", token.grid_row},
                {"grid_column", token.grid_column},
                {"source_normalized_box", {token.x_min, token.y_min, token.x_max, token.y_max}}};
        else
            throw std::runtime_error("unknown image token role");
        tokens.push_back(std::move(item));
    }
    return {{"processed_images", 1},
            {"feature_elements", matrix.values.size()},
            {"last_hidden_state", json_values(matrix.values)},
            {"last_hidden_state_shape", {1, matrix.rows, matrix.columns}},
            {"axes", {"batch", "token", "feature"}},
            {"tokens", std::move(tokens)},
            {"grid_shape", {result.grid_rows(), result.grid_columns()}}};
}
Json image_feature_observation(const trtmc::ImageTokenFeaturesResult& result) {
    return image_token_observation(result);
}
Json image_feature_observation(const trtmc::ImageTokenAndPooledFeaturesResult& result) {
    auto output = image_token_observation(result);
    output["feature_elements"] = result.features().values.size() + result.pooled_values().size();
    output["pooler_output"] = json_values(result.pooled_values());
    output["pooler_output_shape"] = {1, result.pooled_values().size()};
    output["pooling"] = std::string(result.pooling());
    output["normalization"] = std::string(result.normalization());
    return output;
}
Json image_feature_observation(const trtmc::PooledFeaturesResult& result) {
    return {{"processed_images", 1},
            {"feature_elements", result.values().size()},
            {"pooler_output", json_values(result.values())},
            {"pooler_output_shape", {1, result.values().size()}},
            {"pooling", std::string(result.pooling())},
            {"normalization", std::string(result.normalization())}};
}
Json image_feature_observation(const trtmc::SpatialFeaturesResult& result) {
    auto maps = Json::array();
    std::uint64_t elements = 0;
    for (const auto& map : result.maps()) {
        elements += map.count;
        maps.push_back({{"name", std::string(trtmc::detail::string_view(map.name))},
                        {"values", json_values(map.values, map.count)},
                        {"shape", {map.channels, map.height, map.width}},
                        {"axes", {"channel", "y", "x"}},
                        {"stride_y", map.stride_y},
                        {"stride_x", map.stride_x}});
    }
    const auto transform = result.source_to_processed();
    return {{"processed_images", 1},
            {"feature_elements", elements},
            {"maps", std::move(maps)},
            {"processed_image_height", result.processed_image_height()},
            {"processed_image_width", result.processed_image_width()},
            {"source_to_processed",
             {{"scale_x", transform.scale_x},
              {"scale_y", transform.scale_y},
              {"offset_x", transform.offset_x},
              {"offset_y", transform.offset_y},
              {"coordinates", "image_edges"}}}};
}

Json run_extract_features(const trtmc::Model& model, const Json& request, const Timing& timing) {
    check_batch_size(request, 1);
    const auto path = request.at("image_path").get<std::string>();
    std::optional<Image> cached;
    if (!timing.asset_loading_included)
        cached = read_image(path);
    auto run = [&](const auto& task) {
        const auto config = sdk_config(request, task.config_fields(), {"image_path", "batch_size"});
        return measure(
            timing,
            [&]() {
                std::optional<Image> loaded;
                if (!cached)
                    loaded = read_image(path);
                return task.run({sdk_image_view(cached ? *cached : *loaded)}, config);
            },
            [](const auto& result) { return image_feature_observation(result); });
    };
    const auto primary = model.info().bundle_task;
    if (primary == trtmc::ImageToTokenAndPooledFeatures::kTask)
        return run(model.task<trtmc::ImageToTokenAndPooledFeatures>());
    if (primary == trtmc::ImageToTokenFeatures::kTask)
        return run(model.task<trtmc::ImageToTokenFeatures>());
    if (primary == trtmc::ImageToPooledFeatures::kTask)
        return run(model.task<trtmc::ImageToPooledFeatures>());
    if (primary == trtmc::ImageToSpatialFeatures::kTask)
        return run(model.task<trtmc::ImageToSpatialFeatures>());
    throw std::invalid_argument("extract_features requires an image feature Task");
}

Json run_encode(const trtmc::Model& model, const Json& request, const Timing& timing) {
    check_batch_size(request, 1);
    const auto prompt = request.at("prompt").get<std::string>();
    const auto primary = model.info().bundle_task;
    if (primary == trtmc::TextToPooledFeatures::kTask) {
        const auto task = model.task<trtmc::TextToPooledFeatures>();
        const auto config = sdk_config(request, task.config_fields(), {"prompt", "batch_size"});
        return measure(
            timing, [&]() { return task.run({prompt}, config); },
            [](const trtmc::PooledFeaturesResult& result) {
                return Json{{"embedding_vectors", 1},
                            {"embedding_elements", result.values().size()},
                            {"dim", result.values().size()},
                            {"values", json_values(result.values())},
                            {"feature_kind", "pooled"},
                            {"pooling", std::string(result.pooling())},
                            {"normalization", std::string(result.normalization())}};
            });
    }
    if (primary != trtmc::TextToTokenFeatures::kTask)
        throw std::invalid_argument("encode requires a pooled or token feature Task");
    const auto task = model.task<trtmc::TextToTokenFeatures>();
    const auto config = sdk_config(request, task.config_fields(), {"prompt", "batch_size"});
    return measure(
        timing, [&]() { return task.run({prompt}, config); },
        [](const trtmc::TokenFeaturesResult& result) {
            const auto matrix = result.features();
            auto tokens = Json::array();
            for (const auto& token : result.tokens()) {
                Json item{{"token_id", token.token_id},
                          {"input_index", token.input_index},
                          {"token_index", token.token_index}};
                item["byte_offsets"] = token.has_byte_offsets
                                           ? Json::array({token.byte_begin, token.byte_end})
                                           : Json(nullptr);
                tokens.push_back(std::move(item));
            }
            return Json{
                {"embedding_vectors", 1},       {"embedding_elements", matrix.values.size()},
                {"dim", matrix.columns},        {"shape", {matrix.rows, matrix.columns}},
                {"axes", {"token", "feature"}}, {"values", json_values(matrix.values)},
                {"tokens", std::move(tokens)},  {"feature_kind", "token"}};
        });
}

Json run_embed(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (model.info().bundle_task != trtmc::TextToEmbedding::kTask)
        throw std::invalid_argument("embed requires TextToEmbedding");
    check_batch_size(request, 1);
    const auto task = model.task<trtmc::TextToEmbedding>();
    const auto config = sdk_config(request, task.config_fields(), {"prompt", "role", "batch_size"});
    const auto prompt = request.at("prompt").get<std::string>();
    auto role = trtmc::EmbeddingRole::Default;
    if (request.contains("role")) {
        const auto name = request.at("role").get<std::string>();
        if (name == "query")
            role = trtmc::EmbeddingRole::Query;
        else if (name == "document")
            role = trtmc::EmbeddingRole::Document;
        else if (name != "default")
            throw std::invalid_argument("invalid embedding role");
    }
    return measure(
        timing, [&]() { return task.run({prompt, role}, config); },
        [](const trtmc::SemanticEmbeddingResult& result) {
            return Json{{"embedding_vectors", 1},
                        {"embedding_elements", result.values().size()},
                        {"dim", result.values().size()},
                        {"values", json_values(result.values())},
                        {"embedding_space", std::string(result.embedding_space())},
                        {"pooling", std::string(result.pooling())},
                        {"normalization", std::string(result.normalization())}};
        });
}

Json run_rerank(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (model.info().bundle_task != trtmc::TextQueryDocumentsToRelevance::kTask)
        throw std::invalid_argument("document-list rerank requires TextQueryDocumentsToRelevance");
    const auto task = model.task<trtmc::TextQueryDocumentsToRelevance>();
    const auto config = sdk_config(request, task.config_fields(), {"query", "documents"});
    const auto query = request.at("query").get<std::string>();
    const auto documents = request.at("documents").get<std::vector<std::string>>();
    return measure(
        timing, [&]() { return task.run({query, documents}, config); },
        [&](const trtmc::DocumentRelevanceResult& result) {
            return Json{{"documents", documents.size()},
                        {"scores", json_values(result.scores())},
                        {"score_kind", score_kind(result.kind())},
                        {"order", "input_documents"}};
        });
}

Json action_sequence_observation(const trtmc_action_sequence_view_v1& view) {
    const auto& schema = view.schema;
    Json spans = Json::array();
    for (std::uint64_t i = 0; i < view.frame_span_count; ++i)
        spans.push_back({{"begin", view.frame_spans[i].begin}, {"end", view.frame_spans[i].end}});
    return {
        {"action_steps", view.values.rows},
        {"action_dim", view.values.columns},
        {"action_values", view.values.count},
        {"actions", json_values(view.values.data, view.values.count)},
        {"axes", {"step", "action_component"}},
        {"schema",
         {{"domain", std::string(trtmc::detail::string_view(schema.domain))},
          {"component_names", json_strings(schema.component_names)},
          {"units", json_strings(schema.units)},
          {"coordinate_frame", std::string(trtmc::detail::string_view(schema.coordinate_frame))},
          {"normalization", std::string(trtmc::detail::string_view(schema.normalization))}}},
        {"timestamps_seconds",
         json_values(view.timestamps_seconds.data, view.timestamps_seconds.size)},
        {"frame_spans", std::move(spans)}};
}

Json run_control_queue(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::ImageStateActionQueue>();
    const auto fields = task.config_fields();
    const auto config = sdk_config(request, fields, {"observations"});
    const auto& observations = request.at("observations");
    if (!observations.is_array() || observations.empty())
        throw std::invalid_argument("action queue requires nonempty observations");
    std::vector<trtmc::Config> step_config;
    for (const auto& item : observations)
        step_config.push_back(sdk_config(item, fields, {"image_path", "state_path"}));
    using Input = std::pair<Image, std::vector<float>>;
    auto read = [&]() {
        std::vector<Input> inputs;
        inputs.reserve(observations.size());
        for (const auto& item : observations)
            inputs.emplace_back(read_image(item.at("image_path").get<std::string>()),
                                read_float32(item.at("state_path").get<std::string>()));
        return inputs;
    };
    auto views = [](const std::vector<Input>& inputs) {
        std::vector<trtmc::ImageStateObservation> result;
        for (const auto& input : inputs)
            result.push_back(
                {sdk_image_view(input.first), {input.second.data(), input.second.size()}});
        return result;
    };
    std::optional<std::vector<Input>> cached;
    std::vector<trtmc::ImageStateObservation> cached_views;
    if (!timing.asset_loading_included) {
        cached = read();
        cached_views = views(*cached);
    }
    return measure(
        timing,
        [&]() {
            std::optional<std::vector<Input>> loaded;
            std::vector<trtmc::ImageStateObservation> loaded_views;
            if (!cached) {
                loaded = read();
                loaded_views = views(*loaded);
            }
            const auto& input = cached ? cached_views : loaded_views;
            // One observation sequence is one fresh session lifecycle, including release.
            auto session = task.create(config);
            std::vector<trtmc::ActionStepResult> results;
            results.reserve(input.size());
            for (std::size_t i = 0; i < input.size(); ++i)
                results.push_back(session.act(input[i], step_config[i]));
            session.close();
            return results;
        },
        [](const auto& results) {
            Json steps = Json::array();
            for (const auto& result : results) {
                auto step = action_sequence_observation(result.view().action);
                step["within_training_bounds"] = result.within_training_bounds();
                step["started_new_chunk"] = result.started_new_chunk();
                step["inference_ms"] = result.inference_ms();
                steps.push_back(std::move(step));
            }
            return Json{{"action_steps", steps.size()},
                        {"steps", std::move(steps)},
                        {"lifecycle_scope", "fresh_create_act_all_close"}};
        });
}

Json run_control(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (model.info().bundle_task != trtmc::ImageStateToActionChunk::kTask)
        throw std::invalid_argument("control requires a stateless ImageStateToActionChunk Task");
    const auto task = model.task<trtmc::ImageStateToActionChunk>();
    const auto config = sdk_config(request, task.config_fields(), {"image_path", "state_path"});
    const auto image_path = request.at("image_path").get<std::string>();
    const auto state_path = request.at("state_path").get<std::string>();
    auto read = [&]() { return std::make_pair(read_image(image_path), read_float32(state_path)); };
    std::optional<std::pair<Image, std::vector<float>>> cached;
    if (!timing.asset_loading_included)
        cached = read();
    return measure(
        timing,
        [&]() {
            std::optional<std::pair<Image, std::vector<float>>> loaded;
            if (!cached)
                loaded = read();
            const auto& inputs = cached ? *cached : *loaded;
            return task.run(
                {{sdk_image_view(inputs.first), {inputs.second.data(), inputs.second.size()}}},
                config);
        },
        [](const trtmc::ImageStateActionChunkResult& result) {
            auto output = action_sequence_observation(result.view().actions);
            output["within_training_bounds"] = result.within_training_bounds();
            output["inference_ms"] = result.inference_ms();
            return output;
        });
}

float input_float(const Json& request, const char* name, float default_value) {
    if (!request.contains(name))
        return default_value;
    const auto& value = request.at(name);
    if (!value.is_number())
        throw std::invalid_argument(std::string(name) + " must be a number");
    const auto number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
        throw std::invalid_argument(std::string(name) + " must be a finite float32 value");
    return static_cast<float>(number);
}

Json masks_observation(const trtmc_masks_view_v1& view) {
    const char* kind = view.kind == TRTMC_MASK_LOGITS        ? "logits"
                       : view.kind == TRTMC_MASK_PROBABILITY ? "probability"
                                                             : "binary";
    auto boxes = Json::array();
    for (std::uint64_t i = 0; i < view.box_count; ++i)
        boxes.push_back(
            {view.boxes[i].x_min, view.boxes[i].y_min, view.boxes[i].x_max, view.boxes[i].y_max});
    auto proposals = Json::array();
    for (std::uint64_t i = 0; i < view.proposal_count; ++i) {
        const auto& item = view.proposals[i];
        auto points = Json::array();
        for (std::uint64_t j = 0; j < item.seed_point_count; ++j)
            points.push_back({item.seed_points[j].x, item.seed_points[j].y});
        proposals.push_back(
            {{"area", item.area},
             {"crop_box",
              {item.crop_box.x_min, item.crop_box.y_min, item.crop_box.x_max, item.crop_box.y_max}},
             {"seed_points", std::move(points)}});
    }
    return {{"segmented_images", 1},
            {"generated_masks", view.mask_count},
            {"num_masks", view.mask_count},
            {"mask_pixels", view.value_count},
            {"height", view.height},
            {"width", view.width},
            {"mask_kind", kind},
            {"masks", json_values(view.masks, view.value_count)},
            {"iou_scores", json_values(view.predicted_iou, view.iou_count)},
            {"confidence", json_values(view.confidence, view.confidence_count)},
            {"stability_scores", json_values(view.stability, view.stability_count)},
            {"boxes", std::move(boxes)},
            {"box_coordinates", "original_image_pixels_xyxy"},
            {"object_ids", json_values(view.object_ids.data, view.object_ids.size)},
            {"low_res_logits", json_values(view.low_res_logits, view.low_res_count)},
            {"low_res_height", view.low_res_height},
            {"low_res_width", view.low_res_width},
            {"proposals", std::move(proposals)}};
}

Json run_segment(const trtmc::Model& model, const Json& request, const Timing& timing,
                 bool prompted) {
    check_batch_size(request, 1);
    const auto primary = model.info().bundle_task;
    const auto path = request.at("image_path").get<std::string>();
    std::optional<Image> cached;
    if (!timing.asset_loading_included)
        cached = read_image(path);
    auto with_image = [&](const auto& task, const trtmc::Config& config, auto make_input,
                          auto observe) {
        return measure(
            timing,
            [&]() {
                std::optional<Image> loaded;
                if (!cached)
                    loaded = read_image(path);
                return task.run(make_input(cached ? *cached : *loaded), config);
            },
            observe);
    };
    if (!prompted && primary == trtmc::ImageToSemanticSegmentation::kTask) {
        const auto task = model.task<trtmc::ImageToSemanticSegmentation>();
        const auto config = sdk_config(request, task.config_fields(), {"image_path", "batch_size"});
        return with_image(
            task, config,
            [](const auto& image) {
                return trtmc::ImageToSemanticSegmentationRequest{sdk_image_view(image)};
            },
            [](const trtmc::SemanticSegmentationResult& result) {
                const auto& view = result.view();
                return Json{
                    {"segmented_images", 1},
                    {"num_masks", 1},
                    {"mask_pixels", view.pixel_count},
                    {"height", view.height},
                    {"width", view.width},
                    {"mask", json_values(view.labels, view.pixel_count)},
                    {"class_ids", json_values(view.class_ids.data, view.class_ids.size)},
                    {"class_names", json_strings(view.class_names)},
                    {"vocabulary_id", std::string(trtmc::detail::string_view(view.vocabulary_id))},
                    {"ignore_label",
                     view.has_ignore_label ? Json(view.ignore_label) : Json(nullptr)},
                    {"background_label",
                     view.has_background_label ? Json(view.background_label) : Json(nullptr)},
                    {"class_scores", json_values(view.class_scores, view.score_count)},
                    {"score_height", view.score_height},
                    {"score_width", view.score_width},
                    {"score_kind", score_kind(view.score_kind)},
                    {"class_score_axes", {"class", "height", "width"}}};
            });
    }
    if (primary == trtmc::ImagePointsToMasks::kTask) {
        if (request.contains("prompt"))
            throw std::invalid_argument("text prompts require ImageTextToInstanceMasks");
        if (!prompted && (request.contains("point_x") || request.contains("point_y") ||
                          request.contains("is_foreground")))
            throw std::invalid_argument("explicit point controls require segment_prompted");
        const float x = input_float(request, "point_x", 0.5F);
        const float y = input_float(request, "point_y", 0.5F);
        if (request.contains("is_foreground") && !request.at("is_foreground").is_boolean())
            throw std::invalid_argument("is_foreground must be boolean");
        const bool foreground = request.value("is_foreground", true);
        const auto task = model.task<trtmc::ImagePointsToMasks>();
        const auto config =
            sdk_config(request, task.config_fields(),
                       {"image_path", "batch_size", "point_x", "point_y", "is_foreground"});
        // This point lives through each synchronous call, including its borrowed Span.
        trtmc::PointPrompt point{};
        return with_image(
            task, config,
            [&](const auto& image) {
                point = {{std::floor(x * image.width), std::floor(y * image.height)}, foreground};
                return trtmc::ImagePointsToMasksRequest{sdk_image_view(image), {&point, 1}};
            },
            [&](const trtmc::MasksResult& result) {
                const auto& view = result.view();
                auto output = masks_observation(view);
                output["point"] = {{"x", point.point.x},
                                   {"y", point.point.y},
                                   {"foreground", foreground},
                                   {"coordinates", "original_image_pixels"}};
                if (!prompted) {
                    const auto count =
                        view.mask_count ? static_cast<std::uint64_t>(view.height) * view.width : 0;
                    auto mask = Json::array();
                    for (std::uint64_t i = 0; i < count; ++i)
                        mask.push_back(view.masks[i] > 0 ? 1 : 0);
                    output["returned_mask_count"] = view.mask_count;
                    output["mask"] = std::move(mask);
                    output["num_masks"] = view.mask_count ? 1 : 0;
                    output["mask_pixels"] = count;
                    output["selected_mask_index"] = view.mask_count ? Json(0) : Json(nullptr);
                    output["selected_mask_kind"] = "binary";
                }
                return output;
            });
    }
    if (!prompted || primary != trtmc::ImageTextToInstanceMasks::kTask)
        throw std::invalid_argument("segmentation inputs do not match the selected Task");
    for (const auto* key : {"point_x", "point_y", "is_foreground"})
        if (request.contains(key))
            throw std::invalid_argument("text-instance masks do not accept point controls");
    const auto prompt = request.at("prompt").get<std::string>();
    const auto task = model.task<trtmc::ImageTextToInstanceMasks>();
    const auto config =
        sdk_config(request, task.config_fields(), {"image_path", "batch_size", "prompt"});
    return with_image(
        task, config,
        [&](const auto& image) {
            return trtmc::ImageTextToInstanceMasksRequest{sdk_image_view(image), prompt};
        },
        [](const trtmc::MasksResult& result) { return masks_observation(result.view()); });
}

Json generated_image_observation(const trtmc::ImageResultView& image) {
    if (image.is_worker())
        throw std::runtime_error("worker-only completion has no produced image to benchmark");
    return {{"height", image.height},     {"width", image.width},
            {"channels", image.channels}, {"num_frames", 1},
            {"media_type", "image"},      {"output_elements", image.pixels.size()}};
}
Json generated_image_observation(const trtmc::ImageGenerationResult& image) {
    auto output = generated_image_observation(
        {image.pixels(), image.height(), image.width(), image.channels()});
    output["generated_images"] = 1;
    output["generated_frames"] = 1;
    output["batch_size"] = 1;
    return output;
}
Json generated_video_observation(const trtmc::VideoGenerationResult& video) {
    const auto frames = video.frames();
    if (frames.empty())
        throw std::runtime_error("worker-only completion has no produced video to benchmark");
    auto shapes = Json::array();
    std::uint64_t elements = 0;
    for (const auto& frame : frames) {
        elements += frame.pixel_count;
        shapes.push_back({frame.height, frame.width, frame.channels});
    }
    return {{"generated_images", 1},
            {"batch_size", 1},
            {"generated_frames", frames.size()},
            {"num_frames", frames.size()},
            {"output_elements", elements},
            {"media_type", "video"},
            {"height", frames[0].height},
            {"width", frames[0].width},
            {"channels", frames[0].channels},
            {"frame_shapes", std::move(shapes)},
            {"timestamps_seconds", json_values(video.timestamps_seconds())},
            {"conditioned_prefix_frames", video.conditioned_prefix_frames()},
            {"setup_ms", video.setup_ms()},
            {"inference_ms", video.inference_ms()}};
}

Json run_generate_image(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    const bool video =
        primary == trtmc::TextToVideo::kTask || primary == trtmc::ImageTextActionToVideo::kTask;
    if (request.contains("media_type") &&
        (!request.at("media_type").is_string() ||
         request.at("media_type").get<std::string>() != (video ? "video" : "image")))
        throw std::invalid_argument("media_type must agree with the selected Task");
    if (primary == trtmc::BatchTextToImage::kTask) {
        const auto prompts = request.at("prompt").get<std::vector<std::string>>();
        if (prompts.empty())
            throw std::invalid_argument("image benchmark requires a nonempty prompt batch");
        check_batch_size(request, prompts.size());
        if (request.contains("initial_latents_path"))
            throw std::invalid_argument("a scalar replay path does not define batch replay inputs");
        const auto task = model.task<trtmc::BatchTextToImage>();
        const auto fields = task.config_fields();
        const auto shared = sdk_config(
            request, fields, {"prompt", "seeds", "item_configs", "batch_size", "media_type"});
        const auto seeds = request.value("seeds", Json::array());
        if (!seeds.is_array() || (request.contains("seeds") && seeds.size() != prompts.size()))
            throw std::invalid_argument("seeds must contain one integer per prompt");
        const auto configs = request.value("item_configs", Json::array());
        if (!configs.is_array() ||
            (request.contains("item_configs") && configs.size() != prompts.size()))
            throw std::invalid_argument("item_configs must contain one object per prompt");
        std::vector<trtmc::BatchTextToImageItem> items;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            items.push_back({{prompts[i]}, shared});
            auto add = [&](const trtmc::Config& config) {
                for (const auto& entry : config.entries())
                    items.back().config.add(entry.name, entry.value);
            };
            if (!seeds.empty())
                add(sdk_config({{"seed", seeds[i]}}, fields, {}));
            if (!configs.empty())
                add(sdk_config({{"config", configs[i]}}, fields, {}));
        }
        return measure(
            timing, [&]() { return task.run(items); },
            [](const trtmc::ImageBatchResult& result) {
                auto images = Json::array();
                std::uint64_t elements = 0;
                for (std::uint64_t i = 0; i < result.size(); ++i) {
                    const auto image = result[i];
                    images.push_back(generated_image_observation(image));
                    elements += image.pixels.size();
                }
                Json output{{"generated_images", result.size()},
                            {"batch_size", result.size()},
                            {"generated_frames", result.size()},
                            {"output_elements", elements},
                            {"media_type", "image"},
                            {"images", std::move(images)}};
                if (result.size()) {
                    const auto first = result[0];
                    output["height"] = first.height;
                    output["width"] = first.width;
                    output["channels"] = first.channels;
                    output["num_frames"] = 1;
                }
                return output;
            });
    }
    check_batch_size(request, 1);
    const auto prompt = request.at("prompt").get<std::string>();
    std::vector<std::string> paths;
    if (request.contains("image_path") && request.contains("image_paths"))
        throw std::invalid_argument("use image_path or image_paths, not both");
    if (request.contains("image_path"))
        paths.push_back(request.at("image_path").get<std::string>());
    else if (request.contains("image_paths"))
        paths = request.at("image_paths").get<std::vector<std::string>>();
    const bool edit = primary == trtmc::ImagesTextToImageEdit::kTask;
    const bool world = primary == trtmc::ImageTextActionToVideo::kTask;
    if ((!edit && !world && !paths.empty()) || (edit && paths.empty()) ||
        (world && paths.size() != 1))
        throw std::invalid_argument("conditioning image count does not match the selected Task");
    const auto replay_path = request.value("initial_latents_path", std::string{});
    if (request.contains("initial_latents_path") && replay_path.empty())
        throw std::invalid_argument("initial_latents_path must name a raw float32 file");
    auto read = [&]() {
        std::pair<std::vector<Image>, std::vector<float>> assets;
        for (const auto& path : paths)
            assets.first.push_back(read_image(path));
        if (!replay_path.empty())
            assets.second = read_float32(replay_path);
        return assets;
    };
    using Assets = decltype(read());
    std::optional<Assets> cached;
    if (!timing.asset_loading_included)
        cached = read();
    auto run = [&](const auto& task, const trtmc::Config& config, auto make_input, auto observe) {
        return measure(
            timing,
            [&]() {
                std::optional<Assets> loaded;
                if (!cached)
                    loaded = read();
                return task.run(make_input(cached ? *cached : *loaded), config);
            },
            observe);
    };
    auto observe_image = [](const trtmc::ImageGenerationResult& result) {
        return generated_image_observation(result);
    };
    if (primary == trtmc::TextToImage::kTask) {
        const auto task = model.task<trtmc::TextToImage>();
        const auto config =
            sdk_config(request, task.config_fields(),
                       {"prompt", "batch_size", "media_type", "initial_latents_path"});
        return run(
            task, config,
            [&](const auto& assets) {
                return trtmc::TextToImageRequest{prompt,
                                                 {assets.second.data(), assets.second.size()}};
            },
            observe_image);
    }
    if (edit) {
        const auto task = model.task<trtmc::ImagesTextToImageEdit>();
        const auto config = sdk_config(request, task.config_fields(),
                                       {"prompt", "image_path", "image_paths", "batch_size",
                                        "media_type", "initial_latents_path"});
        return run(
            task, config,
            [&](const auto& assets) {
                trtmc::ImagesTextToImageEditRequest input{
                    {}, prompt, {assets.second.data(), assets.second.size()}};
                for (const auto& image : assets.first)
                    input.images.push_back(sdk_image_view(image));
                return input;
            },
            observe_image);
    }
    if (primary == trtmc::TextToVideo::kTask) {
        const auto task = model.task<trtmc::TextToVideo>();
        const auto config =
            sdk_config(request, task.config_fields(),
                       {"prompt", "batch_size", "media_type", "initial_latents_path"});
        return run(
            task, config,
            [&](const auto& assets) {
                return trtmc::TextToVideoRequest{prompt,
                                                 {assets.second.data(), assets.second.size()}};
            },
            generated_video_observation);
    }
    if (!world)
        throw std::invalid_argument(
            "generate_image inputs do not match an implemented image/video Task");
    const auto action = request.at("action").get<std::string>();
    const auto& raw = request.at("camera_intrinsics");
    if (!raw.is_array() || raw.empty())
        throw std::invalid_argument("camera_intrinsics must be a nonempty numeric array");
    std::vector<float> calibration;
    for (const auto& item : raw)
        calibration.push_back(input_float({{"value", item}}, "value", 0));
    if (calibration.size() == 4) {
        const auto matrix = trtmc::pinhole_intrinsics(calibration[0], calibration[1],
                                                      calibration[2], calibration[3]);
        calibration.assign(matrix.begin(), matrix.end());
    } else if (calibration.size() % 9 != 0)
        throw std::invalid_argument(
            "camera_intrinsics requires fx,fy,cx,cy or row-major 3x3 matrices");
    const auto task = model.task<trtmc::ImageTextActionToVideo>();
    const auto config =
        sdk_config(request, task.config_fields(),
                   {"prompt", "image_path", "image_paths", "action", "camera_intrinsics",
                    "batch_size", "media_type", "initial_latents_path"});
    return run(
        task, config,
        [&](const auto& assets) {
            return trtmc::ImageTextActionToVideoRequest{
                sdk_image_view(assets.first[0]),
                prompt,
                action,
                {{calibration.data(), calibration.size()}, calibration.size() / 9, 9},
                {},
                {assets.second.data(), assets.second.size()}};
        },
        generated_video_observation);
}

Json run_disparity(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (model.info().bundle_task != trtmc::StereoImagesToDisparity::kTask)
        throw std::invalid_argument("disparity requires StereoImagesToDisparity");
    const auto task = model.task<trtmc::StereoImagesToDisparity>();
    const auto config = sdk_config(request, task.config_fields(),
                                   {"left_image_path", "right_image_path", "_artifact_path"});
    const auto left_path = request.at("left_image_path").get<std::string>();
    const auto right_path = request.at("right_image_path").get<std::string>();
    auto read = [&]() {
        auto images = std::make_pair(read_image(left_path), read_image(right_path));
        if (images.first.height != images.second.height ||
            images.first.width != images.second.width)
            throw std::invalid_argument("stereo images must have identical dimensions");
        return images;
    };
    std::optional<std::pair<Image, Image>> cached;
    if (!timing.asset_loading_included)
        cached = read();
    auto invoke = [&]() {
        std::optional<std::pair<Image, Image>> loaded;
        if (!cached)
            loaded = read();
        const auto& images = cached ? *cached : *loaded;
        return task.run({sdk_image_view(images.first), sdk_image_view(images.second)}, config);
    };
    using Result = decltype(invoke());
    std::optional<Result> last;
    for (int i = 0; i < timing.warmup; ++i)
        last = invoke();
    Json observations = Json::array();
    for (int i = 0; i < timing.iterations; ++i) {
        last.reset();
        const auto started = Clock::now();
        auto result = invoke();
        const auto wall_ms = elapsed_ms(started);
        last.emplace(std::move(result));
        observations.push_back({{"runtime_e2e_wall_ms", wall_ms},
                                {"stereo_pairs", 1},
                                {"disparity_pixels", last->view().disparity.count}});
    }
    const auto& map = last->view().disparity;
    const auto path = request.at("_artifact_path").get<std::string>();
    if (map.count >
        static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))
        throw std::runtime_error("disparity artifact is too large");
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(map.data),
                 static_cast<std::streamsize>(map.count * sizeof(float)));
    output.close();
    if (!output)
        throw std::runtime_error("failed to write disparity artifact " + path);
    return {{"observations", std::move(observations)},
            {"output_summary",
             {{"stereo_pairs", 1},
              {"disparity_pixels", map.count},
              {"element_count", map.count},
              {"height", map.rows},
              {"width", map.columns},
              {"disparity_artifact", path},
              {"units", "pixels"},
              {"grid", "left_image"},
              {"convention", "x_left_minus_x_right"}}}};
}

void copy_track_mask(void* destination, const trtmc_track_frame_view_v1& frame) {
    if (!frame.mask_byte_size)
        return;
    if (frame.memory_kind == TRTMC_TRACK_HOST) {
        std::memcpy(destination, frame.masks, static_cast<std::size_t>(frame.mask_byte_size));
        return;
    }
    if (frame.memory_kind != TRTMC_TRACK_CUDA)
        throw std::runtime_error("unknown track mask memory domain");
    auto checked = [](cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("CUDA mask ") + operation + ": " +
                                     cudaGetErrorString(status));
    };
    cudaPointerAttributes attributes{};
    checked(cudaPointerGetAttributes(&attributes, frame.masks), "pointer query");
    if ((attributes.type != cudaMemoryTypeDevice && attributes.type != cudaMemoryTypeManaged) ||
        attributes.device != frame.device_ordinal)
        throw std::runtime_error("CUDA mask pointer does not match its declared device memory");
    int previous = 0;
    checked(cudaGetDevice(&previous), "get device");
    checked(cudaSetDevice(frame.device_ordinal), "set device");
    try {
        // The producer is ready at return; this synchronous copy finishes before close/reuse.
        checked(cudaMemcpy(destination, frame.masks, static_cast<std::size_t>(frame.mask_byte_size),
                           cudaMemcpyDeviceToHost),
                "copy to host");
        checked(cudaSetDevice(previous), "restore device");
    } catch (...) {
        (void)cudaSetDevice(previous);
        throw;
    }
}

struct TrackSnapshot {
    trtmc::TrackClipResult owner;
    trtmc_track_clip_view_v1 metadata;
    std::vector<std::variant<std::vector<std::uint8_t>, std::vector<float>>> masks;
};
TrackSnapshot snapshot_tracks(trtmc::TrackClipResult result) {
    const auto view = result.view();
    TrackSnapshot snapshot{std::move(result), view, {}};
    for (std::uint64_t i = 0; i < view.frame_count; ++i) {
        const auto& frame = view.frames[i];
        if (frame.mask_byte_size > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("track mask exceeds host address space");
        if (frame.element_type == TRTMC_TRACK_UINT8) {
            std::vector<std::uint8_t> values(static_cast<std::size_t>(frame.mask_byte_size));
            copy_track_mask(values.data(), frame);
            snapshot.masks.emplace_back(std::move(values));
        } else if (frame.element_type == TRTMC_TRACK_FLOAT32 &&
                   frame.mask_byte_size % sizeof(float) == 0) {
            std::vector<float> values(
                static_cast<std::size_t>(frame.mask_byte_size / sizeof(float)));
            copy_track_mask(values.data(), frame);
            snapshot.masks.emplace_back(std::move(values));
        } else
            throw std::runtime_error("unknown or misaligned track mask element type");
    }
    return snapshot;
}
Json track_observation(const TrackSnapshot& snapshot, const std::vector<double>& timestamps) {
    const auto& clip =
        snapshot.metadata; // Metadata is owned by the retained result, even after close.
    Json frames = Json::array(), detections = Json::array();
    std::uint64_t elements = 0;
    bool device_copy = false;
    for (std::uint64_t i = 0; i < clip.frame_count; ++i) {
        const auto& frame = clip.frames[i];
        Json masks = std::visit(
            [&](const auto& values) {
                elements += values.size();
                return json_values(values.data(), values.size());
            },
            snapshot.masks[i]);
        Json boxes = Json::array();
        for (std::uint64_t j = 0; j < frame.box_count; ++j)
            boxes.push_back({frame.boxes[j].x_min, frame.boxes[j].y_min, frame.boxes[j].x_max,
                             frame.boxes[j].y_max});
        const char* kind = frame.mask_kind == TRTMC_MASK_LOGITS        ? "logits"
                           : frame.mask_kind == TRTMC_MASK_PROBABILITY ? "probability"
                                                                       : "binary";
        device_copy |= frame.memory_kind == TRTMC_TRACK_CUDA;
        frames.push_back(
            {{"frame_index", frame.frame_index},
             {"height", frame.height},
             {"width", frame.width},
             {"timestamp_seconds", frame.frame_index < timestamps.size()
                                       ? Json(timestamps[frame.frame_index])
                                       : Json(nullptr)},
             {"masks", std::move(masks)},
             {"mask_kind", kind},
             {"mask_element_type", frame.element_type == TRTMC_TRACK_UINT8 ? "uint8" : "float32"},
             {"source_memory", frame.memory_kind == TRTMC_TRACK_HOST ? "host" : "cuda"},
             {"device_ordinal", frame.device_ordinal},
             {"mask_byte_size", frame.mask_byte_size},
             {"object_ids", json_values(frame.object_ids.data, frame.object_ids.size)},
             {"class_ids", json_values(frame.class_ids.data, frame.class_ids.size)},
             {"boxes", std::move(boxes)},
             {"box_coordinates", "original_image_pixels_xyxy"},
             {"detection_scores", json_values(frame.detection_scores, frame.detection_score_count)},
             {"tracking_scores", json_values(frame.tracker_scores, frame.tracker_score_count)},
             {"removed_object_ids",
              json_values(frame.removed_object_ids.data, frame.removed_object_ids.size)},
             {"suppressed_object_ids",
              json_values(frame.suppressed_object_ids.data, frame.suppressed_object_ids.size)}});
    }
    for (std::uint64_t i = 0; i < clip.detection_count; ++i) {
        const auto& value = clip.initial_detections[i];
        detections.push_back({{"frame_index", value.frame_index},
                              {"object_id", value.object_id},
                              {"class_id", value.class_id},
                              {"score", value.score},
                              {"prompt_box",
                               {value.prompt_box.x_min, value.prompt_box.y_min,
                                value.prompt_box.x_max, value.prompt_box.y_max}}});
    }
    return {{"tracked_frames", clip.frame_count},
            {"mask_elements", elements},
            {"frames", std::move(frames)},
            {"initial_detections", std::move(detections)},
            {"device_copy_included", device_copy}};
}

Json run_track_masks(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    const bool detected = primary == trtmc::FramesToDetectedMaskTracks::kTask;
    const bool prompt_frame = primary == trtmc::PromptFrameTextToMaskTracks::kTask;
    if (!detected && !prompt_frame && primary != trtmc::FramesTextToMaskTracks::kTask)
        throw std::invalid_argument("track_masks requires an exact tracking Task");
    if (detected && request.contains("prompt"))
        throw std::invalid_argument("detector tracking has no text prompt input");
    if (!detected && request.contains("device_masks"))
        throw std::invalid_argument("device_masks requires detector tracking");
    if (prompt_frame && request.contains("segment_config"))
        throw std::invalid_argument("prompt-frame Config belongs to creation only");
    const auto paths = request.at("frame_paths").get<std::vector<std::string>>();
    if (paths.empty())
        throw std::invalid_argument("tracking requires a nonempty complete clip");
    const auto prompt = detected ? std::string{} : request.at("prompt").get<std::string>();
    const bool device = request.value("device_masks", false);
    const auto timestamps = request.value("timestamps_seconds", std::vector<double>{});
    auto read = [&]() {
        std::vector<Image> images;
        for (const auto& path : paths)
            images.push_back(read_image(path));
        return images;
    };
    auto video = [&](const std::vector<Image>& images) {
        trtmc::VideoInput result;
        for (const auto& image : images)
            result.frames.push_back(sdk_image_view(image));
        result.timestamps_seconds = timestamps;
        return result;
    };
    std::optional<std::vector<Image>> cached;
    trtmc::VideoInput cached_clip;
    if (!timing.asset_loading_included) {
        cached = read();
        cached_clip = video(*cached);
    }
    struct Result {
        TrackSnapshot final;
        std::optional<TrackSnapshot> prompt;
    };
    auto run = [&](const auto& task, auto invoke) {
        const auto fields = task.config_fields();
        const auto config = sdk_config(
            request, fields,
            {"frame_paths", "timestamps_seconds", "prompt", "device_masks", "segment_config"});
        const auto segment =
            request.contains("segment_config")
                ? sdk_config({{"config", request.at("segment_config")}}, fields, {})
                : trtmc::Config{};
        return measure(
            timing,
            [&]() {
                std::optional<std::vector<Image>> loaded;
                trtmc::VideoInput clip;
                if (!cached) {
                    loaded = read();
                    clip = video(*loaded);
                }
                return invoke(task, config, segment, cached ? cached_clip : clip);
            },
            [&](const Result& result) {
                auto output = track_observation(result.final, timestamps);
                if (result.prompt)
                    output["prompt_snapshot"] = track_observation(*result.prompt, timestamps);
                output["lifecycle_scope"] =
                    prompt_frame
                        ? "fresh_create_accept_first_frame_continue_complete_clip_snapshot_close"
                        : "fresh_create_segment_snapshot_close";
                return output;
            });
    };
    if (detected)
        return run(
            model.task<trtmc::FramesToDetectedMaskTracks>(),
            [&](const auto& task, const auto& config, const auto& segment, const auto& clip) {
                auto session = task.create(config);
                auto result = device ? session.device_masks().segment_device(clip, segment)
                                     : session.segment(clip, segment);
                auto snapshot = snapshot_tracks(std::move(result));
                session.close();
                return Result{std::move(snapshot), {}};
            });
    if (prompt_frame)
        return run(model.task<trtmc::PromptFrameTextToMaskTracks>(),
                   [&](const auto& task, const auto& config, const auto&, const auto& clip) {
                       auto session = task.create(prompt, config);
                       auto first =
                           snapshot_tracks(session.accept_prompt_frame(clip.frames.front()));
                       auto consolidated =
                           snapshot_tracks(session.continue_borrowed(first.owner, clip));
                       session.close();
                       return Result{std::move(consolidated), std::move(first)};
                   });
    return run(model.task<trtmc::FramesTextToMaskTracks>(),
               [&](const auto& task, const auto& config, const auto& segment, const auto& clip) {
                   auto session = task.create(config);
                   auto result = snapshot_tracks(session.segment(clip, prompt, segment));
                   session.close();
                   return Result{std::move(result), {}};
               });
}

struct PreparedCrop {
    trtmc::PoseCropPhase stage;
    std::uint64_t iteration;
    std::vector<float> query_poses;
    trtmc::PoseCrops crops;
};
std::vector<PreparedCrop> read_crop_batches(const Json& input) {
    if (!input.is_array() || input.empty())
        throw std::invalid_argument("preprocessed crop_batches must be nonempty");
    std::vector<PreparedCrop> result;
    for (const auto& entry : input) {
        const auto stage = entry.at("stage").get<std::string>();
        if (stage != "refinement" && stage != "scoring")
            throw std::invalid_argument("unknown crop stage");
        const auto& iteration = entry.at("iteration");
        if (!iteration.is_number_integer() || iteration < 0)
            throw std::invalid_argument("crop iteration must be nonnegative");
        const auto& shape = entry.at("shape");
        if (!shape.is_array() || shape.size() != 4)
            throw std::invalid_argument("crop shape must be [N,H,W,6]");
        for (const auto& axis : shape)
            if (!axis.is_number_integer() || axis <= 0)
                throw std::invalid_argument("crop axes must be positive integers");
        if (shape[3] != 6 || shape[1] > UINT32_MAX || shape[2] > UINT32_MAX)
            throw std::invalid_argument(
                "crop shape must use positive uint32 spatial axes and six channels");
        PreparedCrop crop{stage == "refinement" ? trtmc::PoseCropPhase::Refinement
                                                : trtmc::PoseCropPhase::Scoring,
                          iteration.get<std::uint64_t>(),
                          read_float32(entry.at("query_poses_path").get<std::string>()),
                          {}};
        crop.crops = {read_float32(entry.at("rendered_path").get<std::string>()),
                      read_float32(entry.at("observed_path").get<std::string>()),
                      shape[0].get<std::uint64_t>(),
                      shape[1].get<std::uint32_t>(),
                      shape[2].get<std::uint32_t>(),
                      6};
        result.push_back(std::move(crop));
    }
    return result;
}
struct PreparedPose {
    std::vector<float> poses;
    std::uint64_t count;
    float diameter;
    std::vector<PreparedCrop> batches;
};
PreparedPose read_pose(const Json& input) {
    const auto& count = input.at("hypothesis_count");
    if (!count.is_number_integer() || count <= 0)
        throw std::invalid_argument("hypothesis_count must be positive");
    if (!input.contains("mesh_diameter_meters"))
        throw std::invalid_argument("pose input requires mesh_diameter_meters");
    return {read_float32(input.at("candidate_poses_path").get<std::string>()),
            count.get<std::uint64_t>(), input_float(input, "mesh_diameter_meters", 0),
            read_crop_batches(input.at("crop_batches"))};
}
struct CropQuery {
    trtmc::PoseCropPhase stage;
    std::uint64_t iteration, count;
    std::vector<float> poses;
};
trtmc::PoseCrops consume_crop(const std::vector<PreparedCrop>& batches, std::size_t& cursor,
                              std::vector<CropQuery>& queries, const trtmc::PoseCropQuery& query) {
    if (cursor == batches.size())
        throw std::invalid_argument("no preprocessed crop batch for provider query");
    const auto& batch = batches[cursor];
    if (query.stage != batch.stage || query.iteration != batch.iteration ||
        query.poses.count != batch.crops.count ||
        query.poses.values.size() != batch.query_poses.size() ||
        !std::equal(query.poses.values.begin(), query.poses.values.end(),
                    batch.query_poses.begin()))
        throw std::invalid_argument(
            "preprocessed crop query does not match stage, iteration or actual poses");
    queries.push_back({query.stage,
                       query.iteration,
                       query.poses.count,
                       {query.poses.values.begin(), query.poses.values.end()}});
    ++cursor;
    return batch.crops;
}
struct PoseSnapshot {
    trtmc::RefinedPosesResult result;
    std::vector<CropQuery> queries;
};
Json pose_observation(const PoseSnapshot& result) {
    const auto& view = result.result.view();
    Json queries = Json::array();
    for (const auto& query : result.queries)
        queries.push_back(
            {{"stage", query.stage == trtmc::PoseCropPhase::Refinement ? "refinement" : "scoring"},
             {"iteration", query.iteration},
             {"shape", {query.count, 4, 4}},
             {"poses", query.poses}});
    return {
        {"refined_hypotheses", view.refined_poses.count},
        {"shape", {view.refined_poses.count, 4, 4}},
        {"refined_poses", json_values(view.refined_poses.values, view.refined_poses.value_count)},
        {"scores", json_values(view.scores, view.score_count)},
        {"best_index", view.best_index},
        {"all_poses_rigid", view.all_poses_rigid != 0},
        {"refinement_ms", view.refinement_ms},
        {"scoring_ms", view.scoring_ms},
        {"crop_queries", std::move(queries)}};
}
Json run_refine_pose(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::PoseHypothesesCropsToRefinedPoses>();
    const auto config = sdk_config(
        request, task.config_fields(),
        {"candidate_poses_path", "hypothesis_count", "mesh_diameter_meters", "crop_batches"});
    std::optional<PreparedPose> cached;
    if (!timing.asset_loading_included)
        cached = read_pose(request);
    return measure(
        timing,
        [&]() {
            std::optional<PreparedPose> loaded;
            if (!cached)
                loaded = read_pose(request);
            const auto& input = cached ? *cached : *loaded;
            std::size_t cursor = 0;
            std::vector<CropQuery> queries;
            auto result = task.run({{{input.poses.data(), input.poses.size()}, input.count},
                                    input.diameter,
                                    [&](const auto& query) {
                                        return consume_crop(input.batches, cursor, queries, query);
                                    }},
                                   config);
            if (cursor != input.batches.size())
                throw std::invalid_argument("unused preprocessed crop batches");
            return PoseSnapshot{std::move(result), std::move(queries)};
        },
        pose_observation);
}
Json run_track_pose(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::CropPoseTracking>();
    const auto fields = task.config_fields();
    const auto config = sdk_config(request, fields, {"initialization", "updates"});
    const auto& initial = request.at("initialization");
    const auto& updates = request.at("updates");
    if (!updates.is_array() || updates.empty())
        throw std::invalid_argument("pose tracking requires nonempty updates");
    const auto initial_config = sdk_config(
        initial, fields,
        {"candidate_poses_path", "hypothesis_count", "mesh_diameter_meters", "crop_batches"});
    std::vector<trtmc::Config> update_config;
    for (const auto& item : updates)
        update_config.push_back(sdk_config(item, fields, {"crop_batches"}));
    struct Inputs {
        PreparedPose initial;
        std::vector<std::vector<PreparedCrop>> updates;
    };
    auto read = [&]() {
        Inputs input{read_pose(initial), {}};
        for (const auto& item : updates)
            input.updates.push_back(read_crop_batches(item.at("crop_batches")));
        return input;
    };
    std::optional<Inputs> cached;
    if (!timing.asset_loading_included)
        cached = read();
    return measure(
        timing,
        [&]() {
            std::optional<Inputs> loaded;
            if (!cached)
                loaded = read();
            const auto& input = cached ? *cached : *loaded;
            auto session = task.create(config);
            std::vector<PoseSnapshot> results;
            std::size_t cursor = 0;
            std::vector<CropQuery> queries;
            auto first = session.initialize(
                {{{input.initial.poses.data(), input.initial.poses.size()}, input.initial.count},
                 input.initial.diameter,
                 [&](const auto& query) {
                     return consume_crop(input.initial.batches, cursor, queries, query);
                 }},
                initial_config);
            if (cursor != input.initial.batches.size())
                throw std::invalid_argument("unused initialization crop batches");
            results.push_back({std::move(first), std::move(queries)});
            for (std::size_t i = 0; i < input.updates.size(); ++i) {
                cursor = 0;
                queries.clear();
                auto next = session.track(
                    [&](const auto& query) {
                        return consume_crop(input.updates[i], cursor, queries, query);
                    },
                    update_config[i]);
                if (cursor != input.updates[i].size())
                    throw std::invalid_argument("unused tracking crop batches");
                results.push_back({std::move(next), std::move(queries)});
            }
            session.close();
            return results;
        },
        [](const auto& results) {
            Json updates = Json::array();
            for (std::size_t i = 1; i < results.size(); ++i)
                updates.push_back(pose_observation(results[i]));
            return Json{{"initialization", pose_observation(results.front())},
                        {"pose_updates", updates.size()},
                        {"updates", std::move(updates)},
                        {"lifecycle_scope", "fresh_create_initialize_track_all_close"}};
        });
}

template <class T>
void write_binary_artifact(const std::string& path, const T* values, std::uint64_t count) {
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::runtime_error("binary artifact is too large");
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(reinterpret_cast<const char*>(values),
                 static_cast<std::streamsize>(count * sizeof(T)));
    output.close();
}

Json run_geometry(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::ImageToMetricGeometry>();
    const auto config =
        sdk_config(request, task.config_fields(), {"image_path", "_artifact_prefix"});
    const auto path = request.at("image_path").get<std::string>();
    std::optional<Image> cached;
    if (!timing.asset_loading_included)
        cached = read_image(path);
    auto invoke = [&]() {
        std::optional<Image> loaded;
        if (!cached)
            loaded = read_image(path);
        return task.run({sdk_image_view(cached ? *cached : *loaded)}, config);
    };
    std::optional<trtmc::MetricGeometryResult> last;
    for (int i = 0; i < timing.warmup; ++i)
        last = invoke();
    Json observations = Json::array();
    for (int i = 0; i < timing.iterations; ++i) {
        last.reset();
        const auto started = Clock::now();
        auto result = invoke();
        const auto wall_ms = elapsed_ms(started);
        last.emplace(std::move(result));
        observations.push_back({{"runtime_e2e_wall_ms", wall_ms},
                                {"geometry_images", 1},
                                {"geometry_pixels", last->view().pixel_count}});
    }
    const auto& view = last->view();
    const auto prefix = request.at("_artifact_prefix").get<std::string>();
    write_binary_artifact(prefix + ".points.f32", view.points, view.point_value_count);
    write_binary_artifact(prefix + ".depth.f32", view.depth, view.pixel_count);
    write_binary_artifact(prefix + ".mask.u8", view.valid, view.pixel_count);
    Json intrinsics = Json::array();
    for (int row = 0; row < 3; ++row)
        intrinsics.push_back({view.normalized_intrinsics[row * 3],
                              view.normalized_intrinsics[row * 3 + 1],
                              view.normalized_intrinsics[row * 3 + 2]});
    return {{"observations", std::move(observations)},
            {"output_summary",
             {{"geometry_images", 1},
              {"geometry_pixels", view.pixel_count},
              {"height", view.height},
              {"width", view.width},
              {"point_shape", {view.height, view.width, 3}},
              {"valid_pixels", std::count(view.valid, view.valid + view.pixel_count, 1)},
              {"normalized_intrinsics", std::move(intrinsics)},
              {"units", "meters"},
              {"camera_axes", {"right", "down", "forward"}},
              {"intrinsics_coordinates", "normalized_uv"},
              {"points_artifact", prefix + ".points.f32"},
              {"depth_artifact", prefix + ".depth.f32"},
              {"valid_mask_artifact", prefix + ".mask.u8"}}}};
}

Json run_detect(const trtmc::Model& model, const Json& request, const Timing& timing) {
    if (request.contains("prompt"))
        throw std::invalid_argument("image-only detection has no prompt input");
    const auto task = model.task<trtmc::ImageToBoxes>();
    const auto config = sdk_config(request, task.config_fields(), {"image_path"});
    const auto path = request.at("image_path").get<std::string>();
    std::optional<Image> cached;
    std::optional<trtmc::ImageToBoxesRequest> cached_request;
    if (!timing.asset_loading_included) {
        cached = read_image(path);
        cached_request = trtmc::ImageToBoxesRequest{sdk_image_view(*cached)};
    }
    return measure(
        timing,
        [&]() {
            if (cached)
                return task.run(*cached_request, config);
            const auto input = read_image(path);
            return task.run({sdk_image_view(input)}, config);
        },
        [](const auto& result) {
            const auto& view = result.view();
            Json boxes = Json::array(), scores = Json::array(), classes = Json::array();
            for (std::uint64_t i = 0; i < view.count; ++i) {
                const auto& item = view.boxes[i];
                for (const float coordinate :
                     {item.box.x_min, item.box.y_min, item.box.x_max, item.box.y_max})
                    boxes.push_back(coordinate);
                scores.push_back(item.score);
                classes.push_back(item.class_id);
            }
            return Json{{"detected_images", 1},
                        {"detections", view.count},
                        {"image_height", view.image_height},
                        {"image_width", view.image_width},
                        {"boxes", std::move(boxes)},
                        {"scores", std::move(scores)},
                        {"class_ids", std::move(classes)},
                        {"shape", {view.count, 4}},
                        {"coordinates", "xyxy"},
                        {"units", "pixels"}};
        });
}

Json run_structure(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::MolecularDocumentToStructure>();
    const auto config =
        sdk_config(request, task.config_fields(),
                   {"document_path", "input_encoding", "source_path", "_artifact_prefix"});
    const auto path = request.at("document_path").get<std::string>();
    const auto source_path = request.value("source_path", path);
    std::string encoding;
    if (request.contains("input_encoding"))
        encoding = request.at("input_encoding").get<std::string>();
    else {
        const auto extension = std::filesystem::path(path).extension().string();
        if (extension == ".yaml" || extension == ".yml")
            encoding = "yaml";
        else if (extension == ".json")
            encoding = "json";
        else if (extension == ".b2rq")
            encoding = "b2rq";
        else
            throw std::invalid_argument(
                "unknown structure request extension; specify input_encoding");
    }
    auto read = [&]() {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("cannot read molecular document " + path);
        std::string document{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
        if (input.bad())
            throw std::runtime_error("failed to read molecular document " + path);
        return document;
    };
    auto input_view = [&](const std::string& document) {
        return trtmc::MolecularDocumentToStructureRequest{
            {reinterpret_cast<const std::uint8_t*>(document.data()), document.size()},
            encoding,
            source_path};
    };
    std::optional<std::string> cached;
    std::optional<trtmc::MolecularDocumentToStructureRequest> cached_request;
    std::size_t document_size = 0;
    if (!timing.asset_loading_included) {
        cached = read();
        document_size = cached->size();
        cached_request = input_view(*cached);
    }
    auto invoke = [&]() {
        if (cached)
            return task.run(*cached_request, config);
        const auto document = read();
        document_size = document.size();
        return task.run(input_view(document), config);
    };
    std::optional<trtmc::MolecularStructureResult> last;
    for (int i = 0; i < timing.warmup; ++i)
        last = invoke();
    Json observations = Json::array();
    for (int i = 0; i < timing.iterations; ++i) {
        last.reset();
        const auto started = Clock::now();
        auto result = invoke();
        const auto wall_ms = elapsed_ms(started);
        last.emplace(std::move(result));
        observations.push_back({{"runtime_e2e_wall_ms", wall_ms},
                                {"structures", 1},
                                {"structure_bytes", last->structure().size()},
                                {"document_bytes", document_size}});
    }
    const bool pdb = last->format() == trtmc::MolecularStructureFormat::Pdb;
    const auto prefix = request.at("_artifact_prefix").get<std::string>();
    const auto structure_path = prefix + (pdb ? ".pdb" : ".cif");
    const auto metadata_path = prefix + ".metadata.json";
    // Both result views are written by byte count, including any embedded NUL.
    write_binary_artifact(structure_path, last->structure().data(), last->structure().size());
    write_binary_artifact(metadata_path, last->metadata_json().data(),
                          last->metadata_json().size());
    Json confidence = nullptr;
    if (const auto values = last->confidence())
        confidence = {{"confidence_score", values->confidence_score},
                      {"ptm", values->ptm},
                      {"iptm", values->iptm},
                      {"ligand_iptm", values->ligand_iptm},
                      {"protein_iptm", values->protein_iptm},
                      {"complex_plddt", values->complex_plddt},
                      {"complex_iplddt", values->complex_iplddt},
                      {"plddt", json_values(values->plddt.data, values->plddt.size)}};
    return {{"observations", std::move(observations)},
            {"output_summary",
             {{"structures", 1},
              {"structure_bytes", last->structure().size()},
              {"document_bytes", document_size},
              {"metadata_bytes", last->metadata_json().size()},
              {"format", pdb ? "pdb" : "mmcif"},
              {"input_encoding", encoding},
              {"source_path", source_path},
              {"confidence", std::move(confidence)},
              {"structure_artifact", structure_path},
              {"metadata_artifact", metadata_path}}}};
}

Json run_translate(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::TextTranslation>();
    const trtmc::TextTranslationRequest input{request.at("source_text").get<std::string>(),
                                              language_input(request, "target_language"),
                                              language_input(request, "source_language")};
    const auto config = sdk_config(request, task.config_fields(),
                                   {"source_text", "source_language", "target_language"});
    return measure(timing, [&]() { return task.run(input, config); }, text_observation);
}

std::vector<float> optional_float32(const Json& request, const char* name) {
    return request.contains(name) ? read_float32(request.at(name).get<std::string>())
                                  : std::vector<float>{};
}

Json run_latent_generate(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const bool conditioned =
        model.info().bundle_task == trtmc::LatentConditionedTextGeneration::kTask;
    if (conditioned &&
        (!request.contains("condition_latents_path") || !request.contains("condition_mask_path")))
        throw std::invalid_argument(
            "conditioned generation requires raw condition latents and mask");
    if (!conditioned &&
        (request.contains("condition_latents_path") || request.contains("condition_mask_path")))
        throw std::invalid_argument("raw condition inputs require the conditioned latent Task");
    const auto prompt = request.value("prompt", std::string{});
    struct Inputs {
        std::vector<float> condition, mask, initial, noise, steps;
    };
    auto read = [&]() {
        return Inputs{optional_float32(request, "condition_latents_path"),
                      optional_float32(request, "condition_mask_path"),
                      optional_float32(request, "initial_latents_path"),
                      optional_float32(request, "sde_noises_path"),
                      optional_float32(request, "sampling_steps_path")};
    };
    auto run = [&](const auto& task, auto make_input) {
        const auto fields = task.config_fields();
        const auto config =
            sdk_config(request, fields,
                       {"prompt", "condition_latents_path", "condition_mask_path",
                        "initial_latents_path", "sde_noises_path", "sampling_steps_path"});
        const bool explicit_steps = request.contains("sampling_steps_path");
        if (explicit_steps && std::none_of(fields.begin(), fields.end(), [](const auto& field) {
                return field.name == "sampling_steps" && field.kind == trtmc::ConfigKind::F64List;
            }))
            throw std::invalid_argument(
                "selected Task does not declare sampling_steps as a float list");
        auto options_for = [&](const Inputs& input) {
            auto options = config;
            if (explicit_steps)
                options.add("sampling_steps",
                            std::vector<double>(input.steps.begin(), input.steps.end()));
            return options;
        };
        std::optional<Inputs> cached;
        std::optional<decltype(make_input(Inputs{}))> cached_request;
        trtmc::Config cached_config;
        if (!timing.asset_loading_included) {
            cached = read();
            cached_request = make_input(*cached);
            cached_config = options_for(*cached);
        }
        return measure(
            timing,
            [&]() {
                if (cached)
                    return task.run(*cached_request, cached_config);
                const auto input = read();
                return task.run(make_input(input), options_for(input));
            },
            text_observation);
    };
    auto view = [](const std::vector<float>& values) {
        return trtmc::Span<const float>{values.data(), values.size()};
    };
    if (conditioned)
        return run(model.task<trtmc::LatentConditionedTextGeneration>(), [&](const auto& input) {
            return trtmc::LatentConditionedTextGenerationRequest{
                view(input.condition), view(input.mask), view(input.initial), view(input.noise),
                prompt};
        });
    return run(model.task<trtmc::LatentReplayToText>(), [&](const auto& input) {
        return trtmc::LatentReplayToTextRequest{view(input.initial), prompt, view(input.noise)};
    });
}

Json run_latent_step(const trtmc::Model& model, const Json& request, const Timing& timing,
                     bool decoder) {
    const bool packed = request.contains("branch_path") || request.contains("trunk_path");
    if (packed) {
        if (!request.contains("branch_path") || !request.contains("trunk_path"))
            throw std::invalid_argument("native-packed input requires branch_path and trunk_path");
        for (const auto* key : {"latents_path", "shape", "timestep", "self_condition_path"})
            if (request.contains(key))
                throw std::invalid_argument(
                    "native-packed input cannot include logical latent operands");
    }
    struct Inputs {
        std::vector<float> latent, self;
        std::uint64_t rows{0}, columns{0};
        double timestep{0};
        std::optional<double> guidance;
        trtmc::FloatMatrixView latents() const {
            return {{latent.data(), latent.size()}, rows, columns};
        }
        trtmc::FloatMatrixView self_condition() const {
            return self.empty() ? trtmc::FloatMatrixView{}
                                : trtmc::FloatMatrixView{{self.data(), self.size()}, rows, columns};
        }
    };
    auto read = [&]() {
        Inputs input;
        if (packed) {
            input.latent = read_float32(request.at("branch_path").get<std::string>());
            const auto controls = read_float32(request.at("trunk_path").get<std::string>());
            if (controls.size() != 3 ||
                !std::all_of(controls.begin(), controls.end(),
                             [](float value) { return std::isfinite(value); }))
                throw std::invalid_argument(
                    "trunk must contain finite timestep, guidance and decoder selector");
            if ((controls[2] >= 0.5F) != decoder)
                throw std::invalid_argument(
                    "trunk decoder selector conflicts with the requested latent Task");
            input.timestep = controls[0];
            input.guidance = controls[1];
        } else {
            input.latent = read_float32(request.at("latents_path").get<std::string>());
            input.self = optional_float32(request, "self_condition_path");
            const auto& shape = request.at("shape");
            if (!shape.is_array() || shape.size() != 2 || !shape[0].is_number_integer() ||
                !shape[1].is_number_integer() || shape[0] <= 0 || shape[1] <= 0)
                throw std::invalid_argument(
                    "latent shape must be positive [position,latent_channel]");
            input.rows = shape[0].get<std::uint64_t>();
            input.columns = shape[1].get<std::uint64_t>();
            const auto& time = request.at("timestep");
            if (!time.is_number() || !std::isfinite(time.get<double>()))
                throw std::invalid_argument("timestep must be a finite number");
            input.timestep = time.get<double>();
        }
        return input;
    };
    auto run = [&](const auto& task, auto make_input, auto observe) {
        const auto fields = task.config_fields();
        const auto config = sdk_config(request, fields,
                                       {"branch_path", "trunk_path", "latents_path",
                                        "self_condition_path", "shape", "timestep"});
        if (packed && std::none_of(fields.begin(), fields.end(), [](const auto& field) {
                return field.name == "self_cond_cfg_scale" && field.kind == trtmc::ConfigKind::F64;
            }))
            throw std::invalid_argument(
                "selected Task does not declare self_cond_cfg_scale as a float");
        auto options_for = [&](const Inputs& input) {
            auto options = config;
            if (input.guidance)
                options.add("self_cond_cfg_scale", *input.guidance);
            return options;
        };
        std::optional<Inputs> cached;
        std::optional<decltype(make_input(Inputs{}))> cached_request;
        trtmc::Config cached_config;
        if (!timing.asset_loading_included) {
            cached = read();
            cached_request = make_input(*cached);
            cached_config = options_for(*cached);
        }
        return measure(
            timing,
            [&]() {
                if (cached)
                    return task.run(*cached_request, cached_config);
                const auto input = read();
                return task.run(make_input(input), options_for(input));
            },
            observe);
    };
    auto matrix = [](const auto& value, const char* axis) {
        return Json{{"values", json_values(value.data, value.count)},
                    {"shape", {value.rows, value.columns}},
                    {"axes", {"position", axis}}};
    };
    if (decoder)
        return run(
            model.task<trtmc::LatentToTokenLogits>(),
            [](const auto& input) {
                return trtmc::LatentToTokenLogitsRequest{input.latents(), input.self_condition(),
                                                         input.timestep};
            },
            [&](const auto& result) {
                auto output = matrix(result.view().logits, "vocabulary_id");
                output["kind"] = "token_logits";
                output["logit_elements"] = result.view().logits.count;
                output["vocabulary_id"] =
                    std::string(trtmc::detail::string_view(result.view().vocabulary_id));
                return output;
            });
    return run(
        model.task<trtmc::LatentDenoisingStep>(),
        [](const auto& input) {
            return trtmc::LatentDenoisingStepRequest{input.latents(), input.self_condition(),
                                                     input.timestep};
        },
        [&](const auto& result) {
            auto output = matrix(result.view().latents, "latent_channel");
            output["kind"] = "denoised_latents";
            output["latent_elements"] = result.view().latents.count;
            return output;
        });
}

Json run_generate(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    if (primary == trtmc::UnconditionalTextGeneration::kTask) {
        if (request.contains("prompt"))
            throw std::invalid_argument("unconditional text generation has no prompt input");
        const auto task = model.task<trtmc::UnconditionalTextGeneration>();
        const auto config = sdk_config(request, task.config_fields(), {});
        return measure(timing, [&]() { return task.run(config); }, text_observation);
    }
    if (primary == trtmc::LatentConditionedTextGeneration::kTask ||
        primary == trtmc::LatentReplayToText::kTask)
        return run_latent_generate(model, request, timing);
    const std::string prompt = request.at("prompt").get<std::string>();
    if (request.contains("image_path")) {
        const auto task = model.task<trtmc::ImagesTextToText>();
        const auto config = sdk_config(request, task.config_fields(), {"prompt", "image_path"});
        const auto path = request.at("image_path").get<std::string>();
        std::optional<Image> cached;
        if (!timing.asset_loading_included)
            cached = read_image(path);
        return measure(
            timing,
            [&]() {
                std::optional<Image> loaded;
                if (!cached)
                    loaded = read_image(path);
                const auto& image = cached ? *cached : *loaded;
                return task.run(trtmc::ImagesTextToTextRequest::from_parts(
                                    {trtmc::ImageInput{{image.pixels.data(), image.pixels.size()},
                                                       static_cast<std::uint32_t>(image.height),
                                                       static_cast<std::uint32_t>(image.width)},
                                     trtmc::TextPart{prompt}}),
                                config);
            },
            text_observation);
    }
    auto run = [&](const auto& task, const auto& input) {
        const auto config = sdk_config(request, task.config_fields(), {"prompt"});
        return measure(timing, [&]() { return task.run(input, config); }, text_observation);
    };
    if (primary == trtmc::TextContinuation::kTask ||
        (primary == trtmc::ImagesTextToText::kTask && model.supports<trtmc::TextContinuation>()))
        return run(model.task<trtmc::TextContinuation>(), trtmc::TextContinuationRequest{prompt});
    if (primary == trtmc::ConditionalTextGeneration::kTask)
        return run(model.task<trtmc::ConditionalTextGeneration>(),
                   trtmc::ConditionalTextGenerationRequest{prompt});
    if (primary == trtmc::CorruptedTextReconstruction::kTask)
        return run(model.task<trtmc::CorruptedTextReconstruction>(),
                   trtmc::CorruptedTextReconstructionRequest{prompt});
    if (primary == trtmc::TextSummarization::kTask)
        return run(model.task<trtmc::TextSummarization>(), trtmc::TextSummarizationRequest{prompt});
    throw std::invalid_argument("prompt alone is not a complete generate input for Task '" +
                                primary + "'");
}

struct ForecastHistory {
    std::vector<float> values;
    std::vector<std::uint8_t> mask;
    std::uint64_t rows{0}, columns{0};
    trtmc::SeriesHistory view() const {
        return {{{values.data(), values.size()}, rows, columns}, {mask.data(), mask.size()}};
    }
};

ForecastHistory forecast_history(const Json& request, bool shaped) {
    ForecastHistory history;
    const auto& values = request.at("past_values");
    if (!values.is_array() || values.empty())
        throw std::invalid_argument("past_values must be a nonempty array");
    if (request.contains("observed_mask")) {
        const auto& source = request.at("observed_mask");
        if (!source.is_array() || source.size() != values.size())
            throw std::invalid_argument("observed_mask length must match past_values");
        for (const auto& value : source) {
            if (!value.is_number() || (value != 0 && value != 1))
                throw std::invalid_argument("observed_mask must contain zero or one");
            history.mask.push_back(value == 0 ? 0 : 1);
        }
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i].is_null()) {
            if (history.mask.empty() || history.mask[i] != 0)
                throw std::invalid_argument("a null past value requires observed_mask zero");
            history.values.push_back(std::numeric_limits<float>::quiet_NaN());
        } else {
            if (!values[i].is_number())
                throw std::invalid_argument("past_values must contain numbers or masked nulls");
            const auto value = values[i].get<float>();
            if (!std::isfinite(value))
                throw std::invalid_argument("past_values must be finite or masked nulls");
            history.values.push_back(value);
        }
    }
    if (shaped) {
        const auto& shape = request.at("shape");
        if (!shape.is_array() || shape.size() != 2 || !shape[0].is_number_integer() ||
            !shape[1].is_number_integer() || shape[0] <= 0 || shape[1] <= 0)
            throw std::invalid_argument("shape must be positive [time,channel]");
        history.rows = shape[0].get<std::uint64_t>();
        history.columns = shape[1].get<std::uint64_t>();
        if (history.rows > history.values.size() ||
            history.columns > history.values.size() / history.rows ||
            history.rows * history.columns != history.values.size())
            throw std::invalid_argument("shape does not match past_values length");
    }
    return history;
}

Json run_regress(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto task = model.task<trtmc::SeriesToRegressionDistribution>();
    const auto history = forecast_history(request, request.contains("shape"));
    const auto config =
        sdk_config(request, task.config_fields(), {"past_values", "observed_mask", "shape"});
    return measure(
        timing, [&]() { return task.run({history.view()}, config); },
        [](const auto& result) {
            const auto& view = result.view();
            const char* distribution = nullptr;
            switch (view.distribution) {
            case TRTMC_DISTRIBUTION_NORMAL:
                distribution = "normal";
                break;
            case TRTMC_DISTRIBUTION_STUDENT_T:
                distribution = "student_t";
                break;
            case TRTMC_DISTRIBUTION_NEGATIVE_BINOMIAL:
                distribution = "negative_binomial";
                break;
            default:
                throw std::runtime_error("unknown regression distribution");
            }
            Json parameters = Json::array();
            std::uint64_t elements = 0;
            for (std::uint64_t i = 0; i < view.parameter_count; ++i) {
                const auto& parameter = view.parameters[i];
                parameters.push_back(
                    {{"name", std::string(trtmc::detail::string_view(parameter.name))},
                     {"values", json_values(parameter.values, parameter.target_count)}});
                elements += parameter.target_count;
            }
            return Json{{"distribution", distribution},
                        {"target_count", view.target_count},
                        {"regression_targets", view.target_count},
                        {"parameter_elements", elements},
                        {"axes", {"target"}},
                        {"parameters", std::move(parameters)},
                        {"target_names", json_strings(view.target_names)},
                        {"target_units", json_strings(view.target_units)}};
        });
}

void forecast_axes(Json& output, const trtmc_forecast_axes_v1& axes) {
    output["horizon_steps"] = Json::array();
    for (std::uint64_t i = 0; i < axes.horizon_steps.size; ++i)
        output["horizon_steps"].push_back(axes.horizon_steps.data[i]);
    output["channel_names"] = json_strings(axes.channel_names);
    output["channel_units"] = json_strings(axes.channel_units);
}

Json run_solve(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
    auto point = [](const trtmc_point_forecast_view_v1& view) {
        Json output{{"windows", 1},
                    {"forecast_elements", view.values.count},
                    {"shape", {view.values.rows, view.values.columns}},
                    {"axes", {"horizon", "channel"}},
                    {"values", json_values(view.values.data, view.values.count)}};
        forecast_axes(output, view.axes);
        return output;
    };
    auto quantiles = [](const trtmc_quantile_forecast_view_v1& view) {
        Json output{
            {"windows", 1},
            {"forecast_elements", view.value_count},
            {"shape", {view.quantile_levels.size, view.horizon, view.channels}},
            {"axes", {"quantile", "horizon", "channel"}},
            {"quantile_levels", json_values(view.quantile_levels.data, view.quantile_levels.size)},
            {"values", json_values(view.values, view.value_count)}};
        forecast_axes(output, view.axes);
        return output;
    };
    auto joint = [&](const trtmc_point_and_quantile_forecast_view_v1& view) {
        return Json{{"windows", 1},
                    {"forecast_elements", view.point.values.count + view.quantiles.value_count},
                    {"point", point(view.point)},
                    {"quantiles", quantiles(view.quantiles)}};
    };
    auto batch = [&](const auto& task, auto input, auto observe) {
        if (!request.is_object() || request.size() != 1 || !request.contains("items") ||
            !request["items"].is_array() || request["items"].empty())
            throw std::invalid_argument("batch forecast requires only a nonempty items array");
        const auto& items = request["items"];
        std::vector<ForecastHistory> histories;
        histories.reserve(items.size());
        input.items.reserve(items.size());
        const auto fields = task.config_fields();
        for (std::size_t i = 0; i < items.size(); ++i) {
            try {
                histories.push_back(forecast_history(items[i], true));
                auto config =
                    sdk_config(items[i], fields, {"past_values", "observed_mask", "shape"});
                input.items.push_back({{histories.back().view()}, std::move(config)});
            } catch (const Json::exception& error) {
                throw std::invalid_argument("batch item[" + std::to_string(i) +
                                            "]: " + error.what());
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument("batch item[" + std::to_string(i) +
                                            "]: " + error.what());
            }
        }
        return measure(
            timing, [&]() { return task.run(input); },
            [&](const auto& result) {
                Json outputs = Json::array();
                std::uint64_t elements = 0;
                for (std::uint64_t i = 0; i < result.size(); ++i) {
                    auto output = observe(result[i]);
                    elements += output.at("forecast_elements").template get<std::uint64_t>();
                    outputs.push_back(std::move(output));
                }
                return Json{{"windows", result.size()},
                            {"forecast_elements", elements},
                            {"items", std::move(outputs)}};
            });
    };
    if (primary == trtmc::BatchSeriesToPointForecast::kTask)
        return batch(model.task<trtmc::BatchSeriesToPointForecast>(),
                     trtmc::BatchSeriesToPointForecastRequest{}, point);
    if (primary == trtmc::BatchSeriesToQuantileForecast::kTask)
        return batch(model.task<trtmc::BatchSeriesToQuantileForecast>(),
                     trtmc::BatchSeriesToQuantileForecastRequest{}, quantiles);
    if (primary == trtmc::BatchSeriesToPointAndQuantileForecast::kTask)
        return batch(model.task<trtmc::BatchSeriesToPointAndQuantileForecast>(),
                     trtmc::BatchSeriesToPointAndQuantileForecastRequest{}, joint);
    if (request.contains("items"))
        throw std::invalid_argument("a forecast items array requires a native batch Task");
    const auto owned = forecast_history(request, request.contains("shape"));
    const auto history = owned.view();
    auto run = [&](const auto& task, const auto& input, auto observe) {
        const auto config =
            sdk_config(request, task.config_fields(), {"past_values", "observed_mask", "shape"});
        return measure(timing, [&]() { return task.run(input, config); }, observe);
    };
    if (primary == trtmc::SeriesToPointForecast::kTask)
        return run(model.task<trtmc::SeriesToPointForecast>(),
                   trtmc::SeriesToPointForecastRequest{history},
                   [&](const auto& result) { return point(result.view()); });
    if (primary == trtmc::SeriesToQuantileForecast::kTask)
        return run(model.task<trtmc::SeriesToQuantileForecast>(),
                   trtmc::SeriesToQuantileForecastRequest{history},
                   [&](const auto& result) { return quantiles(result.view()); });
    if (primary == trtmc::SeriesToPointAndQuantileForecast::kTask)
        return run(model.task<trtmc::SeriesToPointAndQuantileForecast>(),
                   trtmc::SeriesToPointAndQuantileForecastRequest{history},
                   [&](const auto& result) { return joint(result.view()); });
    throw std::invalid_argument("forecast history is not a complete solve input for Task '" +
                                primary + "'");
}

Json execute(const Json& request, const std::string& output_path) {
    if (request.at("schema_version").get<int>() != 2)
        throw std::invalid_argument("unsupported worker request schema");
    const std::string bundle = request.at("bundle").get<std::string>();
    const std::string runtime_root = request.value("runtime_root", std::string{});
    const std::string operation = request.at("operation").get<std::string>();
    Json operation_request = request.at("request");
    if (operation == "disparity") {
        std::filesystem::path artifact(output_path);
        artifact.replace_extension(".disparity.f32");
        operation_request["_artifact_path"] = artifact.string();
    }
    if (operation == "geometry") {
        std::filesystem::path artifact(output_path);
        artifact.replace_extension(".geometry");
        operation_request["_artifact_prefix"] = artifact.string();
    }
    if (operation == "predict_structure") {
        std::filesystem::path artifact(output_path);
        artifact.replace_extension(".structure");
        operation_request["_artifact_prefix"] = artifact.string();
    }
    const Timing timing = parse_timing(request.at("measurement"));

    const auto info = trtmc::Bundle::open(bundle).info();
    if (request.contains("expected_family") != request.contains("expected_task"))
        throw std::invalid_argument(
            "bundle identity requires both expected_family and expected_task");
    if (request.contains("expected_family")) {
        const auto family = request.at("expected_family").get<std::string>();
        const auto task = request.at("expected_task").get<std::string>();
        if (info.family != family || info.task != task)
            throw std::invalid_argument("bundle identity mismatch: expected family '" + family +
                                        "' and Task '" + task + "', found family '" + info.family +
                                        "' and Task '" + info.task +
                                        "'; rebuild the managed cache or select a matching bundle");
    }
    const auto& primary = info.task;
    double load_ms = 0;
    Json measured;
    if (!trtmc::app::uses_existing_task_runtime(primary)) {
        trtmc::LoadOptions options;
        options.runtime_root = runtime_root;
        const auto load_started = Clock::now();
        const auto model = trtmc::Model::load(bundle, options);
        load_ms = elapsed_ms(load_started);
        if (operation == "generate")
            measured = run_generate(model, operation_request, timing);
        else if (operation == "translate")
            measured = run_translate(model, operation_request, timing);
        else if (operation == "geometry")
            measured = run_geometry(model, operation_request, timing);
        else if (operation == "regress")
            measured = run_regress(model, operation_request, timing);
        else if (operation == "denoise" || operation == "decode_logits")
            measured =
                run_latent_step(model, operation_request, timing, operation == "decode_logits");
        else if (operation == "solve")
            measured = run_solve(model, operation_request, timing);
        else if (operation == "transcribe")
            measured = run_transcribe(model, operation_request, timing);
        else if (operation == "generate_audio")
            measured = run_generate_audio(model, operation_request, timing);
        else if (operation == "speak")
            measured = run_speak(model, operation_request, timing);
        else if (operation == "speech_dialogue")
            measured = run_speech_dialogue(model, operation_request, timing);
        else if (operation == "disparity")
            measured = run_disparity(model, operation_request, timing);
        else if (operation == "classify")
            measured = run_classify(model, operation_request, timing);
        else if (operation == "detect")
            measured = run_detect(model, operation_request, timing);
        else if (operation == "predict_structure")
            measured = run_structure(model, operation_request, timing);
        else if (operation == "extract_features")
            measured = run_extract_features(model, operation_request, timing);
        else if (operation == "encode")
            measured = run_encode(model, operation_request, timing);
        else if (operation == "embed")
            measured = run_embed(model, operation_request, timing);
        else if (operation == "rerank")
            measured = run_rerank(model, operation_request, timing);
        else if (operation == "control")
            measured = run_control(model, operation_request, timing);
        else if (operation == "control_queue")
            measured = run_control_queue(model, operation_request, timing);
        else if (operation == "track_masks")
            measured = run_track_masks(model, operation_request, timing);
        else if (operation == "refine_pose")
            measured = run_refine_pose(model, operation_request, timing);
        else if (operation == "track_pose")
            measured = run_track_pose(model, operation_request, timing);
        else if (operation == "segment" || operation == "segment_prompted")
            measured =
                run_segment(model, operation_request, timing, operation == "segment_prompted");
        else if (operation == "generate_image")
            measured = run_generate_image(model, operation_request, timing);
        else
            throw std::invalid_argument("semantic benchmark operation is not implemented: " +
                                        operation);
    } else {
        if (runtime_root.empty())
            throw std::invalid_argument("runtime_root is required for an existing bundle mode");
        const auto load_started = Clock::now();
        auto task = trtmc::load_task(bundle, runtime_root);
        load_ms = elapsed_ms(load_started);

        using Runner = Json (*)(trtmc::ITask&, const Json&, const Timing&);
        static const std::unordered_map<std::string, Runner> runners = {
            {"generate", run_generate},
            {"generate_image", run_generate_image},
            {"generate_audio", run_generate_audio},
            {"speak", run_speak},
            {"transcribe", run_transcribe},
            {"segment", run_segment},
            {"segment_prompted", run_segment_prompted},
            {"classify", run_classify},
            {"detect", run_detect},
            {"extract_features", run_extract_features},
            {"disparity", run_disparity},
            {"rerank", run_rerank},
            {"encode", run_encode},
            {"embed", run_embed},
            {"solve", run_solve},
            {"control", run_control},
        };
        const auto runner = runners.find(operation);
        if (runner == runners.end())
            throw std::invalid_argument("unsupported operation: " + operation);
        measured = runner->second(*task, operation_request, timing);
    }
    return {
        {"schema_version", "trtmc.benchmark-worker-result/v2"},
        {"status", "completed"},
        {"case_name", request.at("case_name")},
        {"operation", operation},
        {"task", primary},
        {"timing_scope", "public_task_call_wall"},
        {"observation_serialization_included", false},
        {"asset_loading_included", timing.asset_loading_included},
        {"load_ms", load_ms},
        {"warmup", timing.warmup},
        {"iterations", timing.iterations},
        {"observations", std::move(measured.at("observations"))},
        {"output_summary", std::move(measured.at("output_summary"))},
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string output_path;
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        output_path = arguments.output_path;
        write_json(output_path, execute(read_json(arguments.request_path), output_path));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "trtmc_benchmark_worker: " << error.what() << '\n';
        if (!output_path.empty()) {
            try {
                write_json(output_path, {{"schema_version", "trtmc.benchmark-worker-result/v2"},
                                         {"status", "failed"},
                                         {"error", error.what()}});
            } catch (...) {
            }
        }
        return 1;
    }
}
