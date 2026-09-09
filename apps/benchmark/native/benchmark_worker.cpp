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
#include "trtmc/task.h"
#include "trtmc/text.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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

Json text_observation(const trtmc::TextContinuationResult& result) {
    Json tokens = Json::array();
    for (auto id : result.token_ids())
        tokens.push_back(id);
    return {{"output_tokens", result.token_ids().size()},
            {"token_ids", std::move(tokens)},
            {"setup_ms", result.setup_ms()},
            {"prefill_ms", result.prefill_ms()},
            {"decode_ms", result.decode_ms()},
            {"text", std::string(result.text())}};
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

Json run_transcribe(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
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
            const auto matrix = result.actions();
            const auto& view = result.view();
            const auto& schema = view.actions.schema;
            auto spans = Json::array();
            for (std::uint64_t i = 0; i < view.actions.frame_span_count; ++i)
                spans.push_back({{"begin", view.actions.frame_spans[i].begin},
                                 {"end", view.actions.frame_spans[i].end}});
            return Json{{"action_steps", matrix.rows},
                        {"action_dim", matrix.columns},
                        {"action_values", matrix.values.size()},
                        {"actions", json_values(matrix.values)},
                        {"within_training_bounds", result.within_training_bounds()},
                        {"inference_ms", result.inference_ms()},
                        {"axes", {"step", "action_component"}},
                        {"schema",
                         {{"domain", std::string(trtmc::detail::string_view(schema.domain))},
                          {"component_names", json_strings(schema.component_names)},
                          {"units", json_strings(schema.units)},
                          {"coordinate_frame",
                           std::string(trtmc::detail::string_view(schema.coordinate_frame))},
                          {"normalization",
                           std::string(trtmc::detail::string_view(schema.normalization))}}},
                        {"timestamps_seconds", json_values(view.actions.timestamps_seconds.data,
                                                           view.actions.timestamps_seconds.size)},
                        {"frame_spans", std::move(spans)}};
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

Json run_generate(const trtmc::Model& model, const Json& request, const Timing& timing) {
    const auto primary = model.info().bundle_task;
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
    const Timing timing = parse_timing(request.at("measurement"));

    const auto primary = trtmc::Bundle::open(bundle).info().task;
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
        else if (operation == "solve")
            measured = run_solve(model, operation_request, timing);
        else if (operation == "transcribe")
            measured = run_transcribe(model, operation_request, timing);
        else if (operation == "generate_audio")
            measured = run_generate_audio(model, operation_request, timing);
        else if (operation == "speak")
            measured = run_speak(model, operation_request, timing);
        else if (operation == "disparity")
            measured = run_disparity(model, operation_request, timing);
        else if (operation == "classify")
            measured = run_classify(model, operation_request, timing);
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
