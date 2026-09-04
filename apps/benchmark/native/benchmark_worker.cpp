/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
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

struct Image {
    std::vector<float> pixels;
    std::int32_t height{0};
    std::int32_t width{0};
};

struct Audio {
    std::vector<float> samples;
    std::int32_t sample_rate{0};
};

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
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
        throw std::runtime_error("cannot read image " + path);
    Image image;
    image.width = width;
    image.height = height;
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    image.pixels.resize(count);
    for (std::size_t index = 0; index < count; ++index)
        image.pixels[index] = static_cast<float>(raw[index]) / 255.0F;
    stbi_image_free(raw);
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

template <typename T>
T read_scalar(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input)
        throw std::runtime_error("truncated WAV file");
    return value;
}

Audio read_wav(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read audio " + path);
    char riff[4]{};
    input.read(riff, 4);
    if (std::string(riff, 4) != "RIFF")
        throw std::runtime_error("audio input must be RIFF WAV");
    (void)read_scalar<std::uint32_t>(input);
    char wave[4]{};
    input.read(wave, 4);
    if (std::string(wave, 4) != "WAVE")
        throw std::runtime_error("audio input must be WAVE");

    std::uint16_t format = 0;
    std::uint16_t channels = 0;
    std::uint32_t sample_rate = 0;
    std::uint16_t bits = 0;
    std::vector<char> data;
    while (input) {
        char id[4]{};
        input.read(id, 4);
        if (!input)
            break;
        const std::uint32_t size = read_scalar<std::uint32_t>(input);
        if (std::string(id, 4) == "fmt ") {
            format = read_scalar<std::uint16_t>(input);
            channels = read_scalar<std::uint16_t>(input);
            sample_rate = read_scalar<std::uint32_t>(input);
            input.seekg(6, std::ios::cur);
            bits = read_scalar<std::uint16_t>(input);
            if (size > 16)
                input.seekg(size - 16, std::ios::cur);
        } else if (std::string(id, 4) == "data") {
            data.resize(size);
            input.read(data.data(), static_cast<std::streamsize>(size));
        } else {
            input.seekg(size, std::ios::cur);
        }
        if (size % 2U != 0U)
            input.seekg(1, std::ios::cur);
    }
    if (channels == 0 || sample_rate == 0 || data.empty())
        throw std::runtime_error("WAV file is missing format or sample data");

    Audio audio;
    audio.sample_rate = static_cast<std::int32_t>(sample_rate);
    if (format == 1 && bits == 16) {
        const auto* source = reinterpret_cast<const std::int16_t*>(data.data());
        const std::size_t frames = data.size() / (sizeof(std::int16_t) * channels);
        audio.samples.resize(frames);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0F;
            for (std::uint16_t channel = 0; channel < channels; ++channel)
                sum += static_cast<float>(source[frame * channels + channel]) / 32768.0F;
            audio.samples[frame] = sum / static_cast<float>(channels);
        }
    } else if (format == 3 && bits == 32) {
        const auto* source = reinterpret_cast<const float*>(data.data());
        const std::size_t frames = data.size() / (sizeof(float) * channels);
        audio.samples.resize(frames);
        for (std::size_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0F;
            for (std::uint16_t channel = 0; channel < channels; ++channel)
                sum += source[frame * channels + channel];
            audio.samples[frame] = sum / static_cast<float>(channels);
        }
    } else {
        throw std::runtime_error("only PCM16 and float32 WAV inputs are supported");
    }
    return audio;
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
        const auto started = Clock::now();
        last = invoke();
        Json observation = observe(*last);
        observation["runtime_e2e_wall_ms"] = elapsed_ms(started);
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
        const auto started = Clock::now();
        last = invoke();
        observations.push_back({{"runtime_e2e_wall_ms", elapsed_ms(started)},
                                {"stereo_pairs", 1},
                                {"disparity_pixels", last.disparity.size()}});
    }
    const std::string artifact = request.at("_artifact_path").get<std::string>();
    std::ofstream output(artifact, std::ios::binary);
    output.write(reinterpret_cast<const char*>(last.disparity.data()),
                 static_cast<std::streamsize>(last.disparity.size() * sizeof(float)));
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

Json execute(const Json& request, const std::string& output_path) {
    if (request.at("schema_version").get<int>() != 2)
        throw std::invalid_argument("unsupported worker request schema");
    const std::string bundle = request.at("bundle").get<std::string>();
    const std::string runtime_root = request.at("runtime_root").get<std::string>();
    const std::string operation = request.at("operation").get<std::string>();
    Json operation_request = request.at("request");
    if (operation == "disparity") {
        std::filesystem::path artifact(output_path);
        artifact.replace_extension(".disparity.f32");
        operation_request["_artifact_path"] = artifact.string();
    }
    const Timing timing = parse_timing(request.at("measurement"));

    const auto load_started = Clock::now();
    auto task = trtmc::load_task(bundle, runtime_root);
    const double load_ms = elapsed_ms(load_started);

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
    Json measured = runner->second(*task, operation_request, timing);
    return {
        {"schema_version", "trtmc.benchmark-worker-result/v2"},
        {"status", "completed"},
        {"case_name", request.at("case_name")},
        {"operation", operation},
        {"timing_scope", "public_task_call_wall"},
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
