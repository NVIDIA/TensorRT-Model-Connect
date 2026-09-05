/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"

#include "cli/io.h"
#include "trtmc/runtime/family_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc::cli {

namespace {

namespace fs = std::filesystem;

struct CommandSpec {
    CommandKind kind;
    std::unordered_set<std::string> options;
};

const std::unordered_map<std::string, CommandSpec>& command_specs() {
    static const std::unordered_map<std::string, CommandSpec> specs{
        {"run",
         {CommandKind::kRun,
          {"--prompt",
           "--image",
           "--max-new-tokens",
           "--source-language-token-id",
           "--forced-bos-token-id",
           "--temperature",
           "--top-k",
           "--top-p",
           "--min-p",
           "--seed",
           "--repetition-penalty",
           "--generation-mode",
           "--block-length",
           "--threshold",
           "--use-chat-template",
           "--enable-thinking",
           "--lora-adapter",
           "--lora-adapter-id",
           "--num-steps",
           "--guidance-scale",
           "--cfg-scale",
           "--sde-gamma",
           "--initial-latents-raw",
           "--condition-latents-raw",
           "--condition-mask-raw",
           "--sampling-steps-raw",
           "--sde-noise-raw"}}},
        {"encode", {CommandKind::kEncode, {"--text"}}},
        {"embed", {CommandKind::kEmbed, {"--text"}}},
        {"rerank", {CommandKind::kRerank, {"--query", "--document"}}},
        {"classify", {CommandKind::kClassify, {"--image"}}},
        {"extract-features", {CommandKind::kExtractFeatures, {"--image"}}},
        {"disparity", {CommandKind::kDisparity, {"--left", "--right"}}},
        {"geometry", {CommandKind::kGeometry, {"--image", "--output"}}},
        {"segment", {CommandKind::kSegment, {"--image"}}},
        {"segment-prompted",
         {CommandKind::kSegmentPrompted,
          {"--image", "--prompt", "--point-x", "--point-y", "--foreground"}}},
        {"video-segment", {CommandKind::kVideoSegment, {"--frame", "--prompt"}}},
        {"generate-audio",
         {CommandKind::kGenerateAudio,
          {"--prompt", "--output", "--max-new-tokens", "--talker-max-new-tokens", "--seed",
           "--stream", "--chunk-frames"}}},
        {"transcribe",
         {CommandKind::kTranscribe,
          {"--input", "--max-output-tokens", "--source-language", "--target-language",
           "--translate", "--beam-size", "--length-penalty", "--punctuation", "--timestamps",
           "--max-input-seconds", "--segment-length-seconds", "--segment-min-seconds",
           "--segment-overlap-seconds", "--lcs-merge"}}},
        {"transcribe-batch",
         {CommandKind::kTranscribeBatch,
          {"--input", "--max-output-tokens", "--source-language", "--target-language",
           "--translate", "--beam-size", "--length-penalty", "--punctuation", "--timestamps",
           "--max-input-seconds", "--segment-length-seconds", "--segment-min-seconds",
           "--segment-overlap-seconds", "--lcs-merge"}}},
        {"transcribe-streaming",
         {CommandKind::kTranscribeStreaming,
          {"--input", "--chunk-samples", "--max-new-tokens", "--att-context-left",
           "--att-context-right", "--language"}}},
        {"speak",
         {CommandKind::kSpeak,
          {"--input", "--output", "--max-new-tokens", "--seed", "--tail-frames"}}},
        {"speech-session",
         {CommandKind::kSpeechSession, {"--input", "--output", "--system-prompt", "--timeout-ms"}}},
        {"generate-image",
         {CommandKind::kGenerateImage,
          {"--prompt", "--image", "--output", "--negative-prompt", "--height", "--width",
           "--num-steps", "--seed", "--guidance-scale", "--cfg-scale"}}},
        {"generate-image-batch",
         {CommandKind::kGenerateImageBatch,
          {"--prompts", "--seeds", "--output", "--negative-prompt", "--height", "--width",
           "--num-steps", "--guidance-scale", "--cfg-scale"}}},
        {"generate-video",
         {CommandKind::kGenerateVideo,
          {"--prompt", "--image", "--output", "--negative-prompt", "--height", "--width",
           "--num-steps", "--seed", "--guidance-scale", "--cfg-scale"}}},
        {"solve", {CommandKind::kSolve, {"--branch", "--trunk"}}},
        {"forecast", {CommandKind::kForecast, {"--input", "--mask", "--frequency"}}},
        {"control", {CommandKind::kControl, {"--image", "--state", "--output"}}},
        {"generate-world",
         {CommandKind::kGenerateWorld,
          {"--prompt", "--image", "--action", "--intrinsics", "--num-frames", "--output",
           "--height", "--width", "--num-steps", "--seed", "--guidance-scale", "--cfg-scale"}}},
    };
    return specs;
}

bool is_byok_option(const std::string& option) {
    return option == "--byok-library" || option == "--byok-function" || option == "--byok-name";
}

void load_byok_extension(const Command& command) {
    using LoadKernelFn = const char* (*)(const char*, const char*, const char*) noexcept;
    const fs::path extension = fs::path(command.runtime_root) / "libtrtmc_byok_tvm_ffi.so";
    dlerror();
    void* handle = dlopen(extension.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("unable to load BYOK extension '" + extension.string() +
                                 "': " + (error != nullptr ? error : "unknown dlopen error"));
    }
    static auto* handles = new std::vector<void*>;
    handles->push_back(handle);
    dlerror();
    auto load = reinterpret_cast<LoadKernelFn>(dlsym(handle, "trtmc_load_byok_kernel"));
    if (const char* error = dlerror(); error != nullptr || load == nullptr) {
        throw std::runtime_error("BYOK extension is missing trtmc_load_byok_kernel");
    }
    if (const char* error = load(command.options.at("--byok-library").c_str(),
                                 command.options.at("--byok-function").c_str(),
                                 command.options.at("--byok-name").c_str())) {
        const std::string message = error;
        throw std::runtime_error(message);
    }
}

std::string take_value(int argc, char** argv, int& index, const std::string& option) {
    if (index + 1 >= argc)
        throw std::invalid_argument(option + " requires a value");
    ++index;
    std::string value = argv[index];
    if (value.empty())
        throw std::invalid_argument(option + " requires a non-empty value");
    return value;
}

std::uint64_t parse_byte_size(const std::string& text) {
    std::uint64_t multiplier = 1;
    std::string number = text;
    if (text.size() > 3 && text.compare(text.size() - 3, 3, "GiB") == 0) {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
        number.resize(number.size() - 3);
    } else if (text.size() > 2 && text.compare(text.size() - 2, 2, "GB") == 0) {
        multiplier = 1000ULL * 1000ULL * 1000ULL;
        number.resize(number.size() - 2);
    }
    if (number.empty() || !std::all_of(number.begin(), number.end(), [](unsigned char value) {
            return value >= '0' && value <= '9';
        })) {
        throw std::invalid_argument(
            "--kv-cache-size must be positive integer bytes or a value like 1GB or 1GiB");
    }
    std::size_t consumed = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(number, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument("--kv-cache-size is outside its valid range");
    }
    if (value == 0 || consumed != number.size() ||
        value > std::numeric_limits<std::uint64_t>::max() / multiplier) {
        throw std::invalid_argument("--kv-cache-size is outside its valid range");
    }
    return value * multiplier;
}

std::string require_option(const Command& command, const std::string& option) {
    const auto found = command.options.find(option);
    if (found == command.options.end())
        throw std::invalid_argument(command.name + " requires " + option);
    return found->second;
}

bool has_option(const Command& command, const std::string& option) {
    return command.options.find(option) != command.options.end();
}

std::int32_t parse_int32(const std::string& value, const std::string& option,
                         std::int32_t minimum = std::numeric_limits<std::int32_t>::min()) {
    std::size_t consumed = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " must be an integer");
    }
    if (consumed != value.size() || parsed < minimum ||
        parsed > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(option + " is outside its valid integer range");
    }
    return static_cast<std::int32_t>(parsed);
}

float parse_float(const std::string& value, const std::string& option) {
    std::size_t consumed = 0;
    float parsed = 0.0F;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " must be a number");
    }
    if (consumed != value.size() || !std::isfinite(parsed))
        throw std::invalid_argument(option + " must be a finite number");
    return parsed;
}

bool parse_bool(const std::string& value, const std::string& option) {
    if (value == "true")
        return true;
    if (value == "false")
        return false;
    throw std::invalid_argument(option + " must be true or false");
}

std::int32_t int_option(const Command& command, const std::string& option,
                        std::int32_t default_value,
                        std::int32_t minimum = std::numeric_limits<std::int32_t>::min()) {
    const auto found = command.options.find(option);
    if (found == command.options.end())
        return default_value;
    return parse_int32(found->second, option, minimum);
}

float float_option(const Command& command, const std::string& option, float default_value) {
    const auto found = command.options.find(option);
    if (found == command.options.end())
        return default_value;
    return parse_float(found->second, option);
}

std::int32_t checked_count(std::size_t count, const std::string& field) {
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        throw std::runtime_error(field + " has too many elements");
    return static_cast<std::int32_t>(count);
}

std::vector<float> read_float32_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("unable to open float32 file: " + path);
    const std::streampos end = input.tellg();
    if (end <= 0)
        throw std::runtime_error("float32 file is empty: " + path);
    const auto bytes = static_cast<std::uint64_t>(end);
    if (bytes % sizeof(float) != 0)
        throw std::runtime_error("float32 file size is not a multiple of 4 bytes: " + path);
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        bytes / sizeof(float) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("float32 file is too large: " + path);
    }
    std::vector<float> values(static_cast<std::size_t>(bytes / sizeof(float)));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (input.gcount() != static_cast<std::streamsize>(bytes))
        throw std::runtime_error("float32 file ended while reading: " + path);
    for (const float value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error("float32 file contains a non-finite value: " + path);
    }
    return values;
}

std::vector<std::string> read_nonempty_lines(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("unable to open text file: " + path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) {
        if (!line.empty())
            lines.push_back(std::move(line));
    }
    if (lines.empty())
        throw std::runtime_error("text file has no non-empty lines: " + path);
    return lines;
}

std::vector<std::uint32_t> parse_seeds(const std::string& text) {
    std::vector<std::uint32_t> seeds;
    std::istringstream input(text);
    for (std::string token; std::getline(input, token, ',');) {
        std::size_t consumed = 0;
        unsigned long long value = 0;
        try {
            value = std::stoull(token, &consumed);
        } catch (const std::exception&) {
            throw std::invalid_argument("--seeds must be comma-separated uint32 values");
        }
        if (consumed != token.size() || value > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("--seeds must be comma-separated uint32 values");
        seeds.push_back(static_cast<std::uint32_t>(value));
    }
    if (seeds.empty())
        throw std::invalid_argument("--seeds must not be empty");
    return seeds;
}

io::LoadedImage read_image(const std::string& path) {
    io::LoadedImage image = io::read_image(path);
    if (image.empty())
        throw std::runtime_error("unable to decode image: " + path);
    return image;
}

AudioResult read_audio(const std::string& path) {
    AudioResult audio = io::read_wav(path);
    if (audio.samples.empty() || audio.sample_rate <= 0)
        throw std::runtime_error("WAV contains no audio: " + path);
    return audio;
}

template <typename Interface>
Interface& require_interface(ITask& task) {
    auto* interface = dynamic_cast<Interface*>(&task);
    if (interface == nullptr)
        throw std::runtime_error("loaded family does not implement the requested Task API");
    return *interface;
}

void require_finite(const std::vector<float>& values, const std::string& field) {
    for (const float value : values) {
        if (!std::isfinite(value))
            throw std::runtime_error(field + " contains a non-finite value");
    }
}

void write_json(std::ostream& output, const nlohmann::json& value) {
    output << value.dump() << '\n';
}

template <typename Value>
void write_binary(const fs::path& path, const std::vector<Value>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create output: " + path.string());
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(Value)));
    }
    if (!output)
        throw std::runtime_error("failed to write output: " + path.string());
}

nlohmann::json text_json(const TextResult& result) {
    nlohmann::json value{{"text", result.text},
                         {"token_ids", result.token_ids},
                         {"setup_ms", result.setup_ms},
                         {"prefill_ms", result.prefill_ms},
                         {"decode_ms", result.decode_ms}};
    if (!result.segments.empty()) {
        value["segments"] = nlohmann::json::array();
        for (const auto& segment : result.segments) {
            value["segments"].push_back({{"start_seconds", segment.start_seconds},
                                         {"end_seconds", segment.end_seconds},
                                         {"text", segment.text},
                                         {"token_ids", segment.token_ids}});
        }
    }
    return value;
}

nlohmann::json stream_result_json(const TranscriptionStreamResult& result) {
    return {{"text", result.text},
            {"token_ids", result.token_ids},
            {"is_final", result.is_final},
            {"chunk_index", result.chunk_index},
            {"accepted_samples", result.accepted_samples},
            {"sample_rate", result.sample_rate}};
}

nlohmann::json embedding_json(const EmbeddingResult& result) {
    require_finite(result.data, "embedding");
    return {{"dim", result.dim}, {"values", result.data}};
}

TranscriptionConfig transcription_config(const Command& command, std::int32_t input_sample_rate) {
    TranscriptionConfig config;
    config.input_sample_rate = input_sample_rate;
    config.max_output_tokens =
        int_option(command, "--max-output-tokens", config.max_output_tokens, 1);
    config.beam_size = int_option(command, "--beam-size", config.beam_size, 1);
    config.length_penalty = float_option(command, "--length-penalty", config.length_penalty);
    if (has_option(command, "--source-language"))
        config.source_language = command.options.at("--source-language");
    if (has_option(command, "--target-language"))
        config.target_language = command.options.at("--target-language");
    if (has_option(command, "--translate") &&
        parse_bool(command.options.at("--translate"), "--translate")) {
        config.task = TranscriptionTask::kTranslate;
    }
    if (has_option(command, "--punctuation"))
        config.punctuation = parse_bool(command.options.at("--punctuation"), "--punctuation");
    if (has_option(command, "--timestamps"))
        config.timestamps = parse_bool(command.options.at("--timestamps"), "--timestamps");
    config.max_input_duration_seconds =
        float_option(command, "--max-input-seconds", config.max_input_duration_seconds);
    config.segment_duration_seconds =
        float_option(command, "--segment-length-seconds", config.segment_duration_seconds);
    config.segment_min_duration_seconds =
        float_option(command, "--segment-min-seconds", config.segment_min_duration_seconds);
    config.segment_overlap_seconds =
        float_option(command, "--segment-overlap-seconds", config.segment_overlap_seconds);
    if (has_option(command, "--lcs-merge"))
        config.lcs_merge = parse_bool(command.options.at("--lcs-merge"), "--lcs-merge");
    if (config.length_penalty < 0.0F)
        throw std::invalid_argument("--length-penalty must be non-negative");
    if (has_option(command, "--max-input-seconds") && config.max_input_duration_seconds <= 0.0F)
        throw std::invalid_argument("--max-input-seconds must be positive");
    if (has_option(command, "--segment-length-seconds") && config.segment_duration_seconds <= 0.0F)
        throw std::invalid_argument("--segment-length-seconds must be positive");
    if (has_option(command, "--segment-min-seconds") &&
        config.segment_min_duration_seconds <= 0.0F) {
        throw std::invalid_argument("--segment-min-seconds must be positive");
    }
    if (config.segment_overlap_seconds < 0.0F)
        throw std::invalid_argument("--segment-overlap-seconds must be non-negative");
    return config;
}

ImageGenerationConfig image_config(const Command& command) {
    ImageGenerationConfig config;
    if (has_option(command, "--negative-prompt"))
        config.negative_prompt = command.options.at("--negative-prompt");
    config.height = int_option(command, "--height", 0, 1);
    config.width = int_option(command, "--width", 0, 1);
    config.num_steps = int_option(command, "--num-steps", -1, 1);
    config.seed = int_option(command, "--seed", -1);
    config.guidance_scale = float_option(command, "--guidance-scale", -1.0F);
    config.cfg_scale = float_option(command, "--cfg-scale", -1.0F);
    if (has_option(command, "--guidance-scale") && config.guidance_scale < 0.0F)
        throw std::invalid_argument("--guidance-scale must be non-negative");
    if (has_option(command, "--cfg-scale") && config.cfg_scale < 0.0F)
        throw std::invalid_argument("--cfg-scale must be non-negative");
    return config;
}

void validate_image_result(const ImageResult& result) {
    if (result.height <= 0 || result.width <= 0 || result.channels != 3 || result.num_frames <= 0) {
        throw std::runtime_error("image result must be non-empty RGB HWC/THWC data");
    }
    const auto frame_size =
        static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width) * 3U;
    const auto expected = frame_size * static_cast<std::size_t>(result.num_frames);
    if (result.pixels.size() != expected)
        throw std::runtime_error("image result size does not match its dimensions");
    require_finite(result.pixels, "image result");
}

nlohmann::json write_image(const ImageResult& result, const std::string& path) {
    if (result.pixels.empty() && result.height == 0 && result.width == 0 && result.channels == 3 &&
        result.num_frames == 0) {
        return {{"worker", true}};
    }
    validate_image_result(result);
    if (result.num_frames != 1)
        throw std::runtime_error("generate-image received multiple frames; use generate-video");
    io::save_png(result, path);
    return {{"output", path}, {"height", result.height}, {"width", result.width}};
}

nlohmann::json write_video(const ImageResult& result, const std::string& directory) {
    if (result.pixels.empty() && result.height == 0 && result.width == 0 && result.channels == 3 &&
        result.num_frames == 0) {
        return {{"worker", true}};
    }
    validate_image_result(result);
    fs::create_directories(directory);
    const auto frame_size =
        static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width) * 3U;
    nlohmann::json files = nlohmann::json::array();
    for (std::int32_t frame = 0; frame < result.num_frames; ++frame) {
        std::ostringstream filename;
        filename << "frame-" << std::setw(6) << std::setfill('0') << frame << ".png";
        const fs::path path = fs::path(directory) / filename.str();
        const auto offset = frame_size * static_cast<std::size_t>(frame);
        const auto begin = result.pixels.begin() + static_cast<std::ptrdiff_t>(offset);
        std::vector<float> pixels(begin, begin + static_cast<std::ptrdiff_t>(frame_size));
        io::save_png(path.string(), pixels, result.width, result.height);
        files.push_back(path.string());
    }
    return {{"output", directory},
            {"frames", std::move(files)},
            {"height", result.height},
            {"width", result.width}};
}

ImageResult generate_image(const Command& command, ITask& task) {
    const std::string prompt = require_option(command, "--prompt");
    const ImageGenerationConfig config = image_config(command);
    if (has_option(command, "--image")) {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        return require_interface<IImageEditing>(task).generate_image(
            prompt, image.pixels.data(), image.height, image.width, config);
    }
    return require_interface<IImageGeneration>(task).generate_image(prompt, config);
}

const char* event_kind_name(SpeechSessionEventKind kind) {
    switch (kind) {
    case SpeechSessionEventKind::kAgentAudio:
        return "agent_audio";
    case SpeechSessionEventKind::kAgentText:
        return "agent_text";
    case SpeechSessionEventKind::kUserTranscript:
        return "user_transcript";
    case SpeechSessionEventKind::kTurnStarted:
        return "turn_started";
    case SpeechSessionEventKind::kTurnFinished:
        return "turn_finished";
    case SpeechSessionEventKind::kYielded:
        return "yielded";
    case SpeechSessionEventKind::kCancelled:
        return "cancelled";
    case SpeechSessionEventKind::kReset:
        return "reset";
    case SpeechSessionEventKind::kError:
        return "error";
    case SpeechSessionEventKind::kInputFinished:
        return "input_finished";
    case SpeechSessionEventKind::kUserSpeechStarted:
        return "user_speech_started";
    case SpeechSessionEventKind::kUserSpeechStopped:
        return "user_speech_stopped";
    case SpeechSessionEventKind::kFunctionCall:
        return "function_call";
    case SpeechSessionEventKind::kFunctionCallStarted:
        return "function_call_started";
    case SpeechSessionEventKind::kFunctionResponseFinished:
        return "function_response_finished";
    case SpeechSessionEventKind::kInputCleared:
        return "input_cleared";
    }
    throw std::logic_error("unknown speech event kind");
}

int dispatch_run(const Command& command, ITask& task, std::ostream& output) {
    if (!has_option(command, "--prompt") && !has_option(command, "--initial-latents-raw")) {
        throw std::invalid_argument("run requires --prompt or --initial-latents-raw");
    }
    const std::string prompt =
        has_option(command, "--prompt") ? command.options.at("--prompt") : std::string{};
    TextGenerationConfig config;
    config.source_language_token_id =
        int_option(command, "--source-language-token-id", config.source_language_token_id, 0);
    config.forced_bos_token_id =
        int_option(command, "--forced-bos-token-id", config.forced_bos_token_id, 0);
    config.temperature = float_option(command, "--temperature", 1.0F);
    config.top_k = int_option(command, "--top-k", 1, 0);
    config.top_p = float_option(command, "--top-p", 1.0F);
    config.min_p = float_option(command, "--min-p", 0.0F);
    config.seed = int_option(command, "--seed", -1);
    config.repetition_penalty = float_option(command, "--repetition-penalty", 1.0F);
    if (has_option(command, "--generation-mode"))
        config.text_generation_mode = command.options.at("--generation-mode");
    config.block_length = int_option(command, "--block-length", 0);
    config.confidence_threshold = float_option(command, "--threshold", -1.0F);
    config.num_steps = int_option(command, "--num-steps", -1, 1);
    config.guidance_scale = float_option(command, "--guidance-scale", -1.0F);
    config.cfg_scale = float_option(command, "--cfg-scale", -1.0F);
    config.sde_gamma = float_option(command, "--sde-gamma", -1.0F);
    if (config.temperature < 0.0F)
        throw std::invalid_argument("--temperature must be non-negative");
    if (config.top_p < 0.0F || config.top_p > 1.0F)
        throw std::invalid_argument("--top-p must be in [0, 1]");
    if (config.min_p < 0.0F || config.min_p > 1.0F)
        throw std::invalid_argument("--min-p must be in [0, 1]");
    if (config.repetition_penalty <= 0.0F)
        throw std::invalid_argument("--repetition-penalty must be positive");
    if (has_option(command, "--guidance-scale") && config.guidance_scale < 0.0F)
        throw std::invalid_argument("--guidance-scale must be non-negative");
    if (has_option(command, "--cfg-scale") && config.cfg_scale < 0.0F)
        throw std::invalid_argument("--cfg-scale must be non-negative");
    if (has_option(command, "--sde-gamma") && config.sde_gamma < 0.0F)
        throw std::invalid_argument("--sde-gamma must be non-negative");
    if (has_option(command, "--initial-latents-raw"))
        config.initial_latents = read_float32_file(command.options.at("--initial-latents-raw"));
    if (has_option(command, "--condition-latents-raw")) {
        config.condition_latents = read_float32_file(command.options.at("--condition-latents-raw"));
    }
    if (has_option(command, "--condition-mask-raw"))
        config.condition_mask = read_float32_file(command.options.at("--condition-mask-raw"));
    if (has_option(command, "--sampling-steps-raw"))
        config.sampling_steps = read_float32_file(command.options.at("--sampling-steps-raw"));
    if (has_option(command, "--sde-noise-raw"))
        config.sde_noises = read_float32_file(command.options.at("--sde-noise-raw"));
    if (has_option(command, "--use-chat-template"))
        config.use_chat_template =
            parse_bool(command.options.at("--use-chat-template"), "--use-chat-template");
    if (has_option(command, "--enable-thinking"))
        config.enable_thinking =
            parse_bool(command.options.at("--enable-thinking"), "--enable-thinking");
    const bool has_lora_path = has_option(command, "--lora-adapter");
    const bool has_lora_id = has_option(command, "--lora-adapter-id");
    if (has_lora_path != has_lora_id)
        throw std::invalid_argument("--lora-adapter and --lora-adapter-id must be used together");
    if (has_lora_path) {
        config.lora_adapter_id = command.options.at("--lora-adapter-id");
        require_interface<ILoraAdapterManager>(task).load_lora_adapter(
            config.lora_adapter_id, command.options.at("--lora-adapter"));
    }
    if (!has_option(command, "--image")) {
        auto& interface = require_interface<ITextGeneration>(task);
        config.max_new_tokens =
            int_option(command, "--max-new-tokens", interface.default_max_new_tokens(), 1);
        write_json(output, text_json(interface.generate(prompt, config)));
        return EXIT_SUCCESS;
    }
    const io::LoadedImage image = read_image(require_option(command, "--image"));
    config.max_new_tokens = int_option(command, "--max-new-tokens", 128, 1);
    auto& interface = require_interface<IVisionLanguageGeneration>(task);
    write_json(output, text_json(interface.generate(prompt, image.pixels.data(), image.height,
                                                    image.width, config)));
    return EXIT_SUCCESS;
}

} // namespace

Command parse_args(int argc, char** argv) {
    if (argc < 2)
        throw std::invalid_argument("a command is required");
    const std::string name = argv[1];
    if (name == "help") {
        if (argc != 2)
            throw std::invalid_argument("help does not accept arguments");
        return {CommandKind::kHelp, name, {}, {}, {}, {}, {}, 0, {}, false};
    }
    if (name == "version") {
        if (argc != 2)
            throw std::invalid_argument("version does not accept arguments");
        return {CommandKind::kVersion, name, {}, {}, {}, {}, {}, 0, {}, false};
    }
    if (name == "inspect") {
        if (argc != 3 || std::string(argv[2]).empty())
            throw std::invalid_argument("inspect requires exactly one BUNDLE path");
        return {CommandKind::kInspect, name, argv[2], {}, {}, {}, {}, 0, {}, false};
    }

    const auto spec = command_specs().find(name);
    if (spec == command_specs().end())
        throw std::invalid_argument("unknown command: " + name);
    if (argc < 3 || std::string(argv[2]).empty())
        throw std::invalid_argument(name + " requires a BUNDLE path");

    Command command{spec->second.kind, name, argv[2], {}, {}, {}, {}, 0, {}, false};
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--runtime-root") {
            if (!command.runtime_root.empty())
                throw std::invalid_argument("--runtime-root may be specified only once");
            command.runtime_root = take_value(argc, argv, index, option);
            continue;
        }
        if (option == "--kv-cache-size") {
            if (command.kv_cache_size_bytes != 0)
                throw std::invalid_argument("--kv-cache-size may be specified only once");
            command.kv_cache_size_bytes = parse_byte_size(take_value(argc, argv, index, option));
            continue;
        }
        if (option == "--runtime-cache") {
            if (!command.runtime_cache_path.empty())
                throw std::invalid_argument("--runtime-cache may be specified only once");
            command.runtime_cache_path = take_value(argc, argv, index, option);
            continue;
        }
        if (option == "--cuda-graphs") {
            if (command.cuda_graphs)
                throw std::invalid_argument("--cuda-graphs may be specified only once");
            command.cuda_graphs = true;
            continue;
        }
        if (!is_byok_option(option) && spec->second.options.count(option) == 0)
            throw std::invalid_argument("unknown argument for " + name + ": " + option);
        if (option == "--frame") {
            command.frames.push_back(take_value(argc, argv, index, option));
            continue;
        }
        if (option == "--input" && command.kind == CommandKind::kTranscribeBatch) {
            command.inputs.push_back(take_value(argc, argv, index, option));
            continue;
        }
        if (command.options.count(option) != 0)
            throw std::invalid_argument(option + " may be specified only once");
        command.options.emplace(option, take_value(argc, argv, index, option));
    }
    if (command.runtime_root.empty())
        throw std::invalid_argument("--runtime-root is required for " + name);
    const int byok_option_count = static_cast<int>(command.options.count("--byok-library") +
                                                   command.options.count("--byok-function") +
                                                   command.options.count("--byok-name"));
    if (byok_option_count != 0 && byok_option_count != 3) {
        throw std::invalid_argument(
            "--byok-library, --byok-function, and --byok-name must be used together");
    }
    return command;
}

int dispatch(const Command& command, ITask& task, std::ostream& output) {
    switch (command.kind) {
    case CommandKind::kRun:
        return dispatch_run(command, task, output);
    case CommandKind::kEncode: {
        const auto result =
            require_interface<IEncoding>(task).encode(require_option(command, "--text"));
        write_json(output, embedding_json(result));
        return EXIT_SUCCESS;
    }
    case CommandKind::kEmbed: {
        const auto result =
            require_interface<IEmbedding>(task).embed(require_option(command, "--text"));
        write_json(output, embedding_json(result));
        return EXIT_SUCCESS;
    }
    case CommandKind::kRerank: {
        const float score = require_interface<IReranking>(task).rerank(
            require_option(command, "--query"), require_option(command, "--document"));
        if (!std::isfinite(score))
            throw std::runtime_error("reranking returned a non-finite score");
        write_json(output, {{"score", score}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kClassify: {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        const auto result = require_interface<IImageClassification>(task).classify(
            image.pixels.data(), image.height, image.width);
        require_finite(result.logits, "classification logits");
        if (!std::isfinite(result.top_score))
            throw std::runtime_error("classification returned a non-finite top score");
        write_json(output, {{"logits", result.logits},
                            {"top_class", result.top_class},
                            {"top_score", result.top_score}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kExtractFeatures: {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        const auto result = require_interface<IImageFeatureExtractor>(task).extract_image_features(
            image.pixels.data(), image.height, image.width);
        require_finite(result.last_hidden_state, "last_hidden_state");
        require_finite(result.pooler_output, "pooler_output");
        write_json(output, {{"last_hidden_state", result.last_hidden_state},
                            {"last_hidden_state_shape", result.last_hidden_state_shape},
                            {"pooler_output", result.pooler_output},
                            {"pooler_output_shape", result.pooler_output_shape}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kDisparity: {
        const io::LoadedImage left = read_image(require_option(command, "--left"));
        const io::LoadedImage right = read_image(require_option(command, "--right"));
        if (left.height != right.height || left.width != right.width)
            throw std::invalid_argument("--left and --right images must have the same dimensions");
        const auto result = require_interface<IStereoDisparity>(task).estimate_disparity(
            left.pixels.data(), right.pixels.data(), left.height, left.width);
        require_finite(result.disparity, "disparity");
        write_json(
            output,
            {{"disparity", result.disparity}, {"height", result.height}, {"width", result.width}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kGeometry: {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        const auto result = require_interface<IMonocularGeometry>(task).estimate_geometry(
            image.pixels.data(), image.height, image.width);
        const auto area = static_cast<std::size_t>(result.height) * result.width;
        if (result.height <= 0 || result.width <= 0 || result.points.size() != area * 3U ||
            result.depth.size() != area || result.mask.size() != area) {
            throw std::runtime_error("monocular geometry returned incomplete maps");
        }
        const fs::path directory = require_option(command, "--output");
        fs::create_directories(directory);
        write_binary(directory / "points.f32", result.points);
        write_binary(directory / "depth.f32", result.depth);
        write_binary(directory / "mask.u8", result.mask);
        const nlohmann::json intrinsics = {
            {result.intrinsics[0], result.intrinsics[1], result.intrinsics[2]},
            {result.intrinsics[3], result.intrinsics[4], result.intrinsics[5]},
            {result.intrinsics[6], result.intrinsics[7], result.intrinsics[8]},
        };
        std::ofstream matrix_output(directory / "intrinsics.json");
        if (!matrix_output)
            throw std::runtime_error("failed to create intrinsics.json");
        matrix_output << nlohmann::json{{"height", result.height},
                                        {"width", result.width},
                                        {"intrinsics", intrinsics},
                                        {"normalized", true}}
                             .dump(2)
                      << '\n';
        write_json(
            output,
            {{"output", directory.string()}, {"height", result.height}, {"width", result.width}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kSegment: {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        const auto result = require_interface<ISegmentation>(task).segment(
            image.pixels.data(), image.height, image.width);
        write_json(output,
                   {{"mask", result.mask}, {"height", result.height}, {"width", result.width}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kSegmentPrompted: {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        PromptedSegmentationResult result;
        if (!has_option(command, "--prompt")) {
            const float point_x = float_option(command, "--point-x", 0.5F);
            const float point_y = float_option(command, "--point-y", 0.5F);
            const bool foreground =
                has_option(command, "--foreground")
                    ? parse_bool(command.options.at("--foreground"), "--foreground")
                    : true;
            result = require_interface<IPointPromptedSegmentation>(task).segment_prompted(
                image.pixels.data(), image.height, image.width, point_x, point_y, foreground);
        } else {
            if (has_option(command, "--point-x") || has_option(command, "--point-y") ||
                has_option(command, "--foreground")) {
                throw std::invalid_argument(
                    "text_prompted_segmentation accepts --prompt, not point options");
            }
            result = require_interface<ITextPromptedSegmentation>(task).segment_prompted_text(
                image.pixels.data(), image.height, image.width,
                require_option(command, "--prompt"));
        }
        require_finite(result.masks, "segmentation masks");
        require_finite(result.iou_scores, "segmentation IoU scores");
        require_finite(result.boxes, "segmentation boxes");
        write_json(output, {{"masks", result.masks},
                            {"iou_scores", result.iou_scores},
                            {"boxes", result.boxes},
                            {"num_masks", result.num_masks},
                            {"height", result.height},
                            {"width", result.width}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kVideoSegment: {
        if (command.frames.empty())
            throw std::invalid_argument("video-segment requires at least one --frame");
        std::vector<io::LoadedImage> images;
        images.reserve(command.frames.size());
        for (const auto& path : command.frames)
            images.push_back(read_image(path));
        std::vector<VideoFrameView> frames;
        frames.reserve(images.size());
        for (const auto& image : images) {
            frames.push_back({image.pixels.data(), image.pixels.size(), image.height, image.width,
                              VideoFrameFormat::kRgbFloat32});
        }
        auto session =
            require_interface<IVideoSegmentation>(task).create_video_segmentation_session();
        if (session == nullptr)
            throw std::runtime_error("video segmentation family returned a null session");
        const std::string prompt =
            has_option(command, "--prompt") ? command.options.at("--prompt") : std::string{};
        const auto result = session->segment({std::move(frames), prompt});
        nlohmann::json output_frames = nlohmann::json::array();
        for (const auto& frame : result.frames) {
            require_finite(frame.detection_scores, "video detection scores");
            require_finite(frame.tracking_scores, "video tracking scores");
            require_finite(frame.boxes, "video boxes");
            output_frames.push_back({{"masks", frame.masks},
                                     {"object_ids", frame.object_ids},
                                     {"detection_scores", frame.detection_scores},
                                     {"tracking_scores", frame.tracking_scores},
                                     {"boxes", frame.boxes},
                                     {"num_objects", frame.num_objects},
                                     {"height", frame.height},
                                     {"width", frame.width}});
        }
        write_json(output, {{"frames", std::move(output_frames)}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kGenerateAudio: {
        AudioGenerationConfig config;
        config.max_new_tokens = int_option(command, "--max-new-tokens", 128, 1);
        config.talker_max_new_tokens = int_option(command, "--talker-max-new-tokens", 0, 0);
        config.seed = int_option(command, "--seed", -1);
        const bool streaming = has_option(command, "--stream") &&
                               parse_bool(command.options.at("--stream"), "--stream");
        if (!streaming && has_option(command, "--chunk-frames"))
            throw std::invalid_argument("--chunk-frames requires --stream true");
        const std::string path = require_option(command, "--output");
        if (streaming) {
            const int32_t chunk_frames = int_option(command, "--chunk-frames", 32, 1);
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            if (!stream)
                throw std::runtime_error("unable to open streaming audio output: " + path);
            int32_t sample_rate = 0;
            const int32_t total =
                require_interface<IStreamingAudioGeneration>(task).generate_audio_streaming(
                    require_option(command, "--prompt"), config,
                    [&](const float* samples, int32_t num_samples, int32_t rate) {
                        if (samples == nullptr || num_samples <= 0 || rate <= 0)
                            throw std::runtime_error(
                                "streaming audio family returned an invalid chunk");
                        if (sample_rate != 0 && sample_rate != rate)
                            throw std::runtime_error(
                                "streaming audio sample rate changed mid-stream");
                        sample_rate = rate;
                        stream.write(reinterpret_cast<const char*>(samples),
                                     static_cast<std::streamsize>(num_samples) * sizeof(float));
                        if (!stream)
                            throw std::runtime_error("failed to write streaming audio output: " +
                                                     path);
                    },
                    chunk_frames);
            stream.close();
            if (total <= 0 || sample_rate <= 0)
                throw std::runtime_error("streaming audio family produced no samples");
            write_json(output, {{"output", path},
                                {"format", "float32le"},
                                {"sample_rate", sample_rate},
                                {"num_samples", total}});
            return EXIT_SUCCESS;
        }
        const auto result = require_interface<IAudioGeneration>(task).generate_audio(
            require_option(command, "--prompt"), config);
        require_finite(result.samples, "generated audio");
        io::write_wav(result, path);
        write_json(output, {{"output", path},
                            {"sample_rate", result.sample_rate},
                            {"num_samples", result.samples.size()}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kTranscribe: {
        const AudioResult audio = read_audio(require_option(command, "--input"));
        const TranscriptionConfig config = transcription_config(command, audio.sample_rate);
        const auto result = require_interface<ITranscription>(task).transcribe(
            audio.samples.data(), checked_count(audio.samples.size(), "audio"), config);
        write_json(output, text_json(result));
        return EXIT_SUCCESS;
    }
    case CommandKind::kTranscribeBatch: {
        if (command.inputs.empty())
            throw std::invalid_argument("transcribe-batch requires at least one --input");
        std::vector<TranscriptionRequest> requests;
        requests.reserve(command.inputs.size());
        for (const auto& path : command.inputs) {
            AudioResult audio = read_audio(path);
            TranscriptionConfig config = transcription_config(command, audio.sample_rate);
            requests.push_back({std::move(audio.samples), std::move(config)});
        }
        const auto results =
            require_interface<IBatchTranscription>(task).transcribe_batch(requests);
        if (results.size() != requests.size())
            throw std::runtime_error("batch transcription returned the wrong result count");
        nlohmann::json payload = nlohmann::json::array();
        for (const auto& result : results)
            payload.push_back(text_json(result));
        write_json(output, {{"results", std::move(payload)}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kTranscribeStreaming: {
        const AudioResult audio = read_audio(require_option(command, "--input"));
        TranscriptionStreamConfig config;
        config.input_sample_rate = audio.sample_rate;
        config.max_new_tokens = int_option(command, "--max-new-tokens", config.max_new_tokens, 1);
        config.att_context_left =
            int_option(command, "--att-context-left", config.att_context_left, 1);
        config.att_context_right =
            int_option(command, "--att-context-right", config.att_context_right, 0);
        if (has_option(command, "--language"))
            config.language = command.options.at("--language");
        auto stream =
            require_interface<IStreamingTranscription>(task).create_transcription_stream(config);
        if (stream == nullptr)
            throw std::runtime_error("streaming transcription family returned a null stream");
        const auto chunk_size =
            static_cast<std::size_t>(int_option(command, "--chunk-samples", audio.sample_rate, 1));
        nlohmann::json chunks = nlohmann::json::array();
        for (std::size_t offset = 0; offset < audio.samples.size(); offset += chunk_size) {
            const auto count = std::min(chunk_size, audio.samples.size() - offset);
            chunks.push_back(stream_result_json(stream->accept_audio(
                audio.samples.data() + offset, checked_count(count, "audio chunk"), false)));
        }
        const auto final_result = stream->finish();
        write_json(output,
                   {{"chunks", std::move(chunks)}, {"final", stream_result_json(final_result)}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kSpeak: {
        const AudioResult audio = read_audio(require_option(command, "--input"));
        SpeechToSpeechConfig config;
        config.max_new_tokens = int_option(command, "--max-new-tokens", 128, 1);
        config.seed = int_option(command, "--seed", -1);
        config.tail_frames = int_option(command, "--tail-frames", 0, 0);
        const auto result = require_interface<ISpeechToSpeech>(task).speak(
            audio.samples.data(), checked_count(audio.samples.size(), "audio"), config,
            audio.sample_rate);
        require_finite(result.samples, "generated speech");
        const std::string path = require_option(command, "--output");
        io::write_wav(result, path);
        write_json(output, {{"output", path},
                            {"sample_rate", result.sample_rate},
                            {"num_samples", result.samples.size()}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kSpeechSession: {
        const AudioResult audio = read_audio(require_option(command, "--input"));
        SpeechSessionConfig config;
        config.input_sample_rate = audio.sample_rate;
        if (has_option(command, "--system-prompt"))
            config.system_prompt = command.options.at("--system-prompt");
        auto session =
            require_interface<ISpeechSessionProvider>(task).create_speech_session(config);
        if (session == nullptr)
            throw std::runtime_error("speech family returned a null session");
        session->append_audio(audio.samples.data(), checked_count(audio.samples.size(), "audio"));
        session->finish_input();
        const auto timeout = int_option(command, "--timeout-ms", 30000, 0);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
        std::vector<SpeechSessionEvent> events;
        bool input_finished = false;
        while (!input_finished) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
                throw std::runtime_error("speech session timed out before input_finished");
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
            const auto wait_ms = static_cast<std::int32_t>(std::min<std::int64_t>(remaining, 500));
            auto batch = session->wait_events(wait_ms);
            for (const auto& event : batch) {
                if (event.kind == SpeechSessionEventKind::kInputFinished)
                    input_finished = true;
                if (event.kind == SpeechSessionEventKind::kError)
                    throw std::runtime_error(event.text.empty() ? "speech session failed"
                                                                : event.text);
            }
            events.insert(events.end(), std::make_move_iterator(batch.begin()),
                          std::make_move_iterator(batch.end()));
        }
        auto remaining = session->take_events();
        events.insert(events.end(), std::make_move_iterator(remaining.begin()),
                      std::make_move_iterator(remaining.end()));
        nlohmann::json event_json = nlohmann::json::array();
        std::vector<float> agent_audio;
        std::int32_t agent_sample_rate = 0;
        for (const auto& event : events) {
            event_json.push_back({{"kind", event_kind_name(event.kind)},
                                  {"epoch", event.epoch},
                                  {"sequence", event.sequence},
                                  {"text", event.text},
                                  {"is_final", event.is_final},
                                  {"audio_samples", event.audio_samples.size()}});
            if (!event.audio_samples.empty()) {
                if (agent_sample_rate != 0 && event.sample_rate != agent_sample_rate)
                    throw std::runtime_error("speech session changed output sample rate");
                agent_sample_rate = event.sample_rate;
                agent_audio.insert(agent_audio.end(), event.audio_samples.begin(),
                                   event.audio_samples.end());
            }
        }
        nlohmann::json result{{"events", std::move(event_json)}};
        if (has_option(command, "--output")) {
            if (agent_audio.empty() || agent_sample_rate <= 0)
                throw std::runtime_error("speech session produced no agent audio");
            require_finite(agent_audio, "speech session audio");
            AudioResult generated{std::move(agent_audio), 0, agent_sample_rate};
            generated.num_samples = checked_count(generated.samples.size(), "speech session audio");
            const std::string& path = command.options.at("--output");
            io::write_wav(generated, path);
            result["output"] = path;
        }
        write_json(output, result);
        return EXIT_SUCCESS;
    }
    case CommandKind::kGenerateImage: {
        const std::string path = require_option(command, "--output");
        write_json(output, write_image(generate_image(command, task), path));
        return EXIT_SUCCESS;
    }
    case CommandKind::kGenerateImageBatch: {
        const auto prompts = read_nonempty_lines(require_option(command, "--prompts"));
        const auto seeds = parse_seeds(require_option(command, "--seeds"));
        if (seeds.size() != prompts.size())
            throw std::invalid_argument("--seeds count must match --prompts line count");
        const auto results = require_interface<IImageBatchGeneration>(task).generate_image_batch(
            prompts, seeds, image_config(command));
        if (results.size() != prompts.size())
            throw std::runtime_error("batch image task returned the wrong result count");
        const fs::path directory = require_option(command, "--output");
        fs::create_directories(directory);
        nlohmann::json outputs = nlohmann::json::array();
        for (std::size_t index = 0; index < results.size(); ++index) {
            const fs::path path = directory / (std::to_string(index) + ".png");
            outputs.push_back(write_image(results[index], path.string()));
        }
        write_json(output, {{"outputs", std::move(outputs)}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kGenerateVideo: {
        const std::string directory = require_option(command, "--output");
        write_json(output, write_video(generate_image(command, task), directory));
        return EXIT_SUCCESS;
    }
    case CommandKind::kSolve: {
        const auto branch = read_float32_file(require_option(command, "--branch"));
        const auto trunk = read_float32_file(require_option(command, "--trunk"));
        const auto result = require_interface<INeuralOperator>(task).solve(
            branch.data(), checked_count(branch.size(), "branch input"), trunk.data(),
            checked_count(trunk.size(), "trunk input"));
        write_json(output, embedding_json(result));
        return EXIT_SUCCESS;
    }
    case CommandKind::kForecast: {
        const auto values = read_float32_file(require_option(command, "--input"));
        std::vector<float> mask;
        if (has_option(command, "--mask")) {
            mask = read_float32_file(command.options.at("--mask"));
            if (mask.size() != values.size())
                throw std::invalid_argument("--mask and --input must contain the same float count");
        } else {
            mask.assign(values.size(), 1.0F);
        }
        const auto frequency = int_option(command, "--frequency", 0, 0);
        const auto result = require_interface<ITimeSeriesForecast>(task).forecast(
            {{values.data(), values.size()}, {mask.data(), mask.size()}, frequency});
        require_finite(result.values, "forecast values");
        write_json(output, {{"values", result.values}, {"shape", result.shape}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kControl: {
        const auto image = read_image(require_option(command, "--image"));
        const auto state = read_float32_file(require_option(command, "--state"));
        const RobotObservation observation{
            {image.pixels.data(), image.pixels.size()},
            image.height,
            image.width,
            3,
            {state.data(), state.size()},
        };
        const auto result =
            require_interface<IRobotControl>(task).predict_action_chunk(observation);
        require_finite(result.actions, "robot actions");
        if (has_option(command, "--output"))
            write_binary(command.options.at("--output"), result.actions);
        write_json(output, {{"actions", result.actions},
                            {"num_actions", result.num_actions},
                            {"action_dim", result.action_dim},
                            {"within_training_bounds", result.within_training_bounds},
                            {"inference_ms", result.inference_ms}});
        return EXIT_SUCCESS;
    }
    case CommandKind::kGenerateWorld: {
        const io::LoadedImage image = read_image(require_option(command, "--image"));
        WorldModelRequest request;
        request.prompt = require_option(command, "--prompt");
        request.image = image.pixels;
        request.image_height = image.height;
        request.image_width = image.width;
        if (has_option(command, "--action"))
            request.action = command.options.at("--action");
        if (has_option(command, "--intrinsics"))
            request.camera_intrinsics = read_float32_file(command.options.at("--intrinsics"));
        request.num_frames = int_option(command, "--num-frames", 0, 1);
        request.generation = image_config(command);
        const auto result = require_interface<IWorldModelGeneration>(task).generate_world(request);
        const std::string directory = require_option(command, "--output");
        write_json(output, write_video(result, directory));
        return EXIT_SUCCESS;
    }
    case CommandKind::kHelp:
    case CommandKind::kVersion:
    case CommandKind::kInspect:
        break;
    }
    throw std::logic_error("non-execution command reached Task dispatch");
}

void print_usage(std::ostream& output) {
    output << "Usage:\n"
              "  trtmc version\n"
              "  trtmc inspect BUNDLE\n"
              "  trtmc COMMAND BUNDLE --runtime-root DIR [OPTIONS]\n\n"
              "Execution commands:\n"
              "  run, encode, embed, rerank, classify, extract-features, disparity, geometry,\n"
              "  segment,\n"
              "  segment-prompted, video-segment, generate-audio, transcribe,\n"
              "  transcribe-batch, transcribe-streaming, speak, speech-session, generate-image,\n"
              "  generate-image-batch, generate-video,\n"
              "  solve, forecast, control, generate-world\n\n"
              "Text diffusion replay options:\n"
              "  [--generation-mode MODE] [--block-length N] [--threshold S]\n"
              "  --initial-latents-raw PATH [--condition-latents-raw PATH\n"
              "  --condition-mask-raw PATH] [--sampling-steps-raw PATH]\n"
              "  [--sde-noise-raw PATH] [--num-steps N] [--guidance-scale S]\n"
              "  [--cfg-scale S] [--sde-gamma S]\n\n"
              "Text generation options:\n"
              "  [--source-language-token-id N] [--forced-bos-token-id N]\n\n"
              "Offline transcription options:\n"
              "  [--beam-size N] [--length-penalty F] [--punctuation true|false]\n"
              "  [--max-input-seconds F] [--segment-length-seconds F]\n"
              "  [--segment-min-seconds F] [--segment-overlap-seconds F]\n"
              "  [--lcs-merge true|false]\n\n"
              "BYOK options:\n"
              "  --byok-library DSO --byok-function FUNCTION --byok-name KERNEL\n\n"
              "Runtime-sized KV cache:\n"
              "  [--kv-cache-size BYTES|GB|GiB]\n\n"
              "TensorRT-RTX runtime options:\n"
              "  [--runtime-cache PATH] [--cuda-graphs]\n\n"
              "Execution never searches for runtimes; --runtime-root is always required.\n";
}

int run(int argc, char** argv, std::ostream& output, std::ostream& error) {
    try {
        const Command command = parse_args(argc, argv);
        if (command.kind == CommandKind::kHelp) {
            print_usage(output);
            return EXIT_SUCCESS;
        }
        if (command.kind == CommandKind::kVersion) {
            output << "trtmc " << TRTMC_VERSION_STRING << '\n';
            return EXIT_SUCCESS;
        }
        if (command.kind == CommandKind::kInspect) {
            const BundleInfo bundle = InspectBundle(command.bundle);
            nlohmann::json sections = nlohmann::json::object();
            for (const auto& section : bundle.sections)
                sections[section.name] = {{"offset", section.offset}, {"length", section.length}};
            write_json(output, {{"format", bundle.format},
                                {"family", bundle.family},
                                {"task", bundle.task},
                                {"backend", bundle.backend},
                                {"sections", std::move(sections)}});
            return EXIT_SUCCESS;
        }
        const bool has_byok_library = has_option(command, "--byok-library");
        const bool has_byok_function = has_option(command, "--byok-function");
        const bool has_byok_name = has_option(command, "--byok-name");
        if (has_byok_library || has_byok_function || has_byok_name) {
            if (!(has_byok_library && has_byok_function && has_byok_name)) {
                throw std::invalid_argument(
                    "--byok-library, --byok-function, and --byok-name must be used together");
            }
            load_byok_extension(command);
        }
        std::unique_ptr<ITask> task =
            load_task(command.bundle, command.runtime_root, command.kv_cache_size_bytes,
                      command.runtime_cache_path, command.cuda_graphs);
        return dispatch(command, *task, output);
    } catch (const std::invalid_argument& exception) {
        error << "Error: " << exception.what() << "\n\n";
        print_usage(error);
        return 2;
    } catch (const std::exception& exception) {
        error << "Error: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace trtmc::cli
