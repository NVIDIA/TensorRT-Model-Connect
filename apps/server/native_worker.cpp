/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "server/native_worker.h"

#include "bundle_provenance.h"
#include "cli/io.h"
#include "config.h"
#include "task_runtime.h"
#include "trtmc/audio.hpp"
#include "trtmc/control.hpp"
#include "trtmc/features.hpp"
#include "trtmc/image.hpp"
#include "trtmc/language.hpp"
#include "trtmc/perception.hpp"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/stream.hpp"
#include "trtmc/task.h"
#include "trtmc/text.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace trtmc::server {
namespace {

using Json = nlohmann::json;
constexpr std::size_t kMaxLineBytes = 16U * 1024U * 1024U;

class ProtocolError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

template <typename T>
void assign(const Json& config, const char* name, T& destination) {
    const auto value = config.find(name);
    if (value == config.end())
        return;
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if (!value->is_number_integer())
            throw ProtocolError(std::string("config.") + name + " must be an integer");
        bool in_range = false;
        if (value->is_number_unsigned()) {
            const auto number = value->get<Json::number_unsigned_t>();
            in_range =
                number <= static_cast<Json::number_unsigned_t>(std::numeric_limits<T>::max());
        } else {
            const auto number = value->get<Json::number_integer_t>();
            in_range =
                number >= static_cast<Json::number_integer_t>(std::numeric_limits<T>::min()) &&
                number <= static_cast<Json::number_integer_t>(std::numeric_limits<T>::max());
        }
        if (!in_range)
            throw ProtocolError(std::string("config.") + name + " is out of range");
    }
    destination = value->get<T>();
}

void validate_fields(const Json& object, std::initializer_list<const char*> allowed) {
    for (auto field = object.begin(); field != object.end(); ++field) {
        const bool found = std::any_of(allowed.begin(), allowed.end(),
                                       [&](const char* name) { return field.key() == name; });
        if (!found)
            throw ProtocolError("config." + field.key() + " is unsupported");
    }
}

TextGenerationConfig parse_config(const Json& request, std::int32_t default_tokens) {
    const auto found = request.find("config");
    const Json config = found == request.end() ? Json::object() : *found;
    if (!config.is_object())
        throw ProtocolError("config must be an object");
    validate_fields(config, {"max_new_tokens", "temperature", "top_k", "top_p", "min_p", "seed",
                             "system_prompt", "use_chat_template", "enable_thinking"});

    TextGenerationConfig result;
    result.max_new_tokens = default_tokens > 0 ? default_tokens : 128;
    try {
        assign(config, "max_new_tokens", result.max_new_tokens);
        assign(config, "temperature", result.temperature);
        assign(config, "top_k", result.top_k);
        assign(config, "top_p", result.top_p);
        assign(config, "min_p", result.min_p);
        assign(config, "seed", result.seed);
        assign(config, "system_prompt", result.system_prompt);
        assign(config, "use_chat_template", result.use_chat_template);
        assign(config, "enable_thinking", result.enable_thinking);
    } catch (const nlohmann::json::exception&) {
        throw ProtocolError("generation config contains an invalid value");
    }
    if (result.max_new_tokens <= 0)
        throw ProtocolError("config.max_new_tokens must be positive");
    if (result.top_k < 0)
        throw ProtocolError("config.top_k must be non-negative");
    if (!std::isfinite(result.temperature) || result.temperature < 0.0F)
        throw ProtocolError("config.temperature must be finite and non-negative");
    if (!std::isfinite(result.top_p) || result.top_p < 0.0F || result.top_p > 1.0F)
        throw ProtocolError("config.top_p must be in [0, 1]");
    if (!std::isfinite(result.min_p) || result.min_p < 0.0F || result.min_p > 1.0F)
        throw ProtocolError("config.min_p must be in [0, 1]");
    return result;
}

std::string string_field(const Json& request, const char* name, bool empty_ok = false) {
    const auto value = request.find(name);
    if (value == request.end() || !value->is_string())
        throw ProtocolError(std::string(name) + " must be a string");
    const auto result = value->get<std::string>();
    if (!empty_ok && result.empty())
        throw ProtocolError(std::string(name) + " must not be empty");
    return result;
}

Json error_response(const Json& id, const char* type, const std::string& message) {
    return {{"id", id}, {"ok", false}, {"error", {{"type", type}, {"message", message}}}};
}

bool write_json(std::ostream& output, const Json& value) {
    output << value.dump(-1, ' ', false, Json::error_handler_t::replace) << '\n';
    output.flush();
    return static_cast<bool>(output);
}

Json request_id(const Json& request) {
    if (!request.is_object())
        return nullptr;
    const auto id = request.find("id");
    if (id == request.end() || !id->is_string() || id->get_ref<const std::string&>().empty())
        return nullptr;
    return *id;
}

Config parse_sdk_config(const Json& request, const std::vector<ConfigField>& fields) {
    Config result;
    const auto found = request.find("config");
    if (found == request.end())
        return result;
    if (!found->is_object())
        throw ProtocolError("config must be an object");
    for (auto entry = found->begin(); entry != found->end(); ++entry) {
        const auto field = std::find_if(fields.begin(), fields.end(),
                                        [&](const auto& item) { return item.name == entry.key(); });
        if (field == fields.end())
            throw ProtocolError("config." + entry.key() + " is unsupported");
        if (field->kind == ConfigKind::String && !entry->is_string())
            throw ProtocolError("config." + entry.key() + " must be a string");
        try {
            const auto value =
                field->kind == ConfigKind::String ? entry->get<std::string>() : entry->dump();
            result.add(entry.key(), app::parse_config_value(value, *field));
        } catch (const std::invalid_argument& error) {
            throw ProtocolError(error.what());
        }
    }
    return result;
}

std::int64_t default_token_limit(const std::vector<ConfigField>& fields) {
    for (const auto& field : fields) {
        if (field.name == "max_new_tokens" && field.kind == ConfigKind::I64 &&
            field.default_value) {
            const auto value = field.default_value->get<std::int64_t>();
            if (value > 0)
                return value;
        }
    }
    // Existing HTTP safety default for a family-derived limit. This does not
    // insert a default into the Config passed to the Task.
    return 128;
}

using StreamGenerate =
    std::function<Json(const std::string&, const Json&, const Json&, std::ostream&)>;
using Operations = std::map<std::string, std::function<Json(const Json&)>>;

int run_protocol(const Json& ready, const Operations& operations, std::istream& input,
                 std::ostream& output, StreamGenerate stream_generate = {}) {
    if (!write_json(output, ready))
        return 2;

    std::vector<char> buffer(kMaxLineBytes + 2U);
    while (input.good()) {
        input.getline(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto extracted = input.gcount();
        if (input.eof() && extracted == 0)
            return 0;

        Json id = nullptr;
        bool shutdown = false;
        bool fatal = false;
        Json response;
        try {
            if (input.fail() && !input.eof()) {
                input.clear(input.rdstate() & ~std::ios::failbit);
                input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                throw ProtocolError("request exceeds the 16 MiB JSONL limit");
            }
            std::size_t size = static_cast<std::size_t>(extracted);
            if (!input.eof() && size > 0)
                --size;
            if (size == 0)
                continue;
            Json request;
            try {
                request = Json::parse(std::string_view(buffer.data(), size));
            } catch (const nlohmann::json::parse_error&) {
                throw ProtocolError("request is not valid JSON");
            }
            id = request_id(request);
            if (id.is_null())
                throw ProtocolError("id must be a non-empty string");
            const auto operation = string_field(request, "op");
            if (operation == "shutdown") {
                response = {{"id", id}, {"ok", true}, {"result", {{"status", "shutting_down"}}}};
                shutdown = true;
            } else if (operation == "generate_stream") {
                if (!stream_generate)
                    throw ProtocolError("incremental streaming is unsupported by this model");
                const auto prompt = string_field(request, "prompt", true);
                response = {{"id", id},
                            {"ok", true},
                            {"event", "complete"},
                            {"result", stream_generate(prompt, request, id, output)}};
            } else if (const auto handler = operations.find(operation);
                       handler != operations.end()) {
                response = {{"id", id}, {"ok", true}, {"result", handler->second(request)}};
            } else {
                throw ProtocolError("unknown operation: " + operation);
            }
        } catch (const ProtocolError& error) {
            response = error_response(id, "invalid_request_error", error.what());
        } catch (const Error& error) {
            if (error.code() == TRTMC_INVALID_ARGUMENT || error.code() == TRTMC_INVALID_CONFIG) {
                response = error_response(id, "invalid_request_error",
                                          "request is invalid for this model");
            } else {
                std::cerr << "[trtmc.server.worker] " << error.what() << '\n';
                response = error_response(id, "runtime_error", "native worker operation failed");
                fatal = true;
            }
        } catch (const std::exception& error) {
            std::cerr << "[trtmc.server.worker] " << error.what() << '\n';
            response = error_response(id, "runtime_error", "native worker operation failed");
            fatal = true;
        } catch (...) {
            std::cerr << "[trtmc.server.worker] unknown native worker error\n";
            response = error_response(id, "runtime_error", "native worker operation failed");
            fatal = true;
        }
        if (!write_json(output, response))
            return 2;
        if (fatal)
            return 1;
        if (shutdown)
            return 0;
    }
    return input.bad() ? 2 : 0;
}

template <class Generate>
int run_text_protocol(std::int64_t default_tokens, Generate&& generate, std::istream& input,
                      std::ostream& output, StreamGenerate streaming = {}) {
    Json capabilities = Json::array({"text_generation"});
    if (streaming)
        capabilities.push_back("streaming_text_continuation");
    const Json ready{{"event", "ready"},
                     {"protocol_version", 2},
                     {"capabilities", capabilities},
                     {"default_max_new_tokens", default_tokens}};
    return run_protocol(ready,
                        {{"generate",
                          [&](const Json& request) {
                              return generate(string_field(request, "prompt", true), request);
                          }}},
                        input, output, std::move(streaming));
}

const std::vector<float>& checked_embedding_values(const EmbeddingResult& result) {
    if (result.dim <= 0 || static_cast<std::size_t>(result.dim) != result.data.size())
        throw std::runtime_error("embedding Task returned inconsistent dimensions");
    return result.data;
}

template <class Invoke, class Values>
int run_embedding_protocol(Invoke&& invoke, Values&& extract_values, std::istream& input,
                           std::ostream& output) {
    const Json ready{
        {"event", "ready"}, {"protocol_version", 2}, {"capabilities", Json::array({"embeddings"})}};
    return run_protocol(
        ready,
        {{"embed",
          [&](const Json& request) {
              const auto text = string_field(request, "input");
              const auto started = std::chrono::steady_clock::now();
              auto result = invoke(text);
              const auto wall_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
              const auto values = extract_values(result);
              if (values.empty() || !std::all_of(values.begin(), values.end(),
                                                 [](float v) { return std::isfinite(v); }))
                  throw std::runtime_error("embedding Task returned invalid values");
              return Json{{"values", values},
                          {"dim", values.size()},
                          {"model_call_ms", wall_ms},
                          {"timing_scope", "public_task_call_wall"}};
          }}},
        input, output);
}

template <class Invoke, class Scores>
int run_ranking_protocol(Invoke&& invoke, Scores&& extract_scores, std::istream& input,
                         std::ostream& output) {
    const Json ready{
        {"event", "ready"}, {"protocol_version", 2}, {"capabilities", Json::array({"reranking"})}};
    return run_protocol(
        ready,
        {{"rerank",
          [&](const Json& request) {
              const auto query = string_field(request, "query");
              const auto found = request.find("documents");
              if (found == request.end() || !found->is_array() || found->empty() ||
                  !std::all_of(found->begin(), found->end(), [](const Json& value) {
                      return value.is_string() && !value.get_ref<const std::string&>().empty();
                  }))
                  throw ProtocolError("documents must be a non-empty list of non-empty strings");
              const auto documents = found->get<std::vector<std::string>>();
              const auto started = std::chrono::steady_clock::now();
              auto result = invoke(query, documents);
              const auto wall_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
              const auto scores = extract_scores(result);
              if (scores.size() != documents.size() ||
                  !std::all_of(scores.begin(), scores.end(),
                               [](float score) { return std::isfinite(score); }))
                  throw std::runtime_error("ranking Task returned invalid scores");
              return Json{{"scores", scores},
                          {"model_call_ms", wall_ms},
                          {"timing_scope", "public_task_call_wall"}};
          }}},
        input, output);
}

template <class Invoke, class Text>
Json timed_transcription(Invoke&& invoke, Text&& extract_text) {
    const auto started = std::chrono::steady_clock::now();
    auto result = invoke();
    const auto wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    return {{"text", extract_text(result)},
            {"model_call_ms", wall_ms},
            {"timing_scope", "public_task_call_wall"}};
}

template <class Invoke>
int run_transcription_protocol(Invoke&& invoke, std::istream& input, std::ostream& output) {
    return run_protocol(
        {{"event", "ready"},
         {"protocol_version", 1},
         {"capabilities", Json::array({"audio_transcription"})}},
        {{"transcribe",
          [&](const Json& request) {
              cli::io::LoadedAudio audio;
              const auto path = string_field(request, "audio_path");
              try {
                  audio = cli::io::read_wav_interleaved(path);
              } catch (const std::exception&) {
                  throw ProtocolError("audio must be a readable WAV file");
              }
              if (audio.channels != 1 || audio.sample_rate <= 0 || audio.samples.empty() ||
                  audio.samples.size() >
                      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
                  !std::all_of(audio.samples.begin(), audio.samples.end(),
                               [](float value) { return std::isfinite(value); }))
                  throw ProtocolError(
                      "audio must contain finite mono PCM samples at a positive rate");
              auto result = invoke(audio, request);
              result["audio_sample_count"] = audio.samples.size();
              result["audio_sample_rate"] = audio.sample_rate;
              return result;
          }}},
        input, output);
}

template <class Box>
Json detection_box(float x0, float y0, float x1, float y1, const Box& box, std::int32_t /*height*/,
                   std::int32_t /*width*/) {
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1) ||
        !std::isfinite(box.score) || box.score < 0 || box.score > 1 || box.class_id < 0 ||
        x1 < x0 || y1 < y0)
        throw std::runtime_error("detection Task returned invalid boxes");
    return {{"x_min", x0},
            {"y_min", y0},
            {"x_max", x1},
            {"y_max", y1},
            {"confidence", box.score},
            {"class_id", box.class_id}};
}

template <class Invoke, class Extract>
int run_detection_protocol(Invoke&& invoke, Extract&& extract, std::istream& input,
                           std::ostream& output) {
    return run_protocol({{"event", "ready"},
                         {"protocol_version", 1},
                         {"capabilities", Json::array({"object_detection"})}},
                        {{"detect",
                          [&](const Json& request) {
                              const auto config = request.value("config", Json::object());
                              if (!config.is_object() || !config.empty())
                                  throw ProtocolError("detection config must be empty");
                              cli::io::LoadedImage image;
                              const auto path = string_field(request, "image_path");
                              try {
                                  image = cli::io::read_image(path);
                              } catch (const std::exception&) {
                                  throw ProtocolError("image must be a readable image file");
                              }
                              if (image.height <= 0 || image.width <= 0 || image.pixels.empty())
                                  throw ProtocolError("image must contain RGB pixels");
                              const auto started = std::chrono::steady_clock::now();
                              auto result = invoke(image);
                              const auto wall_ms = std::chrono::duration<double, std::milli>(
                                                       std::chrono::steady_clock::now() - started)
                                                       .count();
                              auto boxes = extract(result, image.height, image.width);
                              return Json{{"boxes", std::move(boxes)},
                                          {"image_height", image.height},
                                          {"image_width", image.width},
                                          {"model_call_ms", wall_ms},
                                          {"timing_scope", "public_task_call_wall"}};
                          }}},
                        input, output);
}

template <class Generate>
int run_vision_protocol(std::int64_t default_tokens, Generate&& generate, std::istream& input,
                        std::ostream& output) {
    return run_protocol({{"event", "ready"},
                         {"protocol_version", 2},
                         {"capabilities", Json::array({"vision_language_generation"})},
                         {"default_max_new_tokens", default_tokens}},
                        {{"generate_vision",
                          [&](const Json& request) {
                              const auto prompt = string_field(request, "prompt");
                              cli::io::LoadedImage image;
                              const auto path = string_field(request, "image_path");
                              try {
                                  image = cli::io::read_image(path);
                              } catch (const std::exception&) {
                                  throw ProtocolError("image must be a readable image file");
                              }
                              if (image.height <= 0 || image.width <= 0 || image.pixels.empty())
                                  throw ProtocolError("image must contain RGB pixels");
                              auto result = generate(prompt, image, request);
                              result["image_height"] = image.height;
                              result["image_width"] = image.width;
                              return result;
                          }}},
                        input, output);
}

std::vector<float> image_latents(const Json& request) {
    if (!request.contains("initial_latents_path"))
        return {};
    const auto path = string_field(request, "initial_latents_path");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(float) != 0)
        throw ProtocolError("initial latents must be a nonempty raw float32 file");
    const auto bytes = static_cast<std::size_t>(file.tellg());
    // Bound the local worker protocol's auxiliary file independently of JSONL size.
    if (bytes > kMaxLineBytes)
        throw ProtocolError("initial latents exceed the worker limit");
    std::vector<float> values(bytes / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file ||
        !std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
        throw ProtocolError("initial latents must contain finite float32 values");
    return values;
}

ImageGenerationConfig image_config(const Json& request) {
    const auto config = request.value("config", Json::object());
    if (!config.is_object())
        throw ProtocolError("config must be an object");
    validate_fields(config, {"seed", "num_steps", "guidance_scale", "cfg_scale", "negative_prompt",
                             "height", "width"});
    ImageGenerationConfig result;
    assign(config, "seed", result.seed);
    assign(config, "num_steps", result.num_steps);
    assign(config, "guidance_scale", result.guidance_scale);
    assign(config, "cfg_scale", result.cfg_scale);
    assign(config, "negative_prompt", result.negative_prompt);
    assign(config, "height", result.height);
    assign(config, "width", result.width);
    if (result.seed < -1 || result.num_steps == 0 || result.num_steps < -1 || result.height < 0 ||
        result.width < 0 || !std::isfinite(result.guidance_scale) ||
        !std::isfinite(result.cfg_scale))
        throw ProtocolError("invalid image generation controls");
    result.initial_latents = image_latents(request);
    return result;
}

template <class Generate>
int run_image_protocol(Generate&& generate, std::istream& input, std::ostream& output,
                       std::string_view capability = "image_generation",
                       std::string_view operation = "generate_image") {
    return run_protocol(
        {{"event", "ready"}, {"protocol_version", 1}, {"capabilities", Json::array({capability})}},
        {{std::string(operation),
          [&](const Json& request) {
              const auto prompt = string_field(request, "prompt");
              const auto path = string_field(request, "output_path");
              // Generation and timing stay with the public Task. PNG encoding is
              // the same codec used by the CLI and benchmark artifact recorder.
              return generate(prompt, path, request);
          }}},
        input, output);
}

template <class Edit>
int run_edit_protocol(Edit&& edit, std::istream& input, std::ostream& output) {
    return run_image_protocol(
        [&](const std::string& prompt, const std::string& path, const Json& request) {
            cli::io::LoadedImage image;
            try {
                image = cli::io::read_image(string_field(request, "image_path"));
            } catch (const std::exception&) {
                throw ProtocolError("image must be a readable image file");
            }
            if (image.height <= 0 || image.width <= 0 || image.pixels.empty())
                throw ProtocolError("image must contain RGB pixels");
            auto result = edit(prompt, image, path, request);
            result["image_height"] = image.height;
            result["image_width"] = image.width;
            return result;
        },
        input, output, "image_edit", "edit_image");
}

Json write_generated_image(const ImageResult& result, const std::string& path, double wall_ms) {
    if (result.num_frames != 1)
        throw std::runtime_error("image endpoint requires one generated frame");
    cli::io::save_png(path, result.pixels, result.width, result.height, result.channels);
    return {{"height", result.height},     {"width", result.width},
            {"channels", result.channels}, {"output_count", 1},
            {"model_call_ms", wall_ms},    {"timing_scope", "public_task_call_wall"}};
}

Json write_video_frames(const ImageResult& result, const std::string& directory,
                        VideoGeometry geometry, double wall_ms) {
    if (result.width != geometry.width || result.height != geometry.height ||
        result.num_frames != geometry.frames || result.channels != 3 ||
        result.pixels.size() !=
            static_cast<std::size_t>(geometry.width) * geometry.height * geometry.frames * 3U ||
        !std::all_of(result.pixels.begin(), result.pixels.end(),
                     [](float v) { return std::isfinite(v); }))
        throw std::runtime_error("video result differs from the built RGB THWC geometry");
    std::filesystem::create_directories(directory);
    const auto frame_size = static_cast<std::size_t>(geometry.width) * geometry.height * 3U;
    for (int frame = 0; frame < geometry.frames; ++frame) {
        std::ostringstream name;
        name << "frame-" << std::setw(6) << std::setfill('0') << frame << ".png";
        const auto begin = result.pixels.begin() + static_cast<std::ptrdiff_t>(frame_size * frame);
        cli::io::save_png(
            (std::filesystem::path(directory) / name.str()).string(),
            std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(frame_size)),
            geometry.width, geometry.height);
    }
    return {{"width", geometry.width},
            {"height", geometry.height},
            {"channels", 3},
            {"num_frames", geometry.frames},
            {"model_call_ms", wall_ms},
            {"timing_scope", "public_task_call_wall"}};
}

int run_existing_worker(ITask& task, std::istream& input, std::ostream& output,
                        VideoGeometry geometry = {}) {
    // Legacy bundle modes select the declared public Task before invocation.
    // Several families implement more C++ interfaces than a given bundle enables.
    if (std::string_view(task.task()) == IImageEditing::kTask) {
        auto* editor = dynamic_cast<IImageEditing*>(&task);
        if (!editor)
            throw std::invalid_argument("bundle does not implement image editing");
        return run_edit_protocol(
            [&](const std::string& prompt, const cli::io::LoadedImage& image,
                const std::string& path, const Json& request) {
                const auto config = image_config(request);
                const auto started = std::chrono::steady_clock::now();
                const auto result = editor->generate_image(prompt, image.pixels.data(),
                                                           image.height, image.width, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                return write_generated_image(result, path, wall_ms);
            },
            input, output);
    }
    if (std::string_view(task.task()) == IImageGeneration::kTask) {
        auto* image = dynamic_cast<IImageGeneration*>(&task);
        if (!image)
            throw std::invalid_argument("bundle does not implement image generation");
        if (geometry.frames > 1) {
            const Json ready{{"event", "ready"},
                             {"protocol_version", 1},
                             {"capabilities", Json::array({"video_generation"})},
                             {"video",
                              {{"width", geometry.width},
                               {"height", geometry.height},
                               {"num_frames", geometry.frames}}}};
            return run_protocol(
                ready,
                {{"generate_video",
                  [&](const Json& request) {
                      const auto prompt = string_field(request, "prompt");
                      const auto path = string_field(request, "output_path");
                      const auto config = image_config(request);
                      if ((config.width && config.width != geometry.width) ||
                          (config.height && config.height != geometry.height))
                          throw ProtocolError("video size must match the built geometry");
                      const auto started = std::chrono::steady_clock::now();
                      const auto result = image->generate_image(prompt, config);
                      const auto wall_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - started)
                                               .count();
                      return write_video_frames(result, path, geometry, wall_ms);
                  }}},
                input, output);
        }
        return run_image_protocol(
            [&](const std::string& prompt, const std::string& path, const Json& request) {
                const auto config = image_config(request);
                const auto started = std::chrono::steady_clock::now();
                const auto result = image->generate_image(prompt, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                return write_generated_image(result, path, wall_ms);
            },
            input, output);
    }
    if (std::string_view(task.task()) == IVisionLanguageGeneration::kTask) {
        auto* vision = dynamic_cast<IVisionLanguageGeneration*>(&task);
        if (!vision)
            throw std::invalid_argument("bundle does not implement vision-language generation");
        const auto* text = dynamic_cast<ITextGeneration*>(&task);
        const auto defaults = text ? text->default_max_new_tokens() : 128;
        return run_vision_protocol(
            defaults,
            [&](const std::string& prompt, const cli::io::LoadedImage& image, const Json& request) {
                const auto config = parse_config(request, defaults);
                if (!config.system_prompt.empty())
                    throw ProtocolError("system prompts are unsupported for image requests");
                const auto started = std::chrono::steady_clock::now();
                const auto result = vision->generate(prompt, image.pixels.data(), image.height,
                                                     image.width, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                return Json{{"text", result.text},
                            {"completion_tokens", result.token_ids.size()},
                            {"setup_ms", result.setup_ms},
                            {"prefill_ms", result.prefill_ms},
                            {"decode_ms", result.decode_ms},
                            {"model_call_ms", wall_ms},
                            {"timing_scope", "public_task_call_wall"}};
            },
            input, output);
    }
    if (std::string_view(task.task()) == IEncoding::kTask) {
        auto* encoder = dynamic_cast<IEncoding*>(&task);
        if (!encoder)
            throw std::invalid_argument("bundle does not implement encoding");
        return run_embedding_protocol(
            [&](const std::string& text) { return encoder->encode(text); },
            checked_embedding_values, input, output);
    }
    if (std::string_view(task.task()) == IEmbedding::kTask) {
        auto* embedding = dynamic_cast<IEmbedding*>(&task);
        if (!embedding)
            throw std::invalid_argument("bundle does not implement embedding");
        return run_embedding_protocol(
            [&](const std::string& text) { return embedding->embed(text); },
            checked_embedding_values, input, output);
    }

    if (std::string_view(task.task()) == IReranking::kTask) {
        auto* ranking = dynamic_cast<IReranking*>(&task);
        if (!ranking)
            throw std::invalid_argument("bundle does not implement reranking");
        return run_ranking_protocol(
            [&](const std::string& query, const std::vector<std::string>& documents) {
                return ranking->rerank_batch(query, documents);
            },
            [](const std::vector<float>& scores) { return scores; }, input, output);
    }
    if (std::string_view(task.task()) == ITranscription::kTask ||
        std::string_view(task.task()) == IStreamingTranscription::kTask) {
        // A streaming-oriented bundle may also implement whole-clip transcription.
        // Invoke that public capability; the family owns its decoding policy.
        auto* transcription = dynamic_cast<ITranscription*>(&task);
        if (!transcription)
            throw std::invalid_argument("bundle does not implement transcription");
        return run_transcription_protocol(
            [&](const cli::io::LoadedAudio& audio, const Json& request) {
                const auto value = request.value("config", Json::object());
                if (!value.is_object())
                    throw ProtocolError("config must be an object");
                validate_fields(value, {"max_output_tokens"});
                TranscriptionConfig config;
                assign(value, "max_output_tokens", config.max_output_tokens);
                if (config.max_output_tokens <= 0)
                    throw ProtocolError("config.max_output_tokens must be positive");
                config.input_sample_rate = audio.sample_rate;
                return timed_transcription(
                    [&] {
                        return transcription->transcribe(
                            audio.samples.data(), static_cast<std::int32_t>(audio.samples.size()),
                            config);
                    },
                    [](const TextResult& result) { return result.text; });
            },
            input, output);
    }
    if (std::string_view(task.task()) == IObjectDetection::kTask) {
        auto* detector = dynamic_cast<IObjectDetection*>(&task);
        if (!detector)
            throw std::invalid_argument("bundle does not implement object detection");
        return run_detection_protocol(
            [&](const cli::io::LoadedImage& image) {
                return detector->detect(image.pixels.data(), image.height, image.width);
            },
            [](const ObjectDetectionResult& result, std::int32_t height, std::int32_t width) {
                if (result.image_height != height || result.image_width != width)
                    throw std::runtime_error("detection Task returned invalid image dimensions");
                auto boxes = Json::array();
                for (const auto& box : result.boxes)
                    boxes.push_back(detection_box(box.x_min, box.y_min, box.x_max, box.y_max, box,
                                                  height, width));
                return boxes;
            },
            input, output);
    }
    return run_text_worker(task, input, output);
}

int run_model_worker(const Model& model, std::istream& input, std::ostream& output) {
    const auto primary = model.info().bundle_task;
    if (primary == ImagesTextToImageEdit::kTask || primary == IImageEditing::kTask) {
        const auto task = model.task<ImagesTextToImageEdit>();
        const auto fields = task.config_fields();
        return run_edit_protocol(
            [&](const std::string& prompt, const cli::io::LoadedImage& image,
                const std::string& path, const Json& request) {
                const auto config = parse_sdk_config(request, fields);
                const auto latents = image_latents(request);
                const ImagesTextToImageEditRequest edit{
                    {ImageInput{{image.pixels.data(), image.pixels.size()},
                                static_cast<std::uint32_t>(image.height),
                                static_cast<std::uint32_t>(image.width)}},
                    prompt,
                    {latents.data(), latents.size()}};
                const auto started = std::chrono::steady_clock::now();
                const auto result = task.run(edit, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                if (result.width() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
                    result.height() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
                    throw std::runtime_error("image dimensions exceed the PNG writer range");
                cli::io::save_png(path, result.pixels(), static_cast<int>(result.width()),
                                  static_cast<int>(result.height()),
                                  static_cast<int>(result.channels()));
                return Json{
                    {"height", result.height()},     {"width", result.width()},
                    {"channels", result.channels()}, {"output_count", 1},
                    {"model_call_ms", wall_ms},      {"timing_scope", "public_task_call_wall"}};
            },
            input, output);
    }
    if (primary == TextToImage::kTask || primary == IImageGeneration::kTask) {
        const auto task = model.task<TextToImage>();
        const auto fields = task.config_fields();
        return run_image_protocol(
            [&](const std::string& prompt, const std::string& path, const Json& request) {
                const auto config = parse_sdk_config(request, fields);
                const auto latents = image_latents(request);
                const auto started = std::chrono::steady_clock::now();
                const auto result = task.run({prompt, {latents.data(), latents.size()}}, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                if (result.width() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
                    result.height() > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
                    throw std::runtime_error("image dimensions exceed the PNG writer range");
                cli::io::save_png(path, result.pixels(), static_cast<int>(result.width()),
                                  static_cast<int>(result.height()),
                                  static_cast<int>(result.channels()));
                return Json{
                    {"height", result.height()},     {"width", result.width()},
                    {"channels", result.channels()}, {"output_count", 1},
                    {"model_call_ms", wall_ms},      {"timing_scope", "public_task_call_wall"}};
            },
            input, output);
    }
    if (primary == ImagesTextToText::kTask || primary == IVisionLanguageGeneration::kTask) {
        const auto task = model.task<ImagesTextToText>();
        const auto fields = task.config_fields();
        return run_vision_protocol(
            default_token_limit(fields),
            [&](const std::string& prompt, const cli::io::LoadedImage& image, const Json& request) {
                const auto config = parse_sdk_config(request, fields);
                const auto input = ImagesTextToTextRequest::from_parts(
                    {ImageInput{{image.pixels.data(), image.pixels.size()},
                                static_cast<std::uint32_t>(image.height),
                                static_cast<std::uint32_t>(image.width)},
                     TextPart{prompt}});
                const auto started = std::chrono::steady_clock::now();
                const auto result = task.run(input, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                return Json{{"text", std::string(result.text())},
                            {"completion_tokens", result.token_ids().size()},
                            {"setup_ms", result.setup_ms()},
                            {"prefill_ms", result.prefill_ms()},
                            {"decode_ms", result.decode_ms()},
                            {"model_call_ms", wall_ms},
                            {"timing_scope", "public_task_call_wall"}};
            },
            input, output);
    }
    if (model.supports<TextToEmbedding>()) {
        const auto task = model.task<TextToEmbedding>();
        return run_embedding_protocol([&](const std::string& text) { return task.run({text}); },
                                      [](const SemanticEmbeddingResult& result) {
                                          return std::vector<float>(result.values().begin(),
                                                                    result.values().end());
                                      },
                                      input, output);
    }
    if (model.supports<TextToPooledFeatures>()) {
        const auto task = model.task<TextToPooledFeatures>();
        return run_embedding_protocol([&](const std::string& text) { return task.run({text}); },
                                      [](const PooledFeaturesResult& result) {
                                          return std::vector<float>(result.values().begin(),
                                                                    result.values().end());
                                      },
                                      input, output);
    }

    if (model.supports<TextQueryDocumentsToRelevance>()) {
        const auto task = model.task<TextQueryDocumentsToRelevance>();
        return run_ranking_protocol(
            [&](const std::string& query, const std::vector<std::string>& documents) {
                return task.run({query, documents});
            },
            [](const DocumentRelevanceResult& result) {
                if (result.kind() != TRTMC_SCORE_LOGIT)
                    throw std::runtime_error("NIM ranking requires raw logit scores");
                return std::vector<float>(result.scores().begin(), result.scores().end());
            },
            input, output);
    }
    if (model.supports<SpeechTranscription>()) {
        const auto task = model.task<SpeechTranscription>();
        const auto fields = task.config_fields();
        return run_transcription_protocol(
            [&](const cli::io::LoadedAudio& audio, const Json& request) {
                auto mapped = request;
                auto supplied = mapped.find("config");
                const auto declares = [&](std::string_view name) {
                    return std::any_of(fields.begin(), fields.end(),
                                       [&](const auto& field) { return field.name == name; });
                };
                if (supplied != mapped.end() && supplied->is_object() &&
                    supplied->contains("max_output_tokens") && !declares("max_output_tokens") &&
                    declares("max_new_tokens")) {
                    if (supplied->contains("max_new_tokens"))
                        throw ProtocolError("duplicate transcription token limit");
                    (*supplied)["max_new_tokens"] = supplied->at("max_output_tokens");
                    supplied->erase("max_output_tokens");
                }
                const auto config = parse_sdk_config(mapped, fields);
                const SpeechTranscriptionRequest input{
                    {{audio.samples.data(), audio.samples.size()},
                     static_cast<std::uint32_t>(audio.sample_rate),
                     1},
                    {}};
                return timed_transcription(
                    [&] { return task.run(input, config); },
                    [](const auto& result) { return std::string(result.text()); });
            },
            input, output);
    }
    if (model.supports<ImageToBoxes>()) {
        const auto task = model.task<ImageToBoxes>();
        return run_detection_protocol(
            [&](const cli::io::LoadedImage& image) {
                return task.run({ImageInput{{image.pixels.data(), image.pixels.size()},
                                            static_cast<std::uint32_t>(image.height),
                                            static_cast<std::uint32_t>(image.width)}});
            },
            [](const DetectedBoxesResult& result, std::int32_t height, std::int32_t width) {
                const auto view = result.view();
                if (view.image_height != static_cast<std::uint32_t>(height) ||
                    view.image_width != static_cast<std::uint32_t>(width) ||
                    (view.count && !view.boxes))
                    throw std::runtime_error("detection Task returned invalid image metadata");
                auto boxes = Json::array();
                for (std::uint64_t i = 0; i < view.count; ++i) {
                    const auto& box = view.boxes[i];
                    boxes.push_back(detection_box(box.box.x_min, box.box.y_min, box.box.x_max,
                                                  box.box.y_max, box, height, width));
                }
                return boxes;
            },
            input, output);
    }
    return run_text_worker(model, input, output);
}

template <class Task, class Request>
int run_sdk_worker(const Task& task, Request&& make_request, std::istream& input,
                   std::ostream& output, const Model* model = nullptr) {
    const auto fields = task.config_fields();
    StreamGenerate streaming;
    if (model != nullptr && model->supports<StreamingTextContinuation>()) {
        const auto stream_task = model->task<StreamingTextContinuation>();
        const auto stream_fields = stream_task.config_fields();
        streaming = [stream_task, stream_fields](const std::string& prompt, const Json& request,
                                                 const Json& id, std::ostream& out) {
            const auto config = parse_sdk_config(request, stream_fields);
            try {
                const TextContinuationRequest input{prompt};
                const auto started = std::chrono::steady_clock::now();
                auto stream = stream_task.start(input, config);
                while (auto event = stream.next()) {
                    if (event->kind() == StreamEventKind::Delta) {
                        if (!write_json(out, {{"id", id},
                                              {"ok", true},
                                              {"event", "delta"},
                                              {"result",
                                               {{"text_delta", std::string(event->text_delta())},
                                                {"token_count", event->token_ids().size()}}}})) {
                            stream.cancel();
                            throw std::runtime_error("stream output pipe closed");
                        }
                    } else if (event->kind() == StreamEventKind::Complete) {
                        const auto wall_ms = std::chrono::duration<double, std::milli>(
                                                 std::chrono::steady_clock::now() - started)
                                                 .count();
                        const auto final = event->final_result();
                        if (!final)
                            throw std::runtime_error("stream completion has no result");
                        const auto result = *final;
                        return Json{
                            {"text", std::string(result.text)},
                            {"completion_tokens", result.token_ids.size()},
                            {"setup_ms", result.setup_ms},
                            {"prefill_ms", result.prefill_ms},
                            {"decode_ms", result.decode_ms},
                            {"model_call_ms", wall_ms},
                            {"timing_scope", "public_task_stream_wall_including_backpressure"}};
                    } else {
                        throw std::runtime_error("native stream was cancelled");
                    }
                }
                throw std::runtime_error("native stream ended without a terminal result");
            } catch (const Error& error) {
                if (error.code() == TRTMC_INVALID_ARGUMENT || error.code() == TRTMC_INVALID_CONFIG)
                    throw ProtocolError("generation request is invalid for this model");
                throw;
            }
        };
    }
    return run_text_protocol(
        default_token_limit(fields),
        [&](const std::string& prompt, const Json& request) {
            const auto config = parse_sdk_config(request, fields);
            try {
                const auto input = make_request(prompt);
                const auto started = std::chrono::steady_clock::now();
                const auto result = task.run(input, config);
                const auto wall_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - started)
                                         .count();
                return Json{{"text", std::string(result.text())},
                            {"completion_tokens", result.token_ids().size()},
                            {"setup_ms", result.setup_ms()},
                            {"prefill_ms", result.prefill_ms()},
                            {"decode_ms", result.decode_ms()},
                            {"model_call_ms", wall_ms},
                            {"timing_scope", "public_task_call_wall"}};
            } catch (const Error& error) {
                if (error.code() == TRTMC_INVALID_ARGUMENT ||
                    error.code() == TRTMC_INVALID_CONFIG) {
                    // Family errors may contain implementation details. Keep
                    // client mistakes recoverable without publishing those details.
                    throw ProtocolError("generation request is invalid for this model");
                }
                throw;
            }
        },
        input, output, std::move(streaming));
}

} // namespace

int run_task_worker(ITask& task, std::istream& input, std::ostream& output,
                    VideoGeometry geometry) {
    return run_existing_worker(task, input, output, geometry);
}

int run_text_worker(ITask& task, std::istream& input, std::ostream& output) {
    auto* text = dynamic_cast<ITextGeneration*>(&task);
    if (text == nullptr)
        throw std::invalid_argument("bundle task does not implement text generation");
    return run_text_protocol(
        text->default_max_new_tokens(),
        [&](const std::string& prompt, const Json& request) {
            const auto config = parse_config(request, text->default_max_new_tokens());
            const auto started = std::chrono::steady_clock::now();
            const auto result = text->generate(prompt, config);
            const auto wall_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();
            return Json{{"text", result.text},
                        {"completion_tokens", result.token_ids.size()},
                        {"setup_ms", result.setup_ms},
                        {"prefill_ms", result.prefill_ms},
                        {"decode_ms", result.decode_ms},
                        {"model_call_ms", wall_ms},
                        {"timing_scope", "public_task_call_wall"}};
        },
        input, output);
}

int run_text_worker(const Model& model, std::istream& input, std::ostream& output) {
    const auto primary = model.info().bundle_task;
    if (primary == ConditionalTextGeneration::kTask)
        return run_sdk_worker(
            model.task<ConditionalTextGeneration>(),
            [](const std::string& prompt) { return ConditionalTextGenerationRequest{prompt}; },
            input, output);
    if (primary == TextTranslation::kTask)
        return run_sdk_worker(
            model.task<TextTranslation>(),
            [](const std::string& prompt) { return TextTranslationRequest{prompt}; }, input,
            output);
    return run_sdk_worker(
        model.task<TextContinuation>(),
        [](const std::string& prompt) { return TextContinuationRequest{prompt}; }, input, output,
        &model);
}

int run_bundle_worker(const std::string& bundle, const LoadOptions& options, std::istream& input,
                      std::ostream& output) {
    const auto primary = Bundle::open(bundle).info().task;
    VideoGeometry geometry;
    if (primary == IImageGeneration::kTask) {
        const auto provenance = app::read_provenance(bundle);
        // Older image bundles and family-owned default dimensions do not need
        // core geometry. Enable video only for an explicit multi-frame build.
        const auto build = provenance.is_object() ? provenance.value("build", Json{}) : Json{};
        const auto integer = [&](const char* name, int fallback) {
            if (!build.is_object() || !build.contains(name) || build.at(name).is_null())
                return fallback;
            int result = fallback;
            try {
                assign(build, name, result);
            } catch (const std::exception&) {
                throw std::invalid_argument(std::string("invalid built geometry field: ") + name);
            }
            return result;
        };
        geometry.frames = integer("video_num_frames", 1);
        if (geometry.frames < 1)
            throw std::invalid_argument("invalid built video frame count");
        if (geometry.frames > 1) {
            geometry.width = integer("image_width", 0);
            geometry.height = integer("image_height", 0);
            if (geometry.width < 1 || geometry.height < 1 ||
                static_cast<std::uint64_t>(geometry.width) * geometry.height >
                    (512U * 1024U * 1024U) / static_cast<std::uint64_t>(geometry.frames))
                throw std::invalid_argument("invalid or excessive built video geometry");
        }
    }
    if (app::uses_existing_task_runtime(primary)) {
        try {
            const auto model = Model::load(bundle, options);
            if (geometry.frames > 1)
                throw std::invalid_argument("SDK image binding cannot serve a video bundle");
            return run_model_worker(model, input, output);
        } catch (const Error& error) {
            // Only pre-SDK families may take this compatibility path. Never
            // retry a malformed SDK binding or a failed Task operation.
            if (error.code() != TRTMC_UNSUPPORTED ||
                std::string(error.what())
                        .find("family has not implemented the Task SDK model interface") ==
                    std::string::npos)
                throw;
        }
        auto task = load_task(bundle, options.runtime_root, options.kv_cache_size_bytes,
                              options.runtime_cache_path, options.cuda_graphs);
        return run_existing_worker(*task, input, output, geometry);
    }
    const auto model = Model::load(bundle, options);
    return run_model_worker(model, input, output);
}

} // namespace trtmc::server
