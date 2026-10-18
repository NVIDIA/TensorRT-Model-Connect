/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"
#include "server/native_worker.h"
#include "trtmc/core.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::filesystem::path write_bundle(const std::filesystem::path& root, const std::string& task,
                                   const std::string& family = "api_fixture",
                                   const Json* provenance = nullptr) {
    const auto path = root / ("server-sdk-" + family + "-" + task + ".bundle");
    const auto metadata = provenance == nullptr ? std::string{} : provenance->dump();
    Json sections{{"engine.plan", {{"offset", 0}, {"length", 4}}}};
    if (provenance != nullptr)
        sections["trtmc.provenance.json"] = {{"offset", 4}, {"length", metadata.size()}};
    const auto header = Json{{"format", 1},
                             {"family", family},
                             {"task", task},
                             {"backend", "fake"},
                             {"sections", std::move(sections)}}
                            .dump();
    std::ofstream file(path, std::ios::binary);
    file.write("BUNDLE\x01\x00", 8);
    const auto size = static_cast<std::uint64_t>(header.size());
    for (int shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((size >> shift) & 0xffU));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write("PLAN", 4);
    file.write(metadata.data(), static_cast<std::streamsize>(metadata.size()));
    require(static_cast<bool>(file), "write SDK worker fixture bundle");
    return path;
}

Json generate(const char* id, Json config = Json::object()) {
    return {{"id", id}, {"op", "generate"}, {"prompt", "hello"}, {"config", std::move(config)}};
}

std::vector<Json> records(const std::string& output) {
    std::istringstream input(output);
    std::vector<Json> result;
    for (std::string line; std::getline(input, line);)
        result.push_back(Json::parse(line));
    return result;
}

struct Run {
    int status;
    std::vector<Json> messages;
};

Run invoke(const std::filesystem::path& bundle, const trtmc::LoadOptions& options,
           const std::vector<Json>& requests, bool direct_model = false) {
    std::ostringstream encoded;
    for (const auto& request : requests)
        encoded << request.dump() << '\n';
    std::istringstream input(encoded.str());
    std::ostringstream output;
    int status;
    if (direct_model) {
        const auto model = trtmc::Model::load(bundle.string(), options);
        status = trtmc::server::run_text_worker(model, input, output);
    } else {
        status = trtmc::server::run_bundle_worker(bundle.string(), options, input, output);
    }
    return {status, records(output.str())};
}

void test_defaults_and_config(const std::filesystem::path& root,
                              const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "text_continuation");
    const auto defaults = invoke(bundle, options, {generate("defaults")}, true);
    require(defaults.status == 0 && defaults.messages.size() == 2, "SDK model overload succeeds");
    const auto& ready = defaults.messages[0];
    require(ready.at("event") == "ready" && ready.at("protocol_version") == 2 &&
                ready.at("capabilities") == Json::array({"text_generation"}) &&
                ready.at("default_max_new_tokens") == 4,
            "ready transport preserves the family token default");
    const auto& result = defaults.messages[1].at("result");
    require(result.at("text") == "hello!|eos" && result.at("completion_tokens") == 2 &&
                result.at("setup_ms") == 0 && result.at("prefill_ms") == 0.75 &&
                result.at("decode_ms") == 4,
            "missing Config values remain family-owned");

    const Json supplied{{"max_new_tokens", 0},
                        {"temperature", 0.25},
                        {"emit_eos", false},
                        {"suffix", std::string("tail\0end", 8)},
                        {"token_biases", {7, -3}},
                        {"schedule", {1.25, 2.5}},
                        {"labels", {"first", "second"}},
                        {"context_limit", 5}};
    const auto explicit_values = invoke(bundle, options, {generate("configured", supplied)});
    require(explicit_values.status == 0 && explicit_values.messages.size() == 2,
            "all seven Config kinds pass through the bundle entry point");
    const auto& configured = explicit_values.messages[1].at("result");
    require(configured.at("text") == std::string("hellotail\0end", 13) + "|first|second|7" &&
                configured.at("setup_ms") == 1.25 && configured.at("prefill_ms") == 0.25 &&
                configured.at("decode_ms") == 0,
            "explicit zero, false, length-delimited strings and arrays are not defaulted");
}

void test_invalid_config_is_nonfatal(const std::filesystem::path& root,
                                     const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "text_continuation");
    const std::vector<Json> bad_configs{
        {{"unknown", 1}},
        {{"max_new_tokens", 1.5}},
        {{"max_new_tokens", true}},
        {{"max_new_tokens", std::uint64_t{18446744073709551615ULL}}},
        {{"max_new_tokens", -1}},
        {{"temperature", 3.0}},
        {{"context_limit", 1}},
        {{"suffix", 7}},
        {{"emit_eos", 1}},
        {{"token_biases", {1.5}}},
        {{"schedule", {"bad"}}},
        {{"labels", {1}}},
        Json::array(),
        nullptr,
    };
    std::vector<Json> requests;
    for (const auto& config : bad_configs)
        requests.push_back(generate("bad", config));
    requests.push_back(generate("after-errors"));
    requests.push_back({{"id", "stop"}, {"op", "shutdown"}});
    const auto run = invoke(bundle, options, requests);
    require(run.status == 0 && run.messages.size() == requests.size() + 1,
            "invalid SDK requests do not retire the worker");
    for (std::size_t i = 0; i < bad_configs.size(); ++i) {
        const auto& error = run.messages[i + 1];
        require(error.at("id") == "bad" && error.at("ok") == false &&
                    error.at("error").at("type") == "invalid_request_error",
                "type, range and unknown-field errors are client errors");
    }
    require(run.messages[bad_configs.size() + 1].at("result").at("text") == "hello!|eos",
            "a valid request still succeeds after invalid Config");
    require(run.messages.back().at("result").at("status") == "shutting_down",
            "SDK worker acknowledges shutdown");
}

void test_task_selection_and_failures(const std::filesystem::path& root,
                                      const trtmc::LoadOptions& options) {
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
             {"conditional_text_generation", "conditional:hello!"},
             {"text_translation", "translation:fixed-src->en:hello!"}}) {
        const auto run =
            invoke(write_bundle(root, item.first, "text_fixture"), options, {generate("selected")});
        require(run.status == 0 && run.messages.size() == 2 &&
                    run.messages[0].at("default_max_new_tokens") == 128 &&
                    run.messages[1].at("result").at("text") == item.second,
                "primary Task selection does not inject undeclared defaults or languages");
    }
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
             {"text_conversation", "language_fixture"}, {"images_text_to_text", "text_fixture"}}) {
        std::istringstream input;
        std::ostringstream output;
        try {
            trtmc::server::run_bundle_worker(write_bundle(root, item.first, item.second).string(),
                                             options, input, output);
            throw std::runtime_error("server selected a secondary Task instead of the primary");
        } catch (const trtmc::Error& error) {
            require(error.code() == TRTMC_UNSUPPORTED && output.str().empty(),
                    "secondary capabilities cannot enable or replace a bundle's primary Task");
        }
    }
    const auto failure = invoke(write_bundle(root, "must_not_run"), options,
                                {generate("failure"), generate("must-not-retry")});
    require(failure.status == 1 && failure.messages.size() == 2 &&
                failure.messages[1].at("error").at("type") == "runtime_error" &&
                failure.messages[1].at("error").at("message") == "native worker operation failed",
            "SDK runtime error is redacted and retires the worker without retry");
    std::istringstream input;
    std::ostringstream output;
    try {
        trtmc::server::run_bundle_worker(write_bundle(root, "disabled").string(), options, input,
                                         output);
        throw std::runtime_error("a model with no enabled text Task was accepted");
    } catch (const trtmc::Error& error) {
        require(error.code() == TRTMC_UNSUPPORTED && output.str().empty(),
                "unavailable SDK Task fails before ready, with no legacy retry");
    }
}
void test_incremental_stream(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "text_continuation", "stream_fixture");
    const auto run = invoke(bundle, options,
                            {{{"id", "stream"},
                              {"op", "generate_stream"},
                              {"prompt", "hello"},
                              {"config", {{"suffix", "!"}}}},
                             {{"id", "stop"}, {"op", "shutdown"}}});
    require(run.status == 0 && run.messages.size() == 4,
            "stream emits one delta and one terminal then shutdown");
    require(run.messages[0].at("protocol_version") == 2 &&
                run.messages[0].at("capabilities") ==
                    Json::array(
                        {"text_generation", "streaming_text_continuation", "text_conversation"}),
            "ready advertises actual SDK stream capability");
    require(run.messages[1].at("id") == "stream" && run.messages[1].at("event") == "delta" &&
                run.messages[1].at("result").at("text_delta") == "hello!",
            "delta precedes completion");
    require(run.messages[2].at("id") == "stream" && run.messages[2].at("event") == "complete" &&
                run.messages[2].at("result").at("text") == "hello!" &&
                run.messages[2].at("result").at("timing_scope") ==
                    "public_task_stream_wall_including_backpressure" &&
                run.messages[2].at("result").at("model_call_ms").get<double>() >= 0,
            "terminal result carries scoped timing");

    const auto invalid_history =
        invoke(bundle, options,
               {{{"id", "bad-history"}, {"op", "converse"}, {"messages", Json::array()}},
                {{"id", "bad-role"},
                 {"op", "converse"},
                 {"messages", Json::array({{{"role", "tool"}, {"content", "unsupported"}}})}},
                {{"id", "after-history-errors"}, {"op", "generate"}, {"prompt", "hello"}}});
    require(invalid_history.status == 0 && invalid_history.messages.size() == 4,
            "invalid conversation requests preserve the worker");
    for (std::size_t index = 1; index < 3; ++index)
        require(invalid_history.messages[index].at("ok") == false &&
                    invalid_history.messages[index].at("error").at("type") ==
                        "invalid_request_error",
                "bad histories are recoverable request errors");
    require(invalid_history.messages.back().at("result").at("text") == "sync",
            "text generation remains usable after conversation errors");
}
void test_audio_task(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "speech_transcription", "audio_fixture");
    const auto path = root / "server-sdk-input.wav";
    const std::vector<float> values{-1.0F, 0.0F, 0.5F, 1.0F};
    trtmc::cli::io::write_wav_interleaved({values.data(), values.size()}, 8000, 1, path.string());
    const auto run = invoke(bundle, options,
                            {{{"id", "clip"},
                              {"op", "transcribe"},
                              {"audio_path", path.string()},
                              {"config", {{"suffix", "?"}, {"max_output_tokens", 7}}}}});
    std::filesystem::remove(path);
    require(run.status == 0 && run.messages.size() == 2 &&
                run.messages[0].at("capabilities") == Json::array({"audio_transcription"}),
            "the declared SDK speech Task is dispatched");
    const auto& result = run.messages[1].at("result");
    require(result.at("text") == "asr:auto?" && result.at("audio_sample_count") == 4 &&
                result.at("audio_sample_rate") == 8000 &&
                result.at("timing_scope") == "public_task_call_wall",
            "SDK defaults, declared Config, transcript and audio identity survive transport");
}

void test_optional_image_geometry(const std::filesystem::path& root,
                                  const trtmc::LoadOptions& options) {
    const auto png = root / "server-sdk-optional-geometry.png";
    const Json request{{"id", "image"},
                       {"op", "generate_image"},
                       {"prompt", "hello"},
                       {"output_path", png.string()}};
    const auto check_image = [&](const Json* provenance) {
        const auto run = invoke(write_bundle(root, "image_generation", "image_fixture", provenance),
                                options, {request});
        require(run.status == 0 && run.messages.size() == 2 &&
                    run.messages[0]["capabilities"] == Json::array({"image_generation"}) &&
                    run.messages[1]["result"]["width"] == 1 &&
                    run.messages[1]["result"]["height"] == 1,
                "optional core geometry preserves the image Task and its own dimensions");
        const auto image = trtmc::cli::io::read_image(png.string());
        require(image.width == 1 && image.height == 1 && image.pixels[0] == 64.0F / 255.0F,
                "legacy image metadata still produces actual complete image pixels");
    };
    check_image(nullptr);
    for (const auto& metadata : std::vector<Json>{
             nullptr,
             Json::array(),
             Json::object(),
             {{"build", nullptr}},
             {{"build", Json::array()}},
             {{"build", Json::object()}},
             {{"build",
               {{"video_num_frames", nullptr},
                {"image_width", nullptr},
                {"image_height", nullptr}}}},
             {{"build", {{"video_num_frames", 1}, {"image_width", "family default"}}}}})
        check_image(&metadata);
    for (const auto& build :
         std::vector<Json>{{{"video_num_frames", 0}},
                           {{"video_num_frames", -1}},
                           {{"video_num_frames", 1.5}},
                           {{"video_num_frames", true}},
                           {{"video_num_frames", "2"}},
                           {{"video_num_frames", std::numeric_limits<std::uint64_t>::max()}},
                           {{"video_num_frames", 3}},
                           {{"video_num_frames", 3}, {"image_width", nullptr}, {"image_height", 4}},
                           {{"video_num_frames", 3}, {"image_width", 8}, {"image_height", -1}},
                           {{"video_num_frames", 3}, {"image_width", 8}, {"image_height", 4.5}},
                           {{"video_num_frames", 2147483647},
                            {"image_width", 2147483647},
                            {"image_height", 2147483647}}}) {
        const Json metadata{{"build", build}};
        try {
            invoke(write_bundle(root, "image_generation", "image_fixture", &metadata), options, {});
            throw std::runtime_error("invalid explicit video geometry was accepted");
        } catch (const std::invalid_argument& error) {
            require(std::string(error.what()).find("built") != std::string::npos,
                    "invalid video metadata fails before Task selection");
        }
    }
    const Json video{{"build", {{"video_num_frames", 3}, {"image_width", 8}, {"image_height", 4}}}};
    try {
        invoke(write_bundle(root, "image_generation", "image_fixture", &video), options, {});
        throw std::runtime_error("an SDK image binding was advertised as a video Task");
    } catch (const std::invalid_argument& error) {
        require(std::string(error.what()) == "SDK image binding cannot serve a video bundle",
                "valid multi-frame metadata selects video before the SDK compatibility guard");
    }
    std::filesystem::remove(png);
}
void test_audio_task_new_token_field(const std::filesystem::path& root,
                                     const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "asr_new_tokens", "audio_fixture");
    const auto path = root / "server-sdk-token-field.wav";
    const std::vector<float> values{-1.0F, 0.0F, 0.5F, 1.0F};
    trtmc::cli::io::write_wav_interleaved({values.data(), values.size()}, 8000, 1, path.string());
    const auto run = invoke(bundle, options,
                            {{{"id", "duplicate"},
                              {"op", "transcribe"},
                              {"audio_path", path.string()},
                              {"config", {{"max_output_tokens", 7}, {"max_new_tokens", 8}}}},
                             {{"id", "wire"},
                              {"op", "transcribe"},
                              {"audio_path", path.string()},
                              {"config", {{"max_output_tokens", 7}}}},
                             {{"id", "declared"},
                              {"op", "transcribe"},
                              {"audio_path", path.string()},
                              {"config", {{"max_new_tokens", 9}}}}});
    std::filesystem::remove(path);
    require(run.status == 0 && run.messages.size() == 4,
            "token-field mapping errors leave the SDK speech worker usable");
    require(run.messages[1].at("ok") == false &&
                run.messages[1].at("error").at("type") == "invalid_request_error",
            "a wire alias cannot overwrite an explicit Task token limit");
    require(run.messages[2].at("result").at("text") == "asr:auto!:7" &&
                run.messages[3].at("result").at("text") == "asr:auto!:9",
            "wire and declared token limits reach the actual SDK speech Task");
}
void test_detection_task(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "image_to_boxes", "image_boxes_fixture");
    const auto path = root / "server-sdk-input.png";
    std::vector<float> pixels(8 * 4 * 3, 0.0F);
    pixels[0] = 1.0F;
    trtmc::cli::io::save_png(path.string(), pixels, 8, 4);
    const auto run = invoke(bundle, options,
                            {{{"id", "image"}, {"op", "detect"}, {"image_path", path.string()}}});
    std::filesystem::remove(path);
    require(run.status == 0 && run.messages.size() == 2 &&
                run.messages[0].at("capabilities") == Json::array({"object_detection"}),
            "the declared SDK image-to-boxes Task is dispatched");
    const auto& result = run.messages[1].at("result");
    require(result.at("boxes").size() == 2 && result.at("boxes")[0].at("class_id") == 42 &&
                result.at("boxes")[0].at("x_min") == -2 && result.at("boxes")[0].at("y_min") == 1 &&
                result.at("boxes")[0].at("confidence") == .75 && result.at("image_width") == 8 &&
                result.at("image_height") == 4 &&
                result.at("timing_scope") == "public_task_call_wall",
            "SDK defaults and complete native image detection results survive transport");
}

void test_vision_task(const std::filesystem::path& root, const trtmc::LoadOptions& options,
                      const std::string& primary) {
    const auto bundle = write_bundle(root, primary, "language_fixture");
    const auto path = root / "server-sdk-vision.png";
    std::vector<float> pixels(8 * 4 * 3, 0.0F);
    pixels[0] = 1.0F;
    trtmc::cli::io::save_png(path.string(), pixels, 8, 4);
    const Json valid{{"id", "vision"},
                     {"op", "generate_vision"},
                     {"image_path", path.string()},
                     {"prompt", "Question?"},
                     {"config", {{"suffix", "tail"}}}};
    auto invalid = valid;
    invalid["config"]["unknown"] = 1;
    const auto run = invoke(bundle, options, {invalid, valid});
    std::filesystem::remove(path);
    require(run.status == 0 && run.messages.size() == 3 &&
                run.messages[0].at("capabilities") == Json::array({"vision_language_generation"}) &&
                run.messages[1].at("error").at("type") == "invalid_request_error",
            "SDK vision capability and family-declared Config are honored");
    const auto& result = run.messages[2].at("result");
    const auto text = result.at("text").get<std::string>();
    require(text.find("I:1;") != std::string::npos &&
                text.find("T:Question?;") != std::string::npos && text.size() >= 4 &&
                text.compare(text.size() - 4, 4, "tail") == 0 && result.at("image_height") == 4 &&
                result.at("image_width") == 8 && result.at("model_call_ms").get<double>() >= 0 &&
                result.at("timing_scope") == "public_task_call_wall",
            "actual public SDK Task receives image and question with its own Config");
}
} // namespace

void test_image_edit_task(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    const auto bundle = write_bundle(root, "images_text_to_image_edit", "image_fixture");
    const auto image = root / "reference.png", noise = root / "edit-noise.f32",
               png = root / "edited.png";
    trtmc::cli::io::save_png(image.string(), std::vector<float>{1, 0, 0}, 1, 1);
    const float values[]{0.25F, 0, 0};
    std::ofstream(noise, std::ios::binary)
        .write(reinterpret_cast<const char*>(values), sizeof(values));
    const auto run = invoke(bundle, options,
                            {{{"id", "edit"},
                              {"op", "edit_image"},
                              {"prompt", "watercolor"},
                              {"image_path", image.string()},
                              {"output_path", png.string()},
                              {"initial_latents_path", noise.string()},
                              {"config", {{"level", 0.0}}}}});
    require(run.status == 0 && run.messages.size() == 2 &&
                run.messages[0]["capabilities"] == Json::array({"image_edit"}),
            "the declared SDK edit Task is selected");
    const auto result = run.messages[1].at("result");
    const auto decoded = trtmc::cli::io::read_image(png.string());
    require(decoded.width == 1 && decoded.height == 1 && decoded.pixels[0] == 64.0F / 255.0F &&
                decoded.pixels[1] == 1 && decoded.pixels[2] == 1 && result["image_width"] == 1 &&
                result["image_height"] == 1 && result["timing_scope"] == "public_task_call_wall",
            "SDK edit consumes the image, explicit Config and actual replay noise");
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        const std::filesystem::path root(argv[1]);
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        test_defaults_and_config(root, options);
        test_invalid_config_is_nonfatal(root, options);
        test_task_selection_and_failures(root, options);
        test_incremental_stream(root, options);
        test_audio_task(root, options);
        test_optional_image_geometry(root, options);
        test_audio_task_new_token_field(root, options);
        test_detection_task(root, options);
        test_vision_task(root, options, "images_text_to_text");
        test_image_edit_task(root, options);
        test_vision_task(root, options, "vision_language_generation");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
