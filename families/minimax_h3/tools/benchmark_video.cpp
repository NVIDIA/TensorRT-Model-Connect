/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"
#include "cli/windows_media.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <chrono>
#include <cwctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <process.h>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

fs::path resolve(const fs::path& directory, const std::string& value) {
    if (value.empty()) throw std::invalid_argument("empty manifest path");
    const auto path = fs::u8path(value);
    return fs::absolute(path.is_absolute() ? path : directory / path).lexically_normal();
}

std::wstring path_key(const fs::path& path) {
    auto key = fs::weakly_canonical(path).wstring();
    std::transform(key.begin(), key.end(), key.begin(), [](wchar_t value) {
        return static_cast<wchar_t>(std::towlower(value));
    });
    return key;
}

void write_json(const fs::path& path, const Json& value) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("cannot open receipt: " + path.u8string());
    stream << value.dump(2) << '\n';
    stream.flush();
    if (!stream) throw std::runtime_error("cannot write receipt: " + path.u8string());
}

trtmc::VideoGenerationMode mode(const Json& item) {
    const auto name = item.at("mode").get<std::string>();
    if (name == "t2va") return trtmc::VideoGenerationMode::kTextToVideoAudio;
    if (name == "fl2va") return trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio;
    if (name == "ref2va") return trtmc::VideoGenerationMode::kReferenceToVideoAudio;
    throw std::invalid_argument("mode must be t2va, fl2va, or ref2va");
}

void validate_item(const Json& item, const fs::path& directory) {
    const std::set<std::string> keys{
        "mode", "prompt", "num_frames", "height", "width", "seed", "num_steps",
        "guidance_scale", "cfg_scale", "first_frame", "last_frame", "references", "output"};
    for (auto it = item.begin(); it != item.end(); ++it)
        if (!keys.count(it.key())) throw std::invalid_argument("unknown request key: " + it.key());
    const auto selected = mode(item);
    if (item.at("prompt").get<std::string>().empty())
        throw std::invalid_argument("prompt must not be empty");
    if (item.contains("height") != item.contains("width"))
        throw std::invalid_argument("height and width must be supplied together");
    const bool endpoints = item.contains("first_frame") || item.contains("last_frame");
    const bool references = item.contains("references") && !item.at("references").empty();
    if ((selected == trtmc::VideoGenerationMode::kTextToVideoAudio && (endpoints || references)) ||
        (selected == trtmc::VideoGenerationMode::kFirstLastFrameToVideoAudio &&
         (!endpoints || references)) ||
        (selected == trtmc::VideoGenerationMode::kReferenceToVideoAudio &&
         (endpoints || !references)))
        throw std::invalid_argument("conditioning inputs do not match the selected mode");
    auto require_input = [&](const std::string& value) {
        if (!fs::is_regular_file(resolve(directory, value)))
            throw std::invalid_argument("input is not a file: " + value);
    };
    for (const auto* key : {"first_frame", "last_frame"})
        if (item.contains(key)) require_input(item.at(key).get<std::string>());
    if (item.contains("references")) {
        if (!item.at("references").is_array())
            throw std::invalid_argument("references must be an ordered array");
        for (const auto& reference : item.at("references")) {
            const auto kind = reference.at("kind").get<std::string>();
            if (kind != "image" && kind != "video" && kind != "audio")
                throw std::invalid_argument("reference kind must be image, video, or audio");
            require_input(reference.at("path").get<std::string>());
        }
    }
}

trtmc::VideoImageInput read_image(const fs::path& path) {
    auto image = trtmc::cli::io::read_image(path.u8string());
    return {std::move(image.pixels), image.height, image.width, 3};
}

trtmc::VideoGenerationRequest prepare(const Json& item, const fs::path& directory,
                                      trtmc::IVideoGeneration& video) {
    // Construct a fresh value for every request; no conditioning survives a mode switch.
    trtmc::VideoGenerationRequest request;
    request.mode = mode(item);
    request.prompt = item.at("prompt").get<std::string>();
    request.config.video_num_frames = item.value("num_frames", 124);
    request.config.height = item.value("height", 0);
    request.config.width = item.value("width", 0);
    request.config.seed = item.value("seed", 0);
    request.config.num_steps = item.value("num_steps", 50);
    request.config.guidance_scale = item.value("guidance_scale", -1.0F);
    request.config.cfg_scale = item.value("cfg_scale", -1.0F);
    if (item.contains("first_frame"))
        request.first_frame = read_image(resolve(directory, item.at("first_frame")));
    if (item.contains("last_frame"))
        request.last_frame = read_image(resolve(directory, item.at("last_frame")));
    const auto policy = video.reference_media_decode_policy();
    for (const auto& input : item.value("references", Json::array())) {
        trtmc::VideoReferenceInput reference;
        const auto path = resolve(directory, input.at("path")).u8string();
        const auto kind = input.at("kind").get<std::string>();
        if (kind == "image") {
            reference.kind = trtmc::VideoReferenceKind::kImage;
            reference.image = read_image(fs::u8path(path));
        } else {
            if (!policy) throw std::runtime_error("family has no reference-media decode policy");
            if (kind == "video") {
                reference.kind = trtmc::VideoReferenceKind::kVideo;
                reference.video = trtmc::cli::read_video_file(path, *policy);
            } else {
                reference.kind = trtmc::VideoReferenceKind::kAudio;
                reference.audio = trtmc::cli::read_audio_file(path, *policy);
            }
        }
        request.references.push_back(std::move(reference));
    }
    return request;
}

int run(const fs::path& manifest_path, bool validate_only) {
    std::ifstream input(manifest_path);
    if (!input) throw std::runtime_error("cannot open manifest");
    const Json manifest = Json::parse(input);
    if (manifest.at("schema_version").get<int>() != 1)
        throw std::invalid_argument("unsupported manifest schema_version");
    const auto directory = fs::absolute(manifest_path).parent_path();
    const auto bundle = resolve(directory, manifest.at("bundle"));
    const auto runtime = resolve(directory, manifest.at("runtime_root"));
    const auto cache = resolve(directory, manifest.at("runtime_cache"));
    if (!fs::is_regular_file(bundle) || !fs::is_directory(runtime))
        throw std::invalid_argument("bundle must be a file and runtime_root a directory");
    const auto& requests = manifest.at("requests");
    if (!requests.is_array() || requests.empty())
        throw std::invalid_argument("requests must be a nonempty array");
    if (fs::exists(cache) && !fs::is_regular_file(cache))
        throw std::invalid_argument("runtime_cache must be a file path");
    auto protect_from_cache = [&](const fs::path& path) {
        if (path_key(cache) == path_key(path) ||
            (fs::exists(cache) && fs::exists(path) && fs::equivalent(cache, path)))
            throw std::invalid_argument("runtime_cache aliases a protected input or output");
    };
    protect_from_cache(bundle);
    protect_from_cache(manifest_path);
    std::set<std::wstring> outputs;
    for (const auto& item : requests) {
        validate_item(item, directory);
        const auto output = resolve(directory, item.at("output"));
        auto receipt = output;
        receipt.replace_extension(".receipt.json");
        if (!trtmc::cli::is_mp4_path(output.u8string()))
            throw std::invalid_argument("output must have an .mp4 extension");
        if (!outputs.insert(path_key(output)).second || fs::exists(output) || fs::exists(receipt))
            throw std::invalid_argument("output/receipt already exists or is duplicated");
        protect_from_cache(output);
        protect_from_cache(receipt);
        for (const auto* key : {"first_frame", "last_frame"})
            if (item.contains(key)) protect_from_cache(resolve(directory, item.at(key)));
        for (const auto& reference : item.value("references", Json::array()))
            protect_from_cache(resolve(directory, reference.at("path")));
    }
    if (validate_only) {
        std::cout << "Validated " << requests.size() << " requests; no task loaded\n";
        return 0;
    }
    fs::create_directories(cache.parent_path());
    const bool cache_existed = fs::exists(cache);
    const auto load_started = Clock::now();
    auto task = trtmc::load_task(bundle.u8string(), runtime.u8string(), 0, cache.u8string(),
                               manifest.value("cuda_graphs", false));
    const double load_ms = milliseconds(load_started);
    auto* video = dynamic_cast<trtmc::IVideoGeneration*>(task.get());
    if (!video) throw std::runtime_error("bundle does not provide IVideoGeneration");
    std::size_t index = 0;
    for (const auto& item : requests) {
        ++index;
        const auto output = resolve(directory, item.at("output"));
        auto receipt_path = output;
        receipt_path.replace_extension(".receipt.json");
        if (fs::exists(output) || fs::exists(receipt_path))
            throw std::runtime_error("refusing to overwrite an output or receipt");
        fs::create_directories(output.parent_path());
        Json receipt{{"schema_version", 1}, {"request_index", index}, {"process_id", _getpid()},
                     {"manifest", fs::absolute(manifest_path).u8string()}, {"request", item},
                     {"bundle", bundle.u8string()}, {"runtime_root", runtime.u8string()},
                     {"runtime_cache", cache.u8string()}, {"same_task_instance", true},
                     {"disk_cache_existed_at_process_start", cache_existed},
                     {"task_load_ms_once", load_ms}, {"cuda_graphs", manifest.value("cuda_graphs", false)},
                     {"timing_scope", "generate_video_ms excludes media preparation and MP4 writing"}};
        std::cerr << "[h3.benchmark] request_begin index=" << index << '\n';
        const auto started = Clock::now();
        const char* phase = "prepare";
        try {
            auto request = prepare(item, directory, *video);
            receipt["prepare_ms"] = milliseconds(started);
            phase = "generate_video";
            const auto generated = Clock::now();
            auto result = video->generate_video(request);
            receipt["generate_video_ms"] = milliseconds(generated);
            if (result.audio.samples.empty())
                throw std::runtime_error("audiovisual request returned no audio");
            phase = "write_mp4";
            const auto encoded = Clock::now();
            trtmc::cli::write_mp4(result, output.u8string());
            receipt["write_mp4_ms"] = milliseconds(encoded);
            receipt["request_wall_ms"] = milliseconds(started);
            receipt["output"] = {{"path", output.u8string()}, {"frames", result.frames.num_frames},
                                 {"width", result.frames.width}, {"height", result.frames.height},
                                 {"fps", result.fps}, {"audio_channels", result.audio.channels},
                                 {"audio_sample_rate", result.audio.sample_rate},
                                 {"audio_samples", result.audio.samples.size()}};
            receipt["status"] = "completed";
            write_json(receipt_path, receipt);
            std::cerr << "[h3.benchmark] request_end index=" << index
                      << " generate_video_ms=" << receipt.at("generate_video_ms") << '\n';
        } catch (const std::exception& error) {
            receipt["status"] = "failed";
            receipt["failed_phase"] = phase;
            receipt["error"] = error.what();
            receipt["request_wall_ms"] = milliseconds(started);
            write_json(receipt_path, receipt);
            throw;
        }
    }
    return 0;
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc == 2 && std::wstring(argv[1]) == L"--help") {
            std::cout << "h3_benchmark_video MANIFEST.json [--validate-only]\n";
            return 0;
        }
        if (argc < 2 || argc > 3 || (argc == 3 && std::wstring(argv[2]) != L"--validate-only"))
            throw std::invalid_argument("usage: h3_benchmark_video MANIFEST.json [--validate-only]");
        return run(fs::path(argv[1]), argc == 3);
    } catch (const std::exception& error) {
        std::cerr << "h3_benchmark_video: " << error.what() << '\n';
        return 1;
    }
}
