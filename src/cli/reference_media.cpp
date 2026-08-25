/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/reference_media.h"

#include "trtmc/trtmc_io.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::cli {
namespace {

using Json = nlohmann::json;

std::runtime_error manifest_error(const std::filesystem::path& manifest_path,
                                  const std::string& message) {
    return std::runtime_error("reference video manifest " + manifest_path.string() + ": " +
                              message);
}

MediaImageInput load_image(const std::filesystem::path& path, const std::string& description) {
    trtmc::io::LoadedImage loaded;
    try {
        loaded = trtmc::io::read_image(path.string());
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to load " + description + " " + path.string() + ": " +
                                 e.what());
    }
    if (loaded.empty() || loaded.height <= 0 || loaded.width <= 0)
        throw std::runtime_error("failed to decode " + description + " " + path.string());

    const auto height = static_cast<std::size_t>(loaded.height);
    const auto width = static_cast<std::size_t>(loaded.width);
    if (height > std::numeric_limits<std::size_t>::max() / width ||
        height * width > std::numeric_limits<std::size_t>::max() / 3U ||
        loaded.pixels.size() != height * width * 3U)
        throw std::runtime_error(description +
                                 " has an invalid RGB pixel buffer: " + path.string());

    MediaImageInput image;
    image.pixels = std::move(loaded.pixels);
    image.height = loaded.height;
    image.width = loaded.width;
    return image;
}

Json parse_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input)
        throw manifest_error(manifest_path, "cannot open file");
    const std::string contents{std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()};
    try {
        return Json::parse(contents);
    } catch (const Json::exception& e) {
        throw manifest_error(manifest_path, std::string("invalid JSON: ") + e.what());
    }
}

void validate_manifest_keys(const Json& manifest, const std::filesystem::path& manifest_path) {
    if (!manifest.is_object())
        throw manifest_error(manifest_path, "root must be a JSON object");
    for (const auto& [key, value] : manifest.items()) {
        (void)value;
        if (key != "fps" && key != "frames" && key != "audio")
            throw manifest_error(manifest_path, "unknown field '" + key + "'");
    }
    if (!manifest.contains("fps"))
        throw manifest_error(manifest_path, "missing required field 'fps'");
    if (!manifest.contains("frames"))
        throw manifest_error(manifest_path, "missing required field 'frames'");
}

std::filesystem::path relative_media_path(const Json& value, const char* field,
                                          const std::filesystem::path& manifest_path) {
    if (!value.is_string())
        throw manifest_error(manifest_path, std::string("'") + field + "' must be a string");
    const auto text = value.get<std::string>();
    if (text.empty())
        throw manifest_error(manifest_path, std::string("'") + field + "' must not be empty");
    const std::filesystem::path relative_path{text};
    if (relative_path.is_absolute())
        throw manifest_error(manifest_path, std::string("'") + field + "' path must be relative");
    return (manifest_path.parent_path() / relative_path).lexically_normal();
}

MediaVideoInput load_video_manifest(const std::filesystem::path& manifest_path) {
    const Json manifest = parse_manifest(manifest_path);
    validate_manifest_keys(manifest, manifest_path);

    const auto& fps_value = manifest.at("fps");
    if (!fps_value.is_number())
        throw manifest_error(manifest_path, "'fps' must be a finite number greater than zero");
    double fps = 0.0;
    try {
        fps = fps_value.get<double>();
    } catch (const Json::exception& e) {
        throw manifest_error(manifest_path, std::string("invalid 'fps': ") + e.what());
    }
    if (!std::isfinite(fps) || fps <= 0.0 || fps > std::numeric_limits<float>::max())
        throw manifest_error(manifest_path, "'fps' must be a finite number greater than zero");

    const auto& frames = manifest.at("frames");
    if (!frames.is_array() || frames.empty())
        throw manifest_error(manifest_path, "'frames' must be a non-empty array");
    if (frames.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw manifest_error(manifest_path, "'frames' contains too many entries");

    MediaVideoInput video;
    video.fps = static_cast<float>(fps);
    for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto field = "frames[" + std::to_string(index) + "]";
        const auto frame_path = relative_media_path(frames.at(index), field.c_str(), manifest_path);
        const auto frame = load_image(frame_path, "reference video frame " + std::to_string(index));
        if (index == 0) {
            video.height = frame.height;
            video.width = frame.width;
            if (frame.pixels.size() > std::numeric_limits<std::size_t>::max() / frames.size())
                throw manifest_error(manifest_path, "combined frame pixel buffer is too large");
            video.pixels.reserve(frame.pixels.size() * frames.size());
        } else if (frame.height != video.height || frame.width != video.width) {
            throw manifest_error(manifest_path, field + " has dimensions " +
                                                    std::to_string(frame.width) + "x" +
                                                    std::to_string(frame.height) + "; expected " +
                                                    std::to_string(video.width) + "x" +
                                                    std::to_string(video.height));
        }
        video.pixels.insert(video.pixels.end(), frame.pixels.begin(), frame.pixels.end());
    }
    video.num_frames = static_cast<int32_t>(frames.size());

    if (manifest.contains("audio")) {
        const auto audio_path = relative_media_path(manifest.at("audio"), "audio", manifest_path);
        try {
            auto soundtrack = trtmc::io::read_wav_multichannel(audio_path.string());
            if (soundtrack.samples.empty())
                throw std::runtime_error("audio contains no samples");
            video.soundtrack = std::move(soundtrack);
        } catch (const std::exception& e) {
            throw manifest_error(manifest_path,
                                 "failed to load 'audio' " + audio_path.string() + ": " + e.what());
        }
    }
    return video;
}

AudioVideoReference load_reference(const ReferenceInput& input) {
    if (input.path.empty())
        throw std::runtime_error("reference media path must not be empty");

    AudioVideoReference reference;
    switch (input.kind) {
    case ReferenceInputKind::kImage:
        reference.kind = AudioVideoReferenceKind::kImage;
        reference.image = load_image(input.path, "reference image");
        return reference;
    case ReferenceInputKind::kAudio:
        reference.kind = AudioVideoReferenceKind::kAudio;
        reference.audio = trtmc::io::read_wav_multichannel(input.path);
        if (reference.audio.samples.empty())
            throw std::runtime_error("reference audio contains no samples: " + input.path);
        return reference;
    case ReferenceInputKind::kVideo:
        reference.kind = AudioVideoReferenceKind::kVideo;
        reference.video = load_video_manifest(input.path);
        return reference;
    }
    throw std::runtime_error("unknown reference input kind");
}

} // namespace

std::vector<AudioVideoReference> load_reference_inputs(const std::vector<ReferenceInput>& inputs) {
    std::vector<AudioVideoReference> references;
    references.reserve(inputs.size());
    for (const auto& input : inputs)
        references.push_back(load_reference(input));
    return references;
}

} // namespace trtmc::cli
