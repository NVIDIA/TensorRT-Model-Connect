/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pure CPU tests for ordered CLI media loading and strict video manifests.

#include "cli/reference_media.h"
#include "test_helpers.h"
#include "trtmc/trtmc_io.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_text(const std::filesystem::path& path, const std::string& contents) {
    std::ofstream output(path, std::ios::binary);
    output << contents;
    if (!output)
        throw std::runtime_error("failed to write test file " + path.string());
}

void write_image(const std::filesystem::path& path, int width, int height,
                 const std::vector<float>& pixels) {
    trtmc::io::save_png(path.string(), pixels, width, height);
}

std::vector<trtmc::AudioVideoReference> load_video(const std::filesystem::path& manifest) {
    return trtmc::cli::load_reference_inputs(
        {{trtmc::cli::ReferenceInputKind::kVideo, manifest.string()}});
}

bool expect_video_error(const std::filesystem::path& manifest, const std::string& contents,
                        const std::string& expected) {
    write_text(manifest, contents);
    try {
        (void)load_video(manifest);
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find(expected) != std::string::npos)
            return true;
        std::cerr << manifest.filename().string() << ": expected error containing '" << expected
                  << "', got '" << e.what() << "'\n";
        return false;
    }
    std::cerr << manifest.filename().string() << ": expected an error\n";
    return false;
}

bool test_ordered_reference_loading_and_relative_video_manifest() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path root{dir.path()};
    const auto frames_dir = root / "frames";
    std::filesystem::create_directories(frames_dir);
    const std::vector<float> red{1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
    const std::vector<float> green{0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    write_image(frames_dir / "red.png", 2, 1, red);
    write_image(frames_dir / "green.png", 2, 1, green);
    write_image(root / "still.png", 2, 1, red);

    trtmc::MultiChannelAudioResult audio;
    audio.samples = {0.1F, 0.2F, -0.1F, -0.2F};
    audio.num_samples = 2;
    audio.sample_rate = 48000;
    audio.num_channels = 2;
    trtmc::io::write_wav(audio, (root / "sound.wav").string());

    const auto manifest = root / "clip.json";
    write_text(
        manifest,
        R"({"fps":12.5,"frames":["frames/red.png","frames/green.png"],"audio":"sound.wav"})");

    const auto references = trtmc::cli::load_reference_inputs(
        {{trtmc::cli::ReferenceInputKind::kAudio, (root / "sound.wav").string()},
         {trtmc::cli::ReferenceInputKind::kImage, (root / "still.png").string()},
         {trtmc::cli::ReferenceInputKind::kVideo, manifest.string()}});
    if (references.size() != 3 || references[0].kind != trtmc::AudioVideoReferenceKind::kAudio ||
        references[1].kind != trtmc::AudioVideoReferenceKind::kImage ||
        references[2].kind != trtmc::AudioVideoReferenceKind::kVideo)
        return false;
    if (references[0].audio.samples != audio.samples || references[0].audio.num_channels != 2 ||
        references[1].image.width != 2 || references[1].image.height != 1)
        return false;

    const auto& video = references[2].video;
    if (video.num_frames != 2 || video.width != 2 || video.height != 1 ||
        std::abs(video.fps - 12.5F) > 1e-6F || video.pixels.size() != 12U ||
        !video.soundtrack.has_value())
        return false;
    for (std::size_t index = 0; index < red.size(); ++index) {
        if (std::abs(video.pixels[index] - red[index]) > 1e-6F ||
            std::abs(video.pixels[red.size() + index] - green[index]) > 1e-6F)
            return false;
    }
    return video.soundtrack->samples == audio.samples && video.soundtrack->num_channels == 2 &&
           video.soundtrack->num_samples == 2 && video.soundtrack->sample_rate == 48000;
}

bool test_manifest_schema_is_strict() {
    trtmc_test::TempDirGuard dir;
    const std::filesystem::path root{dir.path()};
    write_image(root / "frame.png", 2, 1, {1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F});
    write_image(root / "wrong-size.png", 1, 1, {0.0F, 1.0F, 0.0F});

    bool ok = true;
    ok &= expect_video_error(root / "malformed.json", R"({"fps":24,)", "invalid JSON");
    ok &= expect_video_error(root / "root-array.json", R"([])", "root must be a JSON object");
    ok &= expect_video_error(root / "missing-fps.json", R"({"frames":["frame.png"]})",
                             "missing required field 'fps'");
    ok &= expect_video_error(root / "missing-frames.json", R"({"fps":24})",
                             "missing required field 'frames'");
    ok &= expect_video_error(root / "unknown.json",
                             R"({"fps":24,"frames":["frame.png"],"extra":true})",
                             "unknown field 'extra'");
    ok &= expect_video_error(root / "fps-type.json", R"({"fps":"24","frames":["frame.png"]})",
                             "'fps' must be a finite number");
    ok &= expect_video_error(root / "empty-frames.json", R"({"fps":24,"frames":[]})",
                             "'frames' must be a non-empty array");
    ok &= expect_video_error(root / "absolute.json",
                             std::string{"{\"fps\":24,\"frames\":[\""} +
                                 (root / "frame.png").string() + "\"]}",
                             "path must be relative");
    ok &= expect_video_error(root / "mismatch.json",
                             R"({"fps":24,"frames":["frame.png","wrong-size.png"]})",
                             "frames[1] has dimensions 1x1; expected 2x1");
    ok &= expect_video_error(root / "audio-type.json",
                             R"({"fps":24,"frames":["frame.png"],"audio":null})",
                             "'audio' must be a string");

    const auto no_audio = root / "no-audio.json";
    write_text(no_audio, R"({"fps":24,"frames":["frame.png"]})");
    const auto references = load_video(no_audio);
    ok &= references.size() == 1U && !references[0].video.soundtrack.has_value();
    return ok;
}

} // namespace

int main() {
    bool all_passed = true;
    const auto run = [&](const char* name, bool (*test)()) {
        try {
            const bool passed = test();
            std::cout << name << ": " << (passed ? "PASS" : "FAIL") << '\n';
            all_passed &= passed;
        } catch (const std::exception& e) {
            std::cerr << name << ": unexpected exception: " << e.what() << '\n';
            all_passed = false;
        }
    };

    run("ordered_reference_loading_and_relative_video_manifest",
        test_ordered_reference_loading_and_relative_video_manifest);
    run("manifest_schema_is_strict", test_manifest_schema_is_strict);
    return all_passed ? 0 : 1;
}
