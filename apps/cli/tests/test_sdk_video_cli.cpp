/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/io.h"
#include "cli/sdk_dispatch.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace {
using nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
void bundle(const std::filesystem::path& path, std::string mode) {
    const auto header = json{{"format", 1},
                             {"family", "video_fixture"},
                             {"task", std::move(mode)},
                             {"backend", "fake"},
                             {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
                            .dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write("PLAN", 4);
}
struct Run {
    int status;
    std::string output, error;
};
Run run(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& value : args)
        argv.push_back(value.data());
    std::ostringstream output, error;
    const auto status = trtmc::cli::run(static_cast<int>(argv.size()), argv.data(), output, error);
    return {status, output.str(), error.str()};
}
void remove_frames(const std::filesystem::path& directory) {
    for (const auto* name : {"frame-000000.png", "frame-000001.png", "frame-000002.png"})
        std::filesystem::remove(directory / name);
    std::filesystem::remove(directory);
}
void conditioned_video(const std::filesystem::path& root, const std::filesystem::path& model,
                       const std::filesystem::path& output, const std::filesystem::path& initial) {
    const auto image = root / "video_cli_condition.png", four = root / "video_cli_four.f32",
               matrix = root / "video_cli_matrix.f32", per_frame = root / "video_cli_per_frame.f32",
               poses = root / "video_cli_poses.f32", malformed = root / "video_cli_malformed.f32",
               camera_primary = root / "video_cli_camera.bundle";
    const float pixels[] = {0.25F, 0.125F, 0.0625F};
    (void)trtmc::cli::detail::write_image({{pixels, 3}, 1, 1, 3}, image.string());
    const float pinhole[] = {2, 3, 0.25F, 0.5F};
    const float calibration[] = {2, 0, 0.25F, 0, 3, 0.5F, 0, 0, 1};
    const float two_frames[] = {2, 0, 0.25F, 0, 3, 0.5F, 0, 0, 1, 2, 0, 0.25F, 0, 3, 0.5F, 0, 0, 1};
    const float transforms[] = {1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const float invalid[] = {1, 2, 3, 4, 5};
    trtmc::cli::detail::write_binary(four, trtmc::Span<const float>{pinhole});
    trtmc::cli::detail::write_binary(matrix, trtmc::Span<const float>{calibration});
    trtmc::cli::detail::write_binary(per_frame, trtmc::Span<const float>{two_frames});
    trtmc::cli::detail::write_binary(poses, trtmc::Span<const float>{transforms});
    trtmc::cli::detail::write_binary(malformed, trtmc::Span<const float>{invalid});
    bundle(camera_primary, "image_text_camera_trajectory_to_video");
    const auto image_conditioned =
        run({"trtmc", "generate-video", model.string(), "--runtime-root", root.string(), "--prompt",
             "p", "--image", image.string(), "--output", output.string()});
    check(image_conditioned.status == 0,
          "existing image flag selects initial-image video semantics");
    if (image_conditioned.status == 0) {
        const auto frame = trtmc::cli::io::read_image((output / "frame-000000.png").string());
        check(std::abs(frame.pixels[0] - 0.02F) <= 1.0F / 255 &&
                  std::abs(frame.pixels[1] - 0.25F) <= 1.0F / 255,
              "image is a required Task operand rather than a generic Config condition");
    }
    const std::vector<std::string> base{
        "trtmc", "generate-world", model.string(), "--runtime-root", root.string(),  "--prompt",
        "p",     "--image",        image.string(), "--output",       output.string()};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        return run(std::move(args));
    };
    for (const auto& calibration_path : {four, matrix, per_frame}) {
        const auto action = invoke({"--action", "w-2", "--intrinsics", calibration_path.string()});
        check(action.status == 0,
              "SANA action accepts four values, one matrix or per-frame matrices");
        if (action.status == 0) {
            const auto frame = trtmc::cli::io::read_image((output / "frame-000000.png").string());
            const float expected = calibration_path == per_frame ? 0.143F : 0.133F;
            check(std::abs(frame.pixels[0] - 0.08F) <= 1.0F / 255 &&
                      std::abs(frame.pixels[1] - expected) <= 1.0F / 255,
                  "action and explicit calibration layout reach the matching family method");
        }
    }
    for (const bool camera : {false, true}) {
        const auto replay =
            camera ? invoke({"--camera-trajectory", poses.string(), "--intrinsics",
                             per_frame.string(), "--initial-latents-raw", initial.string()})
                   : invoke({"--action", "w-2", "--intrinsics", four.string(),
                             "--initial-latents-raw", initial.string()});
        check(replay.status == 0, "both typed SANA routes preserve initial-latent replay");
        if (replay.status == 0) {
            const auto first = trtmc::cli::io::read_image((output / "frame-000000.png").string());
            const auto next = trtmc::cli::io::read_image((output / "frame-000001.png").string());
            const auto source = trtmc::cli::io::read_image(image.string());
            check(first.pixels == source.pixels && std::abs(next.pixels[0] - 0.3F) <= 1.0F / 255 &&
                      std::abs(next.pixels[1] - 0.5F) <= 1.0F / 255 &&
                      std::abs(next.pixels[2] - 0.7F) <= 1.0F / 255,
                  "family first-frame conditioning survives replay and later values are preserved");
            check(json::parse(replay.output).at("frames").size() == (camera ? 2 : 3),
                  "frame files come from the family result, not a guessed profile/FPS");
        }
    }
    const auto trajectory =
        run({"trtmc", "generate-world", camera_primary.string(), "--runtime-root", root.string(),
             "--prompt", "p", "--image", image.string(), "--camera-trajectory", poses.string(),
             "--intrinsics", four.string(), "--output", output.string()});
    check(trajectory.status == 0, "camera primary and trajectory input need no extra Task flag");
    if (trajectory.status == 0) {
        const auto frame = trtmc::cli::io::read_image((output / "frame-000000.png").string());
        check(
            std::abs(frame.pixels[1] - 0.24F) <= 1.0F / 255,
            "camera matrices retain translation while coordinate/units metadata stay unspecified");
    }
    for (const auto& extra : std::vector<std::vector<std::string>>{
             {"--action", "w-2"},
             {"--intrinsics", four.string()},
             {"--action", "w-2", "--intrinsics", malformed.string()},
             {"--camera-trajectory", malformed.string(), "--intrinsics", four.string()},
             {"--action", "w-2", "--camera-trajectory", poses.string(), "--intrinsics",
              four.string()},
             {"--task", "image_text_action_to_video", "--camera-trajectory", poses.string(),
              "--intrinsics", four.string()},
             {"--task", "image_text_camera_trajectory_to_video", "--action", "w-2", "--intrinsics",
              four.string()}}) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        const auto bad = run(std::move(args));
        check(bad.status != 0 && bad.output.empty(),
              "missing/malformed/conflicting world inputs fail explicitly");
    }
    const auto unsupported_replay =
        run({"trtmc", "generate-video", model.string(), "--runtime-root", root.string(), "--prompt",
             "p", "--image", image.string(), "--initial-latents-raw", initial.string(), "--output",
             output.string()});
    check(unsupported_replay.status != 0 && unsupported_replay.output.empty(),
          "initial-image Task cannot silently discard an unsupported replay operand");
    for (const auto& path : {image, four, matrix, per_frame, poses, malformed, camera_primary})
        std::filesystem::remove(path);
}
void exercise(const std::filesystem::path& root) {
    const auto all = root / "video_cli.bundle", untimed = root / "video_cli_untimed.bundle",
               worker = root / "video_cli_worker.bundle", disabled = root / "video_cli_none.bundle",
               output = root / "video_cli_frames", worker_output = root / "video_cli_worker_frames",
               initial = root / "video_cli_initial.f32";
    bundle(all, "text_to_video");
    bundle(untimed, "unknown_time");
    bundle(worker, "worker");
    bundle(disabled, "none");
    const std::vector<std::string> base{
        "trtmc", "generate-video", all.string(),   "--runtime-root", root.string(), "--prompt",
        "p",     "--output",       output.string()};
    auto invoke = [&](std::initializer_list<std::string> extra) {
        auto args = base;
        args.insert(args.end(), extra.begin(), extra.end());
        return run(std::move(args));
    };
    const auto basic = invoke({});
    check(basic.status == 0, "minimal text-to-video command needs no extra Task selector");
    if (basic.status != 0)
        throw std::runtime_error(basic.error);
    const auto metadata = json::parse(basic.output);
    check(metadata.at("height") == 1 && metadata.at("width") == 1 &&
              metadata.at("frames").size() == 3 &&
              metadata.at("frames").at(0) == (output / "frame-000000.png").string() &&
              metadata.at("timestamps_seconds") == json::array({0, 0.1, 0.3}) &&
              metadata.at("setup_ms") == 0.5 && metadata.at("inference_ms") == 1.5 &&
              !metadata.contains("fps"),
          "existing frame filenames/JSON retain actual nonuniform timeline without fabricated FPS");
    check(trtmc::cli::io::read_image((output / "frame-000001.png").string()).pixels.size() == 3,
          "each reported frame is a readable image, not a placeholder");
    const float values[] = {0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F};
    trtmc::cli::detail::write_binary(initial, trtmc::Span<const float>{values});
    const auto replay = invoke({"--initial-latents-raw", initial.string()});
    check(replay.status == 0,
          "existing raw initial-latent command crosses the typed video request");
    if (replay.status == 0) {
        const auto frame = trtmc::cli::io::read_image((output / "frame-000002.png").string());
        check(std::abs(frame.pixels[0] - 0.5F) <= 1.0F / 255 &&
                  std::abs(frame.pixels[2] - 0.7F) <= 1.0F / 255,
              "family replay values reach real output pixels without shared shape inference");
    }
    const auto no_time = run({"trtmc", "generate-video", untimed.string(), "--runtime-root",
                              root.string(), "--prompt", "p", "--output", output.string()});
    check(no_time.status == 0 && json::parse(no_time.output).at("timestamps_seconds").empty(),
          "missing physical timing stays unspecified");
    const auto participant =
        run({"trtmc", "generate-video", worker.string(), "--runtime-root", root.string(),
             "--prompt", "p", "--output", worker_output.string()});
    check(participant.status == 0 && json::parse(participant.output) == json{{"worker", true}} &&
              !std::filesystem::exists(worker_output),
          "worker completion neither creates a directory nor writes an empty video");
    const auto unsupported = run({"trtmc", "generate-video", disabled.string(), "--runtime-root",
                                  root.string(), "--prompt", "p", "--output", output.string()});
    check(unsupported.status != 0 && unsupported.output.empty(),
          "missing video Task does not retry an image or existing runtime method");
    const auto wrong_task = invoke({"--task", "text_to_image"});
    check(wrong_task.status != 0 && wrong_task.output.empty(),
          "video command does not reinterpret an image Task");
    conditioned_video(root, all, output, initial);
    remove_frames(output);
    for (const auto& path : {all, untimed, worker, disabled, initial})
        std::filesystem::remove(path);
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        ++failures;
    }
    if (!failures)
        std::cout << "ALL PASSED\n";
    return failures ? 1 : 0;
}
