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
void bundle(const std::filesystem::path& path, std::string mode,
            const char* family = "perception_fixture") {
    const auto header = json{
        {"format", 1},
        {"family", family},
        {"task", std::move(mode)},
        {"backend", "fake"},
        {"sections",
         json::object()}}.dump();
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
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
void masks_and_grounding(const std::filesystem::path& root, const std::filesystem::path& model,
                         const std::filesystem::path& image) {
    const auto first = root / "perception_cli_first.bundle",
               empty = root / "perception_cli_empty.bundle",
               points = root / "perception_cli_points.bundle";
    bundle(first, "first_vs_best");
    bundle(empty, "empty_masks");
    bundle(points, "image_points_to_masks");
    const auto semantic = run({"trtmc", "segment", model.string(), "--runtime-root", root.string(),
                               "--image", image.string()});
    check(semantic.status == 0, "semantic segmentation selects its actual class-label Task");
    if (semantic.status == 0) {
        const auto value = json::parse(semantic.output);
        check(value.at("mask").at(0) == 255 && value.at("mask").at(1) == 5 &&
                  value.at("ignore_label") == 255 && value.at("background_label") == 0 &&
                  value.at("class_ids") == json::array({0, 5}) &&
                  value.at("class_scores").size() == 12 && value.at("score_kind") == "logit",
              "class map, ignore/background labels and full score axes are not recomputed or "
              "collapsed");
    }
    const auto center = run({"trtmc", "segment", points.string(), "--runtime-root", root.string(),
                             "--image", image.string()});
    check(center.status == 0 && json::parse(center.output).at("point").at("x") == 1 &&
              json::parse(center.output).at("point").at("y") == 1,
          "SAM-style primary gets the quantized center-point caller helper by default");
    const auto selected = run({"trtmc", "segment", first.string(), "--runtime-root", root.string(),
                               "--image", image.string(), "--task", "image_points_to_masks"});
    check(selected.status == 0, "first-vs-best adversarial mask fixture executes");
    if (selected.status == 0) {
        const auto value = json::parse(selected.output);
        check(value.at("mask") == json::array({0, 1, 0, 1, 0, 1}) &&
                  value.at("selected_mask_index") == 0 &&
                  value.at("iou_scores").at(0) < value.at("iou_scores").at(1) &&
                  value.at("num_masks") == 2 && value.at("masks").size() == 12,
              "caller helper selects the first family mask, not argmax IoU, and retains full raw "
              "output");
    }
    const auto no_masks = run({"trtmc", "segment", empty.string(), "--runtime-root", root.string(),
                               "--image", image.string(), "--task", "image_points_to_masks"});
    check(no_masks.status == 0, "genuinely empty mask results remain representable");
    if (no_masks.status == 0) {
        const auto value = json::parse(no_masks.output);
        check(value.at("num_masks") == 0 && value.at("mask").empty() && value.at("masks").empty() &&
                  value.at("selected_mask_index").is_null(),
              "missing masks are not replaced by a fabricated full-image zero mask");
    }
    const auto prompted = run({"trtmc", "segment-prompted", model.string(), "--runtime-root",
                               root.string(), "--image", image.string(), "--point-x", "0.9",
                               "--point-y", "0.2", "--foreground", "false"});
    check(prompted.status == 0,
          "old fractional point flags cross the typed pixel-coordinate request");
    if (prompted.status == 0) {
        const auto value = json::parse(prompted.output);
        check(value.at("point").at("x") == 2 && value.at("point").at("y") == 0 &&
                  value.at("boxes").at(0) == 2 && value.at("low_res_logits").at(0) == -7 &&
                  value.at("mask_kind") == "logits" && value.at("low_res_width") == 2,
              "point polarity and decoder-space prior logits retain explicit semantics");
    }
    const auto text = run({"trtmc", "segment-prompted", model.string(), "--runtime-root",
                           root.string(), "--image", image.string(), "--prompt", "bird"});
    check(text.status == 0, "text concept uses its independent instance-mask Task");
    if (text.status == 0) {
        const auto value = json::parse(text.output);
        check(value.at("mask_kind") == "binary" && value.at("iou_scores").empty() &&
                  value.at("confidence").size() == 2 &&
                  std::abs(value.at("confidence").at(0).get<float>() - 0.8F) < 1e-6 &&
                  value.at("object_ids") == json::array({100, 101}),
              "SAM3-style detection confidence is not mislabeled as predicted IoU");
    }
    const auto proposals = run({"trtmc", "segment", model.string(), "--runtime-root", root.string(),
                                "--image", image.string(), "--task", "image_to_mask_proposals"});
    check(proposals.status == 0,
          "explicit automatic-mask Task retains its full proposal collection");
    if (proposals.status == 0) {
        const auto value = json::parse(proposals.output);
        check(value.at("proposals").at(0).at("seed_points").size() == 2 &&
                  value.at("proposals").at(0).at("crop_box") == json::array({0, 0, 3, 2}) &&
                  value.at("stability_scores").at(0) == 0.75,
              "proposal area, crop, seeds and stability are not discarded");
    }
    const auto detected = run({"trtmc", "detect", model.string(), "--runtime-root", root.string(),
                               "--image", image.string(), "--prompt", "bird"});
    check(detected.status == 0, "grounding query selects ImageTextToBoxes");
    if (detected.status == 0) {
        const auto value = json::parse(detected.output);
        check(value.at("boxes") == json::array({0.25, 0.5, 2.75, 1.75}) &&
                  value.at("labels") == json::array({"bird"}) &&
                  value.at("scores").at(0).is_null() && !value.contains("classes") &&
                  value.at("parse_complete") == true,
              "grounded labels and unknown confidence do not manufacture class IDs or scores");
    }
    const auto grounded_points =
        run({"trtmc", "detect", model.string(), "--runtime-root", root.string(), "--image",
             image.string(), "--prompt", "bird", "--task", "image_text_to_points"});
    check(grounded_points.status == 0 &&
              json::parse(grounded_points.output).at("points") == json::array({2.5, 1.5}),
          "explicit point grounding preserves its distinct coordinate output");
    const auto unparsed = run({"trtmc", "detect", model.string(), "--runtime-root", root.string(),
                               "--image", image.string(), "--prompt", "malformed"});
    check(unparsed.status == 0 && json::parse(unparsed.output).at("boxes").empty() &&
              json::parse(unparsed.output).at("parse_complete") == false &&
              json::parse(unparsed.output).at("raw_response") == "unparsed output",
          "incomplete family parsing stays visible without fabricated coordinates");
    const auto conflict =
        run({"trtmc", "segment-prompted", model.string(), "--runtime-root", root.string(),
             "--image", image.string(), "--prompt", "bird", "--point-x", "0.5"});
    check(conflict.status != 0 && conflict.output.empty(),
          "text and point prompt contracts are not silently combined");
    for (const auto& path : {first, empty, points})
        std::filesystem::remove(path);
}

void tracked_clip(const std::filesystem::path& root, const std::filesystem::path& image) {
    const auto detected = root / "perception_cli_detected_clip.bundle",
               text = root / "perception_cli_text_clip.bundle",
               two_stage = root / "perception_cli_two_stage.bundle";
    bundle(detected, "frames_to_detected_mask_tracks", "tracking_fixture");
    bundle(text, "frames_text_to_mask_tracks", "tracking_fixture");
    bundle(two_stage, "prompt_frame_text_to_mask_tracks", "tracking_fixture");
    auto arguments = [&](const std::filesystem::path& path) {
        std::vector<std::string> args{"trtmc", "video-segment", path.string(), "--runtime-root",
                                      root.string()};
        for (int i = 0; i < 5; ++i)
            args.insert(args.end(), {"--frame", image.string()});
        return args;
    };
    const auto result = run(arguments(detected));
    check(result.status == 0, "detector tracking receives one complete five-frame clip");
    if (result.status == 0) {
        const auto value = json::parse(result.output);
        check(value.at("frames").size() == 5 && value.at("frames").at(4).at("frame_index") == 4 &&
                  value.at("frames").at(0).at("object_ids") == json::array({7}) &&
                  value.at("frames").at(0).at("class_ids") == json::array({2}) &&
                  value.at("frames").at(0).at("mask_element_type") == "uint8" &&
                  value.at("initial_detections").at(0).at("prompt_box") ==
                      json::array({0, 0, 3, 2}),
              "whole-clip masks, persistent identities and initial detector evidence survive");
    }
    auto text_args = arguments(text);
    text_args.insert(text_args.end(), {"--prompt", "bird"});
    const auto text_result = run(std::move(text_args));
    check(text_result.status == 0, "text tracking delegates whole-clip composition to the family");
    if (text_result.status == 0) {
        const auto value = json::parse(text_result.output);
        const auto& frame = value.at("frames").at(0);
        check(value.at("frames").size() == 5 && frame.at("masks").at(0) == 1 &&
                  frame.at("removed_object_ids") == json::array({19}) &&
                  frame.at("suppressed_object_ids") == json::array({23}) &&
                  frame.at("detection_scores").at(0) == 0.875 &&
                  frame.at("tracking_scores").at(0) == 0.625 &&
                  frame.at("mask_element_type") == "float32" &&
                  frame.at("timestamp_seconds").is_null() && !value.contains("fps"),
              "family final tracks retain score kinds, removal/suppression and unknown physical "
              "times");
    }
    auto invalid = arguments(two_stage);
    invalid.insert(invalid.end(), {"--prompt", "bird"});
    const auto rejected = run(std::move(invalid));
    check(rejected.status != 0 && rejected.output.empty(),
          "whole-clip command does not emulate an independently declared two-stage lifecycle");
    const auto short_clip = run({"trtmc", "video-segment", detected.string(), "--runtime-root",
                                 root.string(), "--frame", image.string()});
    check(short_clip.status != 0 && short_clip.output.empty(),
          "family clip-length constraints are not padded by CLI");
    for (const auto& path : {detected, text, two_stage})
        std::filesystem::remove(path);
}
void exercise(const std::filesystem::path& root) {
    const auto all = root / "perception_cli.bundle",
               restricted = root / "perception_cli_pose.bundle",
               left = root / "perception_cli_left.png", right = root / "perception_cli_right.png",
               small = root / "perception_cli_small.png",
               directory = root / "perception_cli_geometry";
    bundle(all, "image_to_metric_geometry");
    bundle(restricted, "pose_only");
    const std::vector<float> a(18, 0.75F), b(18, 0.25F), one(3, 0.5F);
    (void)trtmc::cli::detail::write_image({{a.data(), a.size()}, 2, 3, 3}, left.string());
    (void)trtmc::cli::detail::write_image({{b.data(), b.size()}, 2, 3, 3}, right.string());
    (void)trtmc::cli::detail::write_image({{one.data(), one.size()}, 1, 1, 3}, small.string());
    const auto stereo = run({"trtmc", "disparity", all.string(), "--runtime-root", root.string(),
                             "--left", left.string(), "--right", right.string()});
    check(stereo.status == 0, "minimal stereo CLI reaches the public disparity Task");
    if (stereo.status != 0)
        throw std::runtime_error(stereo.error);
    const auto result = json::parse(stereo.output);
    const auto expected = trtmc::cli::io::read_image(left.string()).pixels[0] -
                          trtmc::cli::io::read_image(right.string()).pixels[0];
    check(result.at("height") == 2 && result.at("width") == 3 &&
              result.at("disparity").size() == 6 &&
              std::abs(result.at("disparity").at(0).get<float>() - expected) < 1e-6 &&
              result.at("units") == "pixels" && result.at("grid") == "left_image",
          "disparity keeps left/right roles, actual values and typed pixel-grid semantics");
    const std::vector<std::string> base{"trtmc",          "geometry",    all.string(),
                                        "--runtime-root", root.string(), "--image",
                                        left.string(),    "--output",    directory.string()};
    auto geometry = run(base);
    check(geometry.status == 0, "minimal geometry CLI keeps meaningful family defaults");
    if (geometry.status != 0)
        throw std::runtime_error(geometry.error);
    check(json::parse(geometry.output).at("height") == 2 &&
              std::filesystem::file_size(directory / "points.f32") == 18 * sizeof(float) &&
              std::filesystem::file_size(directory / "depth.f32") == 6 * sizeof(float) &&
              std::filesystem::file_size(directory / "mask.u8") == 6,
          "geometry retains the existing complete binary map files");
    {
        float invalid_depth = 0;
        std::ifstream depth(directory / "depth.f32", std::ios::binary);
        depth.read(reinterpret_cast<char*>(&invalid_depth), sizeof(invalid_depth));
        std::ifstream mask(directory / "mask.u8", std::ios::binary);
        check(std::isinf(invalid_depth) && mask.get() == 0 && mask.get() == 1,
              "invalid geometry infinity and its authoritative mask are not replaced by zeros");
        std::ifstream input(directory / "intrinsics.json");
        const auto calibration = json::parse(input);
        check(calibration.at("normalized") == true && calibration.at("intrinsics").size() == 3 &&
                  std::abs(calibration.at("intrinsics").at(0).at(0).get<float>() - 0.6F) < 1e-6,
              "normalized intrinsics retain the family default and existing file schema");
    }
    auto explicit_fov = base;
    explicit_fov.insert(explicit_fov.end(), {"--set", "fov_x=90"});
    check(run(std::move(explicit_fov)).status == 0,
          "known FoV passes through family-declared Config");
    {
        std::ifstream input(directory / "intrinsics.json");
        check(std::abs(json::parse(input).at("intrinsics").at(0).at(0).get<float>() - 0.9F) < 1e-6,
              "explicit FoV is not overwritten by an application default");
    }
    const auto mismatch = run({"trtmc", "disparity", all.string(), "--runtime-root", root.string(),
                               "--left", left.string(), "--right", small.string()});
    check(mismatch.status != 0 && mismatch.output.empty(),
          "mismatched stereo grids fail before inference");
    const auto unsupported =
        run({"trtmc", "geometry", restricted.string(), "--runtime-root", root.string(), "--image",
             left.string(), "--output", directory.string()});
    check(unsupported.status != 0 && unsupported.output.empty(),
          "missing geometry Task has no old-method fallback");
    masks_and_grounding(root, all, left);
    tracked_clip(root, left);
    for (const auto* name : {"points.f32", "depth.f32", "mask.u8", "intrinsics.json"})
        std::filesystem::remove(directory / name);
    std::filesystem::remove(directory);
    for (const auto& path : {all, restricted, left, right, small})
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
