/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/video.hpp"

#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace trtmc::cli {
namespace {
using detail::has_option;
using detail::require_option;

ImageInput image_view(const io::LoadedImage& image) {
    return {{image.pixels.data(), image.pixels.size()},
            static_cast<std::uint32_t>(image.height),
            static_cast<std::uint32_t>(image.width)};
}

std::vector<float> initial_latents(const Command& command) {
    return has_option(command, "--initial-latents-raw")
               ? detail::read_float32_file(command.options.at("--initial-latents-raw"))
               : std::vector<float>{};
}

std::vector<float> intrinsics(const Command& command) {
    auto values = detail::read_float32_file(require_option(command, "--intrinsics"));
    if (values.size() == 4) {
        const auto matrix = pinhole_intrinsics(values[0], values[1], values[2], values[3]);
        values.assign(matrix.begin(), matrix.end());
    } else if (values.size() % 9 != 0) {
        throw std::invalid_argument(
            "--intrinsics must contain fx,fy,cx,cy or row-major 3x3 matrices");
    }
    return values;
}

nlohmann::json write_video(const VideoGenerationResult& video, const std::string& directory) {
    const auto frames = video.frames();
    // The C result has already validated the exact worker-completion sentinel.
    // It carries no image/frame payload; never create an empty output directory.
    if (frames.empty())
        return {{"worker", true}};
    std::filesystem::create_directories(directory);
    nlohmann::json files = nlohmann::json::array();
    for (std::size_t frame = 0; frame < frames.size(); ++frame) {
        std::ostringstream name;
        name << "frame-" << std::setw(6) << std::setfill('0') << frame << ".png";
        const auto path = (std::filesystem::path(directory) / name.str()).string();
        const auto& image = frames[frame];
        (void)detail::write_image({{image.pixels, static_cast<std::size_t>(image.pixel_count)},
                                   image.height,
                                   image.width,
                                   image.channels},
                                  path);
        files.push_back(path);
    }
    nlohmann::json times = nlohmann::json::array();
    for (const auto time : video.timestamps_seconds())
        times.push_back(time);
    return {{"output", directory},
            {"frames", std::move(files)},
            {"height", frames[0].height},
            {"width", frames[0].width},
            {"channels", frames[0].channels},
            {"timestamps_seconds", std::move(times)},
            {"conditioned_prefix_frames", video.conditioned_prefix_frames()},
            {"setup_ms", video.setup_ms()},
            {"inference_ms", video.inference_ms()}};
}

void generate_world(const Command& command, const Model& model, std::string_view id,
                    std::ostream& output) {
    const bool camera = id == ImageTextCameraTrajectoryToVideo::kTask;
    if ((camera && has_option(command, "--action")) ||
        (!camera && has_option(command, "--camera-trajectory")))
        throw std::invalid_argument(
            "--action and --camera-trajectory select different video Tasks");
    const auto image = detail::read_image(require_option(command, "--image"));
    const auto prompt = require_option(command, "--prompt");
    const auto calibration = intrinsics(command);
    const FloatMatrixView matrix{
        {calibration.data(), calibration.size()}, calibration.size() / 9, 9};
    const auto initial = initial_latents(command);
    const auto directory = require_option(command, "--output");
    if (camera) {
        const auto poses =
            detail::read_float32_file(require_option(command, "--camera-trajectory"));
        if (poses.size() % 16 != 0)
            throw std::invalid_argument(
                "--camera-trajectory requires row-major 4x4 camera-to-world matrices");
        const auto task = model.task<ImageTextCameraTrajectoryToVideo>();
        const auto config =
            detail::task_config(command, task.config_fields(),
                                {"--image", "--prompt", "--intrinsics", "--camera-trajectory",
                                 "--output", "--initial-latents-raw"});
        const CameraTrajectoryView trajectory{
            {{poses.data(), poses.size()}, poses.size() / 16, 16}, {}, "", ""};
        // Matrix shape comes from its declared file format. Coordinates, units,
        // timing and the model profile are not guessed by this application.
        const auto result = task.run(
            {image_view(image), prompt, trajectory, matrix, {initial.data(), initial.size()}},
            config);
        detail::write_json(output, write_video(result, directory));
    } else {
        const auto task = model.task<ImageTextActionToVideo>();
        const auto config = detail::task_config(command, task.config_fields(),
                                                {"--image", "--prompt", "--action", "--intrinsics",
                                                 "--output", "--initial-latents-raw"});
        const auto result = task.run({image_view(image),
                                      prompt,
                                      require_option(command, "--action"),
                                      matrix,
                                      std::nullopt,
                                      {initial.data(), initial.size()}},
                                     config);
        detail::write_json(output, write_video(result, directory));
    }
}
} // namespace

std::string_view video_task_for_command(const Command& command, const Model& model) {
    if (!command.selected_task.empty())
        return {};
    if (command.kind == CommandKind::kGenerateVideo) {
        if (has_option(command, "--image") ||
            model.info().bundle_task == InitialImageTextToVideo::kTask)
            return InitialImageTextToVideo::kTask;
        return TextToVideo::kTask;
    }
    if (command.kind == CommandKind::kGenerateWorld) {
        if (has_option(command, "--action") && has_option(command, "--camera-trajectory"))
            throw std::invalid_argument("--action and --camera-trajectory are mutually exclusive");
        if (has_option(command, "--camera-trajectory"))
            return ImageTextCameraTrajectoryToVideo::kTask;
        if (has_option(command, "--action"))
            return ImageTextActionToVideo::kTask;
        return model.info().bundle_task == ImageTextCameraTrajectoryToVideo::kTask
                   ? ImageTextCameraTrajectoryToVideo::kTask
                   : ImageTextActionToVideo::kTask;
    }
    return {};
}

bool dispatch_sdk_video(const Command& command, const Model& model, std::string_view id,
                        std::ostream& output) {
    if (command.kind == CommandKind::kGenerateWorld &&
        (id == ImageTextActionToVideo::kTask || id == ImageTextCameraTrajectoryToVideo::kTask)) {
        generate_world(command, model, id, output);
        return true;
    }
    if (command.kind == CommandKind::kGenerateVideo && id == InitialImageTextToVideo::kTask) {
        if (has_option(command, "--initial-latents-raw"))
            throw std::invalid_argument("InitialImageTextToVideo does not accept a replay operand");
        const auto image = detail::read_image(require_option(command, "--image"));
        const auto task = model.task<InitialImageTextToVideo>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--image", "--prompt", "--output"});
        const auto result =
            task.run({image_view(image), require_option(command, "--prompt")}, config);
        detail::write_json(output, write_video(result, require_option(command, "--output")));
        return true;
    }
    if (command.kind != CommandKind::kGenerateVideo || id != TextToVideo::kTask)
        return false;
    if (has_option(command, "--image"))
        throw std::invalid_argument("an initial image requires an image-conditioned video Task");
    const auto task = model.task<TextToVideo>();
    const auto config = detail::task_config(command, task.config_fields(),
                                            {"--prompt", "--output", "--initial-latents-raw"});
    const auto initial = initial_latents(command);
    const auto result =
        task.run({require_option(command, "--prompt"), {initial.data(), initial.size()}}, config);
    detail::write_json(output, write_video(result, require_option(command, "--output")));
    return true;
}

} // namespace trtmc::cli
