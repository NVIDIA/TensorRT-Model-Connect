/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>

namespace trtmc::cli {
namespace {
using detail::has_option;
using detail::require_option;
} // namespace

namespace detail {
std::vector<std::string> image_paths(const Command& command) {
    if (has_option(command, "--image") && has_option(command, "--images"))
        throw std::invalid_argument("use --image or --images, not both");
    if (has_option(command, "--image"))
        return {command.options.at("--image")};
    if (!has_option(command, "--images"))
        return {};
    const auto input = nlohmann::json::parse(command.options.at("--images"), nullptr, false);
    if (!input.is_array() || input.empty())
        throw std::invalid_argument("--images must be a nonempty JSON array of image paths");
    std::vector<std::string> paths;
    for (const auto& path : input) {
        if (!path.is_string() || path.get_ref<const std::string&>().empty())
            throw std::invalid_argument("--images entries must be nonempty paths");
        paths.push_back(path.get<std::string>());
    }
    return paths;
}
} // namespace detail

namespace {
using detail::image_paths;
ImageInput view(const io::LoadedImage& image) {
    return {Span<const float>{image.pixels.data(), image.pixels.size()},
            static_cast<std::uint32_t>(image.height), static_cast<std::uint32_t>(image.width)};
}

ImageResultView view(const ImageGenerationResult& image) {
    return {image.pixels(), image.height(), image.width(), image.channels()};
}
} // namespace

namespace detail {
nlohmann::json write_image(const ImageResultView& image, const std::string& path) {
    if (image.is_worker())
        return {{"worker", true}};
    if (image.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        image.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("image dimensions exceed the PNG writer's range");
    io::save_png(path, image.pixels, static_cast<int>(image.width), static_cast<int>(image.height),
                 static_cast<int>(image.channels));
    return {{"output", path}, {"height", image.height}, {"width", image.width}};
}
} // namespace detail

std::string_view image_task_for_command(const Command& command, const Model& model) {
    if (!command.selected_task.empty())
        return {};
    if (command.kind == CommandKind::kGenerateImageBatch)
        return BatchTextToImage::kTask;
    if (command.kind != CommandKind::kGenerateImage)
        return {};
    if (has_option(command, "--mask"))
        return MaskedImageTextToImage::kTask;
    if (has_option(command, "--image") || has_option(command, "--images"))
        return ImagesTextToImageEdit::kTask;
    const auto primary = model.info().bundle_task;
    if (primary == ImagesTextToImageEdit::kTask)
        return ImagesTextToImageEdit::kTask;
    if (primary == MaskedImageTextToImage::kTask)
        return MaskedImageTextToImage::kTask;
    return TextToImage::kTask;
}

bool dispatch_sdk_image(const Command& command, const Model& model, std::string_view id,
                        std::ostream& output) {
    if (command.kind == CommandKind::kGenerateImageBatch && id == BatchTextToImage::kTask) {
        const auto task = model.task<BatchTextToImage>();
        const auto fields = task.config_fields();
        const auto config =
            detail::task_config(command, fields, {"--prompts", "--seeds", "--output"});
        const auto prompts = detail::read_lines(require_option(command, "--prompts"));
        std::vector<std::uint32_t> seeds;
        if (has_option(command, "--seeds")) {
            seeds = detail::parse_seeds(command.options.at("--seeds"));
            if (seeds.size() != prompts.size())
                throw std::invalid_argument("--seeds count must match --prompts line count");
            const auto field =
                std::find_if(fields.begin(), fields.end(), [](const ConfigField& item) {
                    return item.name == "seed" && item.kind == ConfigKind::I64;
                });
            if (field == fields.end())
                throw std::invalid_argument(
                    "selected image batch Task does not declare seed as int64");
        }
        std::vector<BatchTextToImageItem> requests;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            requests.push_back({{prompts[i]}, config});
            if (!seeds.empty())
                requests.back().config.add("seed", static_cast<std::int64_t>(seeds[i]));
        }
        const auto result = task.run(requests);
        const std::filesystem::path directory = require_option(command, "--output");
        nlohmann::json items = nlohmann::json::array();
        for (std::uint64_t i = 0; i < result.size(); ++i) {
            const auto image = result[i];
            if (!image.is_worker())
                std::filesystem::create_directories(directory);
            items.push_back(
                detail::write_image(image, (directory / (std::to_string(i) + ".png")).string()));
        }
        detail::write_json(output, {{"outputs", std::move(items)}});
        return true;
    }
    if (command.kind != CommandKind::kGenerateImage ||
        (id != TextToImage::kTask && id != ImagesTextToImageEdit::kTask &&
         id != MaskedImageTextToImage::kTask))
        return false;
    const auto prompt = require_option(command, "--prompt");
    const auto path = require_option(command, "--output");
    const auto initial =
        has_option(command, "--initial-latents-raw")
            ? detail::read_float32_file(command.options.at("--initial-latents-raw"))
            : std::vector<float>{};
    const Span<const float> latents{initial.data(), initial.size()};
    if (id == TextToImage::kTask) {
        const auto task = model.task<TextToImage>();
        const auto config = detail::task_config(command, task.config_fields(),
                                                {"--prompt", "--output", "--initial-latents-raw"});
        const auto result = task.run({prompt, latents}, config);
        detail::write_json(output, detail::write_image(view(result), path));
        return true;
    }
    const auto paths = image_paths(command);
    std::vector<io::LoadedImage> images;
    for (const auto& source : paths)
        images.push_back(detail::read_image(source));
    if (id == ImagesTextToImageEdit::kTask) {
        const auto task = model.task<ImagesTextToImageEdit>();
        const auto config = detail::task_config(
            command, task.config_fields(),
            {"--prompt", "--output", "--image", "--images", "--initial-latents-raw"});
        ImagesTextToImageEditRequest request{{}, prompt, latents};
        for (const auto& image : images)
            request.images.push_back(view(image));
        const auto result = task.run(request, config);
        detail::write_json(output, detail::write_image(view(result), path));
        return true;
    }
    if (images.size() != 1)
        throw std::invalid_argument("masked image generation requires exactly one source image");
    const auto mask = detail::read_float32_file(require_option(command, "--mask"));
    const auto task = model.task<MaskedImageTextToImage>();
    const auto config = detail::task_config(
        command, task.config_fields(), {"--prompt", "--output", "--image", "--images", "--mask"});
    const auto result = task.run({view(images[0]),
                                  {{mask.data(), mask.size()},
                                   static_cast<std::uint32_t>(images[0].height),
                                   static_cast<std::uint32_t>(images[0].width)},
                                  prompt},
                                 config);
    detail::write_json(output, detail::write_image(view(result), path));
    return true;
}
} // namespace trtmc::cli
