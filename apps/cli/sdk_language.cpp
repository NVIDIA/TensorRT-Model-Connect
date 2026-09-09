/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/language.hpp"

#include <nlohmann/json.hpp>

namespace trtmc::cli {

std::string_view language_task_for_command(const Command& command, const Model& model) {
    if (command.kind != CommandKind::kRun || !command.selected_task.empty())
        return {};
    if (detail::has_option(command, "--image") || detail::has_option(command, "--images"))
        return ImagesTextToText::kTask;
    // A multimodal bundle can also explicitly provide ordinary text generation.
    // This is selection before execution, never a retry of a failed media call.
    const auto primary = model.info().bundle_task;
    if (primary == ImagesTextToText::kTask && model.supports<TextContinuation>())
        return TextContinuation::kTask;
    return {};
}

bool dispatch_sdk_language(const Command& command, const Model& model, std::string_view id,
                           std::ostream& output) {
    if (command.kind != CommandKind::kRun || id != ImagesTextToText::kTask)
        return false;
    const auto task = model.task<ImagesTextToText>();
    const auto config =
        detail::task_config(command, task.config_fields(), {"--prompt", "--image", "--images"});
    const auto prompt = detail::require_option(command, "--prompt");
    std::vector<io::LoadedImage> images;
    for (const auto& path : detail::image_paths(command))
        images.push_back(detail::read_image(path));
    std::vector<ImagesTextPart> parts;
    for (const auto& image : images) {
        parts.emplace_back(ImageInput{Span<const float>{image.pixels.data(), image.pixels.size()},
                                      static_cast<std::uint32_t>(image.height),
                                      static_cast<std::uint32_t>(image.width)});
    }
    parts.emplace_back(TextPart{prompt});
    const auto result = task.run(ImagesTextToTextRequest::from_parts(std::move(parts)), config);
    detail::write_json(output, detail::text_json(result));
    return true;
}

} // namespace trtmc::cli
