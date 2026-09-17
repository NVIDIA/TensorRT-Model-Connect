/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/numeric.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace trtmc::cli {
namespace {
using detail::has_option;

std::vector<float> operand(const Command& command, const char* name) {
    if (!has_option(command, name))
        return {};
    return detail::read_float32_file(command.options.at(name));
}

Config replay_config(const Command& command, const std::vector<ConfigField>& fields) {
    auto config =
        detail::task_config(command, fields,
                            {"--prompt", "--initial-latents-raw", "--condition-latents-raw",
                             "--condition-mask-raw", "--sde-noise-raw", "--sampling-steps-raw"});
    if (has_option(command, "--sampling-steps-raw")) {
        const auto field = std::find_if(fields.begin(), fields.end(), [](const ConfigField& item) {
            return item.name == "sampling_steps";
        });
        if (field == fields.end() || field->kind != ConfigKind::F64List)
            throw std::invalid_argument(
                "selected model/Task does not declare sampling_steps as a float list");
        const auto raw = operand(command, "--sampling-steps-raw");
        config.add("sampling_steps", std::vector<double>(raw.begin(), raw.end()));
        // Preserve an earlier --set sampling_steps entry: family rejects duplicates.
    }
    return config;
}
} // namespace

// Input-driven selection, not a retry after failure. The root dispatcher calls
// this before choosing the bundle's declared primary Task ID for ordinary input.
std::string_view replay_task_for_inputs(const Command& command) {
    if (command.kind != CommandKind::kRun)
        return {};
    const bool condition = has_option(command, "--condition-latents-raw");
    const bool mask = has_option(command, "--condition-mask-raw");
    const bool initial = has_option(command, "--initial-latents-raw");
    const bool noise = has_option(command, "--sde-noise-raw");
    if (condition != mask)
        throw std::invalid_argument(
            "--condition-latents-raw and --condition-mask-raw must be supplied together");
    const auto inferred = condition
                              ? LatentConditionedTextGeneration::kTask
                              : (initial || noise ? LatentReplayToText::kTask : std::string_view{});
    if (!command.selected_task.empty()) {
        if (!inferred.empty() && command.selected_task != inferred)
            throw std::invalid_argument(
                "raw replay inputs conflict with the explicitly selected Task");
        return {};
    }
    return inferred;
}

bool dispatch_sdk_replay(const Command& command, const Model& model, std::string_view id,
                         std::ostream& output) {
    if (command.kind != CommandKind::kRun ||
        (id != LatentReplayToText::kTask && id != LatentConditionedTextGeneration::kTask))
        return false;
    const auto inferred = replay_task_for_inputs(command);
    if (!inferred.empty() && inferred != id)
        throw std::invalid_argument("raw replay inputs do not match the selected Task");
    // All owned buffers stay alive through the synchronous public SDK call.
    const auto initial = operand(command, "--initial-latents-raw");
    const auto noises = operand(command, "--sde-noise-raw");
    const auto prompt = has_option(command, "--prompt") ? command.options.at("--prompt") : "";
    if (id == LatentConditionedTextGeneration::kTask) {
        const auto task = model.task<LatentConditionedTextGeneration>();
        const auto condition = operand(command, "--condition-latents-raw");
        const auto mask = operand(command, "--condition-mask-raw");
        const auto config = replay_config(command, task.config_fields());
        const auto result = task.run({{condition.data(), condition.size()},
                                      {mask.data(), mask.size()},
                                      {initial.data(), initial.size()},
                                      {noises.data(), noises.size()},
                                      prompt},
                                     config);
        detail::write_json(output, detail::text_json(result));
    } else {
        if (has_option(command, "--condition-latents-raw") ||
            has_option(command, "--condition-mask-raw"))
            throw std::invalid_argument("raw condition inputs require the conditioned latent Task");
        const auto task = model.task<LatentReplayToText>();
        const auto config = replay_config(command, task.config_fields());
        const auto result = task.run(
            {{initial.data(), initial.size()}, prompt, {noises.data(), noises.size()}}, config);
        detail::write_json(output, detail::text_json(result));
    }
    return true;
}

} // namespace trtmc::cli
