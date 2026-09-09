/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/action.hpp"

#include <cmath>
#include <nlohmann/json.hpp>

namespace trtmc::cli {
namespace {
nlohmann::json strings(trtmc_strings_view input) {
    auto output = nlohmann::json::array();
    for (uint64_t i = 0; i < input.size; ++i)
        output.push_back(std::string(trtmc::detail::string_view(input.data[i])));
    return output;
}
} // namespace

bool dispatch_sdk_action(const Command& command, const Model& model, std::string_view task_id,
                         std::ostream& output) {
    if (task_id != ImageStateToActionChunk::kTask && task_id != ImageStateActionQueue::kTask)
        return false;
    if (command.kind != CommandKind::kControl)
        throw std::invalid_argument("command '" + command.name + "' does not accept Task '" +
                                    std::string(task_id) + "'");
    if (task_id == ImageStateActionQueue::kTask)
        throw std::invalid_argument("control predicts a stateless action chunk; use the Task SDK "
                                    "session API for a persistent action queue");
    const auto task = model.task<ImageStateToActionChunk>();
    const auto config =
        detail::task_config(command, task.config_fields(), {"--image", "--state", "--output"});
    const auto image = detail::read_image(detail::require_option(command, "--image"));
    const auto state = detail::read_float32_file(detail::require_option(command, "--state"));
    const ImageStateObservation observation{ImageInput({image.pixels.data(), image.pixels.size()},
                                                       static_cast<uint32_t>(image.height),
                                                       static_cast<uint32_t>(image.width)),
                                            {state.data(), state.size()}};
    const auto result = task.run({observation}, config);
    const auto view = result.view();
    const auto matrix = result.actions();
    auto actions = nlohmann::json::array();
    for (const auto value : matrix.values) {
        if (!std::isfinite(value))
            throw std::runtime_error("action output contains a non-finite value");
        actions.push_back(value);
    }
    auto timestamps = nlohmann::json::array();
    for (uint64_t i = 0; i < view.actions.timestamps_seconds.size; ++i)
        timestamps.push_back(view.actions.timestamps_seconds.data[i]);
    auto associations = nlohmann::json::array();
    for (uint64_t i = 0; i < view.actions.frame_span_count; ++i) {
        const auto span = view.actions.frame_spans[i];
        associations.push_back({{"begin", span.begin}, {"end", span.end}});
    }
    const auto schema = view.actions.schema;
    if (detail::has_option(command, "--output"))
        detail::write_binary(detail::require_option(command, "--output"), matrix.values);
    detail::write_json(
        output,
        {{"task", task_id},
         {"actions", std::move(actions)},
         {"num_actions", matrix.rows},
         {"action_dim", matrix.columns},
         {"within_training_bounds", result.within_training_bounds()},
         {"inference_ms", result.inference_ms()},
         {"axes", {"step", "action_component"}},
         {"schema",
          {{"domain", std::string(trtmc::detail::string_view(schema.domain))},
           {"component_names", strings(schema.component_names)},
           {"units", strings(schema.units)},
           {"coordinate_frame", std::string(trtmc::detail::string_view(schema.coordinate_frame))},
           {"normalization", std::string(trtmc::detail::string_view(schema.normalization))}}},
         {"timestamps_seconds", std::move(timestamps)},
         {"frame_spans", std::move(associations)}});
    return true;
}
} // namespace trtmc::cli
