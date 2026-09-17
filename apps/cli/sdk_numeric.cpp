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
using nlohmann::json;
std::string_view forecast_id(std::string_view id) {
    for (const auto candidate :
         {SeriesToPointForecast::kTask, SeriesToQuantileForecast::kTask,
          SeriesToPointAndQuantileForecast::kTask, SeriesToRegressionDistribution::kTask,
          SeriesToRegressionValues::kTask})
        if (candidate == id)
            return candidate;
    return {};
}
struct LoadedSeries {
    std::vector<float> values;
    std::vector<uint8_t> observed;
    SeriesHistory view() const {
        return SeriesHistory::from_flat({values.data(), values.size()},
                                        {observed.data(), observed.size()});
    }
};
LoadedSeries read_series(const Command& command) {
    LoadedSeries input;
    input.values = detail::read_float32_file(detail::require_option(command, "--input"));
    if (detail::has_option(command, "--mask")) {
        const auto mask = detail::read_float32_file(detail::require_option(command, "--mask"));
        if (mask.size() != input.values.size())
            throw std::invalid_argument("--mask must match the input element count");
        input.observed.reserve(mask.size());
        for (const auto value : mask) {
            if (value != 0 && value != 1)
                throw std::invalid_argument("--mask must contain binary observed values");
            input.observed.push_back(value == 0 ? 0 : 1);
        }
    }
    return input;
}
std::vector<float> values(Span<const float> view) {
    std::vector<float> out;
    if (!view.empty())
        out.assign(view.begin(), view.end());
    detail::require_finite(out, "numeric output");
    return out;
}
json names(trtmc_strings_view view) {
    auto output = json::array();
    for (uint64_t i = 0; i < view.size; ++i)
        output.push_back(std::string(trtmc::detail::string_view(view.data[i])));
    return output;
}
json axes(const trtmc_forecast_axes_v1& view) {
    auto steps = json::array();
    for (uint64_t i = 0; i < view.horizon_steps.size; ++i)
        steps.push_back(view.horizon_steps.data[i]);
    return {{"horizon_steps", std::move(steps)},
            {"channel_names", names(view.channel_names)},
            {"channel_units", names(view.channel_units)}};
}
json point_json(const trtmc_point_forecast_view_v1& result) {
    auto output = axes(result.axes);
    output["kind"] = "point";
    output["values"] = values({result.values.data, static_cast<size_t>(result.values.count)});
    output["shape"] = {result.values.rows, result.values.columns};
    output["axes"] = {"horizon", "channel"};
    return output;
}
json quantile_json(const trtmc_quantile_forecast_view_v1& result) {
    auto output = axes(result.axes);
    auto levels = json::array();
    for (uint64_t i = 0; i < result.quantile_levels.size; ++i)
        levels.push_back(result.quantile_levels.data[i]);
    output["kind"] = "quantiles";
    output["values"] = values({result.values, static_cast<size_t>(result.value_count)});
    output["shape"] = {result.quantile_levels.size, result.horizon, result.channels};
    output["axes"] = {"quantile", "horizon", "channel"};
    output["quantile_levels"] = std::move(levels);
    return output;
}
json joint_json(const trtmc_point_and_quantile_forecast_view_v1& result) {
    return {{"kind", "point_and_quantiles"},
            {"point", point_json(result.point)},
            {"quantiles", quantile_json(result.quantiles)}};
}
json regression_json(const trtmc_regression_distribution_view_v1& result) {
    const char* distribution = nullptr;
    switch (result.distribution) {
    case TRTMC_DISTRIBUTION_NORMAL:
        distribution = "normal";
        break;
    case TRTMC_DISTRIBUTION_STUDENT_T:
        distribution = "student_t";
        break;
    case TRTMC_DISTRIBUTION_NEGATIVE_BINOMIAL:
        distribution = "negative_binomial";
        break;
    default:
        throw std::runtime_error("unknown regression distribution");
    }
    auto parameters = json::object();
    for (uint64_t i = 0; i < result.parameter_count; ++i) {
        const auto& parameter = result.parameters[i];
        parameters[std::string(trtmc::detail::string_view(parameter.name))] =
            values({parameter.values, static_cast<size_t>(parameter.target_count)});
    }
    return {
        {"kind", "regression_distribution"},          {"distribution", distribution},
        {"parameters", std::move(parameters)},        {"target_count", result.target_count},
        {"target_names", names(result.target_names)}, {"target_units", names(result.target_units)}};
}
struct SolveControls {
    double timestep;
    double guidance;
    std::string_view task;
};
SolveControls solve_controls(const Command& command) {
    const auto controls = detail::read_float32_file(detail::require_option(command, "--trunk"));
    if (controls.size() != 3)
        throw std::invalid_argument(
            "--trunk must contain timestep, self-conditioning guidance and decoder selector");
    const auto task = controls[2] >= 0.5F ? LatentToTokenLogits::kTask : LatentDenoisingStep::kTask;
    if (!command.selected_task.empty() && command.selected_task != task)
        throw std::invalid_argument("--task conflicts with the decoder selector in --trunk");
    return {controls[0], controls[1], task};
}
Config solve_config(const Command& command, const std::vector<ConfigField>& fields,
                    double guidance) {
    auto config = detail::task_config(command, fields, {"--branch", "--trunk"});
    const auto field = std::find_if(fields.begin(), fields.end(), [](const ConfigField& field) {
        return field.name == "self_cond_cfg_scale" && field.kind == ConfigKind::F64;
    });
    if (field == fields.end())
        throw std::invalid_argument(
            "selected solve Task does not declare self_cond_cfg_scale as a float");
    config.add("self_cond_cfg_scale", guidance);
    // Do not overwrite --set: a duplicate control reaches the family parser.
    return config;
}
json matrix_json(const trtmc_f32_matrix_view_v1& matrix, const char* axis) {
    return {{"dim", matrix.columns},
            {"values", values({matrix.data, static_cast<size_t>(matrix.count)})},
            {"shape", {matrix.rows, matrix.columns}},
            {"axes", {"position", axis}}};
}
} // namespace

std::string_view numeric_task_for_command(const Command& command, const Model& model) {
    if (command.kind == CommandKind::kSolve)
        return solve_controls(command).task;
    if (command.kind != CommandKind::kForecast || !command.selected_task.empty())
        return {};
    if (const auto primary = forecast_id(model.info().bundle_task); !primary.empty())
        return primary;
    std::string_view selected;
    for (const auto& task : model.tasks()) {
        const auto candidate = forecast_id(task.id);
        if (candidate.empty())
            continue;
        if (!selected.empty())
            throw std::invalid_argument("forecast has multiple supported outputs; choose --task");
        selected = candidate;
    }
    return selected;
}
bool dispatch_sdk_numeric(const Command& command, const Model& model, std::string_view id,
                          std::ostream& output) {
    if (command.kind == CommandKind::kSolve) {
        const auto controls = solve_controls(command);
        if (id != controls.task)
            throw std::invalid_argument("selected Task conflicts with --trunk controls");
        const auto branch = detail::read_float32_file(detail::require_option(command, "--branch"));
        json result;
        if (id == LatentDenoisingStep::kTask) {
            const auto task = model.task<LatentDenoisingStep>();
            const auto config = solve_config(command, task.config_fields(), controls.guidance);
            const auto value = task.run(LatentDenoisingStepRequest::native_packed(
                                            {branch.data(), branch.size()}, controls.timestep),
                                        config);
            result = matrix_json(value.view().latents, "latent_channel");
            result["kind"] = "denoised_latents";
        } else {
            const auto task = model.task<LatentToTokenLogits>();
            const auto config = solve_config(command, task.config_fields(), controls.guidance);
            const auto value = task.run(LatentToTokenLogitsRequest::native_packed(
                                            {branch.data(), branch.size()}, controls.timestep),
                                        config);
            result = matrix_json(value.view().logits, "vocabulary_id");
            result["kind"] = "token_logits";
            result["vocabulary_id"] =
                std::string(trtmc::detail::string_view(value.view().vocabulary_id));
        }
        result["task"] = id;
        detail::write_json(output, result);
        return true;
    }
    if (id == BatchSeriesToPointForecast::kTask || id == BatchSeriesToQuantileForecast::kTask ||
        id == BatchSeriesToPointAndQuantileForecast::kTask)
        throw std::invalid_argument(
            "native forecast batches require independent requests; use the Task SDK batch API");
    if (forecast_id(id).empty())
        return false;
    if (command.kind != CommandKind::kForecast)
        throw std::invalid_argument("selected forecast Task requires the forecast command");
    const auto input = read_series(command);
    json result;
    if (id == SeriesToPointForecast::kTask) {
        const auto task = model.task<SeriesToPointForecast>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--input", "--mask"});
        result = point_json(task.run({input.view()}, config).view());
    } else if (id == SeriesToQuantileForecast::kTask) {
        const auto task = model.task<SeriesToQuantileForecast>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--input", "--mask"});
        result = quantile_json(task.run({input.view()}, config).view());
    } else if (id == SeriesToPointAndQuantileForecast::kTask) {
        const auto task = model.task<SeriesToPointAndQuantileForecast>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--input", "--mask"});
        result = joint_json(task.run({input.view()}, config).view());
    } else if (id == SeriesToRegressionValues::kTask) {
        const auto task = model.task<SeriesToRegressionValues>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--input", "--mask"});
        const auto result_owner = task.run({input.view()}, config);
        const auto& view = result_owner.view();
        result = {
            {"kind", "regression_values"},
            {"values", values({view.values.data, static_cast<std::size_t>(view.values.size)})},
            {"target_count", view.values.size},
            {"axes", {"target"}},
            {"target_names", names(view.target_names)},
            {"target_units", names(view.target_units)}};
    } else {
        const auto task = model.task<SeriesToRegressionDistribution>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--input", "--mask"});
        result = regression_json(task.run({input.view()}, config).view());
    }
    result["task"] = id;
    detail::write_json(output, result);
    return true;
}
} // namespace trtmc::cli
