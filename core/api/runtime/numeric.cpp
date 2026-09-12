/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/numeric.h"

#include "api_internal.h"
#include "trtmc/numeric.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace trtmc::api {
namespace {

std::size_t elements(std::uint64_t rows, std::uint64_t columns) {
    require(rows && columns, "matrix dimensions must be positive");
    checked_size(rows, columns);
    const auto count = rows * columns;
    return checked_size(count, sizeof(float));
}
void output_check(bool ok, const char* message) {
    if (!ok)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
} // namespace

internal::FloatMatrixView matrix_input(const trtmc_f32_matrix_view_v1& input, bool optional) {
    if (optional && input.count == 0) {
        require(input.rows == 0 && input.columns == 0, "empty matrix must have zero dimensions");
        return {};
    }
    require(input.count == elements(input.rows, input.columns),
            "matrix count does not match shape");
    return {checked_span(input.data, input.count), input.rows, input.columns};
}
trtmc_f32_matrix_view_v1 matrix_result_view(const internal::FloatMatrix& value) {
    std::size_t count;
    try {
        count = elements(value.rows, value.columns);
    } catch (const ApiFailure&) {
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "invalid family matrix shape"};
    }
    output_check(value.values.size() == count, "family matrix storage does not match shape");
    return {value.values.data(), value.values.size(), value.rows, value.columns};
}
namespace {
internal::SeriesHistory history(const trtmc_series_request_v1& input) {
    internal::FloatMatrixView values;
    if (input.past_values.rows == 0 && input.past_values.columns == 0) {
        const auto flat = checked_span(input.past_values.data, input.past_values.count);
        require(!flat.empty(), "series history must not be empty");
        values = {flat, 0, 0}; // The family, not this converter, resolves its channel count.
    } else {
        values = matrix_input(input.past_values, false);
    }
    auto observed = checked_span(input.observed, input.observed_count);
    require(observed.empty() || observed.size() == values.values.size(),
            "observed mask shape mismatch");
    for (const auto value : observed)
        require(value <= 1, "observed mask must contain zero or one");
    return {values, observed};
}
void matching(const internal::FloatMatrixView& required,
              const internal::FloatMatrixView& optional) {
    require(optional.values.empty() ||
                (required.rows == optional.rows && required.columns == optional.columns),
            "latent condition and initial/self-condition shapes must match");
}
internal::LatentConditionedTextGenerationRequest
conditioned(const trtmc_latent_conditioned_text_request_v1& in) {
    auto cond = checked_span(in.condition_latents.data, in.condition_latents.size);
    auto mask = checked_span(in.condition_mask.data, in.condition_mask.size);
    require(!cond.empty() && !mask.empty(), "latent conditioning requires condition and mask");
    return {cond, mask, checked_span(in.initial_latents.data, in.initial_latents.size),
            checked_span(in.sde_noises.data, in.sde_noises.size), string_view(in.prompt)};
}
internal::LatentReplayToTextRequest replay(const trtmc_latent_replay_text_request_v1& in) {
    auto initial = checked_span(in.initial_latents.data, in.initial_latents.size);
    auto noises = checked_span(in.sde_noises.data, in.sde_noises.size);
    require(!initial.empty() || !noises.empty(),
            "latent replay requires initial latents or SDE noise");
    return {initial, string_view(in.prompt), noises};
}
template <class Request>
Request step(const trtmc_latent_step_request_v1& in) {
    const bool packed = in.latents.rows == 0 && in.latents.columns == 0;
    internal::FloatMatrixView latent;
    if (packed) {
        const auto values = checked_span(in.latents.data, in.latents.count);
        require(!values.empty(), "native-packed latent input must not be empty");
        latent = {values, 0, 0};
    } else {
        latent = matrix_input(in.latents, false);
    }
    auto self = matrix_input(in.self_condition, true);
    if (packed)
        require(self.values.empty(), "native-packed input cannot have separate self-conditioning");
    else
        matching(latent, self);
    require(std::isfinite(in.timestep), "timestep must be finite");
    return {latent, self, in.timestep};
}

std::vector<trtmc_string_view> names(const std::vector<std::string>& values) {
    std::vector<trtmc_string_view> out;
    out.reserve(values.size());
    for (const auto& value : values)
        out.push_back(borrowed_string(value));
    return out;
}
struct AxesViews {
    AxesViews(const internal::ForecastAxes& axes, std::uint64_t horizon, std::uint64_t channels)
        : channel_names(names(axes.channel_names)), channel_units(names(axes.channel_units)) {
        output_check(axes.horizon_steps.size() == horizon, "family omitted horizon-step indices");
        output_check(channel_names.empty() || channel_names.size() == channels,
                     "channel name count mismatch");
        output_check(channel_units.empty() || channel_units.size() == channels,
                     "channel unit count mismatch");
        std::int64_t previous = 0;
        for (auto value : axes.horizon_steps) {
            output_check(value > previous, "horizon steps must be positive and increasing");
            previous = value;
        }
        view = {{axes.horizon_steps.data(), axes.horizon_steps.size()},
                {channel_names.data(), channel_names.size()},
                {channel_units.data(), channel_units.size()}};
    }
    std::vector<trtmc_string_view> channel_names, channel_units;
    trtmc_forecast_axes_v1 view{};
};
struct PointStorage final : ResultStorage {
    explicit PointStorage(internal::PointForecastResult result)
        : value(std::move(result)), axes(value.axes, value.values.rows, value.values.columns),
          view{matrix_result_view(value.values), axes.view} {}
    internal::PointForecastResult value;
    AxesViews axes;
    trtmc_point_forecast_view_v1 view;
};
struct QuantileStorage final : ResultStorage {
    explicit QuantileStorage(internal::QuantileForecastResult result)
        : value(std::move(result)), axes(value.axes, value.horizon, value.channels) {
        output_check(!value.quantile_levels.empty(), "family omitted quantile levels");
        std::size_t count;
        try {
            const auto matrix_count = elements(value.horizon, value.channels);
            checked_size(value.quantile_levels.size(), matrix_count);
            count = value.quantile_levels.size() * matrix_count;
        } catch (const ApiFailure&) {
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "invalid quantile result shape"};
        }
        output_check(value.values.size() == count,
                     "quantile result count does not match Q,H,C shape");
        double previous = -1;
        for (auto level : value.quantile_levels) {
            output_check(std::isfinite(level) && level >= 0 && level <= 1 && level > previous,
                         "quantile levels must increase within zero and one");
            previous = level;
        }
        view = {value.values.data(),
                value.values.size(),
                {value.quantile_levels.data(), value.quantile_levels.size()},
                value.horizon,
                value.channels,
                axes.view};
    }
    internal::QuantileForecastResult value;
    AxesViews axes;
    trtmc_quantile_forecast_view_v1 view{};
};
struct PointAndQuantileStorage final : ResultStorage {
    explicit PointAndQuantileStorage(internal::PointAndQuantileForecastResult result)
        : point(std::move(result.point)), quantiles(std::move(result.quantiles)) {
        output_check(point.value.values.rows == quantiles.value.horizon &&
                         point.value.values.columns == quantiles.value.channels,
                     "joint forecast components have different horizon/channel shapes");
        output_check(point.value.axes.horizon_steps == quantiles.value.axes.horizon_steps &&
                         point.value.axes.channel_names == quantiles.value.axes.channel_names &&
                         point.value.axes.channel_units == quantiles.value.axes.channel_units,
                     "joint forecast components have different horizon/channel metadata");
        view = {point.view, quantiles.view};
    }
    PointStorage point;
    QuantileStorage quantiles;
    trtmc_point_and_quantile_forecast_view_v1 view{};
};
struct RegressionStorage final : ResultStorage {
    explicit RegressionStorage(internal::RegressionDistributionResult result)
        : value(std::move(result)), target_names(names(value.target_names)),
          target_units(names(value.target_units)) {
        output_check(value.target_count > 0, "regression target count must be positive");
        output_check(target_names.empty() || target_names.size() == value.target_count,
                     "target name count mismatch");
        output_check(target_units.empty() || target_units.size() == value.target_count,
                     "target unit count mismatch");
        std::set<std::string> expected;
        switch (value.distribution) {
        case internal::DistributionKind::Normal:
            expected = {"location", "scale"};
            break;
        case internal::DistributionKind::StudentT:
            expected = {"degrees_of_freedom", "location", "scale"};
            break;
        case internal::DistributionKind::NegativeBinomial:
            expected = {"total_count", "logits"};
            break;
        default:
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "unknown regression distribution"};
        }
        for (const auto& parameter : value.parameters) {
            output_check(expected.erase(parameter.name) == 1 &&
                             parameter.values.size() == value.target_count,
                         "distribution parameter names or target shape mismatch");
            for (auto number : parameter.values) {
                output_check(std::isfinite(number), "distribution parameter must be finite");
                if (parameter.name == "scale" || parameter.name == "total_count")
                    output_check(number > 0, "distribution scale/count must be positive");
                if (parameter.name == "degrees_of_freedom")
                    output_check(number > 2, "Student-t degrees_of_freedom must exceed two");
            }
            parameters.push_back({borrowed_string(parameter.name), parameter.values.data(),
                                  parameter.values.size()});
        }
        output_check(expected.empty(), "family omitted named distribution parameters");
        view = {static_cast<std::uint32_t>(value.distribution),
                value.target_count,
                parameters.data(),
                parameters.size(),
                {target_names.data(), target_names.size()},
                {target_units.data(), target_units.size()}};
    }
    internal::RegressionDistributionResult value;
    std::vector<trtmc_string_view> target_names, target_units;
    std::vector<trtmc_distribution_parameter_v1> parameters;
    trtmc_regression_distribution_view_v1 view{};
};
struct RegressionValuesStorage final : ResultStorage {
    explicit RegressionValuesStorage(internal::RegressionValuesResult result)
        : value(std::move(result)), target_names(names(value.target_names)),
          target_units(names(value.target_units)) {
        output_check(!value.values.empty(), "regression values require at least one target");
        output_check(target_names.empty() || target_names.size() == value.values.size(),
                     "regression target name count mismatch");
        output_check(target_units.empty() || target_units.size() == value.values.size(),
                     "regression target unit count mismatch");
        for (const auto number : value.values)
            output_check(std::isfinite(number), "regression target values must be finite");
        view = {{value.values.data(), value.values.size()},
                {target_names.data(), target_names.size()},
                {target_units.data(), target_units.size()}};
    }
    internal::RegressionValuesResult value;
    std::vector<trtmc_string_view> target_names, target_units;
    trtmc_regression_values_view_v1 view{};
};
struct DenoisedStorage final : ResultStorage {
    explicit DenoisedStorage(internal::DenoisedLatentsResult result)
        : value(std::move(result)), view{matrix_result_view(value.latents)} {}
    internal::DenoisedLatentsResult value;
    trtmc_denoised_latents_view_v1 view;
};
struct LogitsStorage final : ResultStorage {
    explicit LogitsStorage(internal::LatentTokenLogitsResult result)
        : value(std::move(result)),
          view{matrix_result_view(value.logits), borrowed_string(value.vocabulary_id)} {
        output_check(!value.vocabulary_id.empty(), "latent logits require vocabulary identity");
    }
    internal::LatentTokenLogitsResult value;
    trtmc_latent_token_logits_view_v1 view;
};

template <class Store, class View>
trtmc_status TRTMC_CALL view_result(const trtmc_result* result, View* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "numeric result view is null");
        *out = require_result<Store>(result).view;
    });
}
template <class Interface, class Store, class Request, class Convert>
trtmc_status run(trtmc_model* model, const Request* request, const trtmc_config_view_v1* config,
                 trtmc_result** out, trtmc_error** error, Convert convert) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out, "numeric request or result output is null");
        const std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        const auto input = convert(*request);
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model), internal::contract_key<Interface>(),
                             options.view());
        *out = make_result<Store>(family.run(input, options.view()));
    });
}
#define TRTMC_NUMERIC_RUN(Name, Interface, Store, CRequest, Conversion)                            \
    trtmc_status TRTMC_CALL Name(trtmc_model* model, const CRequest* input,                        \
                                 const trtmc_config_view_v1* config, trtmc_result** out,           \
                                 trtmc_error** error) noexcept {                                   \
        return run<internal::Interface, Store>(                                                    \
            model, input, config, out, error, [](const CRequest& request) { return Conversion; }); \
    }
TRTMC_NUMERIC_RUN(point, ISeriesToPointForecast, PointStorage, trtmc_series_request_v1,
                  internal::SeriesToPointForecastRequest{history(request)})
TRTMC_NUMERIC_RUN(quantile, ISeriesToQuantileForecast, QuantileStorage, trtmc_series_request_v1,
                  internal::SeriesToQuantileForecastRequest{history(request)})
TRTMC_NUMERIC_RUN(point_quantile, ISeriesToPointAndQuantileForecast, PointAndQuantileStorage,
                  trtmc_series_request_v1,
                  internal::SeriesToPointAndQuantileForecastRequest{history(request)})
TRTMC_NUMERIC_RUN(regression, ISeriesToRegressionDistribution, RegressionStorage,
                  trtmc_series_request_v1,
                  internal::SeriesToRegressionDistributionRequest{history(request)})
TRTMC_NUMERIC_RUN(regression_values, ISeriesToRegressionValues, RegressionValuesStorage,
                  trtmc_series_request_v1,
                  internal::SeriesToRegressionValuesRequest{history(request)})
TRTMC_NUMERIC_RUN(condition_text, ILatentConditionedTextGeneration, TextResultStorage,
                  trtmc_latent_conditioned_text_request_v1, conditioned(request))
TRTMC_NUMERIC_RUN(replay_text, ILatentReplayToText, TextResultStorage,
                  trtmc_latent_replay_text_request_v1, replay(request))
TRTMC_NUMERIC_RUN(denoise, ILatentDenoisingStep, DenoisedStorage, trtmc_latent_step_request_v1,
                  step<internal::LatentDenoisingStepRequest>(request))
TRTMC_NUMERIC_RUN(logits, ILatentToTokenLogits, LogitsStorage, trtmc_latent_step_request_v1,
                  step<internal::LatentToTokenLogitsRequest>(request))
#undef TRTMC_NUMERIC_RUN

template <class Function>
auto forecast_item(size_t index, Function function) {
    try {
        return function();
    } catch (const internal::ConfigError& failure) {
        throw OwnedApiFailure{TRTMC_INVALID_CONFIG,
                              "batch item[" + std::to_string(index) + "]: " + failure.what()};
    } catch (const ApiFailure& failure) {
        throw OwnedApiFailure{failure.status,
                              "batch item[" + std::to_string(index) + "]: " + failure.message};
    } catch (const OwnedApiFailure& failure) {
        throw OwnedApiFailure{failure.status,
                              "batch item[" + std::to_string(index) + "]: " + failure.message};
    }
    // bad_alloc propagates to the existing guard without a formatting fallback.
}
template <class Store>
struct ForecastBatchStorage final : ResultStorage {
    template <class Results>
    explicit ForecastBatchStorage(Results values) {
        items.reserve(values.size());
        for (size_t index = 0; index < values.size(); ++index)
            items.push_back(forecast_item(
                index, [&] { return std::make_unique<Store>(std::move(values[index])); }));
    }
    std::vector<std::unique_ptr<Store>> items;
};
template <class Store>
trtmc_status TRTMC_CALL forecast_batch_count(const trtmc_result* result, uint64_t* out,
                                             trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out != nullptr, "forecast batch count output is null");
        *out = require_result<ForecastBatchStorage<Store>>(result).items.size();
    });
}
template <class Store, class View>
trtmc_status TRTMC_CALL forecast_batch_item_view(const trtmc_result* result, uint64_t index,
                                                 View* out, trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "forecast batch item output is null");
        const auto& batch = require_result<ForecastBatchStorage<Store>>(result);
        require(index < batch.items.size(), "forecast batch item index is out of range");
        *out = batch.items[index]->view;
    });
}
template <class Interface, class Store>
trtmc_status TRTMC_CALL run_forecast_batch(trtmc_model* model,
                                           const trtmc_batch_series_request_v1* input,
                                           trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "forecast batch request and output are required");
        const auto source = checked_span(input->items, input->count);
        require(!source.empty(), "forecast batch requires at least one item");
        using Request = typename Interface::Request;
        std::vector<typename Request::Item> items;
        std::vector<ConvertedConfig> configs;
        items.reserve(source.size());
        configs.reserve(source.size());
        for (size_t i = 0; i < source.size(); ++i) {
            forecast_item(i, [&] {
                configs.emplace_back(&source[i].config);

                items.push_back({{history(source[i].input)}, configs.back().view()});
            });
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        validate_batch_configs(model_owner(model), internal::contract_key<Interface>(), configs);
        auto results = family.run_batch(Request{{items.data(), items.size()}});
        output_check(results.size() == items.size(),
                     "forecast batch result count differs from input count");
        *out = make_result<ForecastBatchStorage<Store>>(std::move(results));
    });
}
const trtmc_batch_series_to_point_forecast_api_v1 batch_series_to_point_forecast_api = {
    {1, 0, sizeof(trtmc_batch_series_to_point_forecast_api_v1)},
    run_forecast_batch<internal::IBatchSeriesToPointForecast, PointStorage>,
    forecast_batch_count<PointStorage>,
    forecast_batch_item_view<PointStorage, trtmc_point_forecast_view_v1>};
static_assert(offsetof(trtmc_batch_series_to_point_forecast_api_v1, header) == 0);
const trtmc_batch_series_to_quantile_forecast_api_v1 batch_series_to_quantile_forecast_api = {
    {1, 0, sizeof(trtmc_batch_series_to_quantile_forecast_api_v1)},
    run_forecast_batch<internal::IBatchSeriesToQuantileForecast, QuantileStorage>,
    forecast_batch_count<QuantileStorage>,
    forecast_batch_item_view<QuantileStorage, trtmc_quantile_forecast_view_v1>};
static_assert(offsetof(trtmc_batch_series_to_quantile_forecast_api_v1, header) == 0);
const trtmc_batch_series_to_point_and_quantile_forecast_api_v1
    batch_series_to_point_and_quantile_forecast_api = {
        {1, 0, sizeof(trtmc_batch_series_to_point_and_quantile_forecast_api_v1)},
        run_forecast_batch<internal::IBatchSeriesToPointAndQuantileForecast,
                           PointAndQuantileStorage>,
        forecast_batch_count<PointAndQuantileStorage>,
        forecast_batch_item_view<PointAndQuantileStorage,
                                 trtmc_point_and_quantile_forecast_view_v1>};
static_assert(offsetof(trtmc_batch_series_to_point_and_quantile_forecast_api_v1, header) == 0);

const trtmc_series_to_point_forecast_api_v1 point_api{
    {1, 0, sizeof(point_api)}, point, view_result<PointStorage, trtmc_point_forecast_view_v1>};
const trtmc_series_to_quantile_forecast_api_v1 quantile_api{
    {1, 0, sizeof(quantile_api)},
    quantile,
    view_result<QuantileStorage, trtmc_quantile_forecast_view_v1>};
const trtmc_series_to_point_and_quantile_forecast_api_v1 point_quantile_api{
    {1, 0, sizeof(point_quantile_api)},
    point_quantile,
    view_result<PointAndQuantileStorage, trtmc_point_and_quantile_forecast_view_v1>};
const trtmc_series_to_regression_distribution_api_v1 regression_api{
    {1, 0, sizeof(regression_api)},
    regression,
    view_result<RegressionStorage, trtmc_regression_distribution_view_v1>};
const trtmc_series_to_regression_values_api_v1 regression_values_api{
    {1, 0, sizeof(regression_values_api)},
    regression_values,
    view_result<RegressionValuesStorage, trtmc_regression_values_view_v1>};
const trtmc_latent_conditioned_text_generation_api_v1 condition_api{
    {1, 0, sizeof(condition_api)}, condition_text, text_result_view};
const trtmc_latent_replay_to_text_api_v1 replay_api{
    {1, 0, sizeof(replay_api)}, replay_text, text_result_view};
const trtmc_latent_denoising_step_api_v1 denoise_api{
    {1, 0, sizeof(denoise_api)},
    denoise,
    view_result<DenoisedStorage, trtmc_denoised_latents_view_v1>};
const trtmc_latent_to_token_logits_api_v1 logits_api{
    {1, 0, sizeof(logits_api)},
    logits,
    view_result<LogitsStorage, trtmc_latent_token_logits_view_v1>};

} // namespace
Span<const TaskBinding> numeric_task_bindings() noexcept {
    static const TaskBinding bindings[] = {
        {internal::IBatchSeriesToPointForecast::kTask, 1, 0,
         &batch_series_to_point_forecast_api.header},
        {internal::IBatchSeriesToQuantileForecast::kTask, 1, 0,
         &batch_series_to_quantile_forecast_api.header},
        {internal::IBatchSeriesToPointAndQuantileForecast::kTask, 1, 0,
         &batch_series_to_point_and_quantile_forecast_api.header},
        {internal::ISeriesToPointAndQuantileForecast::kTask, 1, 0, &point_quantile_api.header},
        {internal::ISeriesToPointForecast::kTask, 1, 0, &point_api.header},
        {internal::ISeriesToQuantileForecast::kTask, 1, 0, &quantile_api.header},
        {internal::ISeriesToRegressionDistribution::kTask, 1, 0, &regression_api.header},
        {internal::ISeriesToRegressionValues::kTask, 1, 0, &regression_values_api.header},
        {internal::ILatentConditionedTextGeneration::kTask, 1, 0, &condition_api.header},
        {internal::ILatentReplayToText::kTask, 1, 0, &replay_api.header},
        {internal::ILatentDenoisingStep::kTask, 1, 0, &denoise_api.header},
        {internal::ILatentToTokenLogits::kTask, 1, 0, &logits_api.header},
    };
    return {bindings};
}
} // namespace trtmc::api
