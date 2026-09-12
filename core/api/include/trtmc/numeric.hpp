/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/matrix.hpp"
#include "trtmc/numeric.h"
#include "trtmc/text.hpp"

#include <algorithm>
#include <cmath>

namespace trtmc {

struct SeriesHistory {
    FloatMatrixView past_values;
    Span<const std::uint8_t> observed;
    // The family derives dimensions from its loaded bundle; no shared schema lookup.
    static SeriesHistory from_flat(Span<const float> values,
                                   Span<const std::uint8_t> mask = {}) noexcept {
        return {{values, 0, 0}, mask};
    }
};
struct SeriesToPointForecastRequest {
    SeriesHistory history;
};
struct SeriesToQuantileForecastRequest {
    SeriesHistory history;
};
struct SeriesToPointAndQuantileForecastRequest {
    SeriesHistory history;
};
struct SeriesToRegressionDistributionRequest {
    SeriesHistory history;
};
struct SeriesToRegressionValuesRequest {
    SeriesHistory history;
};
// Borrow generation operands until the synchronous call returns. Layout and
// schedule validation belong to the loaded family, not this wrapper.
struct LatentConditionedTextGenerationRequest {
    Span<const float> condition_latents;
    Span<const float> condition_mask;
    Span<const float> initial_latents;
    Span<const float> sde_noises;
    std::string prompt{};
};
struct LatentReplayToTextRequest {
    Span<const float> initial_latents;
    std::string prompt;
    Span<const float> sde_noises;
};
struct LatentDenoisingStepRequest {
    FloatMatrixView latents;
    FloatMatrixView self_condition;
    double timestep;
    // Input already has the loaded family's native packing, including any
    // self-conditioning channels; the family validates the exact buffer length.
    static LatentDenoisingStepRequest native_packed(Span<const float> input, double time) noexcept {
        return {{input, 0, 0}, {}, time};
    }
};
struct LatentToTokenLogitsRequest {
    FloatMatrixView latents;
    FloatMatrixView self_condition;
    double timestep;
    static LatentToTokenLogitsRequest native_packed(Span<const float> input, double time) noexcept {
        return {{input, 0, 0}, {}, time};
    }
};

namespace detail {
inline trtmc_series_request_v1 numeric_series(const SeriesHistory& in) {
    return {in.past_values.c_view(), in.observed.data(), in.observed.size()};
}
inline trtmc_series_request_v1 numeric_request(const SeriesToPointForecastRequest& in) {
    return numeric_series(in.history);
}
inline trtmc_series_request_v1 numeric_request(const SeriesToQuantileForecastRequest& in) {
    return numeric_series(in.history);
}
inline trtmc_series_request_v1 numeric_request(const SeriesToPointAndQuantileForecastRequest& in) {
    return numeric_series(in.history);
}
inline trtmc_series_request_v1 numeric_request(const SeriesToRegressionDistributionRequest& in) {
    return numeric_series(in.history);
}
inline trtmc_series_request_v1 numeric_request(const SeriesToRegressionValuesRequest& in) {
    return numeric_series(in.history);
}
inline trtmc_latent_conditioned_text_request_v1
numeric_request(const LatentConditionedTextGenerationRequest& in) {
    return {{in.condition_latents.data(), in.condition_latents.size()},
            {in.condition_mask.data(), in.condition_mask.size()},
            {in.initial_latents.data(), in.initial_latents.size()},
            {in.sde_noises.data(), in.sde_noises.size()},
            c_string(in.prompt)};
}
inline trtmc_latent_replay_text_request_v1 numeric_request(const LatentReplayToTextRequest& in) {
    return {{in.initial_latents.data(), in.initial_latents.size()},
            c_string(in.prompt),
            {in.sde_noises.data(), in.sde_noises.size()}};
}
inline trtmc_latent_step_request_v1 numeric_request(const LatentDenoisingStepRequest& in) {
    return {in.latents.c_view(), in.self_condition.c_view(), in.timestep};
}
inline trtmc_latent_step_request_v1 numeric_request(const LatentToTokenLogitsRequest& in) {
    return {in.latents.c_view(), in.self_condition.c_view(), in.timestep};
}

template <class Result, class Table, class Request>
Result numeric_call(const std::shared_ptr<ModelState>& state, const Table* table,
                    const Request& request, const Config& config) {
    const auto input = numeric_request(request);
    if constexpr (std::is_same_v<Result, TextContinuationResult>) {
        return run_text(state, table, input, config);
    } else {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = table->run(state->handle, &input, &options, &raw, &error);
        ResultOwner owner(state, raw);
        check(state->api, status, error);
        return Result(std::move(owner), table->result_view);
    }
}
} // namespace detail

using PointForecastResult = detail::ViewResult<trtmc_point_forecast_view_v1>;
using QuantileForecastResult = detail::ViewResult<trtmc_quantile_forecast_view_v1>;
using PointAndQuantileForecastResult =
    detail::ViewResult<trtmc_point_and_quantile_forecast_view_v1>;
using RegressionDistributionResult = detail::ViewResult<trtmc_regression_distribution_view_v1>;
using RegressionValuesResult = detail::ViewResult<trtmc_regression_values_view_v1>;
using DenoisedLatentsResult = detail::ViewResult<trtmc_denoised_latents_view_v1>;
using LatentTokenLogitsResult = detail::ViewResult<trtmc_latent_token_logits_view_v1>;

struct QuantileSummary {
    double level;
    uint64_t horizon, channels;
    std::vector<float> values; // [horizon,channel], not an average over either axis.
    std::vector<int64_t> horizon_steps;
    std::vector<std::string> channel_names, channel_units;
};

// Caller-side interpolation along the declared Q axis only. Exact levels copy
// raw floats; interior levels use adjacent declared grid points. No extrapolation,
// endpoint clamping or quantile-crossing repair. The returned data/axes are owned.
inline QuantileSummary interpolate_quantile(const trtmc_quantile_forecast_view_v1& input,
                                            double level) {
    const auto require = [](bool condition, const char* message) {
        if (!condition)
            throw Error(TRTMC_INVALID_ARGUMENT, message);
    };
    const auto limit = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    const auto count = input.quantile_levels.size;
    require(count && input.horizon && input.channels && input.quantile_levels.data,
            "quantile summary requires a nonempty Q,H,C forecast");
    require(count <= limit / sizeof(double) &&
                input.horizon <= limit / sizeof(float) / input.channels,
            "quantile forecast axes exceed host storage");
    const auto plane = input.horizon * input.channels;
    require(count <= limit / sizeof(float) / plane && input.value_count == count * plane &&
                input.values,
            "quantile forecast values do not match Q,H,C axes");
    require(std::isfinite(level) && level >= 0 && level <= 1,
            "target quantile must be finite and within [0,1]");
    const auto* grid = input.quantile_levels.data;
    for (uint64_t i = 0; i < count; ++i)
        require(std::isfinite(grid[i]) && grid[i] >= 0 && grid[i] <= 1 &&
                    (i == 0 || grid[i] > grid[i - 1]),
                "quantile levels must strictly increase within [0,1]");
    require(level >= grid[0] && level <= grid[count - 1],
            "target quantile is outside the supplied grid");
    for (uint64_t i = 0; i < input.value_count; ++i)
        require(std::isfinite(input.values[i]), "quantile summary requires finite forecast values");
    const auto& axes = input.axes;
    require(axes.horizon_steps.size == input.horizon && axes.horizon_steps.data &&
                input.horizon <= limit / sizeof(int64_t),
            "quantile horizon-step count must match the horizon axis");
    int64_t previous = 0;
    for (uint64_t i = 0; i < input.horizon; ++i) {
        require(axes.horizon_steps.data[i] > previous,
                "horizon steps must be positive and increasing");
        previous = axes.horizon_steps.data[i];
    }
    const auto copy_names = [&](trtmc_strings_view names) {
        require((names.size == 0 || names.size == input.channels) &&
                    names.size <= limit / sizeof(trtmc_string_view) && (!names.size || names.data),
                "quantile channel metadata must match the channel axis");
        std::vector<std::string> result;
        result.reserve(static_cast<size_t>(names.size));
        for (uint64_t i = 0; i < names.size; ++i) {
            const auto value = names.data[i];
            require(value.size <= limit && (!value.size || value.data),
                    "invalid quantile channel string");
            result.emplace_back(value.size ? value.data : "", static_cast<size_t>(value.size));
        }
        return result;
    };
    QuantileSummary result{level,
                           input.horizon,
                           input.channels,
                           {},
                           {axes.horizon_steps.data, axes.horizon_steps.data + input.horizon},
                           copy_names(axes.channel_names),
                           copy_names(axes.channel_units)};
    const auto upper = static_cast<uint64_t>(std::lower_bound(grid, grid + count, level) - grid);
    if (grid[upper] == level) {
        result.values.assign(input.values + upper * plane, input.values + (upper + 1) * plane);
    } else {
        const auto lower = upper - 1;
        const double weight = (level - grid[lower]) / (grid[upper] - grid[lower]);
        result.values.reserve(static_cast<size_t>(plane));
        for (uint64_t i = 0; i < plane; ++i) {
            const double a = input.values[lower * plane + i], b = input.values[upper * plane + i];
            result.values.push_back(static_cast<float>(a + weight * (b - a)));
        }
    }
    return result;
}
inline QuantileSummary interpolate_quantile(const QuantileForecastResult& input, double level) {
    return interpolate_quantile(input.view(), level);
}
// This is a quantile-derived median, never a model's independent mean/point head.
inline QuantileSummary median_forecast(const trtmc_quantile_forecast_view_v1& input) {
    return interpolate_quantile(input, 0.5);
}
inline QuantileSummary median_forecast(const QuantileForecastResult& input) {
    return median_forecast(input.view());
}

#define TRTMC_NUMERIC_WRAPPER(Name, Id, Api, Request, Result)                                      \
    class Name {                                                                                   \
      public:                                                                                      \
        static constexpr std::string_view kTask = Id;                                              \
        static constexpr std::uint32_t kMajor = 1, kMinor = 0;                                     \
        std::vector<ConfigField> config_fields() const {                                           \
            return detail::config_fields(state_, kTask, kMajor, kMinor);                           \
        }                                                                                          \
        Result run(const Request& request, const Config& config = {}) const {                      \
            return detail::numeric_call<Result>(state_, api_, request, config);                    \
        }                                                                                          \
        static void validate_table(const trtmc_api_header* table) {                                \
            if (!table || table->major != kMajor || table->minor != kMinor ||                      \
                table->byte_size < sizeof(Api))                                                    \
                throw Error(TRTMC_VERSION_MISMATCH, "incompatible numeric Task table");            \
        }                                                                                          \
                                                                                                   \
      private:                                                                                     \
        friend class Model;                                                                        \
        Name(std::shared_ptr<detail::ModelState> state, const trtmc_api_header* table) noexcept    \
            : state_(std::move(state)), api_(reinterpret_cast<const Api*>(table)) {}               \
        std::shared_ptr<detail::ModelState> state_;                                                \
        const Api* api_;                                                                           \
    };
TRTMC_NUMERIC_WRAPPER(SeriesToPointForecast, TRTMC_TASK_SERIES_TO_POINT_FORECAST,
                      trtmc_series_to_point_forecast_api_v1, SeriesToPointForecastRequest,
                      PointForecastResult)
TRTMC_NUMERIC_WRAPPER(SeriesToQuantileForecast, TRTMC_TASK_SERIES_TO_QUANTILE_FORECAST,
                      trtmc_series_to_quantile_forecast_api_v1, SeriesToQuantileForecastRequest,
                      QuantileForecastResult)
TRTMC_NUMERIC_WRAPPER(SeriesToPointAndQuantileForecast,
                      TRTMC_TASK_SERIES_TO_POINT_AND_QUANTILE_FORECAST,
                      trtmc_series_to_point_and_quantile_forecast_api_v1,
                      SeriesToPointAndQuantileForecastRequest, PointAndQuantileForecastResult)
TRTMC_NUMERIC_WRAPPER(SeriesToRegressionDistribution, TRTMC_TASK_SERIES_TO_REGRESSION_DISTRIBUTION,
                      trtmc_series_to_regression_distribution_api_v1,
                      SeriesToRegressionDistributionRequest, RegressionDistributionResult)
TRTMC_NUMERIC_WRAPPER(SeriesToRegressionValues, TRTMC_TASK_SERIES_TO_REGRESSION_VALUES,
                      trtmc_series_to_regression_values_api_v1, SeriesToRegressionValuesRequest,
                      RegressionValuesResult)
TRTMC_NUMERIC_WRAPPER(LatentConditionedTextGeneration,
                      TRTMC_TASK_LATENT_CONDITIONED_TEXT_GENERATION,
                      trtmc_latent_conditioned_text_generation_api_v1,
                      LatentConditionedTextGenerationRequest, TextContinuationResult)
TRTMC_NUMERIC_WRAPPER(LatentReplayToText, TRTMC_TASK_LATENT_REPLAY_TO_TEXT,
                      trtmc_latent_replay_to_text_api_v1, LatentReplayToTextRequest,
                      TextContinuationResult)
TRTMC_NUMERIC_WRAPPER(LatentDenoisingStep, TRTMC_TASK_LATENT_DENOISING_STEP,
                      trtmc_latent_denoising_step_api_v1, LatentDenoisingStepRequest,
                      DenoisedLatentsResult)
TRTMC_NUMERIC_WRAPPER(LatentToTokenLogits, TRTMC_TASK_LATENT_TO_TOKEN_LOGITS,
                      trtmc_latent_to_token_logits_api_v1, LatentToTokenLogitsRequest,
                      LatentTokenLogitsResult)
#undef TRTMC_NUMERIC_WRAPPER

struct BatchSeriesToPointForecastItem {
    SeriesToPointForecastRequest input;
    Config config{};
};
struct BatchSeriesToPointForecastRequest {
    std::vector<BatchSeriesToPointForecastItem> items;
};
struct BatchSeriesToQuantileForecastItem {
    SeriesToQuantileForecastRequest input;
    Config config{};
};
struct BatchSeriesToQuantileForecastRequest {
    std::vector<BatchSeriesToQuantileForecastItem> items;
};
struct BatchSeriesToPointAndQuantileForecastItem {
    SeriesToPointAndQuantileForecastRequest input;
    Config config{};
};
struct BatchSeriesToPointAndQuantileForecastRequest {
    std::vector<BatchSeriesToPointAndQuantileForecastItem> items;
};

namespace detail {
template <class View>
class ForecastBatchResult {
  public:
    using Count = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t*, trtmc_error**);
    using ItemView = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t, View*, trtmc_error**);
    ForecastBatchResult(ResultOwner owner, Count count, ItemView item)
        : owner_(std::move(owner)), item_(item) {
        trtmc_error* error = nullptr;
        const auto status = count(owner_.get(), &count_, &error);
        check(owner_.api(), status, error);
    }
    ForecastBatchResult(const ForecastBatchResult&) = delete;
    ForecastBatchResult& operator=(const ForecastBatchResult&) = delete;
    ForecastBatchResult(ForecastBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), item_(other.item_),
          count_(std::exchange(other.count_, 0)) {}
    ForecastBatchResult& operator=(ForecastBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            item_ = other.item_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    uint64_t size() const noexcept { return count_; }
    View operator[](uint64_t index) const {
        View out{};
        trtmc_error* error = nullptr;
        const auto status = item_(owner_.get(), index, &out, &error);
        check(owner_.api(), status, error);
        return out;
    }

  private:
    ResultOwner owner_;
    ItemView item_;
    uint64_t count_{0};
};
template <class Traits>
class ForecastBatchTask {
  public:
    static constexpr std::string_view kTask = Traits::kTask;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = ForecastBatchResult<typename Traits::View>;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(typename Traits::Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible batch forecast Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    Result run(const Request& input) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_series_item_v1> items;
        configs.reserve(input.items.size());
        items.reserve(input.items.size());
        for (const auto& item : input.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({numeric_request(item.input), configs.back().view()});
        }
        const trtmc_batch_series_request_v1 request{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &request, &raw, &error);
        ResultOwner owner(model_, raw);
        check(model_->api, status, error);
        return Result(std::move(owner), api_->result_count, api_->result_item_view);
    }

  private:
    friend class ::trtmc::Model;
    ForecastBatchTask(std::shared_ptr<ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)), api_(reinterpret_cast<const typename Traits::Table*>(table)) {}
    std::shared_ptr<ModelState> model_;
    const typename Traits::Table* api_;
};
struct BatchSeriesToPointForecastTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_SERIES_TO_POINT_FORECAST;
    using Request = BatchSeriesToPointForecastRequest;
    using Table = trtmc_batch_series_to_point_forecast_api_v1;
    using View = trtmc_point_forecast_view_v1;
};
struct BatchSeriesToQuantileForecastTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_SERIES_TO_QUANTILE_FORECAST;
    using Request = BatchSeriesToQuantileForecastRequest;
    using Table = trtmc_batch_series_to_quantile_forecast_api_v1;
    using View = trtmc_quantile_forecast_view_v1;
};
struct BatchSeriesToPointAndQuantileForecastTraits {
    static constexpr std::string_view kTask =
        TRTMC_TASK_BATCH_SERIES_TO_POINT_AND_QUANTILE_FORECAST;
    using Request = BatchSeriesToPointAndQuantileForecastRequest;
    using Table = trtmc_batch_series_to_point_and_quantile_forecast_api_v1;
    using View = trtmc_point_and_quantile_forecast_view_v1;
};
} // namespace detail
using BatchSeriesToPointForecast =
    detail::ForecastBatchTask<detail::BatchSeriesToPointForecastTraits>;
using BatchSeriesToQuantileForecast =
    detail::ForecastBatchTask<detail::BatchSeriesToQuantileForecastTraits>;
using BatchSeriesToPointAndQuantileForecast =
    detail::ForecastBatchTask<detail::BatchSeriesToPointAndQuantileForecastTraits>;
} // namespace trtmc
