/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/internal/matrix.h"
#include "trtmc/internal/text.h"

namespace trtmc::internal {

struct SeriesHistory {
    // [time,input_channel], oldest first. Both dimensions may be zero for a
    // nonempty flat history; only the family resolves its bundle-owned shape.
    // A partially specified shape is invalid. Other matrix contracts are unchanged.
    FloatMatrixView past_values;
    Span<const std::uint8_t> observed; // Empty or same element count; 1 means observed.
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

struct ForecastAxes {
    std::vector<std::int64_t> horizon_steps; // Positive offsets after last input step.
    std::vector<std::string> channel_names;  // Empty means unspecified, otherwise one per channel.
    std::vector<std::string> channel_units;
};
struct PointForecastResult {
    FloatMatrix values; // [horizon,channel].
    ForecastAxes axes;
};
struct QuantileForecastResult {
    std::vector<float> values; // [quantile,horizon,channel], contiguous.
    std::vector<double> quantile_levels;
    std::uint64_t horizon{0};
    std::uint64_t channels{0};
    ForecastAxes axes;
};
struct PointAndQuantileForecastResult {
    // The same family evaluation produces both outputs; the point estimate
    // is not inferred from a median quantile by shared code.
    PointForecastResult point;
    QuantileForecastResult quantiles;
};
enum class DistributionKind : std::uint32_t { Normal = 1, StudentT = 2, NegativeBinomial = 3 };
struct DistributionParameter {
    std::string name;
    std::vector<float> values; // [target], never a forecast horizon.
};
struct RegressionDistributionResult {
    DistributionKind
        distribution{}; // Zero is deliberately invalid until the family chooses a head.
    std::uint64_t target_count{0};
    std::vector<DistributionParameter> parameters;
    std::vector<std::string> target_names; // Empty when checkpoint does not identify targets.
    std::vector<std::string> target_units;
};
struct RegressionValuesResult {
    std::vector<float> values; // [target], finite point predictions, not future horizon.
    std::vector<std::string> target_names; // Empty or exactly one name per target.
    std::vector<std::string> target_units; // Empty means unspecified.
};

// Generation buffers use the loaded family's documented float32 layout.
// The family validates its latent dimensions and resolved schedule; shared
// code does not infer them from image sizes or family configuration files.
struct LatentConditionedTextGenerationRequest {
    Span<const float> condition_latents; // Required family-layout [position,latent_channel].
    Span<const float> condition_mask;    // Required [position]; family validates weights.
    Span<const float> initial_latents;   // Empty selects the family's initial-state policy.
    Span<const float> sde_noises;        // Empty selects family noise; otherwise step-major.
    std::string_view prompt{};           // Preserved; the family owns raw-condition precedence.
};
struct LatentReplayToTextRequest {
    Span<const float> initial_latents; // Optional; initial or SDE noise must be supplied.
    std::string_view prompt; // Empty = no text condition; nonempty is encoded by the family.
    Span<const float> sde_noises;
};
struct LatentDenoisingStepRequest {
    // Positive shape: logical [position,latent_channel]. Nonempty 0/0 shape:
    // family-native already-packed input, including self-conditioning if the
    // bundle requires it; separate self_condition must then be empty.
    FloatMatrixView latents;
    FloatMatrixView self_condition; // Empty or same shape; packing is family-owned.
    double timestep;
};
struct LatentToTokenLogitsRequest {
    FloatMatrixView latents; // Same logical/native-packed distinction as the denoising step.
    FloatMatrixView self_condition;
    double timestep;
};
struct DenoisedLatentsResult {
    FloatMatrix latents;
};
struct LatentTokenLogitsResult {
    FloatMatrix logits; // [position,vocabulary], unnormalized logits.
    std::string vocabulary_id;
};

#define TRTMC_NUMERIC_INTERFACE(Name, Id, Request, Result)                                         \
    class I##Name {                                                                                \
      public:                                                                                      \
        using TaskInterface = I##Name;                                                             \
        static constexpr std::string_view kTask = Id;                                              \
        virtual ~I##Name() = default;                                                              \
        virtual Result run(const Request&, ConfigView) = 0;                                        \
    };
TRTMC_NUMERIC_INTERFACE(SeriesToPointForecast, "series_to_point_forecast",
                        SeriesToPointForecastRequest, PointForecastResult)
TRTMC_NUMERIC_INTERFACE(SeriesToQuantileForecast, "series_to_quantile_forecast",
                        SeriesToQuantileForecastRequest, QuantileForecastResult)
TRTMC_NUMERIC_INTERFACE(SeriesToPointAndQuantileForecast, "series_to_point_and_quantile_forecast",
                        SeriesToPointAndQuantileForecastRequest, PointAndQuantileForecastResult)
TRTMC_NUMERIC_INTERFACE(SeriesToRegressionDistribution, "series_to_regression_distribution",
                        SeriesToRegressionDistributionRequest, RegressionDistributionResult)
TRTMC_NUMERIC_INTERFACE(SeriesToRegressionValues, "series_to_regression_values",
                        SeriesToRegressionValuesRequest, RegressionValuesResult)
TRTMC_NUMERIC_INTERFACE(LatentConditionedTextGeneration, "latent_conditioned_text_generation",
                        LatentConditionedTextGenerationRequest, TextResult)
TRTMC_NUMERIC_INTERFACE(LatentReplayToText, "latent_replay_to_text", LatentReplayToTextRequest,
                        TextResult)
TRTMC_NUMERIC_INTERFACE(LatentDenoisingStep, "latent_denoising_step", LatentDenoisingStepRequest,
                        DenoisedLatentsResult)
TRTMC_NUMERIC_INTERFACE(LatentToTokenLogits, "latent_to_token_logits", LatentToTokenLogitsRequest,
                        LatentTokenLogitsResult)
#undef TRTMC_NUMERIC_INTERFACE

// Each item is an independent series, never another input channel.
// Native batching and per-item config preflight belong to the family.
struct BatchSeriesToPointForecastItem {
    SeriesToPointForecastRequest input;
    ConfigView config;
};
struct BatchSeriesToPointForecastRequest {
    using Item = BatchSeriesToPointForecastItem;
    Span<const Item> items;
};
class IBatchSeriesToPointForecast {
  public:
    using TaskInterface = IBatchSeriesToPointForecast;
    static constexpr std::string_view kTask = "batch_series_to_point_forecast";
    using Request = BatchSeriesToPointForecastRequest;
    virtual ~IBatchSeriesToPointForecast() = default;
    virtual std::vector<PointForecastResult>
    run_batch(const BatchSeriesToPointForecastRequest&) = 0;
};
struct BatchSeriesToQuantileForecastItem {
    SeriesToQuantileForecastRequest input;
    ConfigView config;
};
struct BatchSeriesToQuantileForecastRequest {
    using Item = BatchSeriesToQuantileForecastItem;
    Span<const Item> items;
};
class IBatchSeriesToQuantileForecast {
  public:
    using TaskInterface = IBatchSeriesToQuantileForecast;
    static constexpr std::string_view kTask = "batch_series_to_quantile_forecast";
    using Request = BatchSeriesToQuantileForecastRequest;
    virtual ~IBatchSeriesToQuantileForecast() = default;
    virtual std::vector<QuantileForecastResult>
    run_batch(const BatchSeriesToQuantileForecastRequest&) = 0;
};
struct BatchSeriesToPointAndQuantileForecastItem {
    SeriesToPointAndQuantileForecastRequest input;
    ConfigView config;
};
struct BatchSeriesToPointAndQuantileForecastRequest {
    using Item = BatchSeriesToPointAndQuantileForecastItem;
    Span<const Item> items;
};
class IBatchSeriesToPointAndQuantileForecast {
  public:
    using TaskInterface = IBatchSeriesToPointAndQuantileForecast;
    static constexpr std::string_view kTask = "batch_series_to_point_and_quantile_forecast";
    using Request = BatchSeriesToPointAndQuantileForecastRequest;
    virtual ~IBatchSeriesToPointAndQuantileForecast() = default;
    virtual std::vector<PointAndQuantileForecastResult>
    run_batch(const BatchSeriesToPointAndQuantileForecastRequest&) = 0;
};

} // namespace trtmc::internal
