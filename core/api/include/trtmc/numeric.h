/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_NUMERIC_H
#define TRTMC_NUMERIC_H
#include "trtmc/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* [time,input_channel], oldest first. For nonempty data, rows=columns=0
     * delegates shape resolution to the family/bundle, not the shared runtime.
     * Exactly one missing dimension is invalid. This rule applies only here. */
    trtmc_f32_matrix_view_v1 past_values;
    const uint8_t* observed; /* Empty or past_values.count binary entries; 1 = observed. */
    uint64_t observed_count;
} trtmc_series_request_v1;

typedef struct {
    trtmc_i64_view horizon_steps; /* Positive offsets from final input step, one per horizon row. */
    trtmc_strings_view channel_names; /* Empty = unspecified; otherwise one per channel. */
    trtmc_strings_view channel_units;
} trtmc_forecast_axes_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 values; /* [horizon,channel]. */
    trtmc_forecast_axes_v1 axes;
} trtmc_point_forecast_view_v1;
typedef struct {
    const float* values; /* [quantile,horizon,channel], contiguous row-major. */
    uint64_t value_count;
    trtmc_f64_view quantile_levels; /* Ascending levels in [0,1]. */
    uint64_t horizon;
    uint64_t channels;
    trtmc_forecast_axes_v1 axes;
} trtmc_quantile_forecast_view_v1;
typedef struct {
    trtmc_point_forecast_view_v1 point;
    trtmc_quantile_forecast_view_v1 quantiles;
} trtmc_point_and_quantile_forecast_view_v1;

enum {
    TRTMC_DISTRIBUTION_NORMAL = 1,
    TRTMC_DISTRIBUTION_STUDENT_T = 2,
    TRTMC_DISTRIBUTION_NEGATIVE_BINOMIAL = 3
};
typedef struct {
    trtmc_string_view name;
    const float* values;
    uint64_t target_count;
} trtmc_distribution_parameter_v1;
/* Named parameters: Normal = location,scale; StudentT = degrees_of_freedom,
 * location,scale; NegativeBinomial = total_count,logits. Their vectors are
 * indexed by target, never future horizon. No normalization is added here. */
typedef struct {
    uint32_t distribution;
    uint64_t target_count;
    const trtmc_distribution_parameter_v1* parameters;
    uint64_t parameter_count;
    trtmc_strings_view target_names;
    trtmc_strings_view target_units;
} trtmc_regression_distribution_view_v1;

/* Finite point predictions indexed by target, never horizon or distribution.
 * Names/units are empty when unspecified; otherwise each has values.size entries.
 * All views borrow the result handle and survive model release. */
typedef struct {
    trtmc_f32_view values;
    trtmc_strings_view target_names;
    trtmc_strings_view target_units;
} trtmc_regression_values_view_v1;

/* Borrowed float32 generation operands use the loaded family's documented
 * layout. Family validates latent dimensions, mask weights and resolved
 * schedule/noise counts. Empty optional buffers select family policy; shared
 * code never reads family configuration to guess a shape or initializes noise. */
typedef struct {
    trtmc_f32_view condition_latents; /* Required [position,latent_channel]. */
    trtmc_f32_view condition_mask;    /* Required [position]. */
    trtmc_f32_view initial_latents;   /* Optional initial-state replay. */
    trtmc_f32_view sde_noises;        /* Optional [noise_step,position,latent_channel]. */
    trtmc_string_view prompt;         /* Preserved; raw-condition precedence is family-owned. */
} trtmc_latent_conditioned_text_request_v1;
typedef struct {
    trtmc_f32_view initial_latents; /* Optional; initial or SDE noise must be nonempty. */
    trtmc_string_view prompt; /* Empty or an actual text condition, never silently discarded. */
    trtmc_f32_view sde_noises;
} trtmc_latent_replay_text_request_v1;
typedef struct {
    /* Positive dimensions describe logical latents. Nonempty 0/0 means
     * family-native already-packed input (not automatic logical shape
     * inference) and requires empty self_condition. No other matrix rule changes. */
    trtmc_f32_matrix_view_v1 latents;
    trtmc_f32_matrix_view_v1 self_condition; /* Empty or same shape. */
    double timestep; /* The Task chooses denoiser/decoder behavior, not a mode flag. */
} trtmc_latent_step_request_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 latents;
} trtmc_denoised_latents_view_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 logits; /* [position,vocabulary], unnormalized. */
    trtmc_string_view vocabulary_id;
} trtmc_latent_token_logits_view_v1;

#define TRTMC_TASK_SERIES_TO_POINT_FORECAST "series_to_point_forecast"
#define TRTMC_TASK_SERIES_TO_QUANTILE_FORECAST "series_to_quantile_forecast"
#define TRTMC_TASK_SERIES_TO_POINT_AND_QUANTILE_FORECAST "series_to_point_and_quantile_forecast"
#define TRTMC_TASK_SERIES_TO_REGRESSION_DISTRIBUTION "series_to_regression_distribution"
#define TRTMC_TASK_SERIES_TO_REGRESSION_VALUES "series_to_regression_values"
#define TRTMC_TASK_LATENT_CONDITIONED_TEXT_GENERATION "latent_conditioned_text_generation"
#define TRTMC_TASK_LATENT_REPLAY_TO_TEXT "latent_replay_to_text"
#define TRTMC_TASK_LATENT_DENOISING_STEP "latent_denoising_step"
#define TRTMC_TASK_LATENT_TO_TOKEN_LOGITS "latent_to_token_logits"

#define TRTMC_NUMERIC_API(Name, Request, View)                                                     \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* run)(trtmc_model*, const Request*, const trtmc_config_view_v1*,   \
                                      trtmc_result**, trtmc_error**);                              \
        trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, View*, trtmc_error**);          \
    } Name;
TRTMC_NUMERIC_API(trtmc_series_to_point_forecast_api_v1, trtmc_series_request_v1,
                  trtmc_point_forecast_view_v1)
TRTMC_NUMERIC_API(trtmc_series_to_quantile_forecast_api_v1, trtmc_series_request_v1,
                  trtmc_quantile_forecast_view_v1)
/* The family returns both components from the same evaluation, with matching
 * horizon/channel axes. No shared median conversion or second Task invocation.
 * Frequency remains a family-declared I64 config key, not another input field. */
TRTMC_NUMERIC_API(trtmc_series_to_point_and_quantile_forecast_api_v1, trtmc_series_request_v1,
                  trtmc_point_and_quantile_forecast_view_v1)
TRTMC_NUMERIC_API(trtmc_series_to_regression_distribution_api_v1, trtmc_series_request_v1,
                  trtmc_regression_distribution_view_v1)
TRTMC_NUMERIC_API(trtmc_series_to_regression_values_api_v1, trtmc_series_request_v1,
                  trtmc_regression_values_view_v1)
TRTMC_NUMERIC_API(trtmc_latent_conditioned_text_generation_api_v1,
                  trtmc_latent_conditioned_text_request_v1, trtmc_text_result_view_v1)
TRTMC_NUMERIC_API(trtmc_latent_replay_to_text_api_v1, trtmc_latent_replay_text_request_v1,
                  trtmc_text_result_view_v1)
TRTMC_NUMERIC_API(trtmc_latent_denoising_step_api_v1, trtmc_latent_step_request_v1,
                  trtmc_denoised_latents_view_v1)
TRTMC_NUMERIC_API(trtmc_latent_to_token_logits_api_v1, trtmc_latent_step_request_v1,
                  trtmc_latent_token_logits_view_v1)
#undef TRTMC_NUMERIC_API

// Request layouts stay unchanged: frequency is per-item family Config.
typedef struct {
    trtmc_series_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_series_item_v1;
typedef struct {
    const trtmc_batch_series_item_v1* items;
    uint64_t count;
} trtmc_batch_series_request_v1;
// Nonempty batches return every item in order or fail without a result handle.
// This does not roll back already executed family effects. Views borrow the
// complete owned batch and survive model release until result_release.
#define TRTMC_TASK_BATCH_SERIES_TO_POINT_FORECAST "batch_series_to_point_forecast"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_series_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_point_forecast_view_v1*, trtmc_error**);
} trtmc_batch_series_to_point_forecast_api_v1;
#define TRTMC_TASK_BATCH_SERIES_TO_QUANTILE_FORECAST "batch_series_to_quantile_forecast"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_series_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_quantile_forecast_view_v1*, trtmc_error**);
} trtmc_batch_series_to_quantile_forecast_api_v1;
#define TRTMC_TASK_BATCH_SERIES_TO_POINT_AND_QUANTILE_FORECAST                                     \
    "batch_series_to_point_and_quantile_forecast"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_series_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_point_and_quantile_forecast_view_v1*,
                                               trtmc_error**);
} trtmc_batch_series_to_point_and_quantile_forecast_api_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_NUMERIC_H */
