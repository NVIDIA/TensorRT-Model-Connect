# Caller-side Task helpers

These helpers compile into your application. They do not add a runtime Task,
load a tokenizer, call a model again, or change family configuration.

## Rank masked-token candidates

The family supplies vocabulary logits and the selected mask positions. Pass
candidate IDs to your application's actual decoder for that vocabulary; the SDK
does not infer one from the family name.

```cpp
#include <trtmc/features.hpp>
#include <iostream>

template<class DecodeToken>
void print_candidates(const trtmc::VocabularyScoresResult& scores,
                      DecodeToken&& decode_token) {
    // The caller's decoder must correspond to scores.vocabulary_id().
    for (const auto& position : trtmc::rank_masked_tokens(scores, 5)) {
        for (const auto& candidate : position.candidates) {
            std::cout << position.position.token_index << ": "
                      << candidate.token_id << " "
                      << decode_token(candidate.token_id) << " logit="
                      << candidate.logit << '\n';
        }
    }
}
```

The output is descending **raw logit**, not probability. Ties use ascending
token ID. The nonnegative candidate limit is capped by the vocabulary size;
zero keeps empty candidate lists. NaN is an error; infinities remain ordered raw
scores. Output values and position metadata survive release of `scores`.

## Summarize an existing quantile forecast

```cpp
#include <trtmc/numeric.hpp>

trtmc::QuantileSummary summarize(const trtmc::QuantileForecastResult& forecast) {
    auto median = trtmc::median_forecast(forecast);
    // median.values[h * median.channels + c] is the requested Q=0.5 value.
    // median.horizon_steps and channel_names/channel_units retain real axes.
    return median;  // Independently owns values and metadata.
}

trtmc::QuantileSummary at_level(const trtmc::QuantileForecastResult& forecast,
                                double level) {
    return trtmc::interpolate_quantile(forecast, level);
}

trtmc::QuantileSummary summarize_joint(
    const trtmc::PointAndQuantileForecastResult& forecast) {
    auto joint = forecast.view();
    // joint.point is the separately returned model prediction. Do not replace
    // it with the quantile-derived median or label that median as a mean.
    return trtmc::median_forecast(joint.quantiles);
}
```

Input layout is `[quantile,horizon,channel]`; summaries retain `[horizon,channel]`.
An exact level copies the original float values. Interior levels are linearly
interpolated using the actual grid spacing, including irregular grids. No
sorting or repair of crossed predictions occurs.

Levels outside the supplied grid are errors, not hidden endpoint clamping or
extrapolation. This also applies when `0.5` is unavailable for the median.
Forecast values must be finite; grid levels must strictly increase within
`[0,1]`. Horizon offsets remain mandatory; unspecified channel names/units stay
empty. Existing batch-item quantile views use the same overload directly.
