/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/numeric.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace {
using namespace trtmc::internal;
class NumericModel final : public IModel,
                           public ISeriesToPointForecast,
                           public ISeriesToQuantileForecast,
                           public ISeriesToPointAndQuantileForecast,
                           public ISeriesToRegressionDistribution,
                           public ISeriesToRegressionValues,
                           public ILatentConditionedTextGeneration,
                           public ILatentReplayToText,
                           public ILatentDenoisingStep,
                           public ILatentToTokenLogits,
                           public IBatchSeriesToPointForecast,
                           public IBatchSeriesToQuantileForecast,
                           public IBatchSeriesToPointAndQuantileForecast {
  public:
    explicit NumericModel(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_.find("regression_values") == 0 || mode_ == ISeriesToRegressionValues::kTask)
            return {bind<ISeriesToRegressionValues>(*this,
                                                    fields_for(ISeriesToRegressionValues::kTask))};
        if (mode_ == "regression_only")
            return {bind<ISeriesToRegressionDistribution>(
                *this, fields_for(ISeriesToRegressionDistribution::kTask))};
        std::vector<TaskInstance> tasks{
            bind<ISeriesToPointForecast>(*this, fields_for(ISeriesToPointForecast::kTask)),
            bind<IBatchSeriesToPointForecast>(*this,
                                              fields_for(IBatchSeriesToPointForecast::kTask)),
            bind<IBatchSeriesToQuantileForecast>(*this,
                                                 fields_for(IBatchSeriesToQuantileForecast::kTask)),
            bind<IBatchSeriesToPointAndQuantileForecast>(
                *this, fields_for(IBatchSeriesToPointAndQuantileForecast::kTask)),
            bind<ISeriesToQuantileForecast>(*this, fields_for(ISeriesToQuantileForecast::kTask)),
            bind<ISeriesToPointAndQuantileForecast>(
                *this, fields_for(ISeriesToPointAndQuantileForecast::kTask)),
            bind<ISeriesToRegressionDistribution>(
                *this, fields_for(ISeriesToRegressionDistribution::kTask)),
            bind<ILatentConditionedTextGeneration>(
                *this, fields_for(ILatentConditionedTextGeneration::kTask)),
            bind<ILatentReplayToText>(*this, fields_for(ILatentReplayToText::kTask)),
            bind<ILatentDenoisingStep>(*this, fields_for(ILatentDenoisingStep::kTask)),
            bind<ILatentToTokenLogits>(*this, fields_for(ILatentToTokenLogits::kTask))};
        if (mode_ == ISeriesToPointForecast::kTask)
            tasks.push_back(bind<ISeriesToRegressionValues>(
                *this, fields_for(ISeriesToRegressionValues::kTask)));
        return tasks;
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view id) const {
        if (id == ISeriesToRegressionValues::kTask) {
            static const ConfigField fields[]{
                {"scale", ConfigKind::F64, ConfigValue{1.0}, "Synthetic target scale"}};
            return fields;
        }
        if (id == ISeriesToPointForecast::kTask || id == ISeriesToQuantileForecast::kTask ||
            id == ISeriesToPointAndQuantileForecast::kTask ||
            id == IBatchSeriesToPointForecast::kTask ||
            id == IBatchSeriesToQuantileForecast::kTask ||
            id == IBatchSeriesToPointAndQuantileForecast::kTask) {
            static const ConfigField default_zero[] = {{"frequency", ConfigKind::I64,
                                                        ConfigValue{std::int64_t{0}},
                                                        "Fixture frequency category"}};
            static const ConfigField default_one[] = {{"frequency", ConfigKind::I64,
                                                       ConfigValue{std::int64_t{1}},
                                                       "Fixture frequency category"}};
            if (mode_ == "frequency_default_one")
                return default_one;
            return default_zero;
        }
        if (id == ISeriesToRegressionDistribution::kTask) {
            static const ConfigField declared[] = {{"distribution", ConfigKind::String,
                                                    ConfigValue{std::string_view{"normal"}},
                                                    "Distribution head"}};
            return declared;
        }
        if (id == ILatentConditionedTextGeneration::kTask || id == ILatentReplayToText::kTask) {
            static const double schedule[] = {0, 0.5, 1};
            static const ConfigField declared[] = {
                {"sampling_steps", ConfigKind::F64List,
                 ConfigValue{trtmc::Span<const double>{schedule}}, "Explicit replay schedule"}};
            return declared;
        }
        if (id == ILatentDenoisingStep::kTask || id == ILatentToTokenLogits::kTask) {
            static const ConfigField declared[] = {{"self_cond_cfg_scale", ConfigKind::F64,
                                                    ConfigValue{1.0},
                                                    "Synthetic native-step guidance control."}};
            return declared;
        }
        return {};
    }
    RegressionValuesResult run(const SeriesToRegressionValuesRequest& request,
                               ConfigView config) override {
        const auto fields = fields_for(ISeriesToRegressionValues::kTask);
        validate_config(fields, config);
        const auto scale = config_get<double>(config, fields, "scale").value();
        if (!std::isfinite(scale))
            throw ConfigError("target scale must be finite");
        const auto input = resolve_history(request.history);
        float sum = 0;
        for (std::size_t i = 0; i < input.past_values.values.size(); ++i)
            if (input.observed.empty() || input.observed[i])
                sum += input.past_values.values[i];
        RegressionValuesResult result{
            {sum * static_cast<float>(scale), static_cast<float>(++regression_evaluations_)},
            {},
            {}};
        if (mode_ == "regression_values_named") {
            result.target_names = {"total", "evaluation"};
            result.target_units = {"unit", "count"};
        } else if (mode_ == "regression_values_bad_names") {
            result.target_names = {"only_one"};
        } else if (mode_ == "regression_values_bad_units") {
            result.target_units = {"only_one"};
        } else if (mode_ == "regression_values_empty") {
            result.values.clear();
        } else if (mode_ == "regression_values_nonfinite") {
            result.values[0] = std::numeric_limits<float>::infinity();
        }
        return result;
    }
    PointForecastResult run(const SeriesToPointForecastRequest& request,
                            ConfigView config) override {
        const auto frequency = forecast_frequency(ISeriesToPointForecast::kTask, config);
        const auto input = resolve_history(request.history);
        const auto channels = input.past_values.columns;
        PointForecastResult result;
        result.values = {std::vector<float>(2 * channels, 3), 2, channels};
        result.values.values[0] =
            input.observed.empty() || input.observed[0] ? input.past_values.values[0] : -10;
        result.values.values[0] += static_cast<float>(frequency * 100);
        result.axes.horizon_steps = {1, 3};
        return result;
    }
    QuantileForecastResult run(const SeriesToQuantileForecastRequest& request,
                               ConfigView config) override {
        const auto frequency = forecast_frequency(ISeriesToQuantileForecast::kTask, config);
        QuantileForecastResult result;
        result.horizon = 2;
        result.channels = resolve_history(request.history).past_values.columns;
        result.quantile_levels = mode_ == "broken_quantile" ? std::vector<double>{0.9, 0.1}
                                                            : std::vector<double>{0.1, 0.9};
        result.values.resize(4 * result.channels);
        for (std::size_t i = 0; i < result.values.size(); ++i)
            result.values[i] = static_cast<float>(i + 1);
        result.values[0] += static_cast<float>(frequency * 100);
        result.axes.horizon_steps = {1, 3};
        return result;
    }
    PointAndQuantileForecastResult run(const SeriesToPointAndQuantileForecastRequest& request,
                                       ConfigView config) override {
        const auto frequency = forecast_frequency(ISeriesToPointAndQuantileForecast::kTask, config);
        const auto input = resolve_history(request.history);
        const auto evaluation = ++forecast_evaluations_;
        auto result = joint_forecast(input, frequency, evaluation);
        if (mode_ == "broken_joint_axes")
            result.quantiles.axes.horizon_steps = {1, 2};
        if (mode_ == "broken_joint_shape") {
            ++result.quantiles.channels;
            result.quantiles.values.resize(3 * result.quantiles.horizon *
                                           result.quantiles.channels);
        }
        return result;
    }

    std::vector<PointForecastResult>
    run_batch(const BatchSeriesToPointForecastRequest& request) override {
        const auto frequencies = batch_frequencies(IBatchSeriesToPointForecast::kTask, request);
        const auto evaluation = ++forecast_evaluations_;
        std::vector<PointForecastResult> results;
        for (std::size_t i = 0; i < request.items.size(); ++i)
            results.push_back(joint_forecast(resolve_history(request.items[i].input.history),
                                             frequencies[i], evaluation)
                                  .point);
        if (mode_ == "broken_batch_count")
            results.pop_back();
        return results;
    }
    std::vector<QuantileForecastResult>
    run_batch(const BatchSeriesToQuantileForecastRequest& request) override {
        const auto frequencies = batch_frequencies(IBatchSeriesToQuantileForecast::kTask, request);
        const auto evaluation = ++forecast_evaluations_;
        std::vector<QuantileForecastResult> results;
        for (std::size_t i = 0; i < request.items.size(); ++i)
            results.push_back(joint_forecast(resolve_history(request.items[i].input.history),
                                             frequencies[i], evaluation)
                                  .quantiles);
        return results;
    }
    std::vector<PointAndQuantileForecastResult>
    run_batch(const BatchSeriesToPointAndQuantileForecastRequest& request) override {
        const auto frequencies =
            batch_frequencies(IBatchSeriesToPointAndQuantileForecast::kTask, request);
        const auto evaluation = ++forecast_evaluations_;
        std::vector<PointAndQuantileForecastResult> results;
        for (std::size_t i = 0; i < request.items.size(); ++i) {
            auto result = joint_forecast(resolve_history(request.items[i].input.history),
                                         frequencies[i], evaluation);
            if (mode_ == "broken_batch_axes" && i == 1)
                result.quantiles.axes.horizon_steps = {1, 2};
            results.push_back(std::move(result));
        }
        return results;
    }

    RegressionDistributionResult run(const SeriesToRegressionDistributionRequest& request,
                                     ConfigView config) override {
        (void)resolve_history(request.history);
        const auto options = resolved(ISeriesToRegressionDistribution::kTask, config);
        const auto kind = config_value_as<std::string_view>(options.at("distribution"));
        RegressionDistributionResult result;
        result.target_count = 2;
        if (kind == "normal") {
            result.distribution = DistributionKind::Normal;
            result.parameters = {{"scale", {0.5F, 1}}, {"location", {10, 20}}};
        } else if (kind == "student_t") {
            result.distribution = DistributionKind::StudentT;
            result.parameters = {
                {"degrees_of_freedom", {3, 4}}, {"location", {10, 20}}, {"scale", {0.5F, 1}}};
        } else if (kind == "negative_binomial") {
            result.distribution = DistributionKind::NegativeBinomial;
            result.parameters = {{"total_count", {2, 3}}, {"logits", {-1, 1}}};
        } else
            throw ConfigError("unsupported fixture distribution");
        if (mode_ == "broken_regression")
            result.parameters[0].name = "horizon";
        return result;
    }
    TextResult run(const LatentConditionedTextGenerationRequest& request,
                   ConfigView config) override {
        check_latents(request.condition_latents, true);
        check_latents(request.initial_latents, false);
        if (request.condition_mask.size() != 2)
            throw std::invalid_argument("fixture condition mask requires two positions");
        const auto schedule =
            check_schedule(ILatentConditionedTextGeneration::kTask, config, request.sde_noises);
        return TextResult{
            "conditioned",
            {static_cast<std::int32_t>(request.condition_latents[0]),
             static_cast<std::int32_t>(request.condition_mask[request.condition_mask.size() - 1] *
                                       100),
             request.initial_latents.empty()
                 ? -1
                 : static_cast<std::int32_t>(request.initial_latents[0]),
             request.sde_noises.empty()
                 ? -1
                 : static_cast<std::int32_t>(request.sde_noises[request.sde_noises.size() - 1]),
             static_cast<std::int32_t>(schedule[1] * 100),
             static_cast<std::int32_t>(request.prompt.size())}};
    }
    TextResult run(const LatentReplayToTextRequest& request, ConfigView config) override {
        check_latents(request.initial_latents, false);
        const auto schedule =
            check_schedule(ILatentReplayToText::kTask, config, request.sde_noises);
        return TextResult{
            std::string(request.prompt) + "|replayed",
            {request.initial_latents.empty()
                 ? -1
                 : static_cast<std::int32_t>(request.initial_latents[0]),
             request.sde_noises.empty()
                 ? -1
                 : static_cast<std::int32_t>(request.sde_noises[request.sde_noises.size() - 1]),
             static_cast<std::int32_t>(schedule[1] * 100)}};
    }
    DenoisedLatentsResult run(const LatentDenoisingStepRequest& request,
                              ConfigView config) override {
        const auto scale = step_scale(ILatentDenoisingStep::kTask, config);
        auto input = step_input(request.latents, request.self_condition);
        DenoisedLatentsResult result{{std::move(input.values), 2, 2}};
        result.latents.values[0] +=
            static_cast<float>(request.timestep) + input.first_self + static_cast<float>(scale - 1);
        return result;
    }
    LatentTokenLogitsResult run(const LatentToTokenLogitsRequest& request,
                                ConfigView config) override {
        const auto scale = step_scale(ILatentToTokenLogits::kTask, config);
        const auto input = step_input(request.latents, request.self_condition);
        LatentTokenLogitsResult result{{std::vector<float>(2 * 3, 0), 2, 3}, "fixture-vocabulary"};
        result.logits.values[0] =
            input.values[0] + input.first_self + static_cast<float>(request.timestep + scale - 1);
        return result;
    }

  private:
    struct StepInput {
        std::vector<float> values;
        float first_self{0};
    };
    static StepInput step_input(const FloatMatrixView& latent, const FloatMatrixView& self) {
        if (latent.rows == 0 && latent.columns == 0) {
            // Synthetic native engine input is [2 positions,2*2 channels],
            // with [latent0,latent1,self0,self1] at each position.
            if (latent.values.size() != 8 || !self.values.empty())
                throw std::invalid_argument("fixture native-packed input requires eight floats and "
                                            "no separate self-condition");
            return {{latent.values[0], latent.values[1], latent.values[4], latent.values[5]},
                    latent.values[2]};
        }
        if (latent.rows != 2 || latent.columns != 2 || latent.values.size() != 4)
            throw std::invalid_argument("fixture logical latent input must be two by two");
        return {{latent.values.begin(), latent.values.end()},
                self.values.empty() ? 0 : self.values[0]};
    }
    double step_scale(std::string_view task, ConfigView config) const {
        const auto options = resolved(task, config);
        const auto value = config_value_as<double>(options.at("self_cond_cfg_scale"));
        if (!std::isfinite(value))
            throw ConfigError("native step guidance must be finite");
        return value;
    }
    SeriesHistory resolve_history(const SeriesHistory& input) const {
        auto result = input;
        if (result.past_values.rows == 0 && result.past_values.columns == 0) {
            if (mode_ == "flat_shape_unsupported")
                throw std::invalid_argument("fixture requires explicit history shape");
            const std::uint64_t channels = mode_ == "flat_channels_two" ? 2 : 1;
            if (result.past_values.values.empty() ||
                result.past_values.values.size() % channels != 0)
                throw std::invalid_argument(
                    "flat history is not divisible by the family channel count");
            result.past_values.rows = result.past_values.values.size() / channels;
            result.past_values.columns = channels;
        } else if (mode_ == "flat_channels_two" && result.past_values.columns != 2) {
            throw std::invalid_argument("explicit history channels differ from the family bundle");
        }
        return result;
    }

    template <class Request>
    std::vector<std::int64_t> batch_frequencies(std::string_view id, const Request& request) const {
        std::vector<std::int64_t> values;
        for (std::size_t i = 0; i < request.items.size(); ++i) {
            try {
                (void)resolve_history(request.items[i].input.history);
                values.push_back(forecast_frequency(id, request.items[i].config));
            } catch (const ConfigError& error) {
                throw ConfigError("batch item[" + std::to_string(i) + "]: " + error.what());
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument("batch item[" + std::to_string(i) +
                                            "]: " + error.what());
            }
        }
        return values;
    }

    std::int64_t forecast_frequency(std::string_view task, ConfigView config) const {
        const auto options = resolved(task, config);
        const auto value = config_value_as<std::int64_t>(options.at("frequency"));
        if (value < 0 || value > 2 || (mode_ == "neutral_frequency" && value != 0))
            throw ConfigError("invalid frequency category for this provider");
        return value;
    }
    static PointAndQuantileForecastResult
    joint_forecast(const SeriesHistory& input, std::int64_t frequency, std::uint64_t evaluation) {
        const auto channels = input.past_values.columns;
        const auto first = input.past_values.values[0];
        const float observed =
            input.observed.empty() || input.observed[0] ? (std::isnan(first) ? -20 : first) : -10;
        const float base =
            observed + static_cast<float>(frequency * 100) + static_cast<float>(evaluation * 1000);
        PointAndQuantileForecastResult result;
        result.point.values = {std::vector<float>(2 * channels), 2, channels};
        result.point.axes.horizon_steps = {1, 3};
        result.quantiles.horizon = 2;
        result.quantiles.channels = channels;
        result.quantiles.quantile_levels = {0.1, 0.5, 0.9};
        result.quantiles.axes = result.point.axes;
        result.quantiles.values.resize(6 * channels);
        const float offsets[] = {-2, 5, 9};
        for (std::uint64_t h = 0; h < 2; ++h)
            for (std::uint64_t c = 0; c < channels; ++c) {
                const auto value = base + static_cast<float>(h * 10 + c) +
                                   (h == 1 ? static_cast<float>(input.past_values.rows) : 0);
                result.point.values.values[h * channels + c] = value;
                for (std::uint64_t q = 0; q < 3; ++q)
                    result.quantiles.values[(q * 2 + h) * channels + c] = value + offsets[q];
            }
        return result;
    }
    std::uint64_t forecast_evaluations_{0};
    std::uint64_t regression_evaluations_{0};
    static void check_latents(trtmc::Span<const float> values, bool required) {
        // The fixture owns its 2x2 layout; no shared API shape inference.
        if ((required || !values.empty()) && values.size() != 4)
            throw std::invalid_argument("fixture generation latents require four floats");
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument("fixture generation latents must be finite");
    }
    std::map<std::string_view, ConfigValue> resolved(std::string_view task_id,
                                                     ConfigView config) const {
        const auto fields = fields_for(task_id);
        std::map<std::string_view, ConfigValue> result;
        for (const auto& field : fields)
            result.emplace(field.name, *field.default_value);
        std::set<std::string_view> supplied;
        for (const auto& entry : config) {
            const auto field = std::find_if(fields.begin(), fields.end(),
                                            [&](const auto& f) { return f.name == entry.name; });
            if (!supplied.insert(entry.name).second || field == fields.end() ||
                field->kind != config_kind(entry.value))
                throw ConfigError("unknown, duplicate, or wrongly typed fixture config");
            result.at(entry.name) = entry.value;
        }
        return result;
    }
    trtmc::Span<const double> check_schedule(std::string_view id, ConfigView config,
                                             trtmc::Span<const float> noise) const {
        const auto options = resolved(id, config);
        const auto schedule =
            config_value_as<trtmc::Span<const double>>(options.at("sampling_steps"));
        if (schedule.size() < 2 ||
            (!noise.empty() && (noise.size() % 4 || noise.size() / 4 != schedule.size() - 2)))
            throw ConfigError("noise replay steps must match the supplied sampling schedule");
        return schedule;
    }
    std::string mode_;
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new NumericModel(context.reader.info().task);
}
