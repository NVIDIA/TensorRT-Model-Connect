/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>

namespace {

using namespace trtmc::internal;
using trtmc::Span;

enum FieldIndex {
    MaxNewTokens,
    Temperature,
    EmitEos,
    Suffix,
    TokenBiases,
    Schedule,
    Labels,
    ContextLimit
};

class FixtureModel final : public IModel, public ITextContinuation {
  public:
    FixtureModel(std::string mode, trtmc::IBackend& backend, std::uint64_t kv_bytes)
        : mode_(std::move(mode)), backend_(backend), kv_bytes_(kv_bytes) {}

    const char* task() const noexcept override { return mode_.c_str(); }

    std::vector<TaskInfo> task_info() const override {
        if (mode_ == "disabled")
            return {};
        return {{ITextContinuation::kTask, 1, 0}};
    }

    std::vector<ConfigField> config_fields(std::string_view task_id) const override {
        if (mode_ == "disabled" || task_id != ITextContinuation::kTask)
            throw UnsupportedTask("fixture bundle has no text continuation");
        return {
            {"max_new_tokens", ConfigKind::I64, ConfigValue{std::int64_t{4}}, "Maximum new tokens"},
            {"temperature", ConfigKind::F64, ConfigValue{0.75}, "Sampling temperature"},
            {"emit_eos", ConfigKind::Bool, ConfigValue{true}, "Emit EOS marker"},
            {"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}}, "Text suffix"},
            {"token_biases", ConfigKind::I64List, ConfigValue{Span<const std::int64_t>{}},
             "Biases"},
            {"schedule", ConfigKind::F64List, ConfigValue{Span<const double>{}}, "Schedule"},
            {"labels", ConfigKind::StringList, ConfigValue{Span<const std::string_view>{}},
             "Labels"},
            {"context_limit", ConfigKind::I64, std::nullopt, "Derived from the loaded bundle"},
        };
    }

    TextResult run(const TextContinuationRequest& request, ConfigView config) override {
        if (mode_ == "disabled")
            throw UnsupportedTask("fixture bundle has no text continuation");
        if (std::string(backend_.name()) != "fake")
            throw std::runtime_error("backend lifetime did not extend to the family call");

        const auto fields = config_fields(ITextContinuation::kTask);
        std::vector<ConfigValue> values;
        values.reserve(fields.size());
        for (const auto& field : fields) {
            // Only context_limit has a computed default; its value comes from
            // this loaded fixture's context rather than the public C layer.
            values.push_back(field.default_value.value_or(
                ConfigValue{static_cast<std::int64_t>(kv_bytes_ ? kv_bytes_ : 64)}));
        }
        std::set<std::string_view> supplied;
        for (const auto& entry : config) {
            if (!supplied.insert(entry.name).second)
                throw ConfigError("duplicate config: " + std::string(entry.name));
            const auto field =
                std::find_if(fields.begin(), fields.end(),
                             [&](const auto& candidate) { return candidate.name == entry.name; });
            if (field == fields.end())
                throw ConfigError("unknown config: " + std::string(entry.name));
            if (config_kind(entry.value) != field->kind)
                throw ConfigError("config type mismatch: " + std::string(entry.name));
            values[static_cast<std::size_t>(field - fields.begin())] = entry.value;
        }
        const auto max_new_tokens = config_value_as<std::int64_t>(values[MaxNewTokens]);
        const auto temperature = config_value_as<double>(values[Temperature]);
        const auto emit_eos = config_value_as<bool>(values[EmitEos]);
        const auto suffix = config_value_as<std::string_view>(values[Suffix]);
        const auto biases = config_value_as<Span<const std::int64_t>>(values[TokenBiases]);
        const auto schedule = config_value_as<Span<const double>>(values[Schedule]);
        const auto labels = config_value_as<Span<const std::string_view>>(values[Labels]);
        const auto context_limit = config_value_as<std::int64_t>(values[ContextLimit]);
        if (max_new_tokens < 0 || max_new_tokens > 128)
            throw ConfigError("max_new_tokens must be between zero and 128");
        if (!std::isfinite(temperature) || temperature < 0.0 || temperature > 2.0)
            throw ConfigError("temperature must be finite and between zero and two");
        const auto input_size =
            std::visit([](const auto& prefix) { return prefix.size(); }, request.prefix);
        if (context_limit < 1 || input_size > static_cast<std::uint64_t>(context_limit))
            throw ConfigError("prefix exceeds context_limit");

        TextResult result;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix)) {
            result.text = *text;
            result.token_ids = {11, 12};
        } else {
            const auto tokens = std::get<Span<const std::int32_t>>(request.prefix);
            result.text = "tokens";
            if (!tokens.empty())
                result.token_ids.assign(tokens.begin(), tokens.end());
        }
        result.text += suffix;
        for (const auto label : labels) {
            result.text += '|';
            result.text += label;
        }
        if (!biases.empty())
            result.text += '|' + std::to_string(biases[0]);
        if (emit_eos)
            result.text += "|eos";
        result.setup_ms = static_cast<double>(kv_bytes_) + (schedule.empty() ? 0.0 : schedule[0]);
        result.prefill_ms = temperature;
        result.decode_ms = static_cast<double>(max_new_tokens);
        result.segments.push_back({0.125, 0.375, std::string("seg\0tail", 8), {41, 42}});
        return result;
    }

  private:
    std::string mode_;
    trtmc::IBackend& backend_;
    std::uint64_t kv_bytes_;
};

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "api_fixture")
        throw std::runtime_error("unexpected API fixture family");
    const auto plan = context.reader.read_section("engine.plan");
    if (std::string(plan.begin(), plan.end()) != "PLAN")
        throw std::runtime_error("fixture bundle payload is wrong");
    return new FixtureModel(context.reader.info().task, context.backend,
                            context.kv_cache_size_bytes);
}
