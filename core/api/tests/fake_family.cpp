/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <string>

namespace {

using namespace trtmc::internal;
using trtmc::Span;

class FixtureModel final : public IModel, public ITextContinuation {
  public:
    FixtureModel(std::string mode, trtmc::IBackend& backend, std::uint64_t kv_bytes)
        : mode_(std::move(mode)), backend_(backend), kv_bytes_(kv_bytes), fields_(make_fields()) {
        if (mode_ == "empty_field")
            fields_[0].name = {};
        if (mode_ == "duplicate_field")
            fields_.push_back(fields_[0]);
        if (mode_ == "wrong_default")
            fields_[0].default_value = ConfigValue{true};
    }

    const char* task() const noexcept override { return mode_.c_str(); }

    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "disabled")
            return {};
        auto task = bind<ITextContinuation>(*this, {fields_.data(), fields_.size()});
        if (mode_ == "unknown_binding")
            task.key.id = "unknown_fixture_task";
        if (mode_ == "wrong_version")
            task.key.minor = 999;
        if (mode_ == "null_binding")
            task.implementation = nullptr;
        if (mode_ == "duplicate_binding")
            return {task, task};
        return {task};
    }

    static std::vector<ConfigField> make_fields() {
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
        if (mode_ == "must_not_run")
            throw std::runtime_error("the fixture execution must not be reached");
        if (mode_ == "disabled")
            throw UnsupportedTask("fixture bundle has no text continuation");
        if (std::string(backend_.name()) != "fake")
            throw std::runtime_error("backend lifetime did not extend to the family call");

        const Span<const ConfigField> fields{fields_.data(), fields_.size()};
        const auto max_new_tokens =
            config_get<std::int64_t>(config, fields, "max_new_tokens").value();
        const auto temperature = config_get<double>(config, fields, "temperature").value();
        const auto emit_eos = config_get<bool>(config, fields, "emit_eos").value();
        const auto suffix = config_get<std::string_view>(config, fields, "suffix").value();
        const auto biases =
            config_get<Span<const std::int64_t>>(config, fields, "token_biases").value();
        const auto schedule = config_get<Span<const double>>(config, fields, "schedule").value();
        const auto labels =
            config_get<Span<const std::string_view>>(config, fields, "labels").value();
        const auto context_limit =
            config_get<std::int64_t>(config, fields, "context_limit")
                .value_or(static_cast<std::int64_t>(kv_bytes_ ? kv_bytes_ : 64));
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
    std::vector<ConfigField> fields_;
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
