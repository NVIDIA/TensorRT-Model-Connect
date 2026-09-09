/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <fstream>
#include <iterator>
#include <map>
#include <string>

namespace {

using namespace trtmc::internal;

class ControlModel final : public IModel,
                           public ITextContinuation,
                           public trtmc::ILoraAdapterManager {
  public:
    explicit ControlModel(const trtmc::FamilyContext& context)
        : mode_(context.reader.info().task), backend_(context.backend) {}

    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInfo> task_info() const override { return {{ITextContinuation::kTask, 1, 0}}; }
    trtmc::ILoraAdapterManager* lora_adapters() noexcept override {
        return mode_ == "enabled" ? this : nullptr;
    }
    std::vector<ConfigField> config_fields(std::string_view id) const override {
        if (id != ITextContinuation::kTask)
            throw UnsupportedTask("unknown control fixture task");
        if (mode_ != "enabled")
            return {};
        return {{"lora_adapter_id", ConfigKind::String, ConfigValue{std::string_view{}},
                 "Choose a loaded, model-local adapter; empty uses the base model"}};
    }

    TextResult run(const TextContinuationRequest& request, ConfigView config) override {
        const auto fields = config_fields(ITextContinuation::kTask);
        std::string_view adapter;
        if (!fields.empty())
            adapter = config_value_as<std::string_view>(*fields[0].default_value);
        if (config.size() > 1)
            throw ConfigError("duplicate adapter configuration");
        for (const auto& entry : config) {
            if (fields.empty() || entry.name != fields[0].name ||
                config_kind(entry.value) != fields[0].kind)
                throw ConfigError("unknown or invalid adapter configuration");
            adapter = config_value_as<std::string_view>(entry.value);
        }
        const auto* prefix = std::get_if<std::string_view>(&request.prefix);
        if (!prefix)
            throw std::invalid_argument("control fixture expects a UTF-8 prefix");
        std::string output(*prefix);
        if (!adapter.empty()) {
            const auto found = adapters_.find(std::string(adapter));
            if (found == adapters_.end())
                throw ConfigError("adapter is not loaded on this model");
            output += '|' + found->second;
        }
        // The existing fake backend exposes the forwarded RTX options. Defer
        // module creation until a call to verify their owned lifetime.
        if (std::string(backend_.name()) == "trt_rtx")
            (void)backend_.create_module(nullptr, 0, {});
        return TextResult{std::move(output), {7}};
    }

    void load_lora_adapter(const std::string& id, const std::string& path) override {
        if (id.empty() || path.empty())
            throw std::invalid_argument("adapter ID and path must be nonempty");
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::invalid_argument("adapter file does not exist");
        adapters_[id] = std::string(std::istreambuf_iterator<char>(input), {});
    }
    void unload_lora_adapter(const std::string& id) override {
        if (adapters_.erase(id) == 0)
            throw std::invalid_argument("adapter is not loaded");
    }
    std::vector<std::string> loaded_lora_adapters() const override {
        std::vector<std::string> ids;
        for (const auto& item : adapters_)
            ids.push_back(item.first);
        return ids;
    }

  private:
    std::string mode_;
    trtmc::IBackend& backend_;
    std::map<std::string, std::string> adapters_;
};

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "control_fixture")
        throw std::invalid_argument("unexpected control fixture family");
    return new ControlModel(context);
}
