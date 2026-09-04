/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_registry.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc {

PipelineRegistry& PipelineRegistry::instance() {
    static PipelineRegistry registry;
    return registry;
}

void PipelineRegistry::register_plugin(const std::string& strategy, IPipelinePlugin* plugin) {
    if (!plugin)
        throw std::invalid_argument("Cannot register null plugin for strategy: " + strategy);
    if (strategy.empty())
        throw std::invalid_argument("Cannot register plugin with empty strategy string");
    registry_[strategy] = plugin;
}

IPipelinePlugin* PipelineRegistry::lookup(const std::string& strategy) const {
    auto it = registry_.find(strategy);
    return (it != registry_.end()) ? it->second : nullptr;
}

std::vector<std::string> PipelineRegistry::registered_strategies() const {
    std::vector<std::string> result;
    result.reserve(registry_.size());
    for (const auto& kv : registry_)
        result.push_back(kv.first);
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace trtmc
