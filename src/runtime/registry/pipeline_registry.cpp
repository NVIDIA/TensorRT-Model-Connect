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
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(TRTMC_LOCKED_H3_RUNTIME)
    if (!locked_registration_open_ || locked_registration_thread_ != std::this_thread::get_id() ||
        locked_registration_strategy_ != strategy || locked_registration_count_ != 0) {
        throw std::runtime_error(
            "locked MiniMax-H3 runtime rejects pipeline registration outside its trusted loader");
    }
    const auto [it, inserted] = registry_.emplace(strategy, plugin);
    if (!inserted)
        throw std::runtime_error("locked MiniMax-H3 runtime refuses duplicate pipeline strategy: " +
                                 strategy);
    ++locked_registration_count_;
#else
    registry_[strategy] = plugin;
#endif
}

void PipelineRegistry::begin_locked_registration(const std::string& strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (locked_registration_open_ || locked_registration_sealed_ ||
        registry_.find(strategy) != registry_.end()) {
        throw std::runtime_error(
            "locked MiniMax-H3 pipeline registration is already active or sealed");
    }
    locked_registration_open_ = true;
    locked_registration_thread_ = std::this_thread::get_id();
    locked_registration_strategy_ = strategy;
    locked_registration_count_ = 0;
}

void PipelineRegistry::finish_locked_registration(const std::string& strategy) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!locked_registration_open_ || locked_registration_thread_ != std::this_thread::get_id() ||
        locked_registration_strategy_ != strategy || locked_registration_count_ != 1 ||
        registry_.find(strategy) == registry_.end()) {
        throw std::runtime_error(
            "locked MiniMax-H3 plugin did not complete its exact trusted registration");
    }
    locked_registration_open_ = false;
    locked_registration_sealed_ = true;
    locked_registration_thread_ = {};
    locked_registration_strategy_.clear();
    locked_registration_count_ = 0;
}

void PipelineRegistry::abort_locked_registration(const std::string& strategy) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!locked_registration_open_ || locked_registration_thread_ != std::this_thread::get_id() ||
        locked_registration_strategy_ != strategy) {
        return;
    }
    if (locked_registration_count_ != 0)
        registry_.erase(strategy);
    locked_registration_open_ = false;
    locked_registration_thread_ = {};
    locked_registration_strategy_.clear();
    locked_registration_count_ = 0;
}

IPipelinePlugin* PipelineRegistry::lookup(const std::string& strategy) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = registry_.find(strategy);
    return (it != registry_.end()) ? it->second : nullptr;
}

std::vector<std::string> PipelineRegistry::registered_strategies() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(registry_.size());
    for (const auto& kv : registry_)
        result.push_back(kv.first);
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace trtmc
