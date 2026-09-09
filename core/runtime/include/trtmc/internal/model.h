/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/config.h"
#include "trtmc/task.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::internal {

struct TaskInfo {
    std::string_view id;
    std::uint32_t major{1};
    std::uint32_t minor{0};
};

// Internal release-coupled model identity, not a public C++ ABI. The existing
// factory and loader keep owning ITask; task() is the bundle's primary mode.
// Semantic Task support is explicitly supplied per loaded model below.
class IModel : public virtual trtmc::ITask {
  public:
    virtual std::vector<TaskInfo> task_info() const = 0;
    virtual std::vector<ConfigField> config_fields(std::string_view task) const = 0;
    // Availability belongs to the loaded family instance, not C++ RTTI alone.
    // The returned interface borrows this model's lifetime.
    virtual trtmc::ILoraAdapterManager* lora_adapters() noexcept { return nullptr; }
};

class ConfigError : public std::invalid_argument {
  public:
    explicit ConfigError(const std::string& message) : std::invalid_argument(message) {}
};

class UnsupportedTask : public std::runtime_error {
  public:
    explicit UnsupportedTask(const std::string& message) : std::runtime_error(message) {}
};

} // namespace trtmc::internal
