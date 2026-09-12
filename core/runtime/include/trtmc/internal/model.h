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
#include <type_traits>
#include <vector>

namespace trtmc::internal {

struct TaskKey {
    std::string_view id;
    std::uint32_t major{1};
    std::uint32_t minor{0};
};

// All currently defined contracts are v1.0. A new compiled contract may
// specialize this function; families never assign their own IDs or versions.
template <class Interface,
          std::enable_if_t<std::is_same_v<Interface, typename Interface::TaskInterface>, int> = 0>
constexpr TaskKey contract_key() noexcept {
    return {Interface::kTask, 1, 0};
}

// Internal, release-coupled metadata. The interface belongs to the loaded
// model. Field storage must remain valid until the Core snapshots the table.
struct TaskInstance {
    TaskKey key;
    void* implementation;
    Span<const ConfigField> fields;
};

// Convert to the requested base interface before erasing its pointer. This
// preserves the correct address for families implementing multiple Tasks.
// The canonical alias rejects deduced or explicit concrete model types even
// when they inherit kTask from a single interface.
template <class Interface,
          std::enable_if_t<std::is_same_v<Interface, typename Interface::TaskInterface>, int> = 0>
TaskInstance bind(Interface& implementation, Span<const ConfigField> fields = {}) noexcept {
    return {contract_key<Interface>(), static_cast<void*>(&implementation), fields};
}

// Internal release-coupled model identity, not a public C++ ABI. The existing
// factory and loader keep owning ITask; task() is the bundle's primary mode.
// Semantic Task support is explicitly supplied per loaded model below.
class IModel : public virtual trtmc::ITask {
  public:
    virtual std::vector<TaskInstance> task_bindings() = 0;
    // Availability belongs to the loaded family instance, not C++ RTTI alone.
    // The returned interface borrows this model's lifetime.
    virtual trtmc::ILoraAdapterManager* lora_adapters() noexcept { return nullptr; }
};

class UnsupportedTask : public std::runtime_error {
  public:
    explicit UnsupportedTask(const std::string& message) : std::runtime_error(message) {}
};

} // namespace trtmc::internal
