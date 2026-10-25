/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/plan_reader.h"
#include "trtmc/runtime/trt_backend.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc {

// Optional capability, separate from IBackend to preserve its existing vtable.
// Ownership transfers to the backend, which may retain the reader with its engine.
class IStreamBackend {
  public:
    virtual ~IStreamBackend() = default;
    virtual bool supports_plan_streaming() const noexcept = 0;
    virtual std::unique_ptr<ITrtModule>
    create_module_from_stream(std::unique_ptr<IPlanReader> plan, const ModuleCreateOptions& options,
                              const std::vector<ModuleExternalBinding>& bindings) = 0;
};

// Families choose when to load an engine. This helper only selects a supported
// input mechanism. A failed streamed load is never retried through buffering.
inline std::unique_ptr<ITrtModule>
create_module_from_reader(IBackend& backend, std::unique_ptr<IPlanReader> plan,
                          const ModuleCreateOptions& options,
                          const std::vector<ModuleExternalBinding>& bindings = {}) {
    if (!plan)
        throw std::invalid_argument("Serialized plan reader must not be null");
    if (auto* streamed = dynamic_cast<IStreamBackend*>(&backend);
        streamed != nullptr && streamed->supports_plan_streaming())
        return streamed->create_module_from_stream(std::move(plan), options, bindings);

    if (plan->size() > std::numeric_limits<std::size_t>::max() ||
        !plan->seek(0, PlanSeekOrigin::begin))
        throw std::runtime_error("Cannot buffer serialized plan source");
    std::vector<char> bytes(static_cast<std::size_t>(plan->size()));
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto count = plan->read(bytes.data() + done, bytes.size() - done);
        if (count == 0 || count > bytes.size() - done)
            throw std::runtime_error("Incomplete serialized plan source");
        done += count;
    }
    return bindings.empty()
               ? backend.create_module(bytes.data(), bytes.size(), options)
               : backend.create_module_prebound(bytes.data(), bytes.size(), options, bindings);
}

} // namespace trtmc
