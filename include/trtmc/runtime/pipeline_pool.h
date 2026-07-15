/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trtmc {

// Owns independent pipeline execution lanes and schedules exclusive access to
// them. Each lease isolates mutable execution-context, stream, KV-cache, and
// adapter-binding state for one in-flight request.
class PipelinePool {
  private:
    struct State;
    class MaintenanceGuard;

  public:
    class Lease {
      public:
        Lease() = default;
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        IPipeline* get() const;
        IPipeline& operator*() const { return *get(); }
        IPipeline* operator->() const { return get(); }
        explicit operator bool() const { return state_ != nullptr; }

      private:
        friend class PipelinePool;
        Lease(std::shared_ptr<State> state, std::size_t index);
        void release();

        std::shared_ptr<State> state_;
        std::size_t index_{0};
    };

    explicit PipelinePool(std::vector<std::unique_ptr<IPipeline>> pipelines);
    ~PipelinePool();
    PipelinePool(PipelinePool&&) noexcept;
    PipelinePool& operator=(PipelinePool&&) noexcept;
    PipelinePool(const PipelinePool&) = delete;
    PipelinePool& operator=(const PipelinePool&) = delete;

    Lease acquire();
    std::optional<Lease> try_acquire();

    std::size_t size() const;
    std::size_t available() const;

    bool supports_lora_adapters() const;
    void load_lora_adapter(const std::string& adapter_id, const std::string& adapter_path);
    void unload_lora_adapter(const std::string& adapter_id);
    std::vector<std::string> loaded_lora_adapters() const;

  private:
    std::shared_ptr<State> state_;
};

} // namespace trtmc
