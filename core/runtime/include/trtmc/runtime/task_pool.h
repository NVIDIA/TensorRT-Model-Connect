/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/task.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trtmc {

// Owns a fixed number of independent Task instances. One move-only lease gives
// one caller exclusive access to one instance until the lease is destroyed.
// The pool coordinates ownership only; it does not batch or schedule requests.
class TaskPool {
  private:
    struct State;

  public:
    class Lease {
      public:
        Lease() = default;
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        ITask* get() const;
        ITask& operator*() const { return *get(); }
        ITask* operator->() const { return get(); }
        explicit operator bool() const noexcept { return state_ != nullptr; }

      private:
        friend class TaskPool;
        Lease(std::shared_ptr<State> state, std::size_t index);
        void release() noexcept;

        std::shared_ptr<State> state_;
        std::size_t index_{0};
    };

    explicit TaskPool(std::vector<std::unique_ptr<ITask>> tasks);
    ~TaskPool();
    TaskPool(TaskPool&&) noexcept;
    TaskPool& operator=(TaskPool&&) noexcept;
    TaskPool(const TaskPool&) = delete;
    TaskPool& operator=(const TaskPool&) = delete;

    Lease acquire();
    std::optional<Lease> try_acquire();
    std::size_t capacity() const;
    std::size_t available() const;

  private:
    std::shared_ptr<State> state_;
};

// Call load_task() count times with the same direct inputs, then own those
// independent instances in one pool.
TaskPool load_task_pool(const std::string& bundle_path, const std::string& runtime_root,
                        std::size_t count, std::uint64_t kv_cache_size_bytes = 0,
                        const std::string& runtime_cache_path = {}, bool cuda_graphs = false);

} // namespace trtmc
