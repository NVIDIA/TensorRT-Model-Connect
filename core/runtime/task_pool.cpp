/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/task_pool.h"

#include "trtmc/runtime/family_loader.h"

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace trtmc {

struct TaskPool::State {
    struct Slot {
        std::unique_ptr<ITask> task;
        bool leased{false};
    };

    explicit State(std::vector<std::unique_ptr<ITask>> tasks) {
        if (tasks.empty())
            throw std::invalid_argument("TaskPool requires at least one task");
        slots.reserve(tasks.size());
        for (auto& task : tasks) {
            if (task == nullptr)
                throw std::invalid_argument("TaskPool cannot contain a null task");
            slots.push_back({std::move(task), false});
        }
        available = slots.size();
    }

    std::vector<Slot> slots;
    std::size_t available{0};
    mutable std::mutex mutex;
    std::condition_variable changed;
};

TaskPool::Lease::Lease(std::shared_ptr<State> state, std::size_t index)
    : state_(std::move(state)), index_(index) {}

TaskPool::Lease::~Lease() {
    release();
}

TaskPool::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)), index_(other.index_) {}

TaskPool::Lease& TaskPool::Lease::operator=(Lease&& other) noexcept {
    if (this == &other)
        return *this;
    release();
    state_ = std::move(other.state_);
    index_ = other.index_;
    return *this;
}

ITask* TaskPool::Lease::get() const {
    if (state_ == nullptr)
        throw std::logic_error("TaskPool lease is empty");
    return state_->slots[index_].task.get();
}

void TaskPool::Lease::release() noexcept {
    if (state_ == nullptr)
        return;
    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        state_->slots[index_].leased = false;
        ++state_->available;
    }
    state_->changed.notify_one();
    state_.reset();
}

TaskPool::TaskPool(std::vector<std::unique_ptr<ITask>> tasks)
    : state_(std::make_shared<State>(std::move(tasks))) {}

TaskPool::~TaskPool() = default;
TaskPool::TaskPool(TaskPool&&) noexcept = default;
TaskPool& TaskPool::operator=(TaskPool&&) noexcept = default;

TaskPool::Lease TaskPool::acquire() {
    if (state_ == nullptr)
        throw std::logic_error("TaskPool is empty");
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] { return state_->available > 0; });
    for (std::size_t index = 0; index < state_->slots.size(); ++index) {
        auto& slot = state_->slots[index];
        if (slot.leased)
            continue;
        slot.leased = true;
        --state_->available;
        return Lease(state_, index);
    }
    throw std::logic_error("TaskPool availability invariant violated");
}

std::optional<TaskPool::Lease> TaskPool::try_acquire() {
    if (state_ == nullptr)
        return std::nullopt;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->available == 0)
        return std::nullopt;
    for (std::size_t index = 0; index < state_->slots.size(); ++index) {
        auto& slot = state_->slots[index];
        if (slot.leased)
            continue;
        slot.leased = true;
        --state_->available;
        return Lease(state_, index);
    }
    throw std::logic_error("TaskPool availability invariant violated");
}

std::size_t TaskPool::capacity() const {
    if (state_ == nullptr)
        return 0;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->slots.size();
}

std::size_t TaskPool::available() const {
    if (state_ == nullptr)
        return 0;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->available;
}

TaskPool load_task_pool(const std::string& bundle_path, const std::string& runtime_root,
                        std::size_t count, std::uint64_t kv_cache_size_bytes,
                        const std::string& runtime_cache_path, bool cuda_graphs) {
    if (count == 0)
        throw std::invalid_argument("TaskPool count must be positive");
    std::vector<std::unique_ptr<ITask>> tasks;
    tasks.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        tasks.push_back(load_task(bundle_path, runtime_root, kv_cache_size_bytes,
                                  runtime_cache_path, cuda_graphs));
    }
    return TaskPool(std::move(tasks));
}

} // namespace trtmc
