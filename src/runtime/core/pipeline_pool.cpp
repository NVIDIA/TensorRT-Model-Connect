/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_pool.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace trtmc {

struct PipelinePool::State {
    struct Slot {
        std::unique_ptr<IPipeline> pipeline;
        bool leased{false};
    };

    explicit State(std::vector<std::unique_ptr<IPipeline>> pipelines) {
        if (pipelines.empty())
            throw std::invalid_argument("PipelinePool requires at least one pipeline");
        slots.reserve(pipelines.size());
        for (auto& pipeline : pipelines) {
            if (!pipeline)
                throw std::invalid_argument("PipelinePool cannot contain a null pipeline");
            slots.push_back(Slot{std::move(pipeline), false});
        }
        available = slots.size();
    }

    std::vector<Slot> slots;
    std::size_t available{0};
    bool maintenance{false};
    mutable std::mutex mutex;
    std::condition_variable changed;
};

namespace {

bool contains_adapter(IPipeline& pipeline, const std::string& adapter_id) {
    const auto ids = pipeline.loaded_lora_adapters();
    return std::find(ids.begin(), ids.end(), adapter_id) != ids.end();
}

void rollback_loaded_adapters(const std::vector<IPipeline*>& loaded,
                              const std::string& adapter_id) noexcept {
    for (auto iterator = loaded.rbegin(); iterator != loaded.rend(); ++iterator) {
        try {
            if (contains_adapter(**iterator, adapter_id))
                (*iterator)->unload_lora_adapter(adapter_id);
        } catch (...) {
            // Preserve the original load failure. A later explicit unload can
            // clean up a plugin that also failed rollback.
        }
    }
}

} // namespace

class PipelinePool::MaintenanceGuard {
  public:
    explicit MaintenanceGuard(const std::shared_ptr<State>& state) : state_(state) {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->changed.wait(lock, [&] { return !state_->maintenance; });
        state_->maintenance = true;
        state_->changed.wait(lock, [&] { return state_->available == state_->slots.size(); });
    }

    ~MaintenanceGuard() {
        {
            const std::lock_guard<std::mutex> lock(state_->mutex);
            state_->maintenance = false;
        }
        state_->changed.notify_all();
    }

  private:
    std::shared_ptr<State> state_;
};

PipelinePool::Lease::Lease(std::shared_ptr<State> state, std::size_t index)
    : state_(std::move(state)), index_(index) {}

PipelinePool::Lease::~Lease() {
    release();
}

PipelinePool::Lease::Lease(Lease&& other) noexcept
    : state_(std::move(other.state_)), index_(other.index_) {}

PipelinePool::Lease& PipelinePool::Lease::operator=(Lease&& other) noexcept {
    if (this == &other)
        return *this;
    release();
    state_ = std::move(other.state_);
    index_ = other.index_;
    return *this;
}

IPipeline* PipelinePool::Lease::get() const {
    if (!state_)
        throw std::logic_error("PipelinePool lease is empty");
    return state_->slots[index_].pipeline.get();
}

void PipelinePool::Lease::release() {
    if (!state_)
        return;
    {
        const std::lock_guard<std::mutex> lock(state_->mutex);
        state_->slots[index_].leased = false;
        ++state_->available;
    }
    state_->changed.notify_all();
    state_.reset();
}

PipelinePool::PipelinePool(std::vector<std::unique_ptr<IPipeline>> pipelines)
    : state_(std::make_shared<State>(std::move(pipelines))) {}

PipelinePool::~PipelinePool() = default;
PipelinePool::PipelinePool(PipelinePool&&) noexcept = default;
PipelinePool& PipelinePool::operator=(PipelinePool&&) noexcept = default;

PipelinePool::Lease PipelinePool::acquire() {
    if (!state_)
        throw std::logic_error("PipelinePool is empty");
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->changed.wait(lock, [&] { return !state_->maintenance && state_->available > 0; });
    for (std::size_t index = 0; index < state_->slots.size(); ++index) {
        auto& slot = state_->slots[index];
        if (slot.leased)
            continue;
        slot.leased = true;
        --state_->available;
        return Lease(state_, index);
    }
    throw std::logic_error("PipelinePool availability invariant violated");
}

std::optional<PipelinePool::Lease> PipelinePool::try_acquire() {
    if (!state_)
        return std::nullopt;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->maintenance || state_->available == 0)
        return std::nullopt;
    for (std::size_t index = 0; index < state_->slots.size(); ++index) {
        auto& slot = state_->slots[index];
        if (slot.leased)
            continue;
        slot.leased = true;
        --state_->available;
        return Lease(state_, index);
    }
    return std::nullopt;
}

std::size_t PipelinePool::size() const {
    if (!state_)
        return 0;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->slots.size();
}

std::size_t PipelinePool::available() const {
    if (!state_)
        return 0;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->available;
}

bool PipelinePool::supports_lora_adapters() const {
    if (!state_)
        return false;
    const std::lock_guard<std::mutex> lock(state_->mutex);
    return std::all_of(state_->slots.begin(), state_->slots.end(), [](const State::Slot& slot) {
        return slot.pipeline->supports_lora_adapters();
    });
}

void PipelinePool::load_lora_adapter(const std::string& adapter_id,
                                     const std::string& adapter_path) {
    if (!state_)
        throw std::logic_error("PipelinePool is empty");
    if (adapter_id.empty())
        throw std::invalid_argument("PipelinePool adapter ID must not be empty");
    MaintenanceGuard guard(state_);
    for (const auto& slot : state_->slots) {
        if (!slot.pipeline->supports_lora_adapters())
            throw std::runtime_error(std::string(slot.pipeline->pipeline_type()) +
                                     " does not support dynamic LoRA adapters");
    }
    std::vector<IPipeline*> loaded;
    try {
        for (const auto& slot : state_->slots) {
            if (contains_adapter(*slot.pipeline, adapter_id))
                continue;
            slot.pipeline->load_lora_adapter(adapter_id, adapter_path);
            loaded.push_back(slot.pipeline.get());
        }
    } catch (...) {
        rollback_loaded_adapters(loaded, adapter_id);
        throw;
    }
}

void PipelinePool::unload_lora_adapter(const std::string& adapter_id) {
    if (!state_)
        throw std::logic_error("PipelinePool is empty");
    MaintenanceGuard guard(state_);
    bool found = false;
    for (const auto& slot : state_->slots) {
        if (!contains_adapter(*slot.pipeline, adapter_id))
            continue;
        slot.pipeline->unload_lora_adapter(adapter_id);
        found = true;
    }
    if (!found)
        throw std::invalid_argument("PipelinePool: unknown adapter ID '" + adapter_id + "'");
}

std::vector<std::string> PipelinePool::loaded_lora_adapters() const {
    if (!state_)
        return {};
    MaintenanceGuard guard(state_);
    return state_->slots.front().pipeline->loaded_lora_adapters();
}

} // namespace trtmc
