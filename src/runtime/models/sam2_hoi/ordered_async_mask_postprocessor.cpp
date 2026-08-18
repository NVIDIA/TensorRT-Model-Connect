/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/ordered_async_mask_postprocessor.h"

#include <stdexcept>
#include <system_error>
#include <utility>

namespace trtmc::sam2_hoi {

OrderedAsyncMaskPostprocessor::OrderedAsyncMaskPostprocessor(std::size_t max_in_flight,
                                                             Launcher launcher)
    : max_in_flight_(max_in_flight), launcher_(std::move(launcher)) {
    if (max_in_flight_ == 0U)
        throw std::invalid_argument("SAM2 HOI mask postprocess window must be positive");
    if (!launcher_) {
        launcher_ = [](CallbackHandle callback, std::size_t) {
            return std::async(std::launch::async,
                              [callback = std::move(callback)]() mutable { return (*callback)(); });
        };
    }
}

OrderedAsyncMaskPostprocessor::~OrderedAsyncMaskPostprocessor() noexcept {
    (void)join_pending_noexcept();
}

void OrderedAsyncMaskPostprocessor::submit(std::size_t index, Callback callback) {
    if (index != next_to_submit_)
        throw std::invalid_argument("SAM2 HOI mask postprocess submissions must be ordered");
    if (full())
        throw std::logic_error("SAM2 HOI mask postprocess window is full");
    if (!callback)
        throw std::invalid_argument("SAM2 HOI mask postprocess callback is empty");

    auto callback_handle = make_callback_handle(index, std::move(callback));
    auto future = launch_callback(index, std::move(callback_handle));
    append_pending(index, std::move(future));
    ++next_to_submit_;
}

OrderedAsyncMaskPostprocessor::CallbackHandle
OrderedAsyncMaskPostprocessor::make_callback_handle(std::size_t index, Callback callback) {
    try {
        return std::make_shared<Callback>(std::move(callback));
    } catch (...) {
        rethrow_earliest({index, std::current_exception()}, join_pending_noexcept());
    }
}

std::future<OrderedAsyncMaskPostprocessor::Output>
OrderedAsyncMaskPostprocessor::launch_callback(std::size_t index, CallbackHandle callback) {
    try {
        return launcher_(callback, index);
    } catch (const std::system_error& error) {
        if (error.code() != std::make_error_code(std::errc::resource_unavailable_try_again))
            rethrow_earliest({index, std::current_exception()}, join_pending_noexcept());
        try {
            return std::async(std::launch::deferred,
                              [callback = std::move(callback)]() mutable { return (*callback)(); });
        } catch (...) {
            rethrow_earliest({index, std::current_exception()}, join_pending_noexcept());
        }
    } catch (...) {
        rethrow_earliest({index, std::current_exception()}, join_pending_noexcept());
    }
}

void OrderedAsyncMaskPostprocessor::append_pending(std::size_t index, std::future<Output> future) {
    try {
        pending_.push_back({index, std::move(future)});
    } catch (...) {
        const auto submit_failure = std::current_exception();
        try {
            (void)future.get();
        } catch (...) {
        }
        rethrow_earliest({index, submit_failure}, join_pending_noexcept());
    }
}

std::optional<OrderedAsyncMaskPostprocessor::IndexedFailure>
OrderedAsyncMaskPostprocessor::join_pending_noexcept() noexcept {
    std::optional<IndexedFailure> first_failure;
    while (!pending_.empty()) {
        const std::size_t index = pending_.front().index;
        auto future = std::move(pending_.front().future);
        pending_.pop_front();
        try {
            (void)future.get();
        } catch (...) {
            if (!first_failure.has_value())
                first_failure = IndexedFailure{index, std::current_exception()};
        }
    }
    return first_failure;
}

[[noreturn]] void OrderedAsyncMaskPostprocessor::rethrow_earliest(
    IndexedFailure failure, const std::optional<IndexedFailure>& pending_failure) {
    if (pending_failure.has_value() && pending_failure->index < failure.index)
        std::rethrow_exception(pending_failure->error);
    std::rethrow_exception(failure.error);
}

OrderedAsyncMaskPostprocessor::Result OrderedAsyncMaskPostprocessor::take_next() {
    if (pending_.empty())
        throw std::out_of_range("SAM2 HOI mask postprocess window is empty");
    if (pending_.front().index != next_to_consume_)
        throw std::logic_error("SAM2 HOI mask postprocess window lost input order");

    const std::size_t index = pending_.front().index;
    auto future = std::move(pending_.front().future);
    pending_.pop_front();
    Output output;
    try {
        output = future.get();
    } catch (...) {
        rethrow_earliest({index, std::current_exception()}, join_pending_noexcept());
    }
    ++next_to_consume_;
    return {index, std::move(output)};
}

} // namespace trtmc::sam2_hoi
