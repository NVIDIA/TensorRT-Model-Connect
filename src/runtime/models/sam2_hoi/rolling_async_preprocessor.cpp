/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/rolling_async_preprocessor.h"

#include <exception>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace trtmc::sam2_hoi {

RollingAsyncPreprocessor::RollingAsyncPreprocessor(std::size_t count, std::size_t max_in_flight,
                                                   Callback callback, Launcher launcher)
    : count_(count), max_in_flight_(max_in_flight), callback_(std::move(callback)),
      launcher_(std::move(launcher)) {
    if (max_in_flight_ == 0U)
        throw std::invalid_argument("SAM2 HOI preprocess lookahead must be positive");
    if (!callback_)
        throw std::invalid_argument("SAM2 HOI preprocess callback is empty");
    if (!launcher_) {
        launcher_ = [](const Callback& callback, std::size_t index) {
            return std::async(std::launch::async,
                              [callback, index]() mutable { return callback(index); });
        };
    }

    try {
        while (next_to_launch_ < count_ && pending_.size() < max_in_flight_)
            launch_one();
    } catch (...) {
        rethrow_earliest({next_to_launch_, std::current_exception()}, join_pending_noexcept());
    }
}

RollingAsyncPreprocessor::~RollingAsyncPreprocessor() noexcept {
    (void)join_pending_noexcept();
}

void RollingAsyncPreprocessor::launch_one() {
    const std::size_t index = next_to_launch_;
    std::future<Output> future;
    try {
        future = launcher_(callback_, index);
    } catch (const std::system_error& error) {
        if (error.code() != std::make_error_code(std::errc::resource_unavailable_try_again))
            throw;
        // Thread exhaustion must not turn an otherwise valid request into a
        // failure. Run this item lazily on the ordered consumer instead.
        future = std::async(std::launch::deferred,
                            [callback = callback_, index]() mutable { return callback(index); });
    }
    pending_.push_back({index, std::move(future)});
    ++next_to_launch_;
}

std::optional<RollingAsyncPreprocessor::IndexedFailure>
RollingAsyncPreprocessor::join_pending_noexcept() noexcept {
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

[[noreturn]] void
RollingAsyncPreprocessor::rethrow_earliest(IndexedFailure failure,
                                           const std::optional<IndexedFailure>& pending_failure) {
    if (pending_failure.has_value() && pending_failure->index < failure.index)
        std::rethrow_exception(pending_failure->error);
    std::rethrow_exception(failure.error);
}

RollingAsyncPreprocessor::Output RollingAsyncPreprocessor::take_next() {
    if (empty())
        throw std::out_of_range("SAM2 HOI preprocess lookahead is exhausted");
    if (pending_.empty() || pending_.front().index != next_to_consume_)
        throw std::logic_error("SAM2 HOI preprocess lookahead lost input order");

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

    try {
        if (next_to_launch_ < count_)
            launch_one();
    } catch (...) {
        rethrow_earliest({next_to_launch_, std::current_exception()}, join_pending_noexcept());
    }
    return output;
}

} // namespace trtmc::sam2_hoi
