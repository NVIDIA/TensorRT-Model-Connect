/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <vector>

namespace trtmc::sam2_hoi {

inline constexpr std::size_t kMaxConcurrentMaskPostprocessTasks = 5U;

// Bounded ordered executor for owned mask postprocessing work. The caller
// drains one result before submitting beyond max_in_flight, while destruction
// joins every submitted task on exceptional exits.
class OrderedAsyncMaskPostprocessor final {
  public:
    using Output = std::vector<std::uint8_t>;
    using Callback = std::function<Output()>;
    using CallbackHandle = std::shared_ptr<Callback>;
    using Launcher = std::function<std::future<Output>(CallbackHandle callback, std::size_t index)>;

    struct Result {
        std::size_t index{0U};
        Output output;
    };

    explicit OrderedAsyncMaskPostprocessor(
        std::size_t max_in_flight = kMaxConcurrentMaskPostprocessTasks, Launcher launcher = {});
    ~OrderedAsyncMaskPostprocessor() noexcept;

    OrderedAsyncMaskPostprocessor(const OrderedAsyncMaskPostprocessor&) = delete;
    OrderedAsyncMaskPostprocessor& operator=(const OrderedAsyncMaskPostprocessor&) = delete;
    OrderedAsyncMaskPostprocessor(OrderedAsyncMaskPostprocessor&&) = delete;
    OrderedAsyncMaskPostprocessor& operator=(OrderedAsyncMaskPostprocessor&&) = delete;

    bool empty() const noexcept { return pending_.empty(); }
    bool full() const noexcept { return pending_.size() == max_in_flight_; }
    std::size_t max_in_flight() const noexcept { return max_in_flight_; }

    void submit(std::size_t index, Callback callback);
    Result take_next();

  private:
    struct Pending {
        std::size_t index{0U};
        std::future<Output> future;
    };

    struct IndexedFailure {
        std::size_t index{0U};
        std::exception_ptr error;
    };

    std::optional<IndexedFailure> join_pending_noexcept() noexcept;
    [[noreturn]] static void rethrow_earliest(IndexedFailure failure,
                                              const std::optional<IndexedFailure>& pending_failure);
    CallbackHandle make_callback_handle(std::size_t index, Callback callback);
    std::future<Output> launch_callback(std::size_t index, CallbackHandle callback);
    void append_pending(std::size_t index, std::future<Output> future);

    std::size_t max_in_flight_{0U};
    std::size_t next_to_submit_{0U};
    std::size_t next_to_consume_{0U};
    Launcher launcher_;
    std::deque<Pending> pending_;
};

} // namespace trtmc::sam2_hoi
