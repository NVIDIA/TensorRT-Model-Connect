/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <optional>
#include <vector>

namespace trtmc::sam2_hoi {

inline constexpr std::size_t kMaxConcurrentPreprocessTasks = 5U;

// Keeps a bounded ordered lookahead of CPU preprocessing tasks. A returned
// tensor remains owned by the caller while this object replenishes the pending
// window, so peak prepared storage is the current tensor plus max_in_flight.
class RollingAsyncPreprocessor final {
  public:
    using Output = std::vector<float>;
    using Callback = std::function<Output(std::size_t)>;
    using Launcher =
        std::function<std::future<Output>(const Callback& callback, std::size_t index)>;

    RollingAsyncPreprocessor(std::size_t count, std::size_t max_in_flight, Callback callback,
                             Launcher launcher = {});
    ~RollingAsyncPreprocessor() noexcept;

    RollingAsyncPreprocessor(const RollingAsyncPreprocessor&) = delete;
    RollingAsyncPreprocessor& operator=(const RollingAsyncPreprocessor&) = delete;
    RollingAsyncPreprocessor(RollingAsyncPreprocessor&&) = delete;
    RollingAsyncPreprocessor& operator=(RollingAsyncPreprocessor&&) = delete;

    bool empty() const noexcept { return next_to_consume_ == count_; }
    std::size_t max_in_flight() const noexcept { return max_in_flight_; }

    // Return the next result in input-index order. If preprocessing fails, all
    // launched tasks are joined before the lowest-index failure is rethrown.
    Output take_next();

  private:
    struct Pending {
        std::size_t index{0U};
        std::future<Output> future;
    };

    struct IndexedFailure {
        std::size_t index{0U};
        std::exception_ptr error;
    };

    void launch_one();
    std::optional<IndexedFailure> join_pending_noexcept() noexcept;
    [[noreturn]] static void rethrow_earliest(IndexedFailure failure,
                                              const std::optional<IndexedFailure>& pending_failure);

    std::size_t count_{0U};
    std::size_t max_in_flight_{0U};
    std::size_t next_to_launch_{0U};
    std::size_t next_to_consume_{0U};
    Callback callback_;
    Launcher launcher_;
    std::deque<Pending> pending_;
};

} // namespace trtmc::sam2_hoi
