/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// A range runner for the focused canary frontend tests. It records the ranges
// of every pass the production code shares, and runs the ranges the test
// selects, on the calling thread, last range first. So a test sees exactly
// what the production call hands out, can prove that the result does not
// depend on the order, and can run one range alone to see what it writes.

#include "families/canary/runtime/canary_parallel.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace canary_test {

class RecordingRunner final : public trtmc::canary::RangeRunner {
  public:
    // Whether to run range `range` of the `pass`-th shared pass (from 0).
    using Select = std::function<bool(std::size_t pass, std::size_t range)>;

    explicit RecordingRunner(int32_t workers, Select select = nullptr)
        : workers_(workers), select_(std::move(select)) {}

    int32_t workers() const override { return workers_; }

    void run(const std::vector<trtmc::canary::Range>& ranges,
             const trtmc::canary::RangeFunction& function) override {
        const std::size_t pass = passes.size();
        passes.push_back(ranges);
        for (std::size_t index = ranges.size(); index-- > 0;) {
            if (select_ && !select_(pass, index))
                continue;
            // Worker indices in turn: a pass must not depend on which worker
            // takes which range.
            function(ranges[index].first, ranges[index].second,
                     static_cast<int32_t>(index % static_cast<std::size_t>(workers_)));
        }
    }

    // The ranges of each shared pass, in the order the passes ran.
    std::vector<std::vector<trtmc::canary::Range>> passes;

  private:
    int32_t workers_;
    Select select_;
};

// Whether `ranges` cover [0, count) exactly once, in order, with no empty one.
inline bool covers_once(const std::vector<trtmc::canary::Range>& ranges, int64_t count) {
    int64_t next = 0;
    for (const auto& [begin, end] : ranges) {
        if (begin != next || end <= begin)
            return false;
        next = end;
    }
    return next == count;
}

} // namespace canary_test
