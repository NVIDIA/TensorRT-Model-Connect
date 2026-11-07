/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Row-range parallelism for SAM's host-side resampling passes.
//
// Every destination row is computed by the same expression on the same
// operands whichever worker computes it, so a pass's output does not depend on
// the worker count: splitting rows changes only which thread writes them.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace trtmc {

// A worker takes at least this many destination values of a pass, so a small
// or narrow pass stays on the calling thread (Thor: below this, starting and
// joining a thread costs more than its share of the work saves).
constexpr int64_t kSamMinValuesPerWorker = 128 * 1024;

// The CPUs a cgroup v2 `cpu.max` line ("<quota> <period>" or "max <period>")
// allows, rounded up; 0 when it sets no limit or does not parse.
inline int32_t sam_cpu_max_cpus(const std::string& cpu_max) {
    const char* text = cpu_max.c_str();
    char* end = nullptr;
    const long long quota = std::strtoll(text, &end, 10);
    if (end == text || quota <= 0)
        return 0;
    const char* period_text = end;
    const long long period = std::strtoll(period_text, &end, 10);
    if (end == period_text || period <= 0)
        return 0;
    return static_cast<int32_t>(std::min<long long>((quota + period - 1) / period, 1 << 20));
}

// The CPU quota of this process's cgroup v2 hierarchy: the smallest
// `cpu.max` limit of its cgroup and every ancestor, rounded up to whole CPUs;
// 0 when none applies or it cannot be read (cgroup v1, other systems). Read
// once per process.
inline int32_t sam_cgroup_cpu_quota() {
#if defined(__linux__)
    static const int32_t quota = [] {
        std::ifstream cgroup("/proc/self/cgroup");
        std::string line;
        std::string path;
        while (std::getline(cgroup, line)) {
            if (line.rfind("0::", 0) == 0) {
                path = line.substr(3);
                break;
            }
        }
        if (path.empty() || path[0] != '/')
            return 0;
        int32_t limit = 0;
        while (true) {
            std::ifstream file("/sys/fs/cgroup" + (path == "/" ? std::string() : path) +
                               "/cpu.max");
            std::string cpu_max;
            if (std::getline(file, cpu_max)) {
                const int32_t cpus = sam_cpu_max_cpus(cpu_max);
                if (cpus > 0)
                    limit = limit == 0 ? cpus : std::min(limit, cpus);
            }
            if (path == "/")
                break;
            const auto slash = path.rfind('/');
            path = slash == 0 ? "/" : path.substr(0, slash);
        }
        return limit;
    }();
    return quota;
#else
    return 0;
#endif
}

// The workers a host pass may use: the CPUs this thread may run on (its
// affinity mask on Linux, else the hardware threads), within the cgroup CPU
// quota, at most 8.
inline int32_t sam_host_worker_count() {
    constexpr int32_t kWorkerCap = 8;
    int32_t cpus = 0;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        cpus = CPU_COUNT(&mask);
#endif
    if (cpus <= 0)
        cpus = static_cast<int32_t>(std::thread::hardware_concurrency());
    const int32_t quota = sam_cgroup_cpu_quota();
    if (quota > 0)
        cpus = std::min(cpus, quota);
    return std::clamp(cpus, 1, kWorkerCap);
}

// The rows a worker takes so that it gets at least `min_values` values of
// `row_values` each.
inline int64_t sam_min_rows_per_worker(int64_t row_values,
                                       int64_t min_values = kSamMinValuesPerWorker) {
    const int64_t per_row = std::max<int64_t>(row_values, 1);
    return std::max<int64_t>(1, (std::max<int64_t>(min_values, 1) + per_row - 1) / per_row);
}

// Calls function(begin, end) on contiguous row ranges that cover [0, count)
// exactly once: at most `workers` ranges, each at least `min_rows` rows when
// there is more than one. The calling thread runs the last range. When a
// thread cannot be started, the calling thread runs the ranges left over. An
// exception from any range is rethrown once every started thread has joined.
template <typename Function>
void sam_parallel_for_rows(int64_t count, int64_t min_rows, int32_t workers, Function&& function) {
    if (count <= 0)
        return;
    const int64_t by_rows = std::max<int64_t>(1, count / std::max<int64_t>(1, min_rows));
    const auto ranges =
        static_cast<int32_t>(std::max<int64_t>(1, std::min({int64_t{workers}, by_rows, count})));
    if (ranges == 1) {
        function(int64_t{0}, count);
        return;
    }

    std::vector<std::exception_ptr> errors(static_cast<std::size_t>(ranges));
    const auto run = [&](int32_t range) {
        try {
            function(count * range / ranges, count * (range + 1) / ranges);
        } catch (...) {
            errors[static_cast<std::size_t>(range)] = std::current_exception();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(ranges - 1));
    int32_t started = 0;
    try {
        for (; started < ranges - 1; ++started)
            threads.emplace_back(run, started);
    } catch (...) {
        // Fewer threads than planned: the calling thread runs the rest below.
    }
    for (int32_t range = started; range < ranges; ++range)
        run(range);
    for (auto& thread : threads)
        thread.join();
    for (const auto& error : errors) {
        if (error)
            std::rethrow_exception(error);
    }
}

// Calls pass(item, row_begin, row_end) for each item's share of the flattened
// rows [begin, end), where every item has `rows` rows: flattened row
// index = item * rows + row. Lets one sam_parallel_for_rows pass span the rows
// of several planes.
template <typename Pass>
void sam_for_item_rows(int64_t begin, int64_t end, int64_t rows, Pass&& pass) {
    if (rows <= 0)
        return;
    for (int64_t index = begin; index < end;) {
        const int64_t item = index / rows;
        const int64_t row_begin = index - item * rows;
        const int64_t row_end = std::min(rows, row_begin + (end - index));
        pass(static_cast<std::size_t>(item), static_cast<std::size_t>(row_begin),
             static_cast<std::size_t>(row_end));
        index += row_end - row_begin;
    }
}

} // namespace trtmc
