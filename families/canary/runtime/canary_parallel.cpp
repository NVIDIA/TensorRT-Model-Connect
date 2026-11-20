/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/canary_parallel.h"

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace trtmc {
namespace canary {

int32_t cpu_max_cpus(const std::string& cpu_max) {
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

int32_t cgroup_cpu_quota_at(const std::string& cgroup_file, const std::string& cgroup_root) {
    std::ifstream cgroup(cgroup_file);
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
        std::ifstream file(cgroup_root + (path == "/" ? std::string() : path) + "/cpu.max");
        std::string cpu_max;
        if (std::getline(file, cpu_max)) {
            const int32_t cpus = cpu_max_cpus(cpu_max);
            if (cpus > 0)
                limit = limit == 0 ? cpus : std::min(limit, cpus);
        }
        if (path == "/")
            break;
        const auto slash = path.rfind('/');
        path = slash == 0 ? "/" : path.substr(0, slash);
    }
    return limit;
}

int32_t cgroup_cpu_quota() {
#if defined(__linux__)
    static const int32_t quota = cgroup_cpu_quota_at("/proc/self/cgroup", "/sys/fs/cgroup");
    return quota;
#else
    return 0;
#endif
}

int32_t allowed_cpus() {
    int32_t cpus = 0;
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
        cpus = CPU_COUNT(&mask);
#endif
    if (cpus <= 0)
        cpus = static_cast<int32_t>(std::thread::hardware_concurrency());
    return std::max(cpus, 1);
}

namespace {
constexpr int32_t kWorkerCap = 8;
} // namespace

int32_t worker_count_for(int32_t cpus, int32_t quota) {
    if (quota > 0)
        cpus = std::min(cpus, quota);
    return std::clamp(cpus, 1, kWorkerCap);
}

int32_t worker_count_from_env(const char* value) {
    if (value == nullptr || *value == '\0')
        return 0;
    int32_t count = 0;
    for (const char* digit = value; *digit != '\0'; ++digit) {
        if (*digit < '0' || *digit > '9')
            return 0;
        count = std::min(count * 10 + (*digit - '0'), kWorkerCap);
    }
    return count;
}

int32_t host_worker_count() {
    const char* value = std::getenv(kHostWorkersEnv);
    const int32_t asked = worker_count_from_env(value);
    if (asked > 0)
        return asked;
    if (value != nullptr && *value != '\0') {
        static std::once_flag named;
        std::call_once(named, [value] {
            std::cerr << "[canary] ignoring " << kHostWorkersEnv << "=" << value
                      << ": not the digits of a count of at least 1" << std::endl;
        });
    }
    return worker_count_for(allowed_cpus(), cgroup_cpu_quota());
}

std::vector<Range> pass_ranges(int64_t count, int64_t min_items, int32_t workers) {
    std::vector<Range> ranges;
    if (count <= 0)
        return ranges;
    const int64_t fit = count / std::max<int64_t>(min_items, 1);
    int64_t parts = 1;
    if (workers > 1 && fit >= kMinRangesToShare)
        parts = std::min(fit, int64_t{workers} * kRangesPerWorker);
    ranges.reserve(static_cast<std::size_t>(parts));
    for (int64_t part = 0; part < parts; ++part)
        ranges.emplace_back(count * part / parts, count * (part + 1) / parts);
    return ranges;
}

void run_pass(const std::vector<Range>& ranges, RangeRunner* runner,
              const RangeFunction& function) {
    if (runner != nullptr && ranges.size() > 1) {
        runner->run(ranges, function);
        return;
    }
    for (const auto& [begin, end] : ranges)
        function(begin, end, 0);
}

RangeRunner* runner_for_segments(RangeRunner* runner, std::size_t segments) {
    return segments == 1 ? runner : nullptr;
}

namespace {

long long this_process() {
#if defined(__unix__) || defined(__APPLE__)
    return static_cast<long long>(getpid());
#else
    return 0;
#endif
}

// The CPU time the calling thread has used; negative where the system has no
// such clock.
std::chrono::nanoseconds thread_cpu_time() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec now{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now) == 0)
        return std::chrono::seconds(now.tv_sec) + std::chrono::nanoseconds(now.tv_nsec);
#endif
    return std::chrono::nanoseconds(-1);
}

} // namespace

HostWorkers::HostWorkers(int32_t workers)
    : sync_(std::make_unique<Sync>()), building_process_(this_process()) {
    const int32_t helpers = std::max(workers, 1) - 1;
    helpers_.reserve(static_cast<std::size_t>(helpers));
    try {
        for (int32_t helper = 0; helper < helpers; ++helper)
            helpers_.emplace_back([this, helper] { helper_loop(helper + 1); });
    } catch (...) {
        // Fewer helpers than asked for: the passes share with those there are.
    }
}

HostWorkers::~HostWorkers() {
    if (!in_building_process()) {
        // A forked child: the helper threads are the parent's. There is none
        // to stop or to join here, and what they sleep on is left as it is
        // (the copy says helpers are waiting on it, and none ever will leave).
        for (auto& helper : helpers_)
            helper.detach();
        static_cast<void>(sync_.release());
        return;
    }
    {
        const std::lock_guard<std::mutex> lock(sync_->mutex);
        stopping_ = true;
    }
    sync_->wake.notify_all();
    for (auto& helper : helpers_)
        helper.join();
}

bool HostWorkers::in_building_process() const {
    return this_process() == building_process_;
}

uint64_t HostWorkers::helper_joins() {
    if (!in_building_process())
        return 0; // a forked child has no helpers, and their mutex is not its own
    const std::lock_guard<std::mutex> lock(sync_->mutex);
    return joins_;
}

void HostWorkers::run(const std::vector<Range>& ranges, const RangeFunction& function) {
    const auto alone = [&] {
        for (const auto& [begin, end] : ranges)
            function(begin, end, 0);
    };
    if (ranges.size() <= 1 || helpers_.empty() || !in_building_process())
        return alone();
    // Resting: this pass is one of those the team stays out of.
    for (int32_t rest = rest_.load(); rest > 0;) {
        if (rest_.compare_exchange_weak(rest, rest - 1))
            return alone();
    }
    if (taken_.exchange(true))
        return alone();
    // The helpers are this pass's until the guard goes.
    struct Release {
        std::atomic<bool>& taken;
        ~Release() { taken.store(false); }
    } release{taken_};

    Pass pass;
    pass.ranges = &ranges;
    pass.function = &function;
    const auto posted = std::chrono::steady_clock::now();
    {
        const std::lock_guard<std::mutex> lock(sync_->mutex);
        pass_ = &pass;
        ++generation_;
    }
    sync_->wake.notify_all();
    const auto cpu_before = thread_cpu_time();
    const std::size_t own_ranges = take_ranges(pass, 0, 0);
    const auto own_cpu = thread_cpu_time() - cpu_before;
    {
        // Every range is taken. A helper that wakes from here on finds no
        // pass; only the helpers that have joined this one are waited for.
        std::unique_lock<std::mutex> lock(sync_->mutex);
        pass_ = nullptr;
        sync_->idle.wait(lock, [&pass] { return pass.helpers == 0; });
    }
    if (cpu_before.count() >= 0) {
        // Alone, the calling thread would have needed about its own CPU time
        // per range for every range. A shared pass that took longer did not
        // have the CPUs to itself: a helper, or this thread, lost its CPU to
        // other work.
        const auto shared = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - posted);
        const auto alone =
            own_cpu * static_cast<int64_t>(ranges.size()) / static_cast<int64_t>(own_ranges);
        lost_ = std::max(std::chrono::nanoseconds(0), lost_ + (shared - alone));
        if (shared > alone + kSlack) {
            rest_length_ = std::clamp(rest_length_ * 2, kFirstRest, kLongestRest);
            rest_.store(rest_length_);
        } else if (lost_.count() == 0) {
            rest_length_ = 0;
        }
    }
    if (pass.error)
        std::rethrow_exception(pass.error);
}

std::size_t HostWorkers::take_ranges(Pass& pass, int32_t worker, std::size_t index) noexcept {
    const std::size_t count = pass.ranges->size();
    std::size_t ran = 0;
    for (; index < count; index = pass.next.fetch_add(1, std::memory_order_relaxed)) {
        ++ran;
        try {
            const auto& [begin, end] = (*pass.ranges)[index];
            (*pass.function)(begin, end, worker);
        } catch (...) {
            const std::lock_guard<std::mutex> lock(sync_->mutex);
            if (!pass.error)
                pass.error = std::current_exception();
            pass.next.store(count, std::memory_order_relaxed);
        }
    }
    return ran;
}

void HostWorkers::helper_loop(int32_t worker) noexcept {
    std::unique_lock<std::mutex> lock(sync_->mutex);
    uint64_t seen = 0;
    while (true) {
        sync_->wake.wait(
            lock, [this, &seen] { return stopping_ || (pass_ != nullptr && generation_ != seen); });
        if (stopping_)
            return;
        seen = generation_;
        Pass& pass = *pass_;
        ++pass.helpers;
        ++joins_;
        lock.unlock();
        take_ranges(pass, worker, pass.next.fetch_add(1, std::memory_order_relaxed));
        lock.lock();
        if (--pass.helpers == 0)
            sync_->idle.notify_all();
    }
}

} // namespace canary
} // namespace trtmc
