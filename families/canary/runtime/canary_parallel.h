/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Range sharing for canary's host-side feature preparation (resampling and
// the mel frontend).
//
// A pass is cut into ranges. The thread that runs the pass takes ranges one
// by one from a shared cursor, and so do the helper threads of its
// HostWorkers that are free. The calling thread never waits for a helper that
// has not joined the pass; it does wait for those that have, and when the
// CPUs are busy with other work a helper, or the calling thread itself, can
// lose its CPU for a scheduler time slice. A pass that took longer shared
// than the calling thread would have needed alone makes the team rest for the
// next passes, which then run on the calling thread alone. Every output value of a
// pass is computed by the same statements on the same operands whichever
// thread computes it, so a pass's output does not depend on how it is cut or
// on which thread takes which range.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace trtmc {
namespace canary {

// The CPUs a cgroup v2 `cpu.max` line ("<quota> <period>" or "max <period>")
// allows, rounded up; 0 when it sets no limit or does not parse.
int32_t cpu_max_cpus(const std::string& cpu_max);

// The CPU quota of a cgroup v2 hierarchy: the smallest `cpu.max` limit of the
// cgroup that `cgroup_file` (a /proc/<pid>/cgroup file) names and of every
// ancestor of it below `cgroup_root`, rounded up to whole CPUs; 0 when none
// applies or it cannot be read (cgroup v1, other systems).
int32_t cgroup_cpu_quota_at(const std::string& cgroup_file, const std::string& cgroup_root);

// cgroup_cpu_quota_at for this process on Linux, read once per process; 0 on
// other systems.
int32_t cgroup_cpu_quota();

// The CPUs the calling thread may run on: its affinity mask on Linux, else
// the hardware threads; at least 1.
int32_t allowed_cpus();

// The threads a request's feature preparation uses on `cpus` allowed CPUs
// under a cgroup quota of `quota` CPUs (0: none): the smaller of the two, at
// least 1, at most 8.
int32_t worker_count_for(int32_t cpus, int32_t quota);

// The environment variable that sets the worker count instead: decimal
// digits only, a count of at least 1, used up to the same cap of 8. 1 keeps
// every pass on the thread that prepares the request.
constexpr const char* kHostWorkersEnv = "TRTMC_CANARY_HOST_WORKERS";

// The worker count an environment value asks for; 0 when it asks for none
// (unset, empty, or anything but the digits of a count of at least 1).
int32_t worker_count_from_env(const char* value);

// The environment's worker count when it gives one, else
// worker_count_for(allowed_cpus(), cgroup_cpu_quota()). A value that is set
// and is no count is named once on stderr and ignored.
int32_t host_worker_count();

using Range = std::pair<int64_t, int64_t>; // [first, second)

// One range of a pass. `worker` is below the runner's workers() and belongs
// to one thread for the length of the call, so a pass can keep scratch state
// per worker.
using RangeFunction = std::function<void(int64_t begin, int64_t end, int32_t worker)>;

// Runs the ranges of a pass.
class RangeRunner {
  public:
    virtual ~RangeRunner() = default;

    // The threads that may run ranges of one pass at the same time, the
    // calling thread included.
    virtual int32_t workers() const = 0;

    // Calls function(begin, end, worker) once for every range, in any order
    // and on any of its threads, and returns when every call has returned.
    // After a call throws, ranges not yet taken may be skipped; the first
    // exception is rethrown once no call is running.
    virtual void run(const std::vector<Range>& ranges, const RangeFunction& function) = 0;
};

// A pass is shared only when at least this many of its smallest ranges fit
// (about a millisecond of work: waking the helpers for less does not pay),
// and it is cut into at most this many ranges per worker: enough ranges for
// late helpers to find work, few enough for a range to outweigh the cost of
// taking it.
constexpr int64_t kMinRangesToShare = 16;
constexpr int64_t kRangesPerWorker = 4;

// Contiguous ranges that cover [0, count) exactly once, each at least
// `min_items` items: one range when `workers` is 1 or less or when fewer than
// kMinRangesToShare such ranges fit, else as many as fit up to
// kRangesPerWorker per worker. Empty when count is not positive.
std::vector<Range> pass_ranges(int64_t count, int64_t min_items, int32_t workers);

// Runs a pass: a single range, or any ranges without a runner, on the calling
// thread as worker 0 and in order; otherwise through the runner.
void run_pass(const std::vector<Range>& ranges, RangeRunner* runner, const RangeFunction& function);

// The runner for the passes of one of `segments` segments that are prepared
// at the same time, each on a thread of its own: the runner for a segment
// alone, none for several. Several segments already use as many threads, and
// helpers beside them take CPUs from the other segments.
RangeRunner* runner_for_segments(RangeRunner* runner, std::size_t segments);

// A team of helper threads that share passes with the threads that call
// run(). The helpers sleep between passes.
//
// A run() takes its ranges itself, on its own thread as worker 0, when
// - the helpers are taken by another pass (another thread's, or the one it
//   was called from): one pass at a time has them;
// - the team is resting. Each shared pass is compared with what the calling
//   thread would have needed alone: its own CPU time per range it ran, times
//   the ranges (it always runs the first range itself). A pass that took
//   more than kSlack longer did not have the CPUs to itself, and rests the
//   team for kFirstRest passes. The rest doubles with every such pass, up to
//   kLongestRest, until the shared passes have saved as much time as those
//   lost; then it starts again at kFirstRest. Where the system has no
//   per-thread CPU clock the team never rests;
// - the process is not the one that built the team (a forked child has no
//   helper threads; its destructor does not join any). The process is known
//   by its id: a child with the builder's id in a nested PID namespace is
//   taken for the builder.
class HostWorkers final : public RangeRunner {
  public:
    // Starts `workers` - 1 helper threads, or as many of them as the system
    // lets it start.
    explicit HostWorkers(int32_t workers);
    ~HostWorkers() override;

    HostWorkers(const HostWorkers&) = delete;
    HostWorkers& operator=(const HostWorkers&) = delete;

    static constexpr std::chrono::microseconds kSlack{500};
    static constexpr int32_t kFirstRest = 4;
    static constexpr int32_t kLongestRest = 1024;

    int32_t workers() const override { return static_cast<int32_t>(helpers_.size()) + 1; }

    void run(const std::vector<Range>& ranges, const RangeFunction& function) override;

    // For the tests: how many times a helper has joined a pass since the team
    // was built (at most once per helper and pass), and how many passes the
    // team still stays out of.
    uint64_t helper_joins();
    int32_t resting() const { return rest_.load(); }

  private:
    struct Pass {
        const std::vector<Range>* ranges;
        const RangeFunction* function;
        std::atomic<std::size_t> next{1}; // range 0 is the calling thread's
        int32_t helpers{0};       // helpers that have joined and not left; guarded by the mutex
        std::exception_ptr error; // the first failure; guarded by the mutex
    };

    // What the helpers sleep on. Apart from the team so that a forked child
    // can leave it alone: destroying a condition variable that the parent's
    // helpers were waiting on would block the child for ever.
    struct Sync {
        std::mutex mutex;
        std::condition_variable wake; // a pass was posted, or the team is stopping
        std::condition_variable idle; // a pass's last helper has left it
    };

    // Runs range `index`, then the ranges it takes from the pass until none
    // is left; the number this call ran.
    std::size_t take_ranges(Pass& pass, int32_t worker, std::size_t index) noexcept;
    void helper_loop(int32_t worker) noexcept;
    bool in_building_process() const;

    std::unique_ptr<Sync> sync_;
    std::vector<std::thread> helpers_;
    long long building_process_{0};  // the process that started the helpers
    std::atomic<bool> taken_{false}; // a run() has the helpers for its pass
    std::atomic<int32_t> rest_{
        0}; // passes the team still stays out of
            // The last rest, and the time the shared passes have lost against their
    // callers alone and not yet saved again; the run() that has the helpers
    // owns both.
    int32_t rest_length_{0};
    std::chrono::nanoseconds lost_{0};
    uint64_t joins_{0}; // guarded by the mutex
    Pass* pass_{nullptr};
    uint64_t generation_{0};
    bool stopping_{false};
};

} // namespace canary
} // namespace trtmc
