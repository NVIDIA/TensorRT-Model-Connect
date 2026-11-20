/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Canary's host range sharing: the worker count follows the CPUs the thread
// may use and the cgroup quota; a pass is cut into ranges that cover it once;
// the team runs every range once, each worker index on one thread at a time,
// with the calling thread taking ranges too; a pass that finds the team taken
// runs on its own thread; a failure surfaces once no range is running.

#include "families/canary/runtime/canary_parallel.h"
#include "families/canary/tests/cpp/canary_range_runners.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using trtmc::canary::HostWorkers;
using trtmc::canary::Range;

int failures = 0;

void check(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void test_cpu_max_line() {
    using trtmc::canary::cpu_max_cpus;
    check(cpu_max_cpus("max 100000") == 0, "cpu.max without a quota sets no limit");
    check(cpu_max_cpus("") == 0, "an empty cpu.max sets no limit");
    check(cpu_max_cpus("quota") == 0, "an unparsable cpu.max sets no limit");
    check(cpu_max_cpus("200000") == 0, "cpu.max without a period sets no limit");
    check(cpu_max_cpus("200000 0") == 0, "cpu.max with a zero period sets no limit");
    check(cpu_max_cpus("-1 100000") == 0, "a negative cpu.max quota sets no limit");
    check(cpu_max_cpus("200000 100000") == 2, "cpu.max of two periods allows two CPUs");
    check(cpu_max_cpus("150000 100000") == 2, "a fractional cpu.max rounds up");
    check(cpu_max_cpus("1000 100000") == 1, "a cpu.max below one CPU allows one");
}

void write_file(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

// A cgroup v2 tree in a temporary directory: the quota is the smallest
// cpu.max of the process's cgroup and of every ancestor, the root included.
void test_cgroup_quota_walks_the_ancestors() {
    using trtmc::canary::cgroup_cpu_quota_at;
    const auto root = std::filesystem::temp_directory_path() /
                      ("canary-cgroup-test-" + std::to_string(
#if defined(__linux__)
                                                   static_cast<long long>(getpid())
#else
                                                   0LL
#endif
                                                       ));
    std::filesystem::remove_all(root);
    const auto tree = root / "fs";
    const auto proc = (root / "cgroup").string();
    write_file(root / "cgroup", "12:cpu:/ignored\n0::/a/b/c\n");

    check(cgroup_cpu_quota_at(proc, tree.string()) == 0, "a tree without cpu.max sets no quota");
    write_file(tree / "a/b/c/cpu.max", "max 100000\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 0, "a cgroup at max sets no quota");
    write_file(tree / "a/b/c/cpu.max", "400000 100000\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 4, "the cgroup's own cpu.max is its quota");
    write_file(tree / "a/cpu.max", "250000 100000\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 3, "a smaller quota of an ancestor wins");
    write_file(tree / "cpu.max", "100000 100000\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 1, "a smaller quota of the root wins");
    write_file(tree / "a/b/cpu.max", "800000 100000\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 1, "a larger quota of an ancestor does not");

    write_file(root / "cgroup", "0::/\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 1,
          "the root cgroup reads the root's cpu.max");
    write_file(root / "cgroup", "5:cpu,cpuacct:/a/b/c\n");
    check(cgroup_cpu_quota_at(proc, tree.string()) == 0, "a cgroup v1 line sets no quota");
    check(cgroup_cpu_quota_at((root / "missing").string(), tree.string()) == 0,
          "an unreadable cgroup file sets no quota");
    std::filesystem::remove_all(root);
}

void test_worker_count_follows_cpus_and_quota() {
    using trtmc::canary::worker_count_for;
    check(worker_count_for(1, 0) == 1, "one allowed CPU gives one worker");
    check(worker_count_for(3, 0) == 3, "three allowed CPUs give three workers");
    check(worker_count_for(28, 0) == 8, "the workers stop at eight");
    check(worker_count_for(28, 2) == 2, "a quota below the CPUs caps the workers");
    check(worker_count_for(2, 6) == 2, "a quota above the CPUs does not raise the workers");
    check(worker_count_for(0, 0) == 1 && worker_count_for(-4, 0) == 1 &&
              worker_count_for(8, -1) == 8,
          "nonsense counts still give a worker and no cap");
    check(trtmc::canary::host_worker_count() ==
              worker_count_for(trtmc::canary::allowed_cpus(), trtmc::canary::cgroup_cpu_quota()),
          "the host's workers are those of its allowed CPUs and its quota");
}

#if defined(__linux__)
// What `call` writes to stderr.
std::string written_to_stderr(const std::function<void()>& call) {
    const auto path = std::filesystem::temp_directory_path() /
                      ("canary-stderr-test-" + std::to_string(static_cast<long long>(getpid())));
    std::cerr.flush();
    std::fflush(stderr);
    const int file = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    const int saved = dup(STDERR_FILENO);
    if (file < 0 || saved < 0 || dup2(file, STDERR_FILENO) < 0)
        return "(stderr could not be redirected)";
    call();
    std::cerr.flush();
    std::fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    close(file);
    std::ifstream stream(path);
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    std::filesystem::remove(path);
    return text;
}
#endif

void test_environment_sets_the_worker_count() {
    using trtmc::canary::worker_count_from_env;
    check(worker_count_from_env(nullptr) == 0 && worker_count_from_env("") == 0,
          "an unset or empty value asks for no worker count");
    check(worker_count_from_env("1") == 1 && worker_count_from_env("3") == 3,
          "a positive integer is the worker count");
    check(worker_count_from_env("8") == 8 && worker_count_from_env("9") == 8 &&
              worker_count_from_env("4000000000") == 8,
          "an asked worker count stops at eight");
    check(worker_count_from_env("0") == 0 && worker_count_from_env("000") == 0 &&
              worker_count_from_env("-2") == 0 && worker_count_from_env("+2") == 0 &&
              worker_count_from_env("two") == 0 && worker_count_from_env("2x") == 0 &&
              worker_count_from_env("1.5") == 0 && worker_count_from_env(" 4") == 0 &&
              worker_count_from_env("4 ") == 0,
          "anything but the digits of a count of at least 1 asks for no worker count");
    check(worker_count_from_env("04") == 4, "leading zeros are digits");
#if defined(__linux__)
    const int32_t automatic = trtmc::canary::host_worker_count();
    setenv(trtmc::canary::kHostWorkersEnv, "1", 1);
    check(trtmc::canary::host_worker_count() == 1,
          "the environment can keep the passes on one thread");
    setenv(trtmc::canary::kHostWorkersEnv, "5", 1);
    check(trtmc::canary::host_worker_count() == 5, "the environment's worker count is the host's");
    // A value that is no count leaves the host's own count and is named on
    // stderr, once per process.
    setenv(trtmc::canary::kHostWorkersEnv, "many", 1);
    int32_t refused = 0;
    const std::string named =
        written_to_stderr([&refused] { refused = trtmc::canary::host_worker_count(); });
    check(refused == automatic, "an environment value that is no count leaves the host's own");
    check(named.find(std::string(trtmc::canary::kHostWorkersEnv) + "=many") != std::string::npos,
          "a refused environment value is named on stderr");
    setenv(trtmc::canary::kHostWorkersEnv, "more", 1);
    const std::string again =
        written_to_stderr([&refused] { refused = trtmc::canary::host_worker_count(); });
    check(refused == automatic && again.empty(), "a refused environment value is named once");
    unsetenv(trtmc::canary::kHostWorkersEnv);
#endif
}

void test_allowed_cpus_follow_the_affinity_mask() {
#if defined(__linux__)
    cpu_set_t original;
    CPU_ZERO(&original);
    if (sched_getaffinity(0, sizeof(original), &original) != 0)
        return;
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &original))
            cpus.push_back(cpu);
    }
    check(trtmc::canary::allowed_cpus() == static_cast<int32_t>(cpus.size()),
          "the allowed CPUs are the affinity mask's");
    for (const std::size_t wanted : {std::size_t{1}, std::size_t{2}, std::size_t{3}}) {
        if (cpus.size() < wanted)
            break;
        cpu_set_t narrowed;
        CPU_ZERO(&narrowed);
        for (std::size_t index = 0; index < wanted; ++index)
            CPU_SET(cpus[index], &narrowed);
        if (sched_setaffinity(0, sizeof(narrowed), &narrowed) != 0)
            break;
        check(trtmc::canary::allowed_cpus() == static_cast<int32_t>(wanted),
              "a mask of " + std::to_string(wanted) + " CPUs allows that many");
        check(trtmc::canary::host_worker_count() ==
                  trtmc::canary::worker_count_for(static_cast<int32_t>(wanted),
                                                  trtmc::canary::cgroup_cpu_quota()),
              "the host's workers follow a mask of " + std::to_string(wanted) + " CPUs");
    }
    sched_setaffinity(0, sizeof(original), &original);
#endif
}

void test_pass_ranges() {
    using trtmc::canary::pass_ranges;
    check(pass_ranges(0, 1, 8).empty() && pass_ranges(-3, 1, 8).empty(), "no items give no range");
    for (const int64_t count : {1, 2, 3, 4, 7, 31, 32, 33, 128, 416, 3000, 16639, 66560}) {
        for (const int64_t min_items : {-1, 0, 1, 2, 5, 8, 512, 2048}) {
            for (const int32_t workers : {-1, 0, 1, 2, 3, 8, 16}) {
                const auto ranges = pass_ranges(count, min_items, workers);
                const std::string name = std::to_string(count) + " items, shares of " +
                                         std::to_string(min_items) + ", " +
                                         std::to_string(workers) + " workers";
                check(canary_test::covers_once(ranges, count),
                      "ranges cover the items once: " + name);
                bool shares = true;
                for (const auto& [begin, end] : ranges)
                    shares = shares && (ranges.size() == 1 || end - begin >= min_items);
                check(shares, "every range of several holds the smallest share: " + name);
                const auto parts = static_cast<int64_t>(ranges.size());
                const int64_t fit = count / std::max<int64_t>(min_items, 1);
                const bool shared = workers > 1 && fit >= trtmc::canary::kMinRangesToShare;
                check(parts ==
                          (shared ? std::min(fit, workers * trtmc::canary::kRangesPerWorker) : 1),
                      "a pass is one range, or when sixteen shares fit as many as fit within "
                      "four per worker: " +
                          name);
            }
        }
    }
    check(pass_ranges(416, 8, 8).size() == 32,
          "416 frames in shares of 8 are 32 ranges on 8 workers");
    check(pass_ranges(416, 8, 2).size() == 8,
          "416 frames in shares of 8 are 8 ranges on 2 workers");
    check(pass_ranges(127, 8, 8).size() == 1, "fewer than sixteen shares stay one range");
    check(pass_ranges(128, 8, 8).size() == 16, "sixteen shares are sixteen ranges");
    check(pass_ranges(416, 8, 1).size() == 1, "one worker keeps a pass in one range");
}

void test_run_pass_without_a_team() {
    std::vector<Range> seen;
    const auto ranges = trtmc::canary::pass_ranges(100, 5, 4);
    trtmc::canary::run_pass(ranges, nullptr, [&](int64_t begin, int64_t end, int32_t worker) {
        seen.emplace_back(begin, end);
        check(worker == 0, "a pass without a runner runs as worker 0");
    });
    check(seen == ranges, "a pass without a runner runs every range in order");

    canary_test::RecordingRunner runner(4);
    int calls = 0;
    trtmc::canary::run_pass({{0, 7}}, &runner, [&](int64_t, int64_t, int32_t) { ++calls; });
    check(calls == 1 && runner.passes.empty(), "a pass of one range does not go to the runner");
    trtmc::canary::run_pass({}, &runner, [&](int64_t, int64_t, int32_t) { ++calls; });
    check(calls == 1, "an empty pass calls nothing");

    using trtmc::canary::runner_for_segments;
    check(runner_for_segments(&runner, 1) == &runner, "a segment alone gets the runner");
    check(runner_for_segments(&runner, 2) == nullptr &&
              runner_for_segments(&runner, 16) == nullptr &&
              runner_for_segments(&runner, 0) == nullptr,
          "segments prepared at the same time get no runner");
    check(runner_for_segments(nullptr, 1) == nullptr, "no runner stays no runner");
}

std::vector<Range> unit_ranges(int64_t count) {
    std::vector<Range> ranges;
    for (int64_t item = 0; item < count; ++item)
        ranges.emplace_back(item, item + 1);
    return ranges;
}

// A range that sleeps: off the CPU, as a thread that lost its CPU is.
void nap(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// A range that works: on the CPU for about that long.
void work(int milliseconds) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    volatile unsigned long spin = 0;
    while (std::chrono::steady_clock::now() < until)
        spin = spin + 1;
}

// What one pass looked like from inside its ranges.
struct PassView {
    std::vector<int> visits;
    std::set<std::thread::id> threads;
    bool bad_worker{false};
    bool shared_worker{false};
    bool only_worker_zero{true};
};

// Runs `ranges` on `team` and records who ran what. When `wait_for` is above
// 1, the ranges wait (polled, at most ten seconds, once) until that many
// threads are inside the pass. (Polled: a timed condition wait trips GCC 11's
// ThreadSanitizer.) Each range then sleeps `nap_ms` and works `work_ms`.
PassView view_pass(HostWorkers& team, const std::vector<Range>& ranges, std::size_t wait_for,
                   int nap_ms = 0, int work_ms = 0) {
    PassView view;
    view.visits.assign(ranges.size(), 0);
    std::vector<std::atomic<int>> busy(static_cast<std::size_t>(team.workers()));
    std::mutex mutex;
    bool gave_up = false;
    team.run(ranges, [&](int64_t begin, int64_t, int32_t worker) {
        const bool in_bounds = worker >= 0 && worker < team.workers();
        const bool shared = in_bounds && busy[static_cast<std::size_t>(worker)].exchange(1) != 0;
        {
            std::unique_lock<std::mutex> lock(mutex);
            ++view.visits[static_cast<std::size_t>(begin)];
            view.bad_worker = view.bad_worker || !in_bounds;
            view.shared_worker = view.shared_worker || shared;
            view.only_worker_zero = view.only_worker_zero && worker == 0;
            view.threads.insert(std::this_thread::get_id());
            for (int waited_ms = 0; !gave_up && view.threads.size() < wait_for; ++waited_ms) {
                lock.unlock();
                nap(1);
                lock.lock();
                gave_up = waited_ms >= 10000;
            }
        }
        if (nap_ms > 0)
            nap(nap_ms);
        if (work_ms > 0)
            work(work_ms);
        if (in_bounds)
            busy[static_cast<std::size_t>(worker)].store(0);
    });
    return view;
}

bool every_range_once(const PassView& view) {
    for (const int count : view.visits) {
        if (count != 1)
            return false;
    }
    return true;
}

// Uses up a rest, so that the next pass has the helpers. A pass whose ranges
// sleep or wait rests the team (it took longer than its caller's own work),
// and a loaded test machine can do the same to any pass.
void end_rest(HostWorkers& team) {
    while (team.resting() > 0)
        team.run(unit_ranges(2), [](int64_t, int64_t, int32_t) {});
}

// Every range runs once. A worker index is on one thread at a time and below
// workers(). The calling thread takes ranges, and so does EVERY helper (the
// ranges wait until all the team's threads are inside the pass). The second
// and third pass start when the helpers are asleep again: they must all be
// woken, and the team must have been given back. A helper joins a pass once.
void test_team_runs_every_range_once() {
    HostWorkers team(4);
    check(team.workers() == 4, "a team of four workers starts three helpers");
    if (team.workers() < 2)
        return; // the system let it start no helper: nothing to share with
    const auto workers = static_cast<std::size_t>(team.workers());
    const auto ranges = unit_ranges(64);
    for (int round = 0; round < 3; ++round) {
        nap(50);
        end_rest(team);
        const uint64_t joins_before = team.helper_joins();
        const auto view = view_pass(team, ranges, workers);
        const std::string pass = ", pass " + std::to_string(round + 1);
        check(every_range_once(view), "every range of a shared pass runs once" + pass);
        check(!view.bad_worker, "worker indices are below the team's workers" + pass);
        check(!view.shared_worker, "a worker index is on one thread at a time" + pass);
        check(view.threads.size() == workers, "every helper takes ranges of a shared pass" + pass);
        check(view.threads.count(std::this_thread::get_id()) == 1,
              "the calling thread takes ranges too" + pass);
        check(team.helper_joins() - joins_before == workers - 1,
              "each helper joins a pass once" + pass);
    }
}

// run() returns only when no range is running, also when the helpers are
// slower than the calling thread.
void test_team_waits_for_the_ranges_its_helpers_hold() {
    HostWorkers team(4);
    if (team.workers() < 2)
        return;
    std::atomic<int> running{0};
    std::atomic<int> finished{0};
    const auto caller = std::this_thread::get_id();
    team.run(unit_ranges(32), [&](int64_t, int64_t, int32_t) {
        ++running;
        if (std::this_thread::get_id() != caller)
            nap(20);
        ++finished;
        --running;
    });
    check(running.load() == 0, "no range is running when a pass returns");
    check(finished.load() == 32, "every range has finished when a pass returns");
}

// A pass that finds the team taken runs all its ranges on the thread that
// called it, as worker 0. Here the second pass starts from inside a range of
// the first, while the helpers with no range of the first are asleep: they
// must stay out of it.
void test_taken_team_leaves_a_pass_to_its_caller() {
    HostWorkers team(4);
    const auto inner = unit_ranges(12);
    std::atomic<int> inner_calls{0};
    std::atomic<bool> inner_alone{true};
    team.run(unit_ranges(2), [&](int64_t, int64_t, int32_t) {
        nap(50);
        const auto caller = std::this_thread::get_id();
        team.run(inner, [&](int64_t, int64_t, int32_t worker) {
            ++inner_calls;
            if (worker != 0 || std::this_thread::get_id() != caller)
                inner_alone = false;
            nap(2);
        });
    });
    check(inner_calls.load() == 2 * 12, "a pass inside a pass runs every range");
    check(inner_alone.load(), "a pass that finds the team taken runs on its caller as worker 0");

    // Two threads at once: both passes run every range once.
    std::vector<std::atomic<int>> first(200);
    std::vector<std::atomic<int>> second(200);
    const auto ranges = unit_ranges(200);
    std::thread other([&] {
        team.run(ranges, [&](int64_t begin, int64_t, int32_t) {
            ++second[static_cast<std::size_t>(begin)];
        });
    });
    team.run(ranges,
             [&](int64_t begin, int64_t, int32_t) { ++first[static_cast<std::size_t>(begin)]; });
    other.join();
    bool once = true;
    for (std::size_t item = 0; item < 200; ++item)
        once = once && first[item].load() == 1 && second[item].load() == 1;
    check(once, "two passes at once on one team each run every range once");
}

// A shared pass that took longer than its caller would have needed alone
// (here: every range is 2 ms of work, and a helper also sleeps 40 ms inside
// its range, as one that lost its CPU would) rests the team: the next
// kFirstRest passes run on the calling thread alone, then the helpers are
// back. The rest doubles with every such pass until the shared passes (here:
// passes of real work, faster shared) have saved the time those lost; then it
// starts again at kFirstRest.
void test_team_rests_after_a_pass_that_sharing_made_slower() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    HostWorkers team(4);
    if (team.workers() < 2)
        return;
    const auto caller = std::this_thread::get_id();
    const auto ranges = unit_ranges(8);
    const auto slowed_pass = [&] {
        // The ranges wait for a helper. Each helper then sleeps 40 ms inside
        // its range and works 2 ms; the calling thread works through the
        // others (alone it would have needed about 16 ms for the eight) and
        // waits for the helpers.
        std::mutex mutex;
        std::set<std::thread::id> threads;
        bool gave_up = false;
        team.run(ranges, [&](int64_t, int64_t, int32_t) {
            {
                std::unique_lock<std::mutex> lock(mutex);
                threads.insert(std::this_thread::get_id());
                for (int waited_ms = 0; !gave_up && threads.size() < 2; ++waited_ms) {
                    lock.unlock();
                    nap(1);
                    lock.lock();
                    gave_up = waited_ms >= 10000;
                }
            }
            if (std::this_thread::get_id() != caller)
                nap(40);
            work(2);
        });
        return threads.size() >= 2;
    };
    const auto rest_passes = [&](int32_t count, const std::string& what) {
        bool alone = true;
        bool complete = true;
        bool counted = true;
        for (int32_t pass = 0; pass < count; ++pass) {
            counted = counted && team.resting() == count - pass;
            const auto view = view_pass(team, ranges, 1, 1);
            alone = alone && view.threads.size() == 1 && view.only_worker_zero &&
                    view.threads.count(caller) == 1;
            complete = complete && every_range_once(view);
        }
        check(counted && team.resting() == 0, "every pass of " + what + " uses up one of it");
        check(alone, "a resting team leaves the passes of " + what + " to the calling thread");
        check(complete, "the passes of " + what + " run every range once");
    };
    // Eight ranges, each 2 ms off the CPU and 3 ms of work: by its work about
    // 24 ms alone, and about 10 ms on four workers (longer than the calling
    // thread's own 6 ms of work, shorter than the pass alone).
    const auto working_pass = [&] { return view_pass(team, ranges, 1, 2, 3); };

    check(team.resting() == 0, "a new team is not resting");
    nap(50);
    check(slowed_pass(), "a helper joins the pass it will slow down");
    check(team.resting() == HostWorkers::kFirstRest,
          "a pass that sharing made slower rests the team");
    rest_passes(HostWorkers::kFirstRest, "the first rest");

    // The first pass after the rest is slowed again: the rest doubles.
    nap(50);
    check(slowed_pass(), "a helper joins the next pass it will slow down");
    check(team.resting() == 2 * HostWorkers::kFirstRest,
          "a slowed pass right after a rest doubles the rest");
    rest_passes(2 * HostWorkers::kFirstRest, "the doubled rest");

    // A pass of real work is faster shared: it leaves the team awake. It has
    // saved about 14 ms, less than the two slowed passes lost (about 50 ms),
    // so the next slowed pass doubles the rest again. On a loaded machine
    // that pass may be slowed after all, so the doubling is judged only when
    // it left the team awake.
    nap(50);
    check(working_pass().threads.size() >= 2, "the helpers are back after a rest");
    bool stayed_awake = team.resting() == 0;
    if (stayed_awake) {
        nap(50);
        check(slowed_pass(), "a helper joins the pass after one that was faster shared");
        check(team.resting() == 4 * HostWorkers::kFirstRest,
              "the rest keeps doubling while the shared passes have not saved the lost time");
    }
    // Twelve passes of real work save about 170 ms, more than the slowed
    // passes lost: the next slowed pass rests the team for kFirstRest passes
    // again. Seen at least once in five tries, whatever the machine's load
    // does to the others.
    bool started_again = false;
    for (int attempt = 0; attempt < 5 && !started_again; ++attempt) {
        end_rest(team);
        bool awake = true;
        for (int pass = 0; pass < 12 && awake; ++pass) {
            working_pass();
            awake = team.resting() == 0;
        }
        stayed_awake = stayed_awake || awake;
        if (!awake)
            continue;
        nap(50);
        started_again = slowed_pass() && team.resting() == HostWorkers::kFirstRest;
    }
    check(stayed_awake, "a pass that was faster shared leaves the team awake");
    check(started_again,
          "the rest starts again at its first length once the shared passes have saved the lost "
          "time");

    // The time a range is off the CPU is not work: a pass whose ranges only
    // sleep took longer than its calling thread's work says it would alone.
    end_rest(team);
    view_pass(team, ranges, 1, 5);
    check(team.resting() > 0,
          "a pass is judged by its calling thread's CPU time, not by the clock");
#endif // without a per-thread CPU clock the team never rests
}

void test_team_rethrows_when_no_range_is_running() {
    HostWorkers team(4);
    for (const int64_t failing : {0, 5, 39}) {
        std::atomic<int> running{0};
        std::atomic<int> started{0};
        bool thrown = false;
        bool quiet = false;
        try {
            team.run(unit_ranges(40), [&](int64_t begin, int64_t, int32_t) {
                ++running;
                ++started;
                if (begin == failing) {
                    --running;
                    throw std::runtime_error("range failed");
                }
                --running;
            });
        } catch (const std::runtime_error&) {
            thrown = true;
            quiet = running.load() == 0;
        }
        check(thrown,
              "a failing range's exception reaches the caller, range " + std::to_string(failing));
        check(quiet, "no range is running when the exception arrives");
        check(started.load() <= 40, "no range runs twice around a failure");
    }
    std::atomic<int> calls{0};
    team.run(unit_ranges(40), [&](int64_t, int64_t, int32_t) { ++calls; });
    check(calls.load() == 40, "the team runs the next pass after a failure");

    // Two ranges fail on two threads, one after the other: the first failure
    // is the one that reaches the caller.
    if (team.workers() < 2)
        return;
    end_rest(team);
    nap(50);
    std::atomic<int> inside{0};
    std::atomic<bool> first_thrown{false};
    std::string message;
    try {
        team.run(unit_ranges(2), [&](int64_t begin, int64_t, int32_t) {
            ++inside;
            if (begin == 0) {
                for (int waited_ms = 0; inside.load() < 2 && waited_ms < 10000; ++waited_ms)
                    nap(1);
                first_thrown = true;
                throw std::runtime_error("first");
            }
            for (int waited_ms = 0; !first_thrown.load() && waited_ms < 10000; ++waited_ms)
                nap(1);
            nap(20);
            throw std::runtime_error("second");
        });
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    check(inside.load() == 2, "both failing ranges of the pass ran");
    check(message == "first",
          "the first failure of a pass is the one rethrown (got '" + message + "')");
}

// Teams come and go without a pass, and a team that could start no helper
// (the thread limit is reached) still runs its passes.
void test_team_without_helpers() {
    for (int round = 0; round < 50; ++round)
        HostWorkers idle(4);
    HostWorkers alone(1);
    check(alone.workers() == 1, "a team of one worker has no helper");
    int calls = 0;
    alone.run(unit_ranges(9), [&](int64_t, int64_t, int32_t worker) {
        ++calls;
        check(worker == 0, "a team of one runs as worker 0");
    });
    check(calls == 9, "a team of one runs every range");

#if defined(__linux__)
    rlimit original{};
    if (getrlimit(RLIMIT_NPROC, &original) != 0)
        return;
    rlimit none = original;
    none.rlim_cur = 1;
    if (setrlimit(RLIMIT_NPROC, &none) != 0)
        return;
    {
        HostWorkers starved(8);
        // A privileged process is not held to the limit: then the helpers start.
        check(starved.workers() >= 1 && starved.workers() <= 8,
              "a team keeps the helpers the system let it start");
        std::vector<std::atomic<int>> visits(32);
        starved.run(unit_ranges(32), [&](int64_t begin, int64_t, int32_t) {
            ++visits[static_cast<std::size_t>(begin)];
        });
        bool once = true;
        for (const auto& count : visits)
            once = once && count.load() == 1;
        check(once, "a team that could not start its helpers runs every range once");
    }
    setrlimit(RLIMIT_NPROC, &original);
#endif
}

// A forked child has the team's memory and none of its helper threads. It
// must be able to run passes (alone), to read the team without its mutex
// (no helper has joined a pass of the child's) and to destroy the team,
// whether the team was idle or inside a pass when the child was made.
void test_team_in_a_forked_child() {
#if defined(__linux__)
    const auto child_exit = [](HostWorkers* team, bool run_first) {
        std::fflush(nullptr);
        const pid_t child = fork();
        if (child == 0) {
            int status = 0;
            if (run_first) {
                int calls = 0;
                bool alone = true;
                const auto thread = std::this_thread::get_id();
                team->run(unit_ranges(40), [&](int64_t, int64_t, int32_t worker) {
                    ++calls;
                    alone = alone && worker == 0 && std::this_thread::get_id() == thread;
                });
                status = (calls == 40 && alone) ? 0 : 3;
            }
            if (team->helper_joins() != 0)
                status = 4;
            delete team; // joins nothing: the helpers are the parent's
            _exit(status);
        }
        if (child < 0)
            return -1;
        // A child that hangs is killed after twenty seconds: -2.
        int status = -1;
        for (int waited_ms = 0; waitpid(child, &status, WNOHANG) != child; ++waited_ms) {
            if (waited_ms >= 20000) {
                kill(child, SIGKILL);
                waitpid(child, &status, 0);
                return -2;
            }
            nap(1);
        }
        return WIFEXITED(status) ? WEXITSTATUS(status) : 1000 + WTERMSIG(status);
    };
    for (const bool run_first : {false, true}) {
        // Deliberately on the heap: the child deletes its copy, the parent its own.
        // The helpers have joined a pass before the fork: the parent's count is not zero.
        auto* idle = new HostWorkers(8);
        end_rest(*idle);
        view_pass(*idle, unit_ranges(16), 2);
        nap(20);
        check(child_exit(idle, run_first) == 0, std::string("a forked child ") +
                                                    (run_first ? "runs a pass and " : "") +
                                                    "destroys an idle team");
        delete idle;

        // Forked from inside a range, while helpers hold ranges and the mutex is in use.
        auto* busy = new HostWorkers(8);
        std::atomic<int> result{-2};
        std::atomic<bool> forked{false};
        busy->run(unit_ranges(64), [&](int64_t begin, int64_t, int32_t) {
            if (begin == 20 && !forked.exchange(true))
                result = child_exit(busy, run_first);
            nap(1);
        });
        check(result.load() == 0, std::string("a child forked inside a pass ") +
                                      (run_first ? "runs a pass and " : "") + "destroys the team");
        delete busy;
    }
#endif
}

} // namespace

int main() {
#if defined(__linux__)
    unsetenv(trtmc::canary::kHostWorkersEnv); // the tests read the host's own count
#endif
    test_cpu_max_line();
    test_cgroup_quota_walks_the_ancestors();
    test_worker_count_follows_cpus_and_quota();
    test_environment_sets_the_worker_count();
    test_allowed_cpus_follow_the_affinity_mask();
    test_pass_ranges();
    test_run_pass_without_a_team();
    test_team_runs_every_range_once();
    test_team_waits_for_the_ranges_its_helpers_hold();
    test_taken_team_leaves_a_pass_to_its_caller();
    test_team_rests_after_a_pass_that_sharing_made_slower();
    test_team_rethrows_when_no_range_is_running();
    test_team_without_helpers();
    test_team_in_a_forked_child();
    return failures;
}
