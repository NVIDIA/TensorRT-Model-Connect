/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/task_pool.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class NamedTask final : public trtmc::ITask {
  public:
    explicit NamedTask(std::string name) : name_(std::move(name)) {}
    const char* task() const noexcept override { return name_.c_str(); }

  private:
    std::string name_;
};

std::vector<std::unique_ptr<trtmc::ITask>> named_tasks(std::size_t count) {
    std::vector<std::unique_ptr<trtmc::ITask>> tasks;
    tasks.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        tasks.push_back(std::make_unique<NamedTask>("task-" + std::to_string(index)));
    return tasks;
}

bool construction_rejects(std::vector<std::unique_ptr<trtmc::ITask>> tasks) {
    try {
        trtmc::TaskPool pool(std::move(tasks));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_capacity_and_leases() {
    trtmc::TaskPool pool(named_tasks(2));
    check(pool.capacity() == 2 && pool.available() == 2, "initial capacity");

    auto first = pool.acquire();
    auto second = pool.acquire();
    check(pool.available() == 0, "leases are exclusive");
    check(first.get() != second.get(), "leases own distinct tasks");
    check(!pool.try_acquire().has_value(), "try_acquire reports exhaustion");

    auto moved = std::move(first);
    check(!first && moved && pool.available() == 0, "lease move preserves ownership");
    moved = {};
    check(pool.available() == 1, "destroying a lease releases one task");
    second = {};
    check(pool.available() == 2, "all tasks return to the pool");
}

void test_blocking_acquire() {
    trtmc::TaskPool pool(named_tasks(1));
    auto occupied = pool.acquire();
    std::mutex mutex;
    std::condition_variable changed;
    bool started = false;
    bool acquired = false;

    std::thread waiter([&] {
        {
            const std::lock_guard<std::mutex> lock(mutex);
            started = true;
        }
        changed.notify_one();
        auto lease = pool.acquire();
        {
            const std::lock_guard<std::mutex> lock(mutex);
            acquired = true;
        }
        changed.notify_one();
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return started; });
        check(!changed.wait_for(lock, std::chrono::milliseconds(20), [&] { return acquired; }),
              "acquire blocks while the only task is leased");
    }
    occupied = {};
    {
        std::unique_lock<std::mutex> lock(mutex);
        check(changed.wait_for(lock, std::chrono::seconds(1), [&] { return acquired; }),
              "released task wakes one waiter");
    }
    waiter.join();
    check(pool.available() == 1, "waiter returns the task after use");
}

void test_lease_outlives_pool() {
    trtmc::TaskPool::Lease lease;
    {
        trtmc::TaskPool pool(named_tasks(1));
        lease = pool.acquire();
    }
    check(lease && std::string(lease->task()) == "task-0", "lease keeps pool state alive");
    lease = {};
}

void write_bundle(const std::filesystem::path& path) {
    const std::string header =
        R"({"format":1,"family":"fake","task":"time_series_forecast","backend":"fake","sections":{"runtime.json":{"offset":0,"length":2},"engine.plan":{"offset":2,"length":4}}})";
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((length >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("{}PLAN", 6);
}

void test_loader_pool(const std::filesystem::path& runtime_root) {
    const auto bundle = runtime_root / "task-pool.bundle";
    write_bundle(bundle);
    bool zero_rejected = false;
    try {
        (void)trtmc::load_task_pool(bundle.string(), runtime_root.string(), 0);
    } catch (const std::invalid_argument&) {
        zero_rejected = true;
    }
    check(zero_rejected, "loader rejects zero capacity");
    {
        auto pool = trtmc::load_task_pool(bundle.string(), runtime_root.string(), 2, 4096);
        check(pool.capacity() == 2 && pool.available() == 2, "loader creates requested capacity");
        auto first = pool.acquire();
        auto second = pool.acquire();
        auto* first_forecast = dynamic_cast<trtmc::ITimeSeriesForecast*>(first.get());
        auto* second_forecast = dynamic_cast<trtmc::ITimeSeriesForecast*>(second.get());
        check(first_forecast != nullptr && second_forecast != nullptr,
              "loader returns the declared Task interface");
        check(first.get() != second.get(), "loader creates independent Task instances");
        if (first_forecast != nullptr && second_forecast != nullptr) {
            const float values[] = {1.0F, 2.0F};
            const auto first_result = first_forecast->forecast({values, {}});
            const auto second_result = second_forecast->forecast({values, {}});
            check(first_result.shape == std::vector<std::int64_t>({4096, 2}) &&
                      second_result.shape == first_result.shape,
                  "loader forwards direct runtime options to every task");
        }
    }
    std::filesystem::remove(bundle);
}

} // namespace

int main(int argc, char** argv) {
    static_assert(!std::is_copy_constructible_v<trtmc::TaskPool>);
    static_assert(std::is_move_constructible_v<trtmc::TaskPool>);
    static_assert(!std::is_copy_constructible_v<trtmc::TaskPool::Lease>);
    static_assert(std::is_move_constructible_v<trtmc::TaskPool::Lease>);

    check(construction_rejects({}), "empty pool is rejected");
    auto null_tasks = named_tasks(1);
    null_tasks.push_back(nullptr);
    check(construction_rejects(std::move(null_tasks)), "null task is rejected");
    test_capacity_and_leases();
    test_blocking_acquire();
    test_lease_outlives_pool();
    if (argc != 2) {
        std::cerr << "usage: test_task_pool RUNTIME_ROOT\n";
        return 2;
    }
    test_loader_pool(argv[1]);
    return failures;
}
