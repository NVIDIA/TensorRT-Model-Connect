/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/runtime/bundle/bundle_format.h"
#include "families/minimax_h3/runtime/plan_loader.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <stdexcept>
#include <unistd.h>

namespace allocation_watch {
// Observe only these two exact vector payload sizes, not allocator RSS or a
// model-memory estimate. No dynamic allocation is used by the observer.
constexpr std::size_t first = 2 * 1024 * 1024 + 17;
constexpr std::size_t second = first + 31;
bool enabled = false;
std::size_t live = 0, peak = 0;
std::array<std::pair<void*, std::size_t>, 8> allocations{};

void* allocate(std::size_t size) {
    void* pointer = std::malloc(size ? size : 1);
    if (!pointer)
        throw std::bad_alloc();
    if (enabled && (size == first || size == second)) {
        for (auto& entry : allocations) {
            if (!entry.first) {
                entry = {pointer, size};
                live += size;
                peak = std::max(peak, live);
                return pointer;
            }
        }
        std::free(pointer);
        throw std::bad_alloc();
    }
    return pointer;
}

void release(void* pointer) noexcept {
    for (auto& entry : allocations) {
        if (entry.first && entry.first == pointer) {
            live -= entry.second;
            entry = {};
            break;
        }
    }
    std::free(pointer);
}
} // namespace allocation_watch

void* operator new(std::size_t size) {
    return allocation_watch::allocate(size);
}
void* operator new[](std::size_t size) {
    return allocation_watch::allocate(size);
}
void operator delete(void* pointer) noexcept {
    allocation_watch::release(pointer);
}
void operator delete[](void* pointer) noexcept {
    allocation_watch::release(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    allocation_watch::release(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    allocation_watch::release(pointer);
}

namespace {
using namespace trtmc;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Fn>
void must_throw(Fn&& function) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, "expected explicit failure");
}

void write_bundle(const std::filesystem::path& path, char first = 'A', bool other_family = false) {
    std::string header =
        R"({"format":1,"family":"minimax_h3","task":"image_generation","backend":"trt","sections":{"a.plan":{"offset":0,"length":)" +
        std::to_string(allocation_watch::first) + R"(},"b.plan":{"offset":)" +
        std::to_string(allocation_watch::first) + R"(,"length":)" +
        std::to_string(allocation_watch::second) + R"(},"empty.plan":{"offset":0,"length":0}}})";
    if (other_family)
        header.replace(header.find("minimax_h3"), 10, "minimax_x3");
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(kBundleMagic), sizeof(kBundleMagic));
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((length >> shift) & 255));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::string a(allocation_watch::first, first), b(allocation_watch::second, 'B');
    file.write(a.data(), static_cast<std::streamsize>(a.size()));
    file.write(b.data(), static_cast<std::streamsize>(b.size()));
    require(file.good(), "write fixture bundle");
}

// Lifecycle doubles only: actual TRT deserialization is covered separately by
// native engine probes. This test observes file reads and temporary ownership.
class Module final : public ITrtModule {
  public:
    explicit Module(bool valid) : valid_(valid) {}
    TensorMap forward(const TensorMap&) override { return {}; }
    DeviceTensorMap forward_device(const DeviceTensorMap&) override { return {}; }
    void forward_device_async(const DeviceTensorMap&) override {}
    void forward_async(const TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<TensorInfo> input_info() const override { return {}; }
    std::vector<TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return false; }
    DType tensor_dtype(const std::string&) const override { return DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 0; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return valid_; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    bool valid_;
};

class Backend final : public IBackend {
  public:
    int calls{0};
    int failure{0};
    cudaStream_t expected_stream{nullptr};
    std::function<void()> during_create;

    std::unique_ptr<ITrtModule> create_module(const void* data, std::size_t size,
                                              const ModuleCreateOptions& options) override {
        ++calls;
        require(options.stream == expected_stream, "preserve caller stream");
        require(allocation_watch::live == size, "exactly one serialized plan is live");
        const char expected = size == allocation_watch::first ? 'A' : 'B';
        const auto* bytes = static_cast<const char*>(data);
        require(std::all_of(bytes, bytes + size, [=](char value) { return value == expected; }),
                "serialized bytes unchanged");
        if (during_create)
            during_create();
        if (failure == 1)
            throw std::runtime_error("fixture backend failure");
        if (failure == 2)
            return nullptr;
        return std::make_unique<Module>(failure != 3);
    }
    std::unique_ptr<ITrtModule>
    create_module_prebound(const void*, std::size_t, const ModuleCreateOptions&,
                           const std::vector<ModuleExternalBinding>&) override {
        return nullptr;
    }
    BackendDualProfileModules create_dual_profile_modules(const void*, std::size_t,
                                                          const ModuleCreateOptions&) override {
        return {};
    }
    const char* name() const override { return "fixture"; }
};

void test_lazy_lifetime(const std::filesystem::path& path) {
    write_bundle(path);
    Backend backend;
    MiniMaxH3ModuleLoader loader;
    allocation_watch::enabled = true;
    {
        const BundleReader reader(path.string());
        auto source = std::make_shared<MiniMaxH3PlanSource>(reader);
        loader = make_minimax_h3_plan_loader(backend, source, {"a.plan", "b.plan"});
        require(allocation_watch::live == 0, "constructing loader reads no plan payloads");
        must_throw([&] { make_minimax_h3_plan_loader(backend, source, {"empty.plan"}); });
        must_throw([&] { make_minimax_h3_plan_loader(backend, source, {"missing.plan"}); });
        must_throw([&] { make_minimax_h3_plan_loader(backend, source, {"a.plan", "a.plan"}); });
        must_throw([&] { make_minimax_h3_plan_loader(backend, source, {}); });
        must_throw([&] { make_minimax_h3_plan_loader(backend, nullptr, {"a.plan"}); });
    }
    auto first = loader("a.plan", nullptr);
    require(first && first->ok() && allocation_watch::live == 0,
            "plan released while module and loader remain alive");
    int stream_marker = 0;
    backend.expected_stream = reinterpret_cast<cudaStream_t>(&stream_marker);
    auto second = loader("b.plan", backend.expected_stream);
    require(second && allocation_watch::live == 0, "second plan released independently");
    backend.expected_stream = nullptr;
    must_throw([&] { loader("undeclared.plan", nullptr); });
    require(backend.calls == 2, "undeclared plan never reaches backend");
    for (int failure = 1; failure <= 3; ++failure) {
        backend.failure = failure;
        must_throw([&] { loader("a.plan", nullptr); });
        require(allocation_watch::live == 0, "all backend failure paths release plan");
    }
    backend.failure = 0;
    require(loader("a.plan", nullptr)->ok(), "retry reads plan afresh");
    require(allocation_watch::peak == allocation_watch::second && allocation_watch::live == 0,
            "peak serialized payload equals largest single plan, not sum");
    allocation_watch::enabled = false;
}

void test_mutation(const std::filesystem::path& path) {
    Backend backend;
    for (int mutation = 0; mutation < 4; ++mutation) {
        write_bundle(path);
        auto source = std::make_shared<MiniMaxH3PlanSource>(BundleReader(path.string()));
        auto loader = make_minimax_h3_plan_loader(backend, source, {"a.plan"});
        if (mutation == 0) {
            const auto replacement = path.string() + ".replacement";
            write_bundle(replacement, 'C');
            std::filesystem::last_write_time(replacement, std::filesystem::last_write_time(path));
            std::filesystem::rename(replacement, path);
        } else if (mutation == 1) {
            std::filesystem::resize_file(path, 16);
        } else if (mutation == 2) {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(-1, std::ios::end);
            file.put('C');
            file.close();
            std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() +
                                                       std::chrono::seconds(2));
        } else {
            std::filesystem::remove(path);
        }
        const auto before = backend.calls;
        must_throw([&] { loader("a.plan", nullptr); });
        require(backend.calls == before, "changed bundle rejected before deserialization");
    }
    // Mutation during synchronous deserialization is rejected as well.
    write_bundle(path);
    auto source = std::make_shared<MiniMaxH3PlanSource>(BundleReader(path.string()));
    auto loader = make_minimax_h3_plan_loader(backend, source, {"a.plan"});
    backend.during_create = [&] { std::filesystem::resize_file(path, 16); };
    allocation_watch::enabled = true;
    must_throw([&] { loader("a.plan", nullptr); });
    require(allocation_watch::live == 0, "late mutation releases temporary plan");
    allocation_watch::enabled = false;

    write_bundle(path);
    const BundleReader stale(path.string());
    write_bundle(path, 'A', true);
    must_throw([&] { MiniMaxH3PlanSource changed(stale); });
}

} // namespace

int main() {
    char pattern[] = "/tmp/trtmc_h3_plan_loader_XXXXXX";
    const auto* directory = mkdtemp(pattern);
    if (!directory)
        return 1;
    try {
        const auto path = std::filesystem::path(directory) / "fixture.bundle";
        test_lazy_lifetime(path);
        test_mutation(path);
        std::filesystem::remove_all(directory);
        std::cout << "MiniMax-H3 lazy plan lifecycle and file identity checks passed\n";
        std::cout << "fixture_serialized_payload_peak_bytes=" << allocation_watch::peak
                  << " retained_bytes=" << allocation_watch::live << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
