/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "core/runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

namespace {
struct Calls {
    std::atomic<int> active{0}, overlaps{0};
};

class Module final : public trtmc::ITrtModule {
  public:
    Module(bool text, Calls& calls) : text_(text), calls_(calls) {}
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        // Per-thread storage avoids a data race even in the unfixed pipeline;
        // the overlap counter independently detects broken engine transactions.
        static thread_local float output;
        if (text_) {
            if (calls_.active.fetch_add(1) != 0)
                ++calls_.overlaps;
            output = static_cast<const int32_t*>(inputs.at("input_ids").data)[0];
        } else {
            output = *static_cast<const float*>(inputs.at("conditional_embeddings").data) +
                     *static_cast<const float*>(inputs.at("pixel_values").data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!text_)
            --calls_.active;
        return {{text_ ? "conditional_embeddings" : "masks",
                 trtmc::Tensor{&output, {1}, trtmc::DType::kFloat32}}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {1}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {1};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  private:
    bool text_;
    Calls& calls_;
};

class Backend final : public trtmc::IBackend {
  public:
    Calls calls;
    std::unique_ptr<trtmc::ITrtModule> create_module(const void* plan, std::size_t,
                                                     const trtmc::ModuleCreateOptions&) override {
        return std::make_unique<Module>(*static_cast<const char*>(plan) == 't', calls);
    }
    std::unique_ptr<trtmc::ITrtModule>
    create_module_prebound(const void*, std::size_t, const trtmc::ModuleCreateOptions&,
                           const std::vector<trtmc::ModuleExternalBinding>&) override {
        return nullptr;
    }
    trtmc::BackendDualProfileModules
    create_dual_profile_modules(const void*, std::size_t,
                                const trtmc::ModuleCreateOptions&) override {
        return {};
    }
    const char* name() const override { return "trt"; }
};

void write_bundle(const std::filesystem::path& path) {
    const std::string tokenizer =
        R"json({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "},{"type":"Lowercase"}]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+"},"behavior":"Removed","invert":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":"</w>","fuse_unk":false,"byte_fallback":false,"vocab":{"a":0,"c":1,"f":2,"i":3,"©":4,"¿":5,"Ã":6,"Ì":7,"Î":8,"Ï":9,"ĥ":10,"ĩ":11,"a</w>":12,"c</w>":13,"f</w>":14,"i</w>":15,"©</w>":16,"¿</w>":17,"Ã</w>":18,"Ì</w>":19,"Î</w>":20,"Ï</w>":21,"ĥ</w>":22,"ĩ</w>":23},"merges":[]}})json";
    const std::string config =
        R"({"image_size":1,"image_mean":[0,0,0],"image_std":[1,1,1],"max_text_len":16,"projection_dim":1})";
    const std::vector<std::pair<std::string, std::string>> sections{{"runtime.json", config},
                                                                    {"tokenizer.json", tokenizer},
                                                                    {"text.plan", "t"},
                                                                    {"mask.plan", "m"}};
    nlohmann::json header{{"format", 1},
                          {"family", "clipseg"},
                          {"task", "text_prompted_segmentation"},
                          {"backend", "trt"}};
    std::string payload;
    for (const auto& [name, value] : sections) {
        header["sections"][name] = {{"offset", payload.size()}, {"length", value.size()}};
        payload += value;
    }
    const auto json = header.dump();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    for (int shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((static_cast<std::uint64_t>(json.size()) >> shift) & 255));
    file << json << payload;
    if (!file)
        throw std::runtime_error("cannot write concurrency fixture");
}
} // namespace

int main() {
    char directory[] = "/tmp/trtmc-clipseg-concurrency-XXXXXX";
    if (!mkdtemp(directory))
        return 1;
    const auto path = std::filesystem::path(directory) / "fixture.bundle";
    write_bundle(path);
    Backend backend;
    const trtmc::BundleReader reader(path.string());
    std::unique_ptr<trtmc::ITask> family(trtmc_create_family({reader, backend}));
    auto* pipeline = dynamic_cast<trtmc::ITextPromptedSegmentation*>(family.get());
    if (!pipeline)
        return 1;
    std::atomic<bool> start{false};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int index = 0; index < 8; ++index)
        threads.emplace_back([&, index] {
            while (!start.load())
                std::this_thread::yield();
            for (int call = 0; call < 8; ++call) {
                const auto value = static_cast<float>(index + call) / 255.0F;
                const float pixels[] = {value, value, value};
                const bool first = (index + call) % 2 == 0;
                const auto result =
                    pipeline->segment_prompted_text(pixels, 1, 1, first ? "a" : "c");
                const float expected = (first ? 12.0F : 13.0F) + value;
                if (result.masks.size() != 1 || std::abs(result.masks[0] - expected) > 1e-6F)
                    ++failures;
            }
        });
    start = true;
    for (auto& thread : threads)
        thread.join();
    std::filesystem::remove_all(directory);
    if (failures || backend.calls.active || backend.calls.overlaps) {
        std::cerr << "concurrent engine transactions overlapped or mixed request results\n";
        return 1;
    }
}
