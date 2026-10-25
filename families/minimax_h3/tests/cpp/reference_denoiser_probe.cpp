/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Real backend/engines/RNG and the production Ref2VA loop, with a declared
// reduced synthetic topology. This does not load Qwen/VAE or trained weights.
#include "families/minimax_h3/runtime/reference_denoiser.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::vector<char> read_bytes(const std::string& name) {
    const auto size = std::filesystem::file_size(name);
    if (size == 0 || size > 16 * 1024 * 1024)
        throw std::invalid_argument("native reference probe input size out of range");
    std::vector<char> bytes(static_cast<std::size_t>(size));
    std::ifstream input(name, std::ios::binary);
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!input)
        throw std::runtime_error("cannot read native reference probe input");
    return bytes;
}

std::vector<float> read_floats(const std::string& name, std::size_t count) {
    const auto bytes = read_bytes(name);
    if (bytes.size() != count * sizeof(float))
        throw std::invalid_argument("native reference probe input shape mismatch");
    std::vector<float> values(count);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

void write_floats(const std::string& name, const std::vector<float>& values) {
    if (std::filesystem::exists(name))
        throw std::invalid_argument("native reference probe will not overwrite results");
    std::ofstream output(name, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output)
        throw std::runtime_error("cannot write native reference probe output");
}

struct Trace {
    std::vector<float> video, audio;
};

// Records already-owned CPU inputs without changing the engine graph,
// scheduler or RNG. Every execution/introspection method reaches the real
// backend module; this is a delegating observer, not an engine double.
class RecordingModule final : public trtmc::ITrtModule {
  public:
    RecordingModule(std::unique_ptr<trtmc::ITrtModule> inner, Trace& trace)
        : inner_(std::move(inner)), trace_(trace) {}
    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        for (const auto& item :
             {std::pair<const char*, std::vector<float>*>{"video_hidden_states", &trace_.video},
              {"audio_hidden_states", &trace_.audio}}) {
            const auto value = inputs.at(item.first);
            if (!value.data || value.dtype != trtmc::DType::kFloat32)
                throw std::runtime_error("trace input ABI mismatch");
            const auto* data = static_cast<const float*>(value.data);
            item.second->insert(item.second->end(), data, data + value.numel());
        }
        return inner_->forward(inputs);
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& x) override {
        return inner_->forward_device(x);
    }
    void forward_device_async(const trtmc::DeviceTensorMap& x) override {
        inner_->forward_device_async(x);
    }
    void forward_async(const trtmc::TensorMap& x) override { inner_->forward_async(x); }
    void sync() override { inner_->sync(); }
    cudaStream_t stream() const override { return inner_->stream(); }
    void enable_cuda_graph() override { inner_->enable_cuda_graph(); }
    bool cuda_graph_active() const override { return inner_->cuda_graph_active(); }
    bool cuda_graph_captured() const override { return inner_->cuda_graph_captured(); }
    int32_t profile_idx() const override { return inner_->profile_idx(); }
    std::vector<trtmc::TensorInfo> input_info() const override { return inner_->input_info(); }
    std::vector<trtmc::TensorInfo> output_info() const override { return inner_->output_info(); }
    bool has_input(const std::string& x) const override { return inner_->has_input(x); }
    bool has_output(const std::string& x) const override { return inner_->has_output(x); }
    trtmc::DType tensor_dtype(const std::string& x) const override {
        return inner_->tensor_dtype(x);
    }
    std::vector<int64_t> tensor_shape(const std::string& x) const override {
        return inner_->tensor_shape(x);
    }
    std::vector<int64_t> input_profile_shape(const std::string& x, int32_t p,
                                             trtmc::ProfileShapeSelector s) const override {
        return inner_->input_profile_shape(x, p, s);
    }
    int32_t optimization_profile_count() const override {
        return inner_->optimization_profile_count();
    }
    void* device_ptr(const std::string& x) const override { return inner_->device_ptr(x); }
    void bind_external(const std::string& x, void* p) override { inner_->bind_external(x, p); }
    void bind_external(const std::string& x, void* p, const std::vector<int64_t>& shape) override {
        inner_->bind_external(x, p, shape);
    }
    int32_t input_rank(const std::string& x) const override { return inner_->input_rank(x); }
    bool input_is_dynamic(const std::string& x) const override {
        return inner_->input_is_dynamic(x);
    }
    void reset_execution_context() override { inner_->reset_execution_context(); }
    void set_timing_label(std::string x) override { inner_->set_timing_label(std::move(x)); }
    bool ok() const override { return inner_->ok(); }
    void keep_alive(std::shared_ptr<void> x) override { inner_->keep_alive(std::move(x)); }

  private:
    std::unique_ptr<trtmc::ITrtModule> inner_;
    Trace& trace_;
};

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7 && argc != 8)
            throw std::invalid_argument(
                "usage: probe RUNTIME_ROOT PLAN_DIR INPUT_DIR "
                "OUTPUT_PREFIX staged|resident SEED [guide|windows|window-prompts]");
        const std::string root = argv[1], plans = argv[2], inputs = argv[3], output = argv[4],
                          mode = argv[5];
        if (mode != "staged" && mode != "resident")
            throw std::invalid_argument("invalid residency");
        const bool guided = argc == 8 && std::string(argv[7]) == "guide";
        const bool window_prompts = argc == 8 && std::string(argv[7]) == "window-prompts";
        const bool windows = argc == 8 && (std::string(argv[7]) == "windows" || window_prompts);
        if (argc == 8 && !guided && !windows)
            throw std::invalid_argument("invalid continuation probe mode");
        std::size_t parsed = 0;
        const uint64_t seed = std::stoull(argv[6], &parsed);
        if (parsed != std::string(argv[6]).size())
            throw std::invalid_argument("invalid seed");
        const auto close_library = [](void* value) { (void)dlclose(value); };
        std::unique_ptr<void, decltype(close_library)> library(
            dlopen((root + "/libtrtmc_backend_trt.so").c_str(), RTLD_NOW | RTLD_LOCAL),
            close_library);
        if (!library)
            throw std::runtime_error(dlerror());
        using Create = trtmc::IBackend* (*)();
        using Destroy = void (*)(trtmc::IBackend*);
        const auto create = reinterpret_cast<Create>(dlsym(library.get(), "trtmc_create_backend"));
        const auto destroy =
            reinterpret_cast<Destroy>(dlsym(library.get(), "trtmc_destroy_backend"));
        if (!create || !destroy)
            throw std::runtime_error("native backend exports missing");
        std::unique_ptr<trtmc::IBackend, Destroy> backend(create(), destroy);
        if (!backend || std::string(backend->name()) != "trt")
            throw std::runtime_error("backend initialization failed");
        using namespace trtmc::minimax_h3;
        std::unordered_map<std::string, int> loads;
        Trace trace;
        ReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t stream) {
            if (name != "adaln.plan" && name != "denoiser.plan")
                throw std::invalid_argument("unexpected reference plan");
            const auto bytes = read_bytes(plans + "/" + name);
            trtmc::ModuleCreateOptions options{};
            options.stream = stream;
            auto module = backend->create_module(bytes.data(), bytes.size(), options);
            if (!module || !module->ok())
                throw std::runtime_error("reference plan initialization failed");
            ++loads[name];
            if (name == "denoiser.plan")
                module = std::make_unique<RecordingModule>(std::move(module), trace);
            return module; // Serialized bytes die here; the real backend owns its engine.
        };
        ReferenceImageConditioning text;
        text.presentation.h3_token_tags = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
        text.presentation.input_ids.assign(15, 42);
        text.presentation.visual_rows = {2, 3, 4, 5, 6, 7, 10, 11};
        text.embeddings = read_floats(inputs + "/text.f32.bin", 15 * 8);
        std::optional<ReferenceImageConditioning> continuation_text;
        int32_t max_text_rows = 15;
        if (window_prompts) {
            const auto bytes = read_bytes(inputs + "/continuation-text-rows.txt");
            const std::string raw(bytes.begin(), bytes.end());
            std::size_t count = 0;
            const auto rows = std::stoi(raw, &count);
            if (count != raw.size() || rows < 14 || rows > 19)
                throw std::invalid_argument("invalid continuation text row fixture");
            continuation_text = text;
            continuation_text->presentation.h3_token_tags.resize(rows, 1);
            continuation_text->presentation.input_ids.assign(rows, 43);
            continuation_text->embeddings =
                read_floats(inputs + "/continuation-text.f32.bin", rows * 8);
            max_text_rows = std::max(15, rows);
        }
        const std::vector<EncodedVisualReference> images = {
            {4, 6, read_floats(inputs + "/reference-0.f32.bin", 24 * 4 * 6)},
            {2, 4, read_floats(inputs + "/reference-1.f32.bin", 24 * 2 * 4)}};
        const ReferenceDenoiserConfig config =
            windows ? ReferenceDenoiserConfig{{37, 4, 6, 207}, 1, 8, 8, 760 + max_text_rows}
                    : ReferenceDenoiserConfig{{2, 4, 6, 3}, 1, 8, 8, guided ? 49 : 41};
        ReferenceLatentGuide guide{{1, 4, 6, 1}, {}, {}};
        if (guided) {
            guide.video_rows = read_floats(inputs + "/guide.video.f32.bin", 6 * 96);
            guide.audio_rows = read_floats(inputs + "/guide.audio.f32.bin", 2 * 32);
        }
        const bool resident = mode == "resident";
        ImageReferenceDenoiser owner(loader, config, nullptr,
                                     resident ? ReferenceDenoiserResidency::Resident
                                              : ReferenceDenoiserResidency::Staged);
        const auto execute = [&] {
            if (windows)
                return owner.run_windows(text, images, seed, 226,
                                         continuation_text ? &*continuation_text : nullptr);
            return guided ? owner.run_continuation(text, images, guide, seed, 17.5, 21)
                          : owner.run(text, images, seed);
        };
        const auto cold = execute();
        write_floats(output + ".cold.video.bin", cold.video_rows);
        write_floats(output + ".cold.audio.bin", cold.audio_rows);
        write_floats(output + ".cold.trace.video.bin", trace.video);
        write_floats(output + ".cold.trace.audio.bin", trace.audio);
        const std::size_t video_rows = guided ? 26 : 20, audio_rows = guided ? 8 : 6;
        const std::size_t video_values = windows ? 49 * (230 + 272) * 96 : 49 * video_rows * 96;
        const std::size_t audio_values = windows ? 49 * (414 + 488) * 32 : 49 * audio_rows * 32;
        if (trace.video.size() != video_values || trace.audio.size() != audio_values)
            throw std::runtime_error("native reference did not execute 49 recorded forwards");
        trace.video.clear();
        trace.audio.clear();
        const auto warm = execute();
        write_floats(output + ".warm.video.bin", warm.video_rows);
        write_floats(output + ".warm.audio.bin", warm.audio_rows);
        write_floats(output + ".warm.trace.video.bin", trace.video);
        write_floats(output + ".warm.trace.audio.bin", trace.audio);
        if (trace.video.size() != video_values || trace.audio.size() != audio_values)
            throw std::runtime_error("native reference warm forward count mismatch");
        const int32_t forwards = windows ? 98 : 49;
        if (cold.forwards != forwards || warm.forwards != forwards ||
            cold.video_rows.size() != (windows ? 402 : 12) * 96 ||
            cold.audio_rows.size() != (windows ? 754 : 6) * 32 ||
            cold.video_rows != warm.video_rows || cold.audio_rows != warm.audio_rows)
            throw std::runtime_error("native reference cold/warm or forward-count contract failed");
        const int32_t expected_loads = windows ? 4 : (resident ? 1 : 2);
        if (loads["adaln.plan"] != expected_loads || loads["denoiser.plan"] != expected_loads)
            throw std::runtime_error(
                "native reference residency did not reuse/release actual modules");
        std::cout << "{\"cold_forwards\":" << cold.forwards
                  << ",\"warm_forwards\":" << warm.forwards
                  << ",\"adaln_loads\":" << loads["adaln.plan"]
                  << ",\"denoiser_loads\":" << loads["denoiser.plan"] << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
