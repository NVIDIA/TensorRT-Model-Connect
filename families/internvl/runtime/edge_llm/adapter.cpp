/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/internvl/runtime/edge_llm/adapter.h"

#include "families/internvl/runtime/edge_llm/request.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <edgellm/cpp/common/logger.h>
#include <edgellm/cpp/runtime/llmInferenceRuntime.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <sys/utsname.h>

namespace trtmc::internvl::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("InternVL Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify InternVL Edge runtime host");
    std::ifstream release("/etc/os-release");
    std::string line, os_version;
    while (std::getline(release, line)) {
        if (line.rfind("VERSION_ID=", 0) == 0) {
            os_version = line.substr(11);
            if (os_version.size() >= 2 && os_version.front() == char(34) &&
                os_version.back() == char(34))
                os_version = os_version.substr(1, os_version.size() - 2);
        }
    }
    int device = 0, cuda_version = 0;
    check_cuda(cudaGetDevice(&device));
    check_cuda(cudaRuntimeGetVersion(&cuda_version));
    cudaDeviceProp gpu{};
    check_cuda(cudaGetDeviceProperties(&gpu, device));
    const int trt_version = getInferLibVersion();
    const std::string trt =
        std::to_string(trt_version / 10000) + "." + std::to_string((trt_version % 10000) / 100) +
        "." + std::to_string(trt_version % 100) + "." + std::to_string(getInferLibBuildVersion());
    const std::string cuda =
        std::to_string(cuda_version / 1000) + "." + std::to_string((cuda_version % 1000) / 10);
    if (target.at("os") != "linux" || target.at("os_version") != os_version ||
        target.at("arch") != host.machine || target.at("sm") != gpu.major * 10 + gpu.minor ||
        target.at("cuda_version") != cuda || target.at("tensorrt_version") != trt)
        throw std::runtime_error(
            "InternVL Edge bundle requires its build GPU and CUDA/TensorRT stack");
}

/// Own extracted engine/checkpoint files until after the Edge runtime is destroyed.
class Artifacts {
  public:
    explicit Artifacts(const BundleReader& bundle, const nlohmann::json& marker) {
        std::set<std::string> names;
        for (const auto& entry : marker.at("artifacts")) {
            const auto name = entry.get<std::string>();
            if (!safe_artifact_path(name) || !names.insert(name).second ||
                !bundle.find_section(name))
                throw std::runtime_error("Invalid InternVL Edge artifact: " + name);
        }
        for (const auto* required :
             {"edge_llm/engine/visual/visual.engine", "edge_llm/engine/visual/config.json",
              "edge_llm/engine/llm.engine", "edge_llm/engine/config.json",
              "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
              "edge_llm/engine/processed_chat_template.json", "edge_llm/checkpoint/config.json"})
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(std::string("Required InternVL Edge artifact missing: ") +
                                         required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-internvl-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create InternVL Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract InternVL Edge artifact: " + name);
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Artifacts() { cleanup(); }
    Artifacts(const Artifacts&) = delete;
    Artifacts& operator=(const Artifacts&) = delete;
    std::string engine() const { return (root_ / "edge_llm/engine").string(); }
    std::string checkpoint() const { return (root_ / "edge_llm/checkpoint").string(); }

  private:
    void cleanup() noexcept {
        std::error_code ignored;
        fs::remove_all(root_, ignored);
    }
    fs::path root_;
};

/// Close the plugin handle after runtime destruction; registrations remain mapped.
struct CloseLibrary {
    void operator()(void* handle) const noexcept {
        if (handle)
            dlclose(handle);
    }
};

/// Initialize the CMake-installed adjacent plugin without process-global environment mutation.
std::unique_ptr<void, CloseLibrary> load_plugin() {
    Dl_info location{};
    if (!dladdr(reinterpret_cast<void*>(&create), &location) || !location.dli_fname)
        throw std::runtime_error("Cannot locate InternVL family library");
    const auto path =
        fs::absolute(location.dli_fname).parent_path() / "libNvInfer_edgellm_plugin.so";
    std::unique_ptr<void, CloseLibrary> plugin(
        dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE));
    if (!plugin)
        throw std::runtime_error("Cannot load CMake-installed Edge plugin: " +
                                 std::string(dlerror()));
    using Initialize = bool (*)(void*, const char*);
    auto initialize = reinterpret_cast<Initialize>(dlsym(plugin.get(), "initEdgellmPlugins"));
    if (!initialize || !initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
        throw std::runtime_error("Cannot initialize InternVL Edge plugin");
    return plugin;
}

/// Stream ownership is independent of construction success and outlives the Edge instance.
class Stream {
  public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&value_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return value_; }

  private:
    cudaStream_t value_{nullptr};
};

/// Validate the admitted raw-tokenizer policy from immutable bundle bytes.
void validate_bundle_tokenizer(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm/engine/tokenizer.json");
    const auto config = bundle.read_section("edge_llm/engine/tokenizer_config.json");
    validate_raw_tokenizer(nlohmann::json::parse(bytes.begin(), bytes.end()),
                           nlohmann::json::parse(config.begin(), config.end()));
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask final : public ITextGeneration, public IVisionLanguageGeneration {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(artifacts_.engine(), artifacts_.engine(),
                   std::unordered_map<std::string, std::string>{}, stream_.get(),
                   trt_edgellm::rt::ContextCacheConfig{}, artifacts_.checkpoint()),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()) {
        validate_bundle_tokenizer(bundle);
    }

    const char* task() const noexcept override { return IVisionLanguageGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override {
        // Some profiles allow inputs up to the entire KV capacity. Reserve one
        // output token there; a prompt occupying every slot still fails validation.
        return std::max(1, capacity_ - input_limit_);
    }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// No-image calls retain native raw semantics, including ignored chat/thinking flags.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        return generate(prompt, nullptr, 0, 0, config);
    }

    /// Invoke the full Edge visual and LLM runtime once; never retry inference natively.
    TextResult generate(const std::string& prompt, const float* pixels, std::int32_t height,
                        std::int32_t width, const TextGenerationConfig& config) override {
        auto bytes = image_bytes(pixels, height, width);
        auto request = make_request(prompt, config, !bytes.empty());
        if (!bytes.empty()) {
            trt_edgellm::rt::Tensor tensor({1, height, width, 3}, trt_edgellm::rt::DeviceType::kCPU,
                                           nvinfer1::DataType::kUINT8);
            std::memcpy(tensor.rawPointer(), bytes.data(), bytes.size());
            request.requests.front().imageBuffers.emplace_back(std::move(tensor));
        }
        const auto requested_budget = request.maxGenerateLength;
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            trt_edgellm::rt::LLMGenerationResponse response{};
            if (!runtime_.handleRequest(request, response, stream_.get()))
                throw std::runtime_error("InternVL Edge generation failed");
            // Actual media expansion plus ORIGINAL budget, even for early EOS.
            validate_response(response, input_limit_, capacity_, requested_budget);
            check_cuda(cudaStreamSynchronize(stream_.get()));
            return {std::move(response.outputTexts.front()), std::move(response.outputIds.front())};
        } catch (...) {
            // Image buffers must outlive outstanding DMA, including a failed request.
            cudaStreamSynchronize(stream_.get());
            throw;
        }
    }

  private:
    // Reverse destruction order keeps weights, plugin and stream alive throughout Edge teardown.
    Artifacts artifacts_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    trt_edgellm::rt::LLMInferenceRuntime runtime_;
    int capacity_;
    int input_limit_;
    std::mutex mutex_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    if (marker.at("version") != 1 || marker.at("edge_revision") != kRevision ||
        marker.at("max_sequence_length").get<int>() <= 1 ||
        marker.at("max_input_length").get<int>() <= 0 ||
        marker.at("max_input_length").get<int>() > marker.at("max_sequence_length").get<int>() ||
        marker.at("max_batch_size") != 1 || marker.at("precision") != "fp16" ||
        marker.at("weight_format") != "fp16" ||
        marker.at("component_weight_formats").at("llm") != marker.at("weight_format") ||
        marker.at("component_weight_formats").at("visual") != "fp16" ||
        marker.at("visual_image_tokens") != 256 || !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid InternVL Edge bundle contract");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::internvl::edge_llm
