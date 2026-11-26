/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/mistral/runtime/edge_llm/adapter.h"

#include "families/mistral/runtime/edge_llm/request.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <sys/utsname.h>
#include <tensorrt_edgellm/common/trtInterop.h>
#include <tensorrt_edgellm/runtime/llmInferenceRuntime.h>
#include <tensorrt_edgellm/runtime/streaming.h>
#include <utility>
#include <vector>

namespace trtmc::mistral::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Mistral Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Mistral Edge runtime host");
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
            "Mistral Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Mistral Edge artifact: " + name);
        }
        const std::vector<std::string> required_files{
            "edge_llm/engine/llm.engine",
            "edge_llm/engine/config.json",
            "edge_llm/engine/tokenizer.json",
            "edge_llm/engine/tokenizer_config.json",
            "edge_llm/engine/chat_template.jinja",
            "edge_llm/checkpoint/config.json",
            "edge_llm/checkpoint/model.safetensors.index.json"};
        for (const auto& required : required_files)
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(std::string("Required Mistral Edge artifact missing: ") +
                                         required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-mistral-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Mistral Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Mistral Edge artifact: " + name);
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
        throw std::runtime_error("Cannot locate Mistral family library");
    const auto path =
        fs::absolute(location.dli_fname).parent_path() / "libNvInfer_edgellm_plugin.so";
    std::unique_ptr<void, CloseLibrary> plugin(
        dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE));
    if (!plugin)
        throw std::runtime_error("Cannot load CMake-installed Edge plugin: " +
                                 std::string(dlerror()));
    using Initialize = bool (*)(void*, const char*);
    auto initialize = reinterpret_cast<Initialize>(dlsym(plugin.get(), "initEdgellmPlugins"));
    auto* logger = static_cast<nvinfer1::ILogger*>(&trt_edgellm::TrtLogger::instance());
    if (!initialize || !initialize(logger, ""))
        throw std::runtime_error("Cannot initialize Mistral Edge plugin");
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

/// Read tokenizer policy using the same bundle byte parser as other family metadata.
std::string bundle_raw_prefix(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm/engine/tokenizer_config.json");
    const auto tokenizer = bundle.read_section("edge_llm/engine/tokenizer.json");
    return raw_prompt_prefix(nlohmann::json::parse(bytes.begin(), bytes.end()),
                             nlohmann::json::parse(tokenizer.begin(), tokenizer.end()));
}

/// Keep the released runtime and its checkpoint-backed weight bindings alive for the task.
std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> make_runtime(const Artifacts& artifacts,
                                                                   cudaStream_t stream) {
    using Runtime = trt_edgellm::rt::LLMInferenceRuntime;
    int device;
    check_cuda(cudaStreamGetDevice(stream, &device));
    trt_edgellm::rt::DeploymentOptions options{};
    options.engineDir = artifacts.engine();
    options.checkpointDir = artifacts.checkpoint();
    options.parallelConfig.devices = {device};
    return std::make_unique<Runtime>(std::move(options), stream);
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask final : public ITextGeneration {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(make_runtime(artifacts_, stream_.get())),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()),
          raw_prefix_(bundle_raw_prefix(bundle)) {}

    std::int32_t default_max_new_tokens() const override { return 128; }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// Invoke Edge once; failures propagate without attempting native inference.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        auto request = make_request(prompt, config, default_max_new_tokens(), raw_prefix_);
        std::lock_guard<std::mutex> lock(mutex_);
        const auto counts = runtime_->countPromptTokens(request);
        if (counts.size() != 1)
            throw std::runtime_error("Mistral Edge returned invalid prompt counts");
        validate_capacity(counts.front(), input_limit_, capacity_, request.maxGenerateLength);
        trt_edgellm::rt::LLMGenerationResponse response{};
        // Complete queued work before response/request storage is destroyed,
        // including exception paths in a persistent task.
        struct Drain {
            cudaStream_t stream;
            ~Drain() { cudaStreamSynchronize(stream); }
        } drain{stream_.get()};
        if (!runtime_->handleRequest(request, response, stream_.get()) ||
            response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
            response.outputIds.front().empty() ||
            response.outputIds.front().size() > static_cast<std::size_t>(request.maxGenerateLength))
            throw std::runtime_error("Mistral Edge generation failed");
        if (response.finishReasons.size() != 1 ||
            (response.finishReasons.front() != trt_edgellm::rt::FinishReason::kEndId &&
             response.finishReasons.front() != trt_edgellm::rt::FinishReason::kLength))
            throw std::runtime_error("Mistral Edge generation did not complete successfully");
        // This API does not expose per-request stage times; zero means unavailable.
        return {std::move(response.outputTexts.front()), std::move(response.outputIds.front())};
    }

  private:
    // Reverse destruction order keeps weights, plugin and stream alive throughout Edge teardown.
    Artifacts artifacts_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime_;
    int capacity_;
    int input_limit_;
    std::string raw_prefix_;
    std::mutex mutex_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    const auto& revision = marker.at("edge_revision");
    // External 0.12 SDK packages attest the release, not a Git checkout.
    const bool compatible =
        marker.at("edge_release") == "0.12.0" && (revision.is_null() || revision == kRevision);
    if (marker.at("version") != 1 || !compatible ||
        marker.at("max_sequence_length").get<int>() <= 1 ||
        marker.at("max_input_length").get<int>() <= 0 ||
        marker.at("max_input_length").get<int>() > marker.at("max_sequence_length").get<int>() ||
        marker.at("max_batch_size") != 1 || marker.at("precision") != "fp16" ||
        marker.at("weight_format") != "gptq_w8a16" || !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid Mistral Edge bundle contract");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::mistral::edge_llm
