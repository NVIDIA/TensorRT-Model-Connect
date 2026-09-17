/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/edge_llm/adapter.h"

#include "families/qwen3_8/runtime/edge_llm/request.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
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
#include <vector>

namespace trtmc::qwen3_8::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.8 Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Qwen3.8 Edge runtime host");
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
            "Qwen3.8 Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Qwen3.8 Edge artifact: " + name);
        }
        std::vector<std::string> required_files{
            "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
            "edge_llm/engine/processed_chat_template.json",
            "edge_llm/engine/trtmc_single_user_prompts.json", "edge_llm/checkpoint/config.json"};

        for (const auto* name :
             {"spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
              "embedding.safetensors", "dspark_heads.safetensors", "dspark_heads_info.json"})
            required_files.push_back(std::string("edge_llm/engine/") + name);
        required_files.push_back("edge_llm/checkpoint/draft/config.json");
        for (const auto& required : required_files)
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error("Required Qwen3.8 Edge artifact missing: " + required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-qwen3_8-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Qwen3.8 Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Qwen3.8 Edge artifact: " + name);
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
        throw std::runtime_error("Cannot locate Qwen3.8 family library");
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
        throw std::runtime_error("Cannot initialize Qwen3.8 Edge plugin");
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

/// Delegate the complete DSpark block7 algorithm to the pinned runtime.
std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> make_runtime(const Artifacts& artifacts,
                                                                   cudaStream_t stream) {
    trt_edgellm::rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    drafting.draftingStep = 1;
    drafting.verifySize = 8;
    drafting.dflashBlockSize = 0;
    drafting.dsparkSchedulerMode = trt_edgellm::rt::DSparkSchedulerMode::kOff;
    drafting.dsparkConfidenceThreshold = 0.0F;
    drafting.dsparkMinProposalLen = 1;
    drafting.dsparkMaxProposalLen = 0;
    return std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
        artifacts.engine(), "", std::unordered_map<std::string, std::string>{}, drafting, stream,
        trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint(),
        (fs::path(artifacts.checkpoint()) / "draft").string());
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask final : public ITextGeneration {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(make_runtime(artifacts_, stream_.get())),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()) {
        std::ifstream input(fs::path(artifacts_.engine()) / "trtmc_single_user_prompts.json");
        source_prompts_ = nlohmann::json::parse(input);
        for (const auto* mode : {"false", "true"})
            for (const auto* part : {"prefix", "suffix"})
                if (!source_prompts_.at(mode).at(part).is_string())
                    throw std::runtime_error("Invalid Qwen3.8 source prompt mapping");
    }

    std::int32_t default_max_new_tokens() const override { return std::min(128, capacity_ - 1); }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// Invoke Edge once; failures propagate without attempting native inference.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        auto effective = config;
        std::string source_prompt = prompt;
        if (config.use_chat_template) {
            const auto& format = source_prompts_.at(config.enable_thinking ? "true" : "false");
            source_prompt = format.at("prefix").get<std::string>() + source_user_content(prompt) +
                            format.at("suffix").get<std::string>();
            effective.use_chat_template = false;
        }
        auto request = make_request(source_prompt, effective, default_max_new_tokens());
        std::lock_guard<std::mutex> lock(mutex_);
        const auto counts = runtime_->countPromptTokens(request);
        if (counts.size() != 1)
            throw std::runtime_error("Qwen3.8 Edge returned invalid prompt counts");
        validate_capacity(counts.front(), input_limit_, capacity_, request.maxGenerateLength);
        trt_edgellm::rt::LLMGenerationResponse response{};
        if (!runtime_->handleRequest(request, response, stream_.get()) ||
            response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
            response.outputIds.front().empty() ||
            response.outputIds.front().size() > static_cast<std::size_t>(request.maxGenerateLength))
            throw std::runtime_error("Qwen3.8 Edge generation failed");
        if (response.finishReasons.size() != 1 ||
            (response.finishReasons.front() != trt_edgellm::rt::FinishReason::kEndId &&
             response.finishReasons.front() != trt_edgellm::rt::FinishReason::kLength))
            throw std::runtime_error("Qwen3.8 Edge generation did not complete successfully");
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
    nlohmann::json source_prompts_;
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
        !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid Qwen3.8 Edge bundle contract");
    const auto variant = marker.value("execution_variant", "");
    if (!valid_execution_variant(variant, marker.value("dspark_block_size", 0)))
        throw std::runtime_error("Unsupported Qwen3.8 Edge execution variant");
    if (marker.value("builder_flow", "") != "onnx" ||
        marker.value("checkpoint_quantization", "") != "nvfp4" ||
        marker.at("max_sequence_length").get<int>() <= 8 ||
        marker.at("max_sequence_length").get<int>() > 1024 ||
        marker.at("target").at("arch") != "x86_64" || marker.at("target").at("sm") != 120)
        throw std::runtime_error("Qwen3.8 DSpark requires the ONNX builder flow");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::qwen3_8::edge_llm
