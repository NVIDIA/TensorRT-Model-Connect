/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_5_moe/runtime/edge_llm/adapter.h"

#include "families/qwen3_5_moe/runtime/edge_llm/media.h"

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

namespace trtmc::qwen3_5_moe::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.5-MoE Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Qwen3.5-MoE Edge runtime host");
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
            "Qwen3.5-MoE Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Qwen3.5-MoE Edge artifact: " + name);
        }
        std::vector<std::string> required{
            "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
            "edge_llm/engine/chat_template.jinja", "edge_llm/checkpoint/config.json",
            "edge_llm/checkpoint/model.safetensors.index.json"};
        if (marker.value("execution_variant", "autoregressive") == "autoregressive") {
            required.insert(required.end(),
                            {"edge_llm/engine/llm.engine", "edge_llm/engine/config.json"});
        } else {
            required.insert(required.end(), {"edge_llm/engine/spec_base.engine",
                                             "edge_llm/engine/spec_draft.engine",
                                             "edge_llm/engine/base_config.json",
                                             "edge_llm/engine/draft_config.json"});
            if (marker.at("execution_variant") != "mtp")
                required.push_back("edge_llm/checkpoint/draft/config.json");
        }
        if (marker.at("task") == IVisionLanguageGeneration::kTask) {
            required.push_back("edge_llm/engine/visual/visual.engine");
            required.push_back("edge_llm/engine/visual/config.json");
        }
        for (const auto& name : required)
            if (!names.count(name) || bundle.find_section(name)->length == 0)
                throw std::runtime_error("Required Qwen3.5-MoE Edge artifact missing: " + name);
        std::string pattern =
            (fs::temp_directory_path() / "trtmc-qwen3_5_moe-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Qwen3.5-MoE Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Qwen3.5-MoE Edge artifact: " + name);
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
        throw std::runtime_error("Cannot locate Qwen3.5-MoE family library");
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
        throw std::runtime_error("Cannot initialize Qwen3.5-MoE Edge plugin");
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
void validate_bundle_tokenizer(const BundleReader& bundle, const nlohmann::json& marker) {
    const auto bytes = bundle.read_section("edge_llm/engine/tokenizer.json");
    const auto config = bundle.read_section("edge_llm/engine/tokenizer_config.json");
    const auto engine =
        bundle.read_section(marker.value("execution_variant", "autoregressive") == "autoregressive"
                                ? "edge_llm/engine/config.json"
                                : "edge_llm/engine/base_config.json");
    validate_raw_tokenizer(nlohmann::json::parse(bytes.begin(), bytes.end()),
                           nlohmann::json::parse(config.begin(), config.end()),
                           nlohmann::json::parse(engine.begin(), engine.end()));
}

/// Delegate the whole speculative algorithm to the pinned Edge runtime.
std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime>
make_runtime(const Artifacts& artifacts, const nlohmann::json& marker, cudaStream_t stream) {
    using Runtime = trt_edgellm::rt::LLMInferenceRuntime;
    const auto media =
        marker.at("task") == IVisionLanguageGeneration::kTask ? artifacts.engine() : "";
    const auto variant = marker.value("execution_variant", "autoregressive");
    if (variant == "dflash" || variant == "dflash_ddtree" || variant == "mtp") {
        trt_edgellm::rt::SpecDecodeDraftingConfig drafting{};
        drafting.draftingTopK = variant == "dflash_ddtree" ? 4 : 1;
        drafting.draftingStep = variant == "mtp" ? 3 : 1;
        drafting.verifySize = variant == "mtp" ? 4 : 16;
        if (variant != "mtp")
            drafting.dflashBlockSize = 16;
        return std::make_unique<Runtime>(
            artifacts.engine(), media, std::unordered_map<std::string, std::string>{}, drafting,
            stream, trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint(),
            variant == "mtp" ? "" : (fs::path(artifacts.checkpoint()) / "draft").string());
    }
    return std::make_unique<Runtime>(artifacts.engine(), media,
                                     std::unordered_map<std::string, std::string>{}, stream,
                                     trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint());
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EngineState {
  public:
    EngineState(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(make_runtime(artifacts_, marker, stream_.get())),
          tree_(marker.value("execution_variant", "autoregressive") == "dflash_ddtree"),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()) {
        validate_bundle_tokenizer(bundle, marker);
    }

    /// Drain outstanding work before destroying the runtime and its weight buffers.
    ~EngineState() { cudaStreamSynchronize(stream_.get()); }

    /// Invoke the full Edge visual and LLM runtime once; never retry inference natively.
    TextResult generate(const std::string& prompt, const float* pixels, std::int32_t height,
                        std::int32_t width, const TextGenerationConfig& config) {
        auto bytes = image_bytes(pixels, height, width);
        auto request = make_request(prompt, config, !bytes.empty());
        if (!bytes.empty()) {
            trt_edgellm::rt::Tensor tensor({1, height, width, 3}, trt_edgellm::rt::DeviceType::kCPU,
                                           nvinfer1::DataType::kUINT8);
            std::memcpy(tensor.rawPointer(), bytes.data(), bytes.size());
            request.requests.front().imageBuffers.emplace_back(std::move(tensor));
        }
        return execute(std::move(request));
    }

    TextResult execute(trt_edgellm::rt::LLMGenerationRequest request) {
        if (tree_ && request.topK != 1 && request.temperature != 0)
            throw std::invalid_argument("Qwen3.5-MoE DFlash DDTree requires greedy generation");
        const auto requested_budget = request.maxGenerateLength;
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            trt_edgellm::rt::LLMGenerationResponse response{};
            if (!runtime_->handleRequest(request, response, stream_.get()))
                throw std::runtime_error("Qwen3.5-MoE Edge generation failed");
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
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime_;
    bool tree_;
    int capacity_;
    int input_limit_;
    std::mutex mutex_;
};
/// Text-only bundles expose only the text interface, with the documented128-token default.
class TextModel final : public ITextGeneration, public api::IModel, public api::ITextContinuation {
  public:
    TextModel(const BundleReader& bundle, const nlohmann::json& marker) : state_(bundle, marker) {}
    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::ITextContinuation>(*this, fields(false))};
    }
    TextResult run(const api::TextContinuationRequest& input, api::ConfigView supplied) override {
        const auto* text = std::get_if<std::string_view>(&input.prefix);
        if (!text)
            throw std::invalid_argument("Qwen3.5-MoE accepts UTF-8 text prefixes only");
        return generate(std::string(*text), generation_config(supplied, false, 128));
    }
    std::int32_t default_max_new_tokens() const override { return 128; }
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        return state_.generate(prompt, nullptr, 0, 0, config);
    }

  private:
    EngineState state_;
};

/// Complete VLM bundles expose only the vision interface; null images retain raw/chat controls.
class VisionModel final : public IVisionLanguageGeneration,
                          public api::IModel,
                          public api::IImagesTextToText,
                          public api::IVideoTextToText,
                          public api::IImageVideoTextToText {
  public:
    VisionModel(const BundleReader& bundle, const nlohmann::json& marker)
        : state_(bundle, marker) {}
    std::vector<api::TaskInstance> task_bindings() override {
        return {api::bind<api::IImagesTextToText>(*this, fields(true)),
                api::bind<api::IVideoTextToText>(*this, fields(true)),
                api::bind<api::IImageVideoTextToText>(*this, fields(true))};
    }
    TextResult run(const api::ImagesTextToTextRequest& input, api::ConfigView supplied) override {
        return state_.execute(media_request(input, generation_config(supplied, true, 128), 128));
    }
    TextResult run(const api::VideoTextToTextRequest& input, api::ConfigView supplied) override {
        return state_.execute(media_request(input, generation_config(supplied, true, 128), 128));
    }
    TextResult run(const api::ImageVideoTextToTextRequest& input,
                   api::ConfigView supplied) override {
        return state_.execute(media_request(input, generation_config(supplied, true, 128), 128));
    }
    TextResult generate(const std::string& prompt, const float* pixels, std::int32_t height,
                        std::int32_t width, const TextGenerationConfig& config) override {
        return state_.generate(prompt, pixels, height, width, config);
    }

  private:
    EngineState state_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    validate_marker(marker, bundle.info().family, bundle.info().task, bundle.info().backend);
    validate_target(marker.at("target"));
    if (marker.at("task") == ITextGeneration::kTask)
        return new TextModel(bundle, marker);
    return new VisionModel(bundle, marker);
}

} // namespace trtmc::qwen3_5_moe::edge_llm
