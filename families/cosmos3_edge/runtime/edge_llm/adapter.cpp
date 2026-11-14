/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/cosmos3_edge/runtime/edge_llm/adapter.h"

#include "families/cosmos3_edge/runtime/edge_llm/request.h"
#include "families/cosmos3_edge/runtime/request.h"

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

namespace trtmc::cosmos3_edge::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Cosmos3-Edge Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Cosmos3-Edge Edge runtime host");
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
            "Cosmos3-Edge Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Cosmos3-Edge Edge artifact: " + name);
        }
        for (const auto* required :
             {"edge_llm/engine/visual/visual.engine", "edge_llm/engine/visual/config.json",
              "edge_llm/engine/llm.engine", "edge_llm/engine/config.json",
              "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
              "edge_llm/engine/chat_template.jinja", "edge_llm/checkpoint/config.json",
              "edge_llm/checkpoint/model.safetensors.index.json"})
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(
                    std::string("Required Cosmos3-Edge Edge artifact missing: ") + required);
        std::string pattern =
            (fs::temp_directory_path() / "trtmc-cosmos3_edge-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Cosmos3-Edge Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Cosmos3-Edge Edge artifact: " + name);
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
        throw std::runtime_error("Cannot locate Cosmos3-Edge family library");
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
        throw std::runtime_error("Cannot initialize Cosmos3-Edge Edge plugin");
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

/// Persistent complete-network runtime. All native task contracts stay family-owned.
class EdgeTask final : public ITextGeneration,
                       public IVisionLanguageGeneration,
                       public IModel,
                       public ITextContinuation,
                       public IImageVideoTextToText,
                       public IImagesTextToText,
                       public IVideoTextToText {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(artifacts_.engine(), artifacts_.engine(),
                   std::unordered_map<std::string, std::string>{}, stream_.get(),
                   trt_edgellm::rt::ContextCacheConfig{}, artifacts_.checkpoint()),
          task_(bundle.info().task), capacity_(marker.at("max_sequence_length")),
          input_limit_(marker.at("max_input_length")) {}
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }
    const char* task() const noexcept override { return task_.c_str(); }
    int32_t default_max_new_tokens() const override { return 128; }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<ITextContinuation>(*this, field_view(false)),
                bind<IImagesTextToText>(*this, field_view()),
                bind<IVideoTextToText>(*this, field_view()),
                bind<IImageVideoTextToText>(*this, field_view())};
    }
    TextResult generate(const std::string& text, const TextGenerationConfig& cfg) override {
        std::vector<ImageVideoTextPart> parts{TextPartView{text}};
        ImageVideoTextMessage message{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&message, 1}, cfg);
    }
    TextResult generate(const std::string& text, const float* pixels, int32_t h, int32_t w,
                        const TextGenerationConfig& cfg) override {
        if (h <= 0 || w <= 0)
            throw std::invalid_argument("Cosmos3-Edge image dimensions must be positive");
        std::vector<ImageVideoTextPart> parts{ImageView{pixels, size_t(h) * w * 3 * sizeof(float),
                                                        uint32_t(h), uint32_t(w), 3,
                                                        ImageFormat::Float32},
                                              TextPartView{text}};
        ImageVideoTextMessage message{MessageRole::User, {parts.data(), parts.size()}};
        return execute({&message, 1}, cfg);
    }
    TextResult run(const TextContinuationRequest& input, ConfigView supplied) override {
        const auto* text = std::get_if<std::string_view>(&input.prefix);
        if (!text)
            throw std::invalid_argument(
                "Cosmos3-Edge continuation accepts text, not raw token IDs");
        return generate(std::string(*text), read_config(supplied, false));
    }
    TextResult run(const ImageVideoTextToTextRequest& input, ConfigView supplied) override {
        return execute(input.messages, read_config(supplied));
    }

    TextResult run(const ImagesTextToTextRequest& input, ConfigView supplied) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Cosmos3-Edge reasoning does not expose tools");
        return media_messages(input.messages, supplied);
    }
    TextResult run(const VideoTextToTextRequest& input, ConfigView supplied) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Cosmos3-Edge reasoning does not expose tools");
        return media_messages(input.messages, supplied);
    }
    template <class Part>
    TextResult media_messages(Span<const MediaMessage<Part>> input, ConfigView supplied) {
        std::vector<std::vector<ImageVideoTextPart>> storage;
        std::vector<ImageVideoTextMessage> messages;
        storage.reserve(input.size());
        messages.reserve(input.size());
        for (const auto& source : input) {
            storage.emplace_back();
            for (const auto& part : source.parts) {
                std::visit(
                    [&](const auto& value) {
                        using Value = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<Value, TextPartView> ||
                                      std::is_same_v<Value, ImageView> ||
                                      std::is_same_v<Value, VideoView>)
                            storage.back().emplace_back(value);
                        else
                            throw std::invalid_argument(
                                "Cosmos3-Edge reasoning task requires text or visual media parts");
                    },
                    part);
            }
            messages.push_back({source.role, {storage.back().data(), storage.back().size()}});
        }
        return execute({messages.data(), messages.size()}, read_config(supplied));
    }

  private:
    // Runtime destruction precedes the stream, plugin and extracted checkpoint.
    Artifacts artifacts_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    trt_edgellm::rt::LLMInferenceRuntime runtime_;
    std::string task_;
    int capacity_, input_limit_;
    std::mutex mutex_;

    TextResult execute(Span<const ImageVideoTextMessage> messages,
                       const TextGenerationConfig& cfg) {
        auto request = make_request(messages, cfg, capacity_);
        const auto budget = request.maxGenerateLength;
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            trt_edgellm::rt::LLMGenerationResponse response{};
            if (!runtime_.handleRequest(request, response, stream_.get()))
                throw std::runtime_error("Cosmos3-Edge generation failed");
            validate_response(response, input_limit_, capacity_, budget);
            check_cuda(cudaStreamSynchronize(stream_.get()));
            return {std::move(response.outputTexts.front()), std::move(response.outputIds.front())};
        } catch (...) {
            // The request owns pinned image/video storage used by asynchronous DMA.
            cudaStreamSynchronize(stream_.get());
            throw;
        }
    }
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
        marker.at("execution_variant") != "autoregressive" || !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid Cosmos3-Edge bundle contract");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}
} // namespace trtmc::cosmos3_edge::edge_llm
