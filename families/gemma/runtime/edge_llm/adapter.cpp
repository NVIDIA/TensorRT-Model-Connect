/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/gemma/runtime/edge_llm/adapter.h"

#include "families/gemma/runtime/edge_llm/media.h"

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
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <sys/utsname.h>
#include <vector>

namespace trtmc::gemma::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Gemma4 Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Gemma4 Edge runtime host");
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
            "Gemma4 Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Gemma4 Edge artifact: " + name);
        }
        std::vector<std::string> required_files{
            "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
            "edge_llm/engine/chat_template.jinja", "edge_llm/engine/embedding.safetensors"};
        auto require = [&](const char* name) {
            required_files.push_back(std::string("edge_llm/engine/") + name);
        };
        if (marker.at("execution_variant") == "autoregressive") {
            require("llm.engine");
            require("config.json");
            if (marker.value("ple", false))
                require("ple_embedding.safetensors");
            if (marker.value("vision", false)) {
                require("visual/visual.engine");
                require("visual/config.json");
            }
            if (marker.value("audio", false)) {
                require("audio/audio_encoder.engine");
                require("audio/config.json");
            }
        } else {
            require("spec_base.engine");
            require("spec_draft.engine");
            require("base_config.json");
            require("draft_config.json");
        }
        if (marker.at("execution_variant") == "dspark") {
            require("dspark_heads.safetensors");
            require("dspark_heads_info.json");
        }
        for (const auto& required : required_files)
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error("Required Gemma4 Edge artifact missing: " + required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-gemma-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Gemma4 Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Gemma4 Edge artifact: " + name);
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
    // Edge INFO goes to stdout. Keep the public CLI result channel machine-readable;
    // upstream warnings and errors continue to stderr.
    static std::once_flag logging;
    std::call_once(logging,
                   [] { trt_edgellm::gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING); });
    Dl_info location{};
    if (!dladdr(reinterpret_cast<void*>(&create), &location) || !location.dli_fname)
        throw std::runtime_error("Cannot locate Gemma4 family library");
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
        throw std::runtime_error("Cannot initialize Gemma4 Edge plugin");
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

/// Delegate the complete speculative algorithm to the original Edge runtime.
std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime>
make_runtime(const Artifacts& artifacts, cudaStream_t stream, const nlohmann::json& marker) {
    const bool standalone = marker.at("execution_variant") == "autoregressive";
    const bool dspark = marker.at("execution_variant") == "dspark";
    const bool eagle3 = marker.at("execution_variant") == "eagle3";
    const bool dflash = marker.at("execution_variant") == "dflash";
    const bool media = marker.value("vision", false) || marker.value("audio", false);
    if (standalone)
        return std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
            artifacts.engine(), media ? artifacts.engine() : "",
            std::unordered_map<std::string, std::string>{}, stream,
            trt_edgellm::rt::ContextCacheConfig{});
    trt_edgellm::rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = eagle3 ? 10 : 1;
    drafting.draftingStep = eagle3 ? 6 : (dspark || dflash) ? 1 : 3;
    drafting.verifySize = dflash   ? marker.at("verify_size").get<int>()
                          : eagle3 ? 60
                          : dspark ? 8
                                   : 4;
    drafting.dsparkSchedulerMode = trt_edgellm::rt::DSparkSchedulerMode::kOff;
    return std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
        artifacts.engine(), "", std::unordered_map<std::string, std::string>{}, drafting, stream,
        trt_edgellm::rt::ContextCacheConfig{});
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask final : public ITextGeneration,
                       public api::IModel,
                       public api::ITextContinuation,
                       public api::IImagesTextToText,
                       public api::IAudioTextToText,
                       public api::IImageAudioTextToText {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(make_runtime(artifacts_, stream_.get(), marker)),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()),
          sampling_(allows_sampling(marker.at("execution_variant").get<std::string>())),
          sampled_vanilla_(
              sampling_uses_vanilla(marker.at("execution_variant").get<std::string>())),
          headroom_(marker.at("execution_variant") == "autoregressive" ? 1
                    : marker.at("execution_variant") == "dflash"
                        ? marker.at("verify_size").get<int>()
                    : marker.at("execution_variant") == "eagle3" ? 60
                    : marker.at("execution_variant") == "dspark" ? 8
                                                                 : 4),
          vision_(marker.value("vision", false)), audio_(marker.value("audio", false)),
          primary_(marker.value("task", "text_generation")) {}

    const char* task() const noexcept override { return primary_.c_str(); }

    std::vector<api::TaskInstance> task_bindings() override {
        std::vector<api::TaskInstance> bindings{
            api::bind<api::ITextContinuation>(*this, fields(false))};
        if (vision_)
            bindings.push_back(api::bind<api::IImagesTextToText>(*this, fields(true)));
        if (audio_)
            bindings.push_back(api::bind<api::IAudioTextToText>(*this, fields(true)));
        if (vision_ && audio_)
            bindings.push_back(api::bind<api::IImageAudioTextToText>(*this, fields(true)));
        return bindings;
    }

    TextResult run(const api::TextContinuationRequest& input, api::ConfigView supplied) override {
        const auto* text = std::get_if<std::string_view>(&input.prefix);
        if (!text)
            throw std::invalid_argument("Gemma4 Edge currently accepts UTF-8 text prefixes only");
        return generate(std::string(*text),
                        generation_config(supplied, false, default_max_new_tokens()));
    }

    TextResult run(const api::ImagesTextToTextRequest& input, api::ConfigView config) override {
        if (!input.tools.empty())
            throw std::invalid_argument("Gemma4 media does not map tools");
        return generate_media(input.messages, config);
    }
    TextResult run(const api::AudioTextToTextRequest& input, api::ConfigView config) override {
        return generate_media(input.messages, config);
    }
    TextResult run(const api::ImageAudioTextToTextRequest& input, api::ConfigView config) override {
        return generate_media(input.messages, config);
    }

    std::int32_t default_max_new_tokens() const override {
        return std::min(128, capacity_ - headroom_ - 1);
    }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// Invoke Edge once; failures propagate without attempting native inference.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        if (sampled_vanilla_ && config.temperature > 0 && config.top_k != 1)
            std::cerr
                << "Warning: Gemma Edge 0.11 uses vanilla fallback for this sampled speculative "
                   "request; speculative activity is not qualified. Sampling controls "
                   "are forwarded unchanged.\n";
        return execute(make_request(prompt, config, default_max_new_tokens(), sampling_));
    }

  private:
    template <class Part>
    TextResult generate_media(Span<const api::MediaMessage<Part>> messages,
                              api::ConfigView supplied) {
        const auto config = generation_config(supplied, true, default_max_new_tokens());
        return execute(media_request(messages, config, default_max_new_tokens(), vision_, audio_));
    }

    TextResult execute(trt_edgellm::rt::LLMGenerationRequest request) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& item = request.requests.front();
        if (item.imageBuffers.empty() && item.audioBuffers.empty()) {
            const auto counts = runtime_->countPromptTokens(request);
            if (counts.size() != 1)
                throw std::runtime_error("Gemma4 Edge returned invalid prompt counts");
            validate_capacity(counts.front(), input_limit_, capacity_ - headroom_,
                              request.maxGenerateLength);
        } else {
            // Edge 0.11 cannot count media tokens without executing preprocessing.
            // Reserve the entire input profile plus decode lookahead. Edge rejects
            // expanded prefills exceeding that profile; it cannot silently clip an
            // accepted request's generation budget under this conservative bound.
            validate_capacity(input_limit_, input_limit_, capacity_ - headroom_,
                              request.maxGenerateLength);
        }
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
            throw std::runtime_error("Gemma4 Edge generation failed");
        if (response.finishReasons.size() != 1 ||
            (response.finishReasons.front() != trt_edgellm::rt::FinishReason::kEndId &&
             response.finishReasons.front() != trt_edgellm::rt::FinishReason::kLength))
            throw std::runtime_error("Gemma4 Edge generation did not complete successfully");
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
    bool sampling_;
    bool sampled_vanilla_;
    int headroom_;
    bool vision_;
    bool audio_;
    std::string primary_;
    std::mutex mutex_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    if (marker.at("version") != 1 || marker.at("edge_revision") != kRevision ||
        marker.at("max_sequence_length").get<int>() <= 2 ||
        marker.at("max_input_length").get<int>() <= 0 ||
        marker.at("max_input_length").get<int>() > marker.at("max_sequence_length").get<int>() ||
        marker.at("max_batch_size") != 1 || marker.at("precision") != "fp16" ||
        !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid Gemma4 Edge bundle contract");
    if ((marker.value("execution_variant", "") != "mtp" &&
         marker.value("execution_variant", "") != "dspark" &&
         marker.value("execution_variant", "") != "eagle3" &&
         marker.value("execution_variant", "") != "dflash" &&
         marker.value("execution_variant", "") != "autoregressive") ||
        marker.value("builder_flow", "") != "onnx")
        throw std::runtime_error(
            "Gemma4 requires an autoregressive, MTP, DSpark, EAGLE3 or DFlash ONNX contract");
    if (marker.at("execution_variant") == "dflash" &&
        (marker.value("verify_size", 0) != 7 && marker.value("verify_size", 0) != 16))
        throw std::runtime_error("Invalid Gemma4 DFlash block-size contract");
    const auto primary = marker.value("task", "text_generation");
    if (primary != "text_generation" && primary != "text_continuation" &&
        !(primary == "images_text_to_text" && marker.value("vision", false)) &&
        !(primary == "audio_text_to_text" && marker.value("audio", false)) &&
        !(primary == "image_audio_text_to_text" && marker.value("vision", false) &&
          marker.value("audio", false)))
        throw std::runtime_error("Gemma4 bundle primary task does not match its components");
    if (marker.at("execution_variant") != "autoregressive" &&
        (marker.value("vision", false) || marker.value("audio", false)))
        throw std::runtime_error("Gemma4 paired media execution is not mapped");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::gemma::edge_llm
