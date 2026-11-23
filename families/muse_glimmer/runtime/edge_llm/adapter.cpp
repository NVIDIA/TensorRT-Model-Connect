/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/muse_glimmer/runtime/edge_llm/adapter.h"

#include "families/muse_glimmer/runtime/edge_llm/request.h"
#include "families/muse_glimmer/runtime/sampler.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
#include "families/muse_glimmer/runtime/edge_llm/media.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"

#include <tensorrt_edgellm/common/trtInterop.h>
#include <tensorrt_edgellm/runtime/llmInferenceRuntime.h>
#include <tensorrt_edgellm/runtime/streaming.h>
#else
#include <edgellm/cpp/common/logger.h>
#include <edgellm/cpp/runtime/llmInferenceRuntime.h>
#include <edgellm/cpp/runtime/streaming.h>
#endif
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <sys/utsname.h>
#include <vector>

namespace trtmc::muse_glimmer::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Muse-Glimmer Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Muse-Glimmer Edge runtime host");
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
            "Muse-Glimmer Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Muse-Glimmer Edge artifact: " + name);
        }
        std::vector<std::string> required_files{
            "edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
            "edge_llm/engine/chat_template.jinja", "edge_llm/engine/embedding.safetensors",
            "edge_llm/checkpoint/config.json"};
        if (marker.at("execution_variant") == "dflash" ||
            marker.at("execution_variant") == "dflash_tree") {
            for (const auto* name :
                 {"spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"})
                required_files.push_back(std::string("edge_llm/engine/") + name);
            if (marker.at("dflash_version") == 2)
                required_files.push_back("edge_llm/engine/dflash2_selector.safetensors");
            for (const auto* name : {"base_config.json", "draft_config.json"}) {
                const auto bytes = bundle.read_section(std::string("edge_llm/engine/") + name);
                const auto config = nlohmann::json::parse(bytes.begin(), bytes.end());
                const auto& policy = config.at("dflash_config");
                if (config.at("spec_decode_type") != "dflash" || policy.at("block_size") != 16 ||
                    policy.at("version") != marker.at("dflash_version"))
                    throw std::runtime_error("Muse-Glimmer paired engine contract differs");
                if (marker.at("dflash_version") == 2 && std::string(name) == "draft_config.json" &&
                    policy.value("selector_file", std::string("dflash2_selector.safetensors")) !=
                        "dflash2_selector.safetensors")
                    throw std::runtime_error("Unexpected Muse DFlash2 selector path");
            }
        } else {
            required_files.push_back("edge_llm/engine/llm.engine");
            required_files.push_back("edge_llm/engine/config.json");
        }
        if (marker.contains("vision"))
            for (const auto* name : {"visual.engine", "config.json", "preprocessor_config.json",
                                     "processor_config.json"})
                required_files.push_back(std::string("edge_llm/engine/visual/") + name);
        for (const auto& required : required_files)
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(
                    std::string("Required Muse-Glimmer Edge artifact missing: ") + required);
        std::string pattern =
            (fs::temp_directory_path() / "trtmc-muse_glimmer-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Muse-Glimmer Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Muse-Glimmer Edge artifact: " + name);
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
        throw std::runtime_error("Cannot locate Muse-Glimmer family library");
    const auto path =
        fs::absolute(location.dli_fname).parent_path() / "libNvInfer_edgellm_plugin.so";
    std::unique_ptr<void, CloseLibrary> plugin(
        dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE));
    if (!plugin)
        throw std::runtime_error("Cannot load CMake-installed Edge plugin: " +
                                 std::string(dlerror()));
    using Initialize = bool (*)(void*, const char*);
    auto initialize = reinterpret_cast<Initialize>(dlsym(plugin.get(), "initEdgellmPlugins"));
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
    if (!initialize ||
        !initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::TrtLogger::instance()), ""))
#else
    if (!initialize || !initialize(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
#endif
        throw std::runtime_error("Cannot initialize Muse-Glimmer Edge plugin");
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

/// Use the pinned runtime's complete owning execution mode.
std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime>
make_runtime(const Artifacts& artifacts, const nlohmann::json& marker, cudaStream_t stream) {
    using Runtime = trt_edgellm::rt::LLMInferenceRuntime;
    const auto media_dir = marker.contains("vision") ? artifacts.engine() : std::string{};
    if (marker.at("execution_variant") == "dflash" ||
        marker.at("execution_variant") == "dflash_tree") {
        trt_edgellm::rt::SpecDecodeDraftingConfig config{};
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
        // The release counts the anchor in the verification window but not in proposals.
        const bool tree = marker.at("execution_variant") == "dflash_tree";
        config.draftTopK = tree ? marker.at("dflash_top_k").get<int>() : 1;
        if (!tree)
            config.proposalTokens = 15;
        config.verifyTokens = tree ? marker.at("dflash_verify").get<int>() : 16;
#else
        config.draftingTopK = 1;
        config.draftingStep = 1;
        config.verifySize = 16;
        config.dflashBlockSize = 16;
#endif
        return std::make_unique<Runtime>(
            artifacts.engine(), media_dir, std::unordered_map<std::string, std::string>{}, config,
            stream, trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint());
    }
    return std::make_unique<Runtime>(artifacts.engine(), media_dir,
                                     std::unordered_map<std::string, std::string>{}, stream,
                                     trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint());
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask : public ITextGeneration {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(make_runtime(artifacts_, marker, stream_.get())),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()),
          v1_(marker.value("dflash_version", 0) == 1),
          tree_(marker.at("execution_variant") == "dflash_tree") {}

    std::int32_t default_max_new_tokens() const override { return std::min(512, capacity_ - 1); }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// Invoke Edge once; failures propagate without attempting native inference.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        auto request = make_request(prompt, config, default_max_new_tokens());
        return execute(request, config);
    }

  protected:
    int capacity() const { return capacity_; }

    TextResult execute(trt_edgellm::rt::LLMGenerationRequest& request,
                       const TextGenerationConfig& config) {
        if (tree_) {
            if (!muse_is_greedy(muse_sampling_params_from_config(config, -1)))
                throw std::invalid_argument(
                    "Muse-Glimmer DFlash2 DDTree requires greedy generation");
            // Preserve MC's deterministic result even where the SDK classifies
            // the equivalent parameter spelling as a reason to switch to vanilla.
            request.temperature = 0.0F;
            request.topK = 1;
            request.topP = 1.0F;
        } else if (v1_ && request.temperature != 0.0F) {
            throw std::invalid_argument("Muse-Glimmer DFlash V1 requires greedy generation");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (request.requests.front().imageBuffers.empty()) {
            const auto counts = runtime_->countPromptTokens(request);
            if (counts.size() != 1)
                throw std::runtime_error("Muse-Glimmer Edge returned invalid prompt counts");
            validate_capacity(counts.front(), input_limit_, capacity_, request.maxGenerateLength);
        } else if (request.maxGenerateLength <= 0 || request.maxGenerateLength >= capacity_) {
            throw std::invalid_argument("Muse-Glimmer Edge media output budget exceeds capacity");
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
            throw std::runtime_error("Muse-Glimmer Edge generation failed");
        if (response.finishReasons.size() != 1 ||
            (response.finishReasons.front() != trt_edgellm::rt::FinishReason::kEndId &&
             response.finishReasons.front() != trt_edgellm::rt::FinishReason::kLength))
            throw std::runtime_error("Muse-Glimmer Edge generation did not complete successfully");
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
        if (!request.requests.front().imageBuffers.empty()) {
            // countPromptTokens is text-only. Edge reports its actual expanded media length here.
            if (response.inputTokenCounts.size() != 1)
                throw std::runtime_error("Muse-Glimmer Edge returned no expanded token count");
            validate_capacity(response.inputTokenCounts.front(), input_limit_, capacity_,
                              request.maxGenerateLength);
            if (response.finishReasons.front() == trt_edgellm::rt::FinishReason::kLength &&
                response.outputIds.front().size() != static_cast<size_t>(request.maxGenerateLength))
                throw std::runtime_error("Muse-Glimmer Edge clipped the media output budget");
        }
#endif
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
    bool v1_;
    bool tree_;
    std::mutex mutex_;
};
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
class EdgeMediaTask final : public EdgeTask,
                            public internal::IModel,
                            public internal::ITextContinuation,
                            public internal::IImagesTextToText,
                            public internal::IVideoTextToText {
  public:
    EdgeMediaTask(const BundleReader& b, const nlohmann::json& marker)
        : EdgeTask(b, marker), task_(b.info().task), fps_(marker.at("vision").at("fps")) {}
    const char* task() const noexcept override { return task_.c_str(); }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextContinuation>(*this, media_fields()),
                internal::bind<internal::IImagesTextToText>(*this, media_fields()),
                internal::bind<internal::IVideoTextToText>(*this, media_fields())};
    }
    TextResult run(const internal::TextContinuationRequest& r, internal::ConfigView v) override {
        const auto* text = std::get_if<std::string_view>(&r.prefix);
        if (!text)
            throw std::invalid_argument("Muse-Glimmer text continuation requires UTF-8 text");
        return generate(std::string(*text), media_config(v));
    }
    TextResult run(const internal::ImagesTextToTextRequest& r, internal::ConfigView v) override {
        return run_media(r, v);
    }
    TextResult run(const internal::VideoTextToTextRequest& r, internal::ConfigView v) override {
        return run_media(r, v);
    }

  private:
    template <typename R>
    TextResult run_media(const R& input, internal::ConfigView view) {
        auto cfg = media_config(view);
        auto request =
            media_request(media_turns(input, cfg), cfg, default_max_new_tokens(), capacity(), fps_);
        return execute(request, cfg);
    }
    std::string task_;
    double fps_;
};
#endif
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm.json");
    const auto marker = nlohmann::json::parse(bytes.begin(), bytes.end());
    const auto variant = marker.at("execution_variant").get<std::string>();
    const bool chain =
        variant == "dflash" &&
        (marker.value("dflash_version", 0) == 1 || marker.value("dflash_version", 0) == 2) &&
        marker.value("dflash_block_size", 0) == 16;
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
    const bool paired =
        chain || (variant == "dflash_tree" && marker.value("dflash_version", 0) == 2 &&
                  marker.value("dflash_block_size", 0) == 16 &&
                  marker.value("dflash_top_k", 0) == 4 && marker.value("dflash_verify", 0) == 32);
    const auto& revision = marker.at("edge_revision");
    const bool supported_weights =
        marker.at("weight_format") == "floating" || marker.at("weight_format") == "nvfp4";
    const bool compatible = marker.at("edge_release") == "0.12.0" &&
                            marker.at("builder_flow") == "onnx" &&
                            (revision.is_null() || revision == kRevision);
#else
    const bool paired = chain;
    const bool supported_weights = marker.at("weight_format") == "nvfp4";
    const bool compatible = marker.at("edge_revision") == kRevision;
#endif
    if (marker.at("version") != 1 || !compatible ||
        marker.at("max_sequence_length").get<int>() <= 1 ||
        marker.at("max_input_length").get<int>() <= 0 ||
        marker.at("max_input_length").get<int>() > marker.at("max_sequence_length").get<int>() ||
        marker.at("max_batch_size") != 1 || marker.at("precision") != "fp16" ||
        !supported_weights || (variant != "autoregressive" && !paired) ||
        !marker.at("artifacts").is_array())
        throw std::runtime_error("Invalid Muse-Glimmer Edge bundle contract");
    validate_target(marker.at("target"));
    const auto& task = bundle.info().task;
#ifdef TRTMC_MUSE_GLIMMER_EDGE_012
    if (task == "images_text_to_text" || task == "video_text_to_text") {
        if (marker.at("weight_format") != "floating" || !marker.contains("vision") ||
            !std::isfinite(marker.at("vision").at("fps").get<double>()) ||
            marker.at("vision").at("fps").get<double>() <= 0)
            throw std::runtime_error("Invalid Muse-Glimmer Edge media contract");
        return new EdgeMediaTask(bundle, marker);
    }
#endif
    if (task != "text_generation" || marker.contains("vision"))
        throw std::runtime_error("Unexpected Muse-Glimmer Edge bundle task");
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::muse_glimmer::edge_llm
