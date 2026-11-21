/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8/runtime/edge_llm/adapter.h"

#include "families/qwen3_8/runtime/edge_llm/media.h"

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
void validate_target(const nlohmann::json& target, int parallel) {
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
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count));
    if (count < parallel)
        throw std::runtime_error("Qwen3.8 Edge bundle requires more visible CUDA devices");
    for (int rank = 0; rank < parallel; ++rank) {
        cudaDeviceProp gpu{};
        check_cuda(cudaGetDeviceProperties(&gpu, parallel == 1 ? device : rank));
        if (target.at("sm") != gpu.major * 10 + gpu.minor)
            throw std::runtime_error("Qwen3.8 Edge rank requires its build GPU architecture");
    }
    const int trt_version = getInferLibVersion();
    const std::string trt =
        std::to_string(trt_version / 10000) + "." + std::to_string((trt_version % 10000) / 100) +
        "." + std::to_string(trt_version % 100) + "." + std::to_string(getInferLibBuildVersion());
    const std::string cuda =
        std::to_string(cuda_version / 1000) + "." + std::to_string((cuda_version % 1000) / 10);
    if (target.at("os") != "linux" || target.at("os_version") != os_version ||
        target.at("arch") != host.machine || target.at("cuda_version") != cuda ||
        target.at("tensorrt_version") != trt)
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
            "edge_llm/engine/chat_template.jinja", "edge_llm/checkpoint/config.json"};

        if (marker.value("execution_variant", "") == "autoregressive") {
            const std::vector<std::string> plans =
                marker.value("tensor_parallel_size", 1) == 2
                    ? std::vector<std::string>{"llm_world2_rank0.engine", "llm_world2_rank1.engine",
                                               "config_world2.json"}
                    : std::vector<std::string>{"llm.engine", "config.json"};
            for (const auto& name : plans)
                required_files.push_back("edge_llm/engine/" + name);
        } else if (marker.value("execution_variant", "") == "mtp") {
            for (const auto* name :
                 {"spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json"})
                required_files.push_back(std::string("edge_llm/engine/") + name);
        } else {
            for (const auto* name :
                 {"spec_base.engine", "spec_draft.engine", "base_config.json", "draft_config.json",
                  "embedding.safetensors", "trtmc_single_user_prompts.json"})
                required_files.push_back(std::string("edge_llm/engine/") + name);
            required_files.push_back("edge_llm/checkpoint/draft/config.json");
            if (marker.value("execution_variant", "") == "dspark")
                for (const auto* name : {"dspark_heads.safetensors", "dspark_heads_info.json"})
                    required_files.push_back(std::string("edge_llm/engine/") + name);
        }
        if (marker.value("task", "text_generation") == "vision_language_generation")
            for (const auto* name : {"visual.engine", "config.json", "preprocessor_config.json"})
                required_files.push_back(std::string("edge_llm/engine/visual/") + name);
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

/// Delegate ordinary decoding or the complete paired speculative algorithm to the pinned runtime.
std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime>
make_runtime(const Artifacts& artifacts, const nlohmann::json& marker, cudaStream_t stream) {
    if (marker.value("tensor_parallel_size", 1) == 2) {
        using Runtime = trt_edgellm::rt::LLMInferenceRuntime;
        Runtime::ParallelExecutionConfig config{};
        config.parallelConfig.tensorParallelSize = 2;
        config.parallelConfig.launchMode = trt_edgellm::rt::ParallelLaunchMode::kThread;
        config.checkpointDir = artifacts.checkpoint();
        const auto multimodal =
            marker.value("task", "text_generation") == "vision_language_generation"
                ? artifacts.engine()
                : "";
        return std::make_unique<Runtime>(artifacts.engine(), multimodal,
                                         std::unordered_map<std::string, std::string>{},
                                         std::move(config));
    }
    if (marker.value("execution_variant", "") == "autoregressive") {
        const auto multimodal =
            marker.value("task", "text_generation") == "vision_language_generation"
                ? artifacts.engine()
                : "";
        return std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
            artifacts.engine(), multimodal, std::unordered_map<std::string, std::string>{}, stream,
            trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint());
    }
    trt_edgellm::rt::SpecDecodeDraftingConfig drafting{};
    drafting.draftingTopK = 1;
    const bool mtp = marker.value("execution_variant", "") == "mtp";
    drafting.draftingStep = mtp ? 3 : 1;
    drafting.verifySize = mtp ? 4 : 8;
    drafting.dflashBlockSize = marker.value("execution_variant", "") == "dflash" ? 8 : 0;
    drafting.dsparkSchedulerMode = trt_edgellm::rt::DSparkSchedulerMode::kOff;
    drafting.dsparkConfidenceThreshold = 0.0F;
    drafting.dsparkMinProposalLen = 1;
    drafting.dsparkMaxProposalLen = 0;
    return std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
        artifacts.engine(),
        marker.value("task", "text_generation") == "vision_language_generation" ? artifacts.engine()
                                                                                : "",
        std::unordered_map<std::string, std::string>{}, drafting, stream,
        trt_edgellm::rt::ContextCacheConfig{}, artifacts.checkpoint(),
        mtp ? "" : (fs::path(artifacts.checkpoint()) / "draft").string());
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask final : public ITextGeneration,
                       public IVisionLanguageGeneration,
                       public api::IModel,
                       public api::ITextContinuation,
                       public api::IImagesTextToText,
                       public api::IVideoTextToText,
                       public api::IImageVideoTextToText {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(make_runtime(artifacts_, marker, stream_.get())), task_(bundle.info().task),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()),
          visual_(marker.value("task", "text_generation") == "vision_language_generation"),
          parallel_(marker.value("tensor_parallel_size", 1) == 2),
          paired_(marker.value("execution_variant", "") == "dspark" ||
                  marker.value("execution_variant", "") == "dflash") {
        if (!paired_)
            return;
        std::ifstream input(fs::path(artifacts_.engine()) / "trtmc_single_user_prompts.json");
        source_prompts_ = nlohmann::json::parse(input);
        for (const auto* mode : {"false", "true"})
            for (const auto* part : {"prefix", "suffix"})
                if (!source_prompts_.at(mode).at(part).is_string())
                    throw std::runtime_error("Invalid Qwen3.8 source prompt mapping");
    }

    const char* task() const noexcept override { return task_.c_str(); }

    std::vector<api::TaskInstance> task_bindings() override {
        std::vector<api::TaskInstance> bindings{
            api::bind<api::ITextContinuation>(*this, fields(false))};
        if (visual_) {
            bindings.push_back(api::bind<api::IImagesTextToText>(*this, fields(true)));
            bindings.push_back(api::bind<api::IVideoTextToText>(*this, fields(true)));
            bindings.push_back(api::bind<api::IImageVideoTextToText>(*this, fields(true)));
        }
        return bindings;
    }

    TextResult run(const api::TextContinuationRequest& input, api::ConfigView supplied) override {
        const auto* text = std::get_if<std::string_view>(&input.prefix);
        if (!text)
            throw std::invalid_argument("Qwen3.8 Edge accepts UTF-8 text prefixes only");
        return generate(std::string(*text),
                        generation_config(supplied, false, default_max_new_tokens()));
    }

    TextResult run(const api::ImagesTextToTextRequest& input, api::ConfigView supplied) override {
        if (!visual_)
            throw std::invalid_argument("Qwen3.8 text bundle has no visual engine");
        return execute(media_request(input,
                                     generation_config(supplied, true, default_max_new_tokens()),
                                     default_max_new_tokens()));
    }

    TextResult run(const api::VideoTextToTextRequest& input, api::ConfigView supplied) override {
        if (!visual_)
            throw std::invalid_argument("Qwen3.8 text bundle has no visual engine");
        return execute(media_request(input,
                                     generation_config(supplied, true, default_max_new_tokens()),
                                     default_max_new_tokens()));
    }

    TextResult run(const api::ImageVideoTextToTextRequest& input,
                   api::ConfigView supplied) override {
        if (!visual_)
            throw std::invalid_argument("Qwen3.8 text bundle has no visual engine");
        return execute(media_request(input,
                                     generation_config(supplied, true, default_max_new_tokens()),
                                     default_max_new_tokens()));
    }

    std::int32_t default_max_new_tokens() const override { return std::min(128, capacity_ - 1); }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// Invoke Edge once; failures propagate without attempting native inference.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        return generate(prompt, nullptr, 0, 0, config);
    }

    /// Preserve the original image; preprocessing and visual execution belong to Edge.
    TextResult generate(const std::string& prompt, const float* pixels, std::int32_t height,
                        std::int32_t width, const TextGenerationConfig& config) override {
        auto bytes = image_bytes(pixels, height, width);
        if (!bytes.empty() && !visual_)
            throw std::invalid_argument("Qwen3.8 text bundle does not contain a visual engine");
        auto effective = config;
        std::string source_prompt = prompt;
        if (config.use_chat_template && paired_) {
            const auto& format = source_prompts_.at(config.enable_thinking ? "true" : "false");
            source_prompt = format.at("prefix").get<std::string>() + source_user_content(prompt) +
                            format.at("suffix").get<std::string>();
            effective.use_chat_template = false;
        }
        auto request =
            make_request(source_prompt, effective, default_max_new_tokens(), !bytes.empty());
        if (!bytes.empty()) {
            trt_edgellm::rt::Tensor tensor({1, height, width, 3}, trt_edgellm::rt::DeviceType::kCPU,
                                           nvinfer1::DataType::kUINT8);
            std::memcpy(tensor.rawPointer(), bytes.data(), bytes.size());
            request.requests.front().imageBuffers.emplace_back(std::move(tensor));
        }
        return execute(std::move(request));
    }

  private:
    TextResult execute(trt_edgellm::rt::LLMGenerationRequest request) {
        std::lock_guard<std::mutex> lock(mutex_);
        const bool media = !request.requests.front().imageBuffers.empty();
        if (!media) {
            const auto counts = runtime_->countPromptTokens(request);
            if (counts.size() != 1)
                throw std::runtime_error("Qwen3.8 Edge returned invalid prompt counts");
            validate_capacity(counts.front(), input_limit_, capacity_, request.maxGenerateLength);
        }
        trt_edgellm::rt::LLMGenerationResponse response{};
        // Complete queued work before response/request storage is destroyed,
        // including exception paths in a persistent task.
        struct Drain {
            cudaStream_t stream;
            ~Drain() { cudaStreamSynchronize(stream); }
        } drain{stream_.get()};
        if (!runtime_->handleRequest(request, response, parallel_ ? nullptr : stream_.get()) ||
            response.outputIds.size() != 1 || response.outputTexts.size() != 1 ||
            response.outputIds.front().empty() ||
            response.outputIds.front().size() > static_cast<std::size_t>(request.maxGenerateLength))
            throw std::runtime_error("Qwen3.8 Edge generation failed");
        if (media) {
            if (response.inputTokenCounts.size() != 1)
                throw std::runtime_error("Qwen3.8 Edge returned invalid image prompt counts");
            validate_capacity(response.inputTokenCounts.front(), input_limit_, capacity_,
                              request.maxGenerateLength);
        }
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
    std::string task_;
    int capacity_;
    int input_limit_;
    bool visual_;
    bool parallel_;
    bool paired_;
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
    const auto parallel = marker.value("tensor_parallel_size", 1);
    const auto task = marker.value("task", "text_generation");
    const auto& bundle_task = bundle.info().task;
    if (bundle_task != task && !(task == "vision_language_generation" &&
                                 (variant == "autoregressive" || variant == "mtp") &&
                                 bundle_task == api::IImagesTextToText::kTask))
        throw std::runtime_error("Qwen3.8 Edge bundle task differs from its execution contract");
    if ((task != "text_generation" && task != "vision_language_generation") ||
        (task == "vision_language_generation" &&
         ((variant != "autoregressive" && variant != "mtp") ||
          marker.at("visual_min_image_tokens") != 128 ||
          (marker.at("visual_max_image_tokens") != 1280 &&
           marker.at("visual_max_image_tokens") != marker.at("max_input_length")) ||
          marker.at("visual_max_image_tokens_per_image") != 1280)))
        throw std::runtime_error("Unsupported Qwen3.8 Edge task contract");
    const auto quantization = marker.value("checkpoint_quantization", "");
    const auto sm = marker.at("target").at("sm").get<int>();
    const bool ordinary = variant == "autoregressive" &&
                          marker.value("builder_flow", "") == "experimental" &&
                          marker.at("max_input_length").get<int>() <= 2048 &&
                          marker.at("target").at("arch") == "x86_64" &&
                          ((parallel == 2 && quantization == "none" && sm == 80) ||
                           (parallel == 1 && quantization == "nvfp4" && sm == 120));
    const bool paired =
        valid_execution_variant(variant, marker.value(variant + "_block_size", 0)) &&
        parallel == 1 && marker.value("builder_flow", "") == "onnx" &&
        (quantization == "nvfp4" || (variant == "dflash" && quantization == "int4_awq")) &&
        marker.at("max_sequence_length").get<int>() > 8 &&
        marker.at("max_sequence_length").get<int>() <= (variant == "dspark" ? 1024 : 4096) &&
        marker.at("target").at("arch") == "x86_64" && marker.at("target").at("sm") == 120;
    const bool mtp = variant == "mtp" && parallel == 1 &&
                     (quantization == "nvfp4" || quantization == "int4_awq") && sm == 120 &&
                     marker.at("target").at("arch") == "x86_64" &&
                     marker.value("builder_flow", "") == "experimental" &&
                     marker.at("max_input_length").get<int>() <= 2048 &&
                     marker.at("max_sequence_length").get<int>() > 4 &&
                     marker.value("mtp_draft_steps", 0) == 3 &&
                     marker.value("mtp_verify_size", 0) == 4;
    if (!ordinary && !paired && !mtp)
        throw std::runtime_error("Unsupported Qwen3.8 Edge execution contract");
    validate_target(marker.at("target"), parallel);
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::qwen3_8::edge_llm
