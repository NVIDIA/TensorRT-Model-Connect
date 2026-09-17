/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/nemotron_h/runtime/edge_llm/adapter.h"

#include "families/nemotron_h/runtime/edge_llm/request.h"
#include "families/nemotron_h/runtime/edge_llm/tokenizer.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <edgellm/cpp/common/logger.h>
#include <edgellm/cpp/runtime/llmInferenceRuntime.h>
#include <edgellm/cpp/runtime/modelArtifacts.h>
#include <edgellm/cpp/runtime/streaming.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <stdexcept>
#include <sys/utsname.h>
#include <vector>

namespace trtmc::nemotron_h::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Nemotron-H Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Nemotron-H Edge runtime host");
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
            "Nemotron-H Edge bundle requires its build GPU and CUDA/TensorRT stack");
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
                throw std::runtime_error("Invalid Nemotron-H Edge artifact: " + name);
        }
        const bool paired = marker.value("execution_variant", "autoregressive") == "dflash";
        const std::vector<std::string> plans =
            paired ? std::vector<std::string>{"spec_base.engine", "spec_draft.engine",
                                              "base_config.json", "draft_config.json",
                                              "embedding.safetensors"}
                   : std::vector<std::string>{"llm.engine", "config.json"};
        for (const auto& plan : plans) {
            const auto required = "edge_llm/engine/" + plan;
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error("Required Nemotron-H Edge artifact missing: " + required);
        }
        for (const auto* required :
             {"edge_llm/engine/tokenizer.json", "edge_llm/engine/tokenizer_config.json",
              "edge_llm/engine/processed_chat_template.json", "edge_llm/checkpoint/config.json",
              "edge_llm/runtime_tokenizer/tokenizer.json",
              "edge_llm/runtime_tokenizer/tokenizer_config.json",
              "edge_llm/runtime_tokenizer/processed_chat_template.json"})
            if (!names.count(required) || bundle.find_section(required)->length == 0)
                throw std::runtime_error(
                    std::string("Required Nemotron-H Edge artifact missing: ") + required);
        std::string pattern = (fs::temp_directory_path() / "trtmc-nemotron-h-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Nemotron-H Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section(name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Nemotron-H Edge artifact: " + name);
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
    std::string tokenizer() const { return (root_ / "edge_llm/runtime_tokenizer").string(); }
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
        throw std::runtime_error("Cannot locate Nemotron-H family library");
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
        throw std::runtime_error("Cannot initialize Nemotron-H Edge plugin");
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

/// Drain all queued work before request storage destruction or mutex release.
class RequestDrain {
  public:
    explicit RequestDrain(cudaStream_t stream) : stream_(stream) {}
    ~RequestDrain() noexcept { cudaStreamSynchronize(stream_); }
    void checked() { check_cuda(cudaStreamSynchronize(stream_)); }

  private:
    cudaStream_t stream_;
};

/// The paired engine ABI fixes a linear block16 speculative schedule.
std::optional<trt_edgellm::rt::SpecDecodeDraftingConfig>
drafting_config(const nlohmann::json& marker) {
    if (marker.value("execution_variant", "autoregressive") != "dflash")
        return std::nullopt;
    trt_edgellm::rt::SpecDecodeDraftingConfig config{};
    config.draftingTopK = 1;
    config.draftingStep = 1;
    config.verifySize = 16;
    config.dflashBlockSize = 16;
    return config;
}

/// Use public artifact injection; its coordinator retains this supplied tokenizer.
trt_edgellm::rt::ModelArtifacts
runtime_artifacts(const Artifacts& files, const nlohmann::json& marker, cudaStream_t stream) {
    auto artifacts = trt_edgellm::rt::ModelArtifacts::loadFromEngineDir(
        files.engine(), drafting_config(marker), files.checkpoint(), "", stream);
    artifacts.tokenizer = load_tokenizer(files.tokenizer(),
                                         marker.at("native_eos_token_ids").get<std::vector<int>>());
    artifacts.deployment.base.eosTokenIds =
        marker.at("native_eos_token_ids").get<std::vector<int>>();
    return artifacts;
}

/// Read the original source template resolved into family-owned runtime metadata.
std::string native_chat_format(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm/runtime_tokenizer/tokenizer_config.json");
    const auto metadata = nlohmann::json::parse(bytes.begin(), bytes.end());
    const auto text = metadata.at("chat_template").get<std::string>();
    if (text.empty())
        throw std::runtime_error("Nemotron-H Edge requires resolved source chat_template");
    return nemotron_h_detect_chat_template_format(text);
}

/// Reuse the original family BPE implementation, including its special postprocessor.
std::unique_ptr<ITokenizer> native_tokenizer(const BundleReader& bundle) {
    const auto bytes = bundle.read_section("edge_llm/checkpoint/tokenizer.json");
    return CreateBpeTokenizer(reinterpret_cast<const char*>(bytes.data()), bytes.size(), true);
}

/// Thin persistent Edge API adapter; serialization prevents concurrent use of Edge request state.
class EdgeTask final : public ITextGeneration {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), native_tokenizer_(native_tokenizer(bundle)),
          plugin_(load_plugin()),
          runtime_(runtime_artifacts(artifacts_, marker, stream_.get()), artifacts_.engine(), "",
                   std::unordered_map<std::string, std::string>{}, drafting_config(marker),
                   stream_.get()),
          chat_format_(native_chat_format(bundle)),
          capacity_(marker.at("max_sequence_length").get<int>()),
          input_limit_(marker.at("max_input_length").get<int>()),
          bos_id_(marker.at("native_bos_token_id").get<int>()),
          paired_(marker.value("execution_variant", "autoregressive") == "dflash") {}

    std::int32_t default_max_new_tokens() const override { return kDefaultMaxNewTokens; }

    /// Drain work from failed requests before destroying the runtime and its weight buffers.
    ~EdgeTask() override { cudaStreamSynchronize(stream_.get()); }

    /// Invoke Edge once; failures propagate without attempting native inference.
    TextResult generate(const std::string& prompt, const TextGenerationConfig& config) override {
        auto request = make_request(prompt, config, chat_format_, *native_tokenizer_, bos_id_);
        // The pinned DFlash decoder forces greedy verification. Reject sampling
        // rather than silently replacing caller-requested stochastic semantics.
        if (paired_ && request.temperature != 0.0F)
            throw std::invalid_argument("Nemotron-H DFlash supports greedy generation only");
        const auto count = request.preTokenizedInputIds.front().size();
        if (count == 0)
            return {};
        if (count > static_cast<std::size_t>(input_limit_))
            throw std::invalid_argument("Nemotron-H Edge input exceeds bundle capacity");
        std::lock_guard<std::mutex> lock(mutex_);
        RequestDrain drain(stream_.get());
        validate_capacity(static_cast<int>(count), input_limit_, capacity_,
                          request.maxGenerateLength);
        trt_edgellm::rt::LLMGenerationResponse response{};
        if (!runtime_.handleRequest(request, response, stream_.get()))
            throw std::runtime_error("Nemotron-H Edge generation failed");
        drain.checked();
        validate_response(response, input_limit_, capacity_, request.maxGenerateLength,
                          static_cast<int>(count));
        // This API does not expose per-request stage times; zero means unavailable.
        return {native_tokenizer_->decode(response.outputIds.front()),
                std::move(response.outputIds.front())};
    }

  private:
    // Reverse destruction order keeps weights, plugin and stream alive throughout Edge teardown.
    Artifacts artifacts_;
    std::unique_ptr<ITokenizer> native_tokenizer_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    trt_edgellm::rt::LLMInferenceRuntime runtime_;
    std::string chat_format_;
    int capacity_;
    int input_limit_;
    int bos_id_;
    bool paired_;
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
        !marker.at("artifacts").is_array() || marker.at("tokenizer_policy") != "native_full_eos" ||
        !marker.at("native_bos_token_id").is_number_integer() ||
        !marker.at("native_eos_token_ids").is_array() || marker.at("native_eos_token_ids").empty())
        throw std::runtime_error("Invalid Nemotron-H Edge bundle contract");
    const auto variant = marker.value("execution_variant", "autoregressive");
    if ((variant != "autoregressive" && variant != "dflash") ||
        (variant == "dflash" && (marker.value("builder_flow", "") != "onnx" ||
                                 marker.value("dflash_block_size", 0) != 16)))
        throw std::runtime_error("Invalid Nemotron-H Edge execution variant");
    std::set<int> stops;
    for (const auto& id : marker.at("native_eos_token_ids")) {
        if (!id.is_number_integer() || id.get<int>() < 0 || !stops.insert(id.get<int>()).second)
            throw std::runtime_error("Invalid Nemotron-H Edge full EOS set");
    }
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}

} // namespace trtmc::nemotron_h::edge_llm
