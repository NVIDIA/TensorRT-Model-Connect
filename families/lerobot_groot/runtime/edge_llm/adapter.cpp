/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lerobot_groot/runtime/edge_llm/adapter.h"

#include "families/lerobot_groot/runtime/request.h"
#include "trtmc/internal/model.h"

#include <NvInferRuntime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <tensorrt_edgellm/gr00t_n1d7/runtime/gr00tRuntime.h>
#include <utility>
#include <vector>

namespace trtmc::lerobot_groot::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("GR00T Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify GR00T Edge runtime host");
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
            "GR00T Edge bundle requires its build GPU and CUDA/TensorRT stack");
}

const std::set<std::string> kAssets{
    "action/action.engine",
    "action/action_time_bias.safetensors",
    "action/adanorm_modulation.safetensors",
    "action/config.json",
    "action/policy.json",
    "embed_tokens.safetensors",
    "tokenizer/chat_template.jinja",
    "tokenizer/preprocessor_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "visual/config.json",
    "visual/preprocessor_config.json",
    "visual/processor_config.json",
    "visual/visual.engine",
    "vl_backbone/config.json",
    "vl_backbone/vl_backbone.engine",
};

/// Own extracted engine/checkpoint files until after the Edge runtime is destroyed.
class Artifacts {
  public:
    explicit Artifacts(const BundleReader& bundle, const nlohmann::json& marker) {
        const auto names = marker.at("artifacts").get<std::vector<std::string>>();
        if (names.size() != kAssets.size() ||
            std::set<std::string>(names.begin(), names.end()) != kAssets)
            throw std::runtime_error("Unexpected GR00T Edge asset manifest");
        for (const auto& name : kAssets) {
            const auto* section = bundle.find_section("edge_llm/engine/" + name);
            if (!section || !section->length)
                throw std::runtime_error("Missing GR00T Edge asset: " + name);
        }
        std::string pattern = (fs::temp_directory_path() / "trtmc-groot-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create GR00T Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                if (name == "action/policy.json") {
                    // MC validates and formalizes the instruction. A second pass in Edge
                    // would lowercase the verbatim default for punctuation-only prompts.
                    const auto raw = bundle.read_section("edge_llm/engine/" + name);
                    auto policy = nlohmann::json::parse(raw.begin(), raw.end());
                    auto& vision = policy.at("vision");
                    if (!vision.value("formalize_language", true))
                        throw std::runtime_error("GR00T LIBERO expects language formalization");
                    vision["formalize_language"] = false;
                    output << policy;
                } else {
                    bundle.copy_section("edge_llm/engine/" + name, output);
                }
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract GR00T Edge artifact: " + name);
            }
        } catch (...) {
            cleanup();
            throw;
        }
    }
    ~Artifacts() { cleanup(); }
    Artifacts(const Artifacts&) = delete;
    Artifacts& operator=(const Artifacts&) = delete;
    std::string engine() const { return root_.string(); }

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
        throw std::runtime_error("Cannot locate GR00T family library");
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
        throw std::runtime_error("Cannot initialize GR00T Edge plugin");
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

class EdgeTask final : public internal::IModel, public internal::IImagesStateTextToActionChunk {
  public:
    EdgeTask(const BundleReader& bundle, const nlohmann::json& marker)
        : artifacts_(bundle, marker), plugin_(load_plugin()),
          runtime_(artifacts_.engine(), stream_.get()) {
        const auto vocab = bundle.read_section("edge_llm/engine/tokenizer/tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(vocab.data(), vocab.size());
        if (runtime_.policy().embodimentTag() != "libero_sim" ||
            runtime_.policy().videoKeys() != cameras_ || runtime_.policy().horizon() != 16)
            throw std::runtime_error("GR00T Edge bundle is not the supported LIBERO policy");
    }
    const char* task() const noexcept override {
        return internal::IImagesStateTextToActionChunk::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImagesStateTextToActionChunk>(*this, action_fields())};
    }
    internal::ImagesStateTextActionChunkResult
    run(const internal::ImagesStateTextToActionChunkRequest& request,
        internal::ConfigView config) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto start = std::chrono::steady_clock::now();
        const auto& observation = request.observation;
        const auto options = parse_noise(config, 40 * 132);
        const auto frames = ordered_rgb(observation, cameras_, 256);
        validate_state(observation.state, 8);
        trt_edgellm::gr00t::Gr00tObservation input;
        instruction_tokens(observation.text, true, *tokenizer_, 256, &input.task);
        input.state.assign(observation.state.begin(), observation.state.end());
        for (std::size_t i = 0; i < frames.size(); ++i)
            input.cameras.push_back({cameras_[i], "", frames[i].data(), 256, 256});
        // Edge retains explicit noise, so replace it on every request, including unseeded calls.
        runtime_.runner().setInitialNoise(initial_noise(options, 40 * 132, rng_));
        auto chunk = runtime_.infer(input);
        if (chunk.horizon != 16 || chunk.robotActionDim != 7 ||
            chunk.robotActions.size() != 16 * 7 || chunk.modelHorizon != 40 ||
            chunk.modelActionDim != 132 || chunk.actions.size() != 40 * 132)
            throw std::runtime_error("GR00T Edge returned an invalid action shape");
        const auto finite = [](float value) { return std::isfinite(value); };
        if (!std::all_of(chunk.robotActions.begin(), chunk.robotActions.end(), finite) ||
            !std::all_of(chunk.actions.begin(), chunk.actions.end(), finite))
            throw std::runtime_error("GR00T Edge returned a non-finite action");
        internal::ImagesStateTextActionChunkResult result;
        result.within_training_bounds = true;
        for (int row = 0; row < chunk.horizon; ++row)
            for (int col = 0; col < chunk.robotActionDim; ++col)
                if (std::abs(chunk.actions[row * chunk.modelActionDim + col]) > 1.0F)
                    result.within_training_bounds = false;
        result.actions.values.rows = chunk.horizon;
        result.actions.values.columns = chunk.robotActionDim;
        result.actions.values.values = std::move(chunk.robotActions);
        result.actions.schema = {"lerobot.groot", {}, {}, "", "unnormalized"};
        result.inference_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        return result;
    }

  private:
    const std::vector<std::string> cameras_{"image", "wrist_image"};
    Artifacts artifacts_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    trt_edgellm::gr00t::Gr00tRuntime runtime_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::mt19937_64 rng_{std::random_device{}()};
    std::mutex mutex_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto raw = bundle.read_section("edge_llm.json");
    auto marker = nlohmann::json::parse(raw.begin(), raw.end());
    if (marker.at("version") != 1 || marker.at("edge_release") != "0.12.0" ||
        marker.at("precision") != "fp16" || marker.at("policy") != "libero_sim" ||
        (!marker.at("edge_revision").is_null() &&
         marker.at("edge_revision") != "513dcd18291e587c80928911a480761352f3c2d8"))
        throw std::runtime_error("Unsupported GR00T Edge bundle");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}
} // namespace trtmc::lerobot_groot::edge_llm
