/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lerobot_pi05/runtime/edge_llm/adapter.h"

#include "families/lerobot_pi05/runtime/openpi.h"
#include "families/lerobot_pi05/runtime/request.h"
#include "trtmc/internal/action.h"
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
#include <tensorrt_edgellm/pi05/runtime/pi05Policy.h>
#include <tensorrt_edgellm/pi05/runtime/pi05Runtime.h>
#include <utility>
#include <vector>

namespace trtmc::lerobot_pi05::edge_llm {
namespace {
namespace fs = std::filesystem;

/// Turn CUDA failures into caller-visible load or inference errors.
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Pi0.5 Edge CUDA error: ") +
                                 cudaGetErrorString(result));
}

/// Reject an engine built for a different local GPU or CUDA/TensorRT runtime.
void validate_target(const nlohmann::json& target) {
    utsname host{};
    if (uname(&host) != 0)
        throw std::runtime_error("Cannot identify Pi0.5 Edge runtime host");
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
            "Pi0.5 Edge bundle requires its build GPU and CUDA/TensorRT stack");
}

const std::set<std::string> kAssets{
    "action/action.engine",
    "action/config.json",
    "assets/norm_stats.json",
    "cond/cond.engine",
    "cond/config.json",
    "embed_tokens.safetensors",
    "policy.json",
    "prefix/config.json",
    "prefix/prefix.engine",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "visual/config.json",
    "visual/visual.engine",
};

/// Own extracted engine/checkpoint files until after the Edge runtime is destroyed.
class Artifacts {
  public:
    explicit Artifacts(const BundleReader& bundle, const nlohmann::json& marker) {
        const auto names = marker.at("artifacts").get<std::vector<std::string>>();
        if (names.size() != kAssets.size() ||
            std::set<std::string>(names.begin(), names.end()) != kAssets)
            throw std::runtime_error("Unexpected Pi0.5 Edge asset manifest");
        for (const auto& name : kAssets) {
            const auto* section = bundle.find_section("edge_llm/engine/" + name);
            if (!section || !section->length)
                throw std::runtime_error("Missing Pi0.5 Edge asset: " + name);
        }
        std::string pattern = (fs::temp_directory_path() / "trtmc-pi05-edge-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            throw std::runtime_error("Cannot create Pi0.5 Edge artifact directory");
        root_ = pattern;
        try {
            for (const auto& name : names) {
                const auto destination = root_ / name;
                fs::create_directories(destination.parent_path());
                std::ofstream output(destination, std::ios::binary);
                bundle.copy_section("edge_llm/engine/" + name, output);
                output.close();
                if (!output)
                    throw std::runtime_error("Cannot extract Pi0.5 Edge artifact: " + name);
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
        throw std::runtime_error("Cannot locate Pi0.5 family library");
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
        throw std::runtime_error("Cannot initialize Pi0.5 Edge plugin");
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
        const auto& c = runtime_.contract();
        const bool libero = c.policyConfig == "pi05_libero", droid = c.policyConfig == "pi05_droid";
        if (marker.at("policy") != c.policyConfig ||
            (!libero && !droid && c.policyConfig != "pi05_aloha") || c.modelActionDim != 32 ||
            c.actionHorizon != (libero  ? 10
                                : droid ? 15
                                        : 50) ||
            c.robotActionDim != (libero  ? 7
                                 : droid ? 8
                                         : 14) ||
            c.stateDim != (libero || droid ? 8 : 14) || c.discreteStateInput != !libero ||
            c.maxTokenLen != 200 || c.imageWidth != 224 || c.imageHeight != 224)
            throw std::runtime_error(
                "Pi0.5 Edge contract does not match the requested openpi policy");
    }
    const char* task() const noexcept override {
        return internal::IImagesStateTextToActionChunk::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImagesStateTextToActionChunk>(
            *this, Span<const internal::ConfigField>{kFields, std::size(kFields)})};
    }
    internal::ImagesStateTextActionChunkResult
    run(const internal::ImagesStateTextToActionChunkRequest& request,
        internal::ConfigView config) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto start = std::chrono::steady_clock::now();
        const auto& observation = request.observation;
        const auto& c = runtime_.contract();
        const auto size = static_cast<std::size_t>(c.actionHorizon) * c.modelActionDim;
        const auto options = parse_options(config, size);
        if (observation.text.empty() ||
            observation.state.size() != static_cast<std::size_t>(c.stateDim))
            throw std::invalid_argument("Pi0.5 needs a nonempty task and the policy's state width");
        const auto finite = [](float v) { return std::isfinite(v); };
        if (!std::all_of(observation.state.begin(), observation.state.end(), finite))
            throw std::invalid_argument("Pi0.5 state must be finite");
        trt_edgellm::pi05::Pi05Observation input;
        input.state.assign(observation.state.begin(), observation.state.end());
        input.task.assign(observation.text);
        std::vector<std::vector<std::uint8_t>> owned;
        owned.reserve(observation.images.size());
        for (const auto& camera : observation.images) {
            if (camera.name.empty())
                throw std::invalid_argument("Pi0.5 requires named cameras");
            if (std::find(c.ignoredCameras.begin(), c.ignoredCameras.end(), camera.name) !=
                c.ignoredCameras.end())
                continue;
            const auto* rgb = openpi_rgb(camera.image, owned.emplace_back());
            input.cameras.push_back({std::string(camera.name), "", rgb,
                                     static_cast<int>(camera.image.height),
                                     static_cast<int>(camera.image.width)});
        }
        runtime_.runtime().setInitialNoise(initial_noise(options, size, rng_));
        auto chunk = runtime_.infer(input);
        if (chunk.batch != 1 || chunk.horizon != c.actionHorizon ||
            chunk.modelActionDim != c.modelActionDim ||
            chunk.robotActions.size() !=
                static_cast<std::size_t>(c.actionHorizon) * c.robotActionDim ||
            chunk.normalizedActions.size() != size ||
            !std::all_of(chunk.robotActions.begin(), chunk.robotActions.end(), finite) ||
            !std::all_of(chunk.normalizedActions.begin(), chunk.normalizedActions.end(), finite))
            throw std::runtime_error("Pi0.5 Edge returned an invalid action chunk");
        internal::ImagesStateTextActionChunkResult result;
        result.within_training_bounds = false; // openpi publishes quantiles, not dataset extrema.
        result.actions.values.rows = c.actionHorizon;
        result.actions.values.columns = c.robotActionDim;
        result.actions.values.values = std::move(chunk.robotActions);
        result.actions.schema = {"lerobot.pi05", {}, {}, "", "unnormalized"};
        result.inference_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        return result;
    }

  private:
    Artifacts artifacts_;
    std::unique_ptr<void, CloseLibrary> plugin_;
    Stream stream_;
    trt_edgellm::pi05::Pi05Policy runtime_;
    std::mt19937_64 rng_{std::random_device{}()};
    std::mutex mutex_;
};
} // namespace

ITask* create(const BundleReader& bundle) {
    const auto raw = bundle.read_section("edge_llm.json");
    auto marker = nlohmann::json::parse(raw.begin(), raw.end());
    if (marker.at("version") != 1 || marker.at("edge_release") != "0.12.0" ||
        marker.at("precision") != "fp16" ||
        (!marker.at("edge_revision").is_null() &&
         marker.at("edge_revision") != "513dcd18291e587c80928911a480761352f3c2d8"))
        throw std::runtime_error("Unsupported Pi0.5 Edge bundle");
    validate_target(marker.at("target"));
    return new EdgeTask(bundle, marker);
}
} // namespace trtmc::lerobot_pi05::edge_llm
