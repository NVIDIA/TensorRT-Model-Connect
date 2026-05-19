#include "runtime/deployment/runtime_provider.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace trtmc::deployment {

namespace {

constexpr const char* kEdgeLlmProvider = "tensorrt-edge-llm";
constexpr const char* kEdgeLlmDsoName = "libtrtmc_provider_edgellm.so";

using CreateProviderPipelineFn = IPipeline* (*)(const char*, const char*);
using DestroyProviderPipelineFn = void (*)(IPipeline*);

struct ProviderDso {
    void* handle{nullptr};
    CreateProviderPipelineFn create{nullptr};
    DestroyProviderPipelineFn destroy{nullptr};
};

std::string exe_dir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
        return "";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : "";
}

std::string join_path(const std::string& dir, const std::string& file) {
    if (dir.empty())
        return file;
    if (dir.back() == '/')
        return dir + file;
    return dir + "/" + file;
}

std::vector<std::string> split_paths(const char* value) {
    std::vector<std::string> paths;
    if (value == nullptr || *value == '\0')
        return paths;
    std::string text(value);
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(':', start);
        const auto token =
            text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!token.empty())
            paths.push_back(token);
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return paths;
}

void append_dlopen_error(std::string& tried, const std::string& label) {
    const char* error = dlerror();
    tried += "  " + label + ": " + (error ? error : "unknown dlopen error") + "\n";
}

void* try_open(const std::string& path, const std::string& label, std::string& tried) {
    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
        append_dlopen_error(tried, label);
    return handle;
}

ProviderDso bind_provider_dso(void* handle, const std::string& loaded_from) {
    auto create = reinterpret_cast<CreateProviderPipelineFn>(
        dlsym(handle, "trtmc_create_deployment_provider_pipeline"));
    if (!create) {
        dlclose(handle);
        throw std::runtime_error(loaded_from + " loaded but is missing "
                                               "trtmc_create_deployment_provider_pipeline");
    }
    auto destroy = reinterpret_cast<DestroyProviderPipelineFn>(
        dlsym(handle, "trtmc_destroy_deployment_provider_pipeline"));
    if (!destroy) {
        dlclose(handle);
        throw std::runtime_error(loaded_from + " loaded but is missing "
                                               "trtmc_destroy_deployment_provider_pipeline");
    }
    return ProviderDso{handle, create, destroy};
}

ProviderDso load_edge_llm_provider_dso() {
    std::string tried;

    if (const char* exact = std::getenv("TRTMC_EDGE_LLM_PROVIDER_LIBRARY")) {
        if (*exact != '\0') {
            if (void* handle = try_open(exact, "TRTMC_EDGE_LLM_PROVIDER_LIBRARY", tried))
                return bind_provider_dso(handle, exact);
        }
    }

    const auto exe_path = exe_dir();
    if (!exe_path.empty()) {
        const auto path = join_path(exe_path, kEdgeLlmDsoName);
        if (void* handle = try_open(path, path, tried))
            return bind_provider_dso(handle, path);
    }

    for (const auto& dir : split_paths(std::getenv("TRTMC_PROVIDER_DIR"))) {
        const auto path = join_path(dir, kEdgeLlmDsoName);
        if (void* handle = try_open(path, path, tried))
            return bind_provider_dso(handle, path);
    }

    if (void* handle =
            try_open(kEdgeLlmDsoName, std::string(kEdgeLlmDsoName) + " (default)", tried)) {
        return bind_provider_dso(handle, kEdgeLlmDsoName);
    }

    throw std::runtime_error(
        "TensorRT Edge-LLM runtime provider DSO is not available.\n"
        "Could not load " +
        std::string(kEdgeLlmDsoName) + ":\n" + tried +
        "\nBuild Model-Connect with -DTRTMC_ENABLE_EDGE_LLM_PROVIDER=ON and place " +
        kEdgeLlmDsoName +
        " next to the trtmc binary, in TRTMC_PROVIDER_DIR, in LD_LIBRARY_PATH, or set "
        "TRTMC_EDGE_LLM_PROVIDER_LIBRARY to its full path.");
}

class ProviderDsoPipeline final : public IPipeline {
  public:
    ProviderDsoPipeline(ProviderDso dso, IPipeline* pipeline)
        : dso_(std::move(dso)), pipeline_(pipeline) {}

    ~ProviderDsoPipeline() override {
        if (pipeline_ != nullptr) {
            dso_.destroy(pipeline_);
            pipeline_ = nullptr;
        }
        if (dso_.handle != nullptr) {
            dlclose(dso_.handle);
            dso_.handle = nullptr;
        }
    }

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg) override {
        return pipeline_->generate(prompt, cfg);
    }

    const char* model_id() const override { return pipeline_->model_id(); }

    const char* pipeline_type() const override { return pipeline_->pipeline_type(); }

  private:
    ProviderDso dso_;
    IPipeline* pipeline_{nullptr};
};

std::unique_ptr<IPipeline> create_edge_llm_pipeline(const std::filesystem::path& engine_dir,
                                                    const std::string& bundle_path) {
    auto dso = load_edge_llm_provider_dso();
    IPipeline* pipeline = nullptr;
    try {
        pipeline = dso.create(engine_dir.string().c_str(), bundle_path.c_str());
    } catch (...) {
        dlclose(dso.handle);
        throw;
    }
    if (pipeline == nullptr) {
        dlclose(dso.handle);
        throw std::runtime_error("TensorRT Edge-LLM provider returned null from "
                                 "trtmc_create_deployment_provider_pipeline");
    }
    return std::make_unique<ProviderDsoPipeline>(std::move(dso), pipeline);
}

} // namespace

std::unique_ptr<IPipeline> load_runtime_provider(const Variant& variant,
                                                 const ArtifactStore& artifacts,
                                                 const std::string& bundle_path) {
    if (variant.provider != kEdgeLlmProvider) {
        throw std::runtime_error("No runtime provider registered for deployment provider: " +
                                 variant.provider);
    }

    std::string engine_prefix;
    for (const auto& artifact : variant.artifacts) {
        if (artifact.name == "engine_dir" && artifact.kind == "directory") {
            engine_prefix = artifact.section_prefix;
            break;
        }
    }
    if (engine_prefix.empty()) {
        throw std::runtime_error("TensorRT Edge-LLM variant is missing engine_dir artifact");
    }
    const auto engine_dir = artifacts.materialize_directory(engine_prefix);

    return create_edge_llm_pipeline(engine_dir, bundle_path);
}

} // namespace trtmc::deployment
