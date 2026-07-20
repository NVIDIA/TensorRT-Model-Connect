/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/wan2_2_ti2v/artifact_contract.h"
#include "runtime/models/wan2_2_ti2v/pipeline.h"
#include "runtime/models/wan2_2_ti2v/plugin_contract.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/sha256.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <linux/memfd.h>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using StringExport = const char* (*)();
using IntExport = int (*)();

struct Wan22AotPluginState {
    void* handle{nullptr};
    int backing_fd{-1};
    std::string binary_sha256;
    std::string load_error;
    std::string dependency_runtime_abi;
    std::vector<void*> dependency_handles;
    wan2_2_ti2v::PluginContract contract;
    std::string loaded_runtime_abi;
};

constexpr const char* kWan22PluginSection = "wan2_2_ti2v_plugins.so";

std::mutex& aot_plugin_mutex() {
    static std::mutex mutex;
    return mutex;
}

Wan22AotPluginState& aot_plugin_state() {
    static Wan22AotPluginState state;
    return state;
}

void append_path_list(std::vector<std::filesystem::path>& paths, const char* value) {
    if (value == nullptr)
        return;
    std::string list(value);
    std::size_t begin = 0;
    while (begin <= list.size()) {
        const auto end = list.find(':', begin);
        const auto item = list.substr(begin, end - begin);
        if (!item.empty())
            paths.emplace_back(item);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
}

void append_python_native_directories(std::vector<std::filesystem::path>& paths,
                                      const std::filesystem::path& prefix) {
    std::error_code error;
    const auto lib = prefix / "lib";
    if (!std::filesystem::is_directory(lib, error))
        return;
    for (std::filesystem::directory_iterator iterator(lib, error), end; !error && iterator != end;
         iterator.increment(error)) {
        if (!iterator->is_directory(error) ||
            iterator->path().filename().string().rfind("python", 0) != 0) {
            continue;
        }
        const auto site = iterator->path() / "site-packages";
        paths.push_back(site / "tensorrt_libs");
        paths.push_back(site / "nvidia" / "cudnn" / "lib");
        paths.push_back(site / "nvidia" / "cublas" / "lib");
        paths.push_back(site / "nvidia" / "cuda_runtime" / "lib");
        paths.push_back(site / "nvidia" / "cuda_nvrtc" / "lib");
    }
}

std::set<std::filesystem::path> discover_python_prefixes() {
    std::set<std::filesystem::path> prefixes;
    for (const char* variable : {"VIRTUAL_ENV", "CONDA_PREFIX"}) {
        if (const char* value = std::getenv(variable); value != nullptr && value[0] != '\0')
            prefixes.emplace(value);
    }
    std::vector<std::filesystem::path> executable_paths;
    append_path_list(executable_paths, std::getenv("PATH"));
    for (const auto& path : executable_paths) {
        if (path.filename() == "bin")
            prefixes.insert(path.parent_path());
    }
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error && executable.parent_path().filename() == "bin")
        prefixes.insert(executable.parent_path().parent_path());
    return prefixes;
}

std::vector<std::filesystem::path> packaged_dependency_directories() {
    std::vector<std::filesystem::path> packaged;
    const auto prefixes = discover_python_prefixes();
    for (const auto& prefix : prefixes)
        append_python_native_directories(packaged, prefix);
    return packaged;
}

std::vector<std::filesystem::path> system_dependency_directories() {
    std::vector<std::filesystem::path> system{
        "/usr/local/cuda/lib64",  "/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
        "/lib/aarch64-linux-gnu", "/lib/x86_64-linux-gnu",
    };
    const std::filesystem::path local("/usr/local");
    std::error_code error;
    for (std::filesystem::directory_iterator iterator(local, error), end; !error && iterator != end;
         iterator.increment(error)) {
        if (!iterator->is_directory(error) ||
            iterator->path().filename().string().rfind("cuda", 0) != 0) {
            continue;
        }
        const auto targets = iterator->path() / "targets";
        std::error_code targets_error;
        for (std::filesystem::directory_iterator target(targets, targets_error), target_end;
             !targets_error && target != target_end; target.increment(targets_error)) {
            system.push_back(target->path() / "lib");
        }
    }
    return system;
}

std::vector<std::vector<std::filesystem::path>> dependency_search_tiers() {
    std::vector<std::filesystem::path> configured;
    append_path_list(configured, std::getenv("LD_LIBRARY_PATH"));
    return {std::move(configured), packaged_dependency_directories(),
            system_dependency_directories()};
}

std::optional<std::filesystem::path> resolve_dependency_path(const std::string& soname) {
    for (const auto& tier : dependency_search_tiers()) {
        std::set<std::filesystem::path> matches;
        for (const auto& directory : tier) {
            std::error_code error;
            const auto candidate = directory / soname;
            if (!std::filesystem::is_regular_file(candidate, error))
                continue;
            auto resolved = std::filesystem::canonical(candidate, error);
            matches.insert(error ? candidate : std::move(resolved));
        }
        if (matches.size() == 1)
            return *matches.begin();
        if (matches.size() > 1) {
            std::ostringstream message;
            message << "Wan2.2 dependency " << soname << " is ambiguous in one search tier:";
            for (const auto& path : matches)
                message << ' ' << path;
            throw std::runtime_error(message.str());
        }
    }
    return std::nullopt;
}

void* load_dependency(const std::string& soname) {
    dlerror();
    if (void* handle = dlopen(soname.c_str(), RTLD_NOW | RTLD_GLOBAL); handle != nullptr)
        return handle;
    const char* first_error = dlerror();
    const std::string bare_error = first_error != nullptr ? first_error : "unknown dlopen error";
    const auto path = resolve_dependency_path(soname);
    if (!path) {
        throw std::runtime_error("Unable to resolve Wan2.2 ABI dependency " + soname + ": " +
                                 bare_error);
    }
    dlerror();
    if (void* handle = dlopen(path->c_str(), RTLD_NOW | RTLD_GLOBAL); handle != nullptr)
        return handle;
    const char* path_error = dlerror();
    throw std::runtime_error("Unable to preload Wan2.2 ABI dependency " + soname + " from " +
                             path->string() + ": " +
                             (path_error != nullptr ? path_error : "unknown dlopen error"));
}

void preload_wan22_dependencies(const wan2_2_ti2v::PluginRuntimeAbi& abi,
                                Wan22AotPluginState& state) {
    const auto runtime_abi = wan2_2_ti2v::canonical_runtime_abi(abi);
    if (!state.dependency_runtime_abi.empty()) {
        if (state.dependency_runtime_abi != runtime_abi) {
            throw std::runtime_error("Wan2.2 dependency ABI conflict in this process: loaded=" +
                                     state.dependency_runtime_abi + ", requested=" + runtime_abi);
        }
        if (state.dependency_handles.size() == 5)
            return;
    } else {
        // A partial preload changes process-global CUDA/TRT state even if a
        // later dependency is missing. Pin retries to this exact ABI.
        state.dependency_runtime_abi = runtime_abi;
    }

    const std::vector<std::string> dependencies{
        "libcudart.so." + std::to_string(abi.cuda_major),
        "libcublasLt.so." + std::to_string(abi.cuda_major),
        "libnvrtc.so." + std::to_string(abi.cuda_major),
        "libcudnn.so." + std::to_string(abi.cudnn_major),
        "libnvinfer.so." + std::to_string(abi.tensorrt_major),
    };
    for (std::size_t index = state.dependency_handles.size(); index < dependencies.size();
         ++index) {
        state.dependency_handles.push_back(load_dependency(dependencies[index]));
    }
}

std::string require_export_value(void* handle, const char* symbol) {
    dlerror();
    const auto function = reinterpret_cast<StringExport>(dlsym(handle, symbol));
    const char* error = dlerror();
    if (error != nullptr || function == nullptr) {
        throw std::runtime_error(std::string("Wan2.2 AOT plugin companion is missing export ") +
                                 symbol + (error != nullptr ? std::string(": ") + error : ""));
    }
    const char* value = function();
    if (value == nullptr || value[0] == '\0') {
        throw std::runtime_error(std::string("Wan2.2 AOT plugin companion returned an empty ") +
                                 symbol);
    }
    return value;
}

int require_int_export_value(void* handle, const char* symbol) {
    dlerror();
    const auto function = reinterpret_cast<IntExport>(dlsym(handle, symbol));
    const char* error = dlerror();
    if (error != nullptr || function == nullptr) {
        throw std::runtime_error(std::string("Wan2.2 AOT plugin companion is missing export ") +
                                 symbol + (error != nullptr ? std::string(": ") + error : ""));
    }
    return function();
}

std::string sha256_bytes(const std::vector<char>& bytes) {
    detail::Sha256 digest;
    digest.update(bytes.data(), bytes.size());
    return digest.hex_digest();
}

int create_sealed_plugin_memfd(const std::vector<char>& bytes) {
    if (bytes.empty())
        throw std::runtime_error("Wan2.2 embedded AOT plugin section is empty");

#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif
    unsigned int flags = MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_EXEC;
    int fd = static_cast<int>(syscall(SYS_memfd_create, "trtmc-wan2-2-ti2v-plugins", flags));
    if (fd < 0 && errno == EINVAL) {
        // Kernels older than Linux 6.3 do not know MFD_EXEC. Their default
        // memfd policy permits executable mappings, so retry without it.
        flags = MFD_CLOEXEC | MFD_ALLOW_SEALING;
        fd = static_cast<int>(syscall(SYS_memfd_create, "trtmc-wan2-2-ti2v-plugins", flags));
    }
    if (fd < 0) {
        throw std::runtime_error(std::string("Wan2.2 could not create an in-memory plugin file: ") +
                                 std::strerror(errno));
    }

    std::size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t result =
            ::write(fd, bytes.data() + written, static_cast<size_t>(bytes.size() - written));
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0) {
            const int saved_errno = errno;
            ::close(fd);
            throw std::runtime_error(std::string("Wan2.2 could not materialize its embedded AOT "
                                                 "plugin: ") +
                                     std::strerror(saved_errno));
        }
        written += static_cast<std::size_t>(result);
    }

    const int seals = F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE;
    if (fcntl(fd, F_ADD_SEALS, seals) != 0) {
        const int saved_errno = errno;
        ::close(fd);
        throw std::runtime_error(std::string("Wan2.2 could not seal its embedded AOT plugin: ") +
                                 std::strerror(saved_errno));
    }
    return fd;
}

int32_t current_cuda_architecture() {
    int device = 0;
    cudaDeviceProp properties{};
    const auto device_status = cudaGetDevice(&device);
    if (device_status != cudaSuccess) {
        throw std::runtime_error(std::string("Wan2.2 could not query the active CUDA device: ") +
                                 cudaGetErrorString(device_status));
    }
    const auto properties_status = cudaGetDeviceProperties(&properties, device);
    if (properties_status != cudaSuccess) {
        throw std::runtime_error(std::string("Wan2.2 could not query CUDA device properties: ") +
                                 cudaGetErrorString(properties_status));
    }
    return properties.major * 10 + properties.minor;
}

void validate_companion_exports(void* handle, const wan2_2_ti2v::PluginContract& installed,
                                const std::string& runtime_abi) {
    const int search_path_state =
        require_int_export_value(handle, "trtmc_wan22_plugin_runtime_search_path_state");
    if (search_path_state != 0) {
        throw std::runtime_error(
            "Wan2.2 AOT plugin companion contains DT_RPATH/DT_RUNPATH or could not prove "
            "their absence");
    }
    const auto semantic_abi = require_export_value(handle, "trtmc_wan22_plugin_semantic_abi");
    const auto source_digest = require_export_value(handle, "trtmc_wan22_plugin_source_digest");
    const auto creator_set = require_export_value(handle, "trtmc_wan22_plugin_creator_set");
    if (semantic_abi != installed.semantic_abi || source_digest != installed.source_digest ||
        creator_set != installed.creator_set) {
        throw std::runtime_error(
            "Wan2.2 AOT plugin companion manifest disagrees with its exported fingerprint");
    }
    if (runtime_abi != wan2_2_ti2v::canonical_runtime_abi(installed.runtime_abi)) {
        throw std::runtime_error(
            "Wan2.2 AOT plugin companion was loaded against an incompatible TRT/CUDA/cuDNN ABI");
    }
}

void load_and_validate_wan22_aot_plugin(const PipelineContext& ctx,
                                        const std::string& expected_binary_sha256) {
    // A Wan .trtfb is trusted executable content: the exact AOT library is
    // authenticated against its manifest before these bytes are dlopen'd.
    // Materialize it in a sealed anonymous file so the user still deploys one
    // bundle and no writable cache path can replace the verified image.
    const auto expected = wan2_2_ti2v::parse_bundle_plugin_contract(ctx.config_json);
    const int32_t current_sm = current_cuda_architecture();

    std::lock_guard<std::mutex> lock(aot_plugin_mutex());
    auto& state = aot_plugin_state();
    if (!state.load_error.empty()) {
        throw std::runtime_error("Wan2.2 AOT plugin registry is unusable after an earlier load "
                                 "failure: " +
                                 state.load_error);
    }
    preload_wan22_dependencies(expected.runtime_abi, state);
    if (state.handle == nullptr) {
        if (!ctx.bundle_reader)
            throw std::runtime_error("Wan2.2 requires a pinned bundle reader for its AOT plugin");
        auto plugin_bytes = ctx.bundle_reader->read(kWan22PluginSection);
        const std::string actual_binary_sha256 = sha256_bytes(plugin_bytes);
        if (actual_binary_sha256 != expected_binary_sha256) {
            throw std::runtime_error("Wan2.2 embedded AOT plugin SHA256 mismatch");
        }
        const int backing_fd = create_sealed_plugin_memfd(plugin_bytes);
        const std::string requested_path = "/proc/self/fd/" + std::to_string(backing_fd);
        dlerror();
        void* handle = dlopen(requested_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            const char* error = dlerror();
            const std::string message = error != nullptr ? error : "unknown dlopen error";
            ::close(backing_fd);
            throw std::runtime_error("Unable to load Wan2.2 embedded AOT plugin: " + message);
        }

        // TensorRT creators register during dlopen. Keep the DSO alive for the
        // process lifetime and verify every provenance export before allowing
        // a TensorRT plan to be materialized or deserialized.
        try {
            const auto manifest_json =
                require_export_value(handle, "trtmc_wan22_plugin_manifest_json");
            auto installed = wan2_2_ti2v::parse_companion_plugin_contract(manifest_json);
            const auto loaded_runtime_abi =
                require_export_value(handle, "trtmc_wan22_plugin_runtime_abi");
            validate_companion_exports(handle, installed, loaded_runtime_abi);
            wan2_2_ti2v::validate_plugin_contract(expected, installed, loaded_runtime_abi,
                                                  current_sm);

            state.handle = handle;
            state.backing_fd = backing_fd;
            state.binary_sha256 = actual_binary_sha256;
            state.contract = std::move(installed);
            state.loaded_runtime_abi = loaded_runtime_abi;
        } catch (const std::exception& error) {
            // TensorRT creator registration happens in DSO constructors and
            // cannot be undone. Keep the image mapped so any registry pointer
            // remains valid, poison this process for later Wan loads, and
            // fail closed without attempting dlclose.
            state.handle = handle;
            state.backing_fd = backing_fd;
            state.binary_sha256 = actual_binary_sha256;
            state.load_error = error.what();
            throw;
        }
    } else if (state.binary_sha256 != expected_binary_sha256) {
        // Loading two creator implementations into one TensorRT registry is
        // not reversible. Refuse the second ABI before its static registration
        // can run.
        throw std::runtime_error(
            "Wan2.2 AOT plugin conflict in this process: loaded_sha256=" + state.binary_sha256 +
            ", requested_sha256=" + expected_binary_sha256);
    }

    wan2_2_ti2v::validate_plugin_contract(expected, state.contract, state.loaded_runtime_abi,
                                          current_sm);
}

void validate_wan22_lazy_plan_sections(const PipelineContext& ctx) {
    // Validate the staged contract from header metadata only. This catches an
    // incomplete bundle at load time without materializing any TensorRT plan.
    for (const char* name : {"text_encoder_0_plan", "denoiser_plan", "vae_decoder_plan",
                             "vae_decoder_first_frame_plan"}) {
        bool found = false;
        for (const auto& section : ctx.bundle.info.sections) {
            if (section.name == name) {
                found = section.size != 0;
                break;
            }
        }
        if (!found)
            throw std::runtime_error(std::string("Wan2.2 bundle is missing ") + name);
    }
}

nlohmann::json parse_wan22_config_json(const std::string& config_json) {
    try {
        return nlohmann::json::parse(config_json);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("Invalid Wan2.2 config.json: ") + error.what());
    }
}

const nlohmann::json& require_artifact_sections(const nlohmann::json& config) {
    const auto manifest = config.find("artifact_manifest");
    if (manifest == config.end() || !manifest->is_object() || !manifest->contains("sections") ||
        !(*manifest)["sections"].is_object()) {
        throw std::runtime_error("Wan2.2 config is missing artifact_manifest sections");
    }
    return (*manifest)["sections"];
}

std::string require_artifact_sha256(const nlohmann::json& sections, const char* artifact) {
    const auto entry = sections.find(artifact);
    if (entry == sections.end() || !entry->is_object() || !entry->contains("sha256") ||
        !(*entry)["sha256"].is_string()) {
        throw std::runtime_error(std::string("Wan2.2 artifact_manifest is missing ") + artifact +
                                 " SHA256");
    }
    return (*entry)["sha256"].get<std::string>();
}

std::unordered_map<std::string, std::string>
parse_wan22_artifact_digests(const std::string& config_json) {
    static constexpr const char* kArtifacts[] = {"text_encoder_0_plan", "denoiser_plan",
                                                 "vae_decoder_plan", "vae_decoder_first_frame_plan",
                                                 kWan22PluginSection};
    const auto config = parse_wan22_config_json(config_json);
    const auto& sections = require_artifact_sections(config);

    std::unordered_map<std::string, std::string> result;
    for (const char* artifact : kArtifacts)
        result.emplace(artifact, require_artifact_sha256(sections, artifact));
    return result;
}

Wan22ModuleLoader make_staged_module_loader(const PipelineContext& ctx) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("Wan2.2 requires a TensorRT backend");
    if (!ctx.bundle_reader) {
        throw std::runtime_error(
            "Wan2.2 requires the pinned source bundle reader for staged loading");
    }

    // PipelineContext is factory-owned and expires after create(). Capture
    // every value needed by generation by value. Backends are process-cached
    // by BackendLoader, so the backend pointer remains valid for the pipeline.
    // This is the same open file description that materialized ctx.bundle.
    // Retaining it makes both the eager metadata and every later plan read
    // immune to pathname rename/replacement/unlink.
    auto bundle_reader = ctx.bundle_reader;
    const std::string runtime_cache_path = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    auto plan_digests = parse_wan22_artifact_digests(ctx.config_json);
    return [bundle_reader = std::move(bundle_reader), runtime_cache_path, backend, cuda_graphs,
            plan_digests = std::move(plan_digests)](
               const std::string& section_name, cudaStream_t stream,
               const std::vector<ModuleExternalBinding>& external_bindings)
               -> std::unique_ptr<ITrtModule> {
        // Only one plan payload is resident on the host. TensorRT consumes it
        // synchronously in create_module(); this vector dies before the
        // generation stage receives the module.
        auto plan = bundle_reader->read(section_name);
        if (plan.empty())
            throw std::runtime_error("Wan2.2 bundle section is empty: " + section_name);
        const auto expected_digest = plan_digests.find(section_name);
        if (expected_digest == plan_digests.end()) {
            throw std::runtime_error("Wan2.2 has no authenticated digest for " + section_name);
        }
        detail::Sha256 digest;
        digest.update(plan.data(), plan.size());
        if (digest.hex_digest() != expected_digest->second) {
            throw std::runtime_error("Wan2.2 artifact SHA256 mismatch for " + section_name);
        }
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache_path.c_str();
        options.cuda_graphs = cuda_graphs;
        options.external_bindings = external_bindings;
        auto module = backend->create_module(plan.data(), plan.size(), options);
        if (!module || !module->ok())
            throw std::runtime_error("Wan2.2 could not deserialize " + section_name);
        return module;
    };
}

std::shared_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* tokenizer_json = find_section(bundle, "tokenizer.json");
    if (tokenizer_json == nullptr || tokenizer_json->empty())
        throw std::runtime_error("Wan2.2 bundle is missing tokenizer.json");
    auto tokenizer = CreateUnigramTokenizer(tokenizer_json->data(), tokenizer_json->size(), false);
    if (!tokenizer)
        throw std::runtime_error("Wan2.2 could not create the native UMT5 tokenizer");
    return std::shared_ptr<ITokenizer>(std::move(tokenizer));
}

} // namespace

class Wan22TI2VPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        if (!ctx.bundle_reader)
            throw std::runtime_error("Wan2.2 requires a pinned source bundle reader");
        wan2_2_ti2v::validate_bundle_artifact_provenance(*ctx.bundle_reader, ctx.config_json,
                                                         ctx.config_json.size());
        // Embedded AOT provenance and loaded-library ABI are validated before
        // the staged reader can materialize any TensorRT plan.
        const auto artifact_digests = parse_wan22_artifact_digests(ctx.config_json);
        load_and_validate_wan22_aot_plugin(ctx, artifact_digests.at(kWan22PluginSection));
        validate_wan22_lazy_plan_sections(ctx);
        auto tokenizer = load_tokenizer(ctx.bundle);
        return std::make_unique<Wan22TI2VPipeline>(
            make_staged_module_loader(ctx), std::move(tokenizer),
            parse_wan22_options(ctx.config_json), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_wan2_2_ti2v_plugin, Wan22TI2VPlugin,
                                       "diffusion_wan2_2_ti2v");

} // namespace trtmc
