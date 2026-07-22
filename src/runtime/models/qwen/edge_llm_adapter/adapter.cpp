/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// This file is owned by the Qwen family / TensorRT Edge-LLM adapter. It
// intentionally duplicates downstream translation instead of importing a
// shared optimized-runtime adapter from Model Connect.

#include "runtime/providers/optimized_runtime_factory.h"

#include <nlohmann/json.hpp>

#ifndef TRTMC_QWEN_EDGE_FAKE_RUNTIME
#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <dlfcn.h>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char* kImplementationId = "qwen.tensorrt-edge-llm";
constexpr const char* kRuntimeLibrary = "libtrtmc_impl_qwen_tensorrt_edge_llm.so";
constexpr const char* kPluginLibrary = "libNvInfer_edgellm_plugin.so";
constexpr const char* kEdgeVersion = TRTMC_QWEN_EDGE_VERSION;
constexpr const char* kEdgeCommit = TRTMC_QWEN_EDGE_COMMIT;
constexpr const char* kTensorRtVersion = TRTMC_QWEN_EDGE_TENSORRT_VERSION;
constexpr const char* kCudaVersion = TRTMC_QWEN_EDGE_CUDA_VERSION;
// Edge-LLM's sampling implementation bounds top-k scratch space at 1024 for
// the qualified Qwen profiles. This is adapter policy, not part of the generic
// MC package-private factory contract.
constexpr int32_t kMaxTopK = 1024;
constexpr std::size_t kMaxMetadataBytes = 16U * 1024U * 1024U;

struct TensorRtRuntimeVersion {
    int32_t major{0};
    int32_t minor{0};
    int32_t patch{0};
    int32_t build{0};
};

std::string version_string(const TensorRtRuntimeVersion& version) {
    return std::to_string(version.major) + "." + std::to_string(version.minor) + "." +
           std::to_string(version.patch) + "." + std::to_string(version.build);
}

TensorRtRuntimeVersion loaded_tensorrt_runtime_version() noexcept {
#ifdef TRTMC_QWEN_EDGE_FAKE_RUNTIME
    return {TRTMC_QWEN_EDGE_FAKE_TENSORRT_MAJOR, TRTMC_QWEN_EDGE_FAKE_TENSORRT_MINOR,
            TRTMC_QWEN_EDGE_FAKE_TENSORRT_PATCH, TRTMC_QWEN_EDGE_FAKE_TENSORRT_BUILD};
#else
    // Query the libnvinfer object actually resolved for this DSO. Bundle
    // metadata and the provider's compile-time identity are not runtime proof.
    return {getInferLibMajorVersion(), getInferLibMinorVersion(), getInferLibPatchVersion(),
            getInferLibBuildVersion()};
#endif
}

void require_supported_tensorrt_runtime() {
    const TensorRtRuntimeVersion observed = loaded_tensorrt_runtime_version();
    constexpr TensorRtRuntimeVersion supported{
        TRTMC_QWEN_EDGE_TENSORRT_MAJOR, TRTMC_QWEN_EDGE_TENSORRT_MINOR,
        TRTMC_QWEN_EDGE_TENSORRT_PATCH, TRTMC_QWEN_EDGE_TENSORRT_BUILD};
    if (observed.major != supported.major || observed.minor != supported.minor ||
        observed.patch != supported.patch || observed.build != supported.build) {
        throw std::runtime_error("loaded TensorRT runtime version " + version_string(observed) +
                                 " is unsupported; expected " + version_string(supported));
    }
}

int32_t loaded_cuda_runtime_version() {
#ifdef TRTMC_QWEN_EDGE_FAKE_RUNTIME
    return TRTMC_QWEN_EDGE_FAKE_CUDA_RUNTIME_VERSION;
#else
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess)
        throw std::runtime_error("failed to query loaded CUDA runtime version");
    return runtime_version;
#endif
}

void require_supported_cuda_runtime() {
    constexpr int32_t supported = TRTMC_QWEN_EDGE_CUDA_RUNTIME_VERSION;
    const int32_t observed = loaded_cuda_runtime_version();
    if (observed != supported) {
        throw std::runtime_error("loaded CUDA runtime version " + std::to_string(observed) +
                                 " is unsupported; expected " + std::to_string(supported) +
                                 " (CUDA " + kCudaVersion + ")");
    }
}

void set_error(char* output, size_t capacity, const std::string& message) noexcept {
    if (output == nullptr || capacity == 0)
        return;
    std::snprintf(output, capacity, "%s", message.c_str());
}

std::string required_c_string(const char* value, const char* field) {
    if (value == nullptr || value[0] == '\0')
        throw std::runtime_error(std::string("create request requires ") + field);
    return value;
}

nlohmann::json parse_metadata(const std::string& text) {
    if (text.empty() || text.size() > kMaxMetadataBytes)
        throw std::runtime_error("implementation metadata size is outside the capsule limit");
    std::vector<std::unordered_set<std::string>> object_keys;
    nlohmann::json::parser_callback_t callback = [&](int, nlohmann::json::parse_event_t event,
                                                     nlohmann::json& value) {
        if (event == nlohmann::json::parse_event_t::object_start)
            object_keys.emplace_back();
        if (event == nlohmann::json::parse_event_t::key) {
            const std::string key = value.get<std::string>();
            if (object_keys.empty() || !object_keys.back().insert(key).second)
                throw std::runtime_error("duplicate implementation metadata key: " + key);
        }
        if (event == nlohmann::json::parse_event_t::object_end) {
            if (object_keys.empty())
                throw std::runtime_error("unbalanced implementation metadata object");
            object_keys.pop_back();
        }
        return true;
    };
    try {
        return nlohmann::json::parse(text, callback);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("invalid implementation metadata: ") + error.what());
    }
}

void require_exact_keys(const nlohmann::json& object, std::initializer_list<const char*> expected,
                        const std::string& context) {
    if (!object.is_object())
        throw std::runtime_error(context + " must be an object");
    std::set<std::string> expected_keys;
    for (const char* key : expected)
        expected_keys.insert(key);
    std::set<std::string> actual_keys;
    for (auto item = object.begin(); item != object.end(); ++item)
        actual_keys.insert(item.key());
    if (actual_keys != expected_keys)
        throw std::runtime_error(context + " has an unexpected field set");
}

const nlohmann::json& require_object(const nlohmann::json& object, const char* field,
                                     const char* context) {
    const auto value = object.find(field);
    if (value == object.end() || !value->is_object())
        throw std::runtime_error(std::string(context) + " requires object field '" + field + "'");
    return *value;
}

std::string require_string(const nlohmann::json& object, const char* field,
                           const std::string& context) {
    const auto value = object.find(field);
    if (value == object.end() || !value->is_string() || value->get<std::string>().empty())
        throw std::runtime_error(context + " requires string field '" + field + "'");
    return value->get<std::string>();
}

int32_t require_int32(const nlohmann::json& object, const char* field, const std::string& context) {
    const auto value = object.find(field);
    if (value == object.end() || (!value->is_number_integer() && !value->is_number_unsigned())) {
        throw std::runtime_error(context + " requires integer field '" + field + "'");
    }
    const auto parsed = value->get<std::int64_t>();
    if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max())
        throw std::runtime_error(context + " integer field is outside int32 range");
    return static_cast<int32_t>(parsed);
}

void require_string_value(const nlohmann::json& object, const char* field,
                          const std::string& expected, const std::string& context) {
    const std::string actual = require_string(object, field, context);
    if (actual != expected) {
        throw std::runtime_error(context + " field '" + field + "' mismatch: expected '" +
                                 expected + "', got '" + actual + "'");
    }
}

int32_t require_positive_int32(const nlohmann::json& object, const char* field,
                               const std::string& context) {
    const int32_t value = require_int32(object, field, context);
    if (value <= 0)
        throw std::runtime_error(context + " field '" + field + "' must be positive");
    return value;
}

void require_int_value(const nlohmann::json& object, const char* field, int32_t expected,
                       const std::string& context) {
    const int32_t actual = require_int32(object, field, context);
    if (actual != expected) {
        throw std::runtime_error(context + " field '" + field + "' mismatch: expected " +
                                 std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

void require_sha256(const nlohmann::json& object, const char* field, const std::string& context) {
    const std::string value = require_string(object, field, context);
    const bool valid =
        value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
        });
    if (!valid)
        throw std::runtime_error(context + " field '" + field +
                                 "' must be a lowercase SHA-256 digest");
}

struct CapsuleMetadata {
    std::string gpu_architecture;
    std::string gpu_name;
    int32_t gpu_memory_mib{0};
    int32_t max_cache_length{0};
    int32_t vocab_size{0};
};

CapsuleMetadata
validate_metadata(const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1& request) {
    const std::string request_implementation_id =
        required_c_string(request.implementation_id, "implementation_id");
    const std::string request_profile_id = required_c_string(request.profile_id, "profile_id");
    const std::string request_model_id = required_c_string(request.model_id, "model_id");
    if (request_implementation_id != kImplementationId)
        throw std::runtime_error("create request implementation_id does not match this capsule");
    if (request.implementation_metadata == nullptr || request.implementation_metadata_size == 0)
        throw std::runtime_error("create request requires implementation_metadata");
    const nlohmann::json root = parse_metadata(
        std::string(request.implementation_metadata, request.implementation_metadata_size));
    require_exact_keys(root,
                       {"schema_version", "build_binding", "implementation_id", "profile_id",
                        "operation", "model", "target", "runtime", "artifacts", "limits",
                        "versions", "bundle_info", "bundle_config"},
                       "implementation metadata");
    require_int_value(root, "schema_version", 1, "implementation metadata");
    require_string_value(root, "implementation_id", request_implementation_id,
                         "implementation metadata");
    require_string_value(root, "profile_id", request_profile_id, "implementation metadata");
    require_string_value(root, "operation", "text-generation-v1", "implementation metadata");

    const auto& build_binding = require_object(root, "build_binding", "implementation metadata");
    require_exact_keys(build_binding,
                       {"schema_version", "implementation_id", "manifest_sha256", "request_sha256",
                        "profile_id", "profile_sha256"},
                       "implementation metadata build_binding");
    require_int_value(build_binding, "schema_version", 1, "implementation metadata build_binding");
    require_string_value(build_binding, "implementation_id", request_implementation_id,
                         "implementation metadata build_binding");
    require_string_value(build_binding, "manifest_sha256", TRTMC_QWEN_EDGE_MANIFEST_SHA256,
                         "implementation metadata build_binding");
    require_sha256(build_binding, "request_sha256", "implementation metadata build_binding");
    require_string_value(build_binding, "profile_id", request_profile_id,
                         "implementation metadata build_binding");
    require_sha256(build_binding, "profile_sha256", "implementation metadata build_binding");

    const auto& model = require_object(root, "model", "implementation metadata");
    require_exact_keys(model, {"id", "revision"}, "implementation metadata model");
    require_string_value(model, "id", request_model_id, "implementation metadata model");
    const std::string model_revision =
        require_string(model, "revision", "implementation metadata model");

    const auto& target = require_object(root, "target", "implementation metadata");
    require_string_value(target, "os", "linux", "implementation metadata target");
    require_string_value(target, "architecture", "x86_64", "implementation metadata target");
    require_string_value(target, "platform_kind", "discrete", "implementation metadata target");
    const std::string gpu_architecture =
        require_string(target, "gpu_architecture", "implementation metadata target");
    const std::string gpu_name =
        require_string(target, "gpu_name", "implementation metadata target");
    const int32_t gpu_memory_mib =
        require_positive_int32(target, "gpu_memory_mib", "implementation metadata target");

    const auto& runtime = require_object(root, "runtime", "implementation metadata");
    require_exact_keys(runtime, {"abi", "library", "plugin"}, "implementation metadata runtime");
    require_int_value(runtime, "abi", 1, "implementation metadata runtime");
    require_string_value(runtime, "library", kRuntimeLibrary, "implementation metadata runtime");
    require_string_value(runtime, "plugin", kPluginLibrary, "implementation metadata runtime");

    const auto& artifacts = require_object(root, "artifacts", "implementation metadata");
    require_string_value(artifacts, "layout", "directory-tree-v1",
                         "implementation metadata artifacts");
    require_string_value(artifacts, "engine_dir", "engine.dir",
                         "implementation metadata artifacts");
    require_string_value(artifacts, "runtime_library", kRuntimeLibrary,
                         "implementation metadata artifacts");
    require_string_value(artifacts, "runtime_plugin", kPluginLibrary,
                         "implementation metadata artifacts");

    const auto& limits = require_object(root, "limits", "implementation metadata");
    require_exact_keys(limits,
                       {"max_input_length", "max_cache_length", "max_batch_size", "vocab_size"},
                       "implementation metadata limits");
    const int32_t max_input_length =
        require_positive_int32(limits, "max_input_length", "implementation metadata limits");
    const int32_t max_cache_length =
        require_positive_int32(limits, "max_cache_length", "implementation metadata limits");
    (void)require_positive_int32(limits, "max_batch_size", "implementation metadata limits");
    const int32_t vocab_size =
        require_positive_int32(limits, "vocab_size", "implementation metadata limits");
    if (max_input_length > max_cache_length) {
        throw std::runtime_error(
            "implementation metadata max_input_length exceeds max_cache_length");
    }

    const auto& versions = require_object(root, "versions", "implementation metadata");
    require_exact_keys(versions,
                       {"model_revision", "edge_llm", "edge_llm_commit", "tensorrt", "cuda"},
                       "implementation metadata versions");
    require_string_value(versions, "model_revision", model_revision,
                         "implementation metadata versions");
    require_string_value(versions, "edge_llm", kEdgeVersion, "implementation metadata versions");
    require_string_value(versions, "edge_llm_commit", kEdgeCommit,
                         "implementation metadata versions");
    require_string_value(versions, "tensorrt", kTensorRtVersion,
                         "implementation metadata versions");
    require_string_value(versions, "cuda", kCudaVersion, "implementation metadata versions");

    const auto& bundle_info = require_object(root, "bundle_info", "implementation metadata");
    require_string_value(bundle_info, "model_type", "qwen3", "implementation metadata bundle_info");
    require_string_value(bundle_info, "family", "qwen", "implementation metadata bundle_info");
    require_string_value(bundle_info, "gpu_name", gpu_name, "implementation metadata bundle_info");
    require_int_value(bundle_info, "vocab_size", vocab_size, "implementation metadata bundle_info");
    require_int_value(bundle_info, "max_cache_length", max_cache_length,
                      "implementation metadata bundle_info");

    const auto& bundle_config = require_object(root, "bundle_config", "implementation metadata");
    require_string_value(bundle_config, "model_id", request_model_id,
                         "implementation metadata bundle_config");
    require_string_value(bundle_config, "model_revision", model_revision,
                         "implementation metadata bundle_config");
    require_string_value(bundle_config, "model_type", "qwen3",
                         "implementation metadata bundle_config");
    require_string_value(bundle_config, "family", "qwen", "implementation metadata bundle_config");
    require_string_value(bundle_config, "runtime_provider", request_implementation_id,
                         "implementation metadata bundle_config");
    require_int_value(bundle_config, "max_cache_length", max_cache_length,
                      "implementation metadata bundle_config");

    return CapsuleMetadata{gpu_architecture, gpu_name, gpu_memory_mib, max_cache_length,
                           vocab_size};
}

fs::path require_directory(const fs::path& path, const std::string& description) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(path, error);
    if (error || fs::is_symlink(status) || !fs::is_directory(status))
        throw std::runtime_error(description + " is not a non-symlink directory: " + path.string());
    return path;
}

fs::path adjacent_plugin_path() {
    static const char anchor = 0;
    Dl_info info{};
    if (dladdr(&anchor, &info) == 0 || info.dli_fname == nullptr)
        throw std::runtime_error("unable to determine capsule DSO location");
    return fs::path(info.dli_fname).parent_path() / kPluginLibrary;
}

#ifndef TRTMC_QWEN_EDGE_FAKE_RUNTIME
class ProcessPluginLibrary {
  public:
    ~ProcessPluginLibrary() {
#ifdef RTLD_NODELETE
        if (handle != nullptr)
            (void)dlclose(handle);
#endif
    }
    ProcessPluginLibrary(const ProcessPluginLibrary&) = delete;
    ProcessPluginLibrary& operator=(const ProcessPluginLibrary&) = delete;
    void* handle{nullptr};

  private:
    ProcessPluginLibrary() = default;
    friend void ensure_edge_plugin_loaded();
};

void open_edge_plugin(ProcessPluginLibrary& plugin) {
    const fs::path plugin_path = adjacent_plugin_path();
    std::error_code status_error;
    const fs::file_status status = fs::symlink_status(plugin_path, status_error);
    if (status_error || fs::is_symlink(status) || !fs::is_regular_file(status))
        throw std::runtime_error("capsule plugin is unavailable: " + plugin_path.string());
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
    flags |= RTLD_NODELETE;
#endif
    dlerror();
    plugin.handle = dlopen(plugin_path.c_str(), flags);
    if (plugin.handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error("failed to load capsule plugin: " +
                                 std::string(error ? error : "unknown dlopen error"));
    }
}

void initialize_edge_plugin(ProcessPluginLibrary& plugin) {
    using InitPluginsFn = bool (*)(void*, const char*);
    dlerror();
    auto* symbol = dlsym(plugin.handle, "initEdgellmPlugins");
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || symbol == nullptr)
        throw std::runtime_error("capsule plugin is missing initEdgellmPlugins");
    auto init_plugins = reinterpret_cast<InitPluginsFn>(symbol);
    if (!init_plugins(static_cast<nvinfer1::ILogger*>(&trt_edgellm::gLogger), ""))
        throw std::runtime_error("initEdgellmPlugins returned false");
}

void configure_edge_logging() noexcept {
    // MC's public CLI reserves stdout for the generated result. The pinned Edge-LLM API
    // sends INFO messages to stdout by default, which would make the unchanged
    // Python wrapper return runtime logs as part of the generated text. Keep
    // warnings and errors, but suppress INFO and VERBOSE output.
    trt_edgellm::gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
}

void ensure_edge_plugin_loaded() {
    static std::once_flag once;
    static std::exception_ptr initialization_error;
    static ProcessPluginLibrary plugin;
    std::call_once(once, [] {
        try {
            configure_edge_logging();
            open_edge_plugin(plugin);
            initialize_edge_plugin(plugin);
        } catch (...) {
            initialization_error = std::current_exception();
        }
    });
    if (initialization_error)
        std::rethrow_exception(initialization_error);
}

class CudaDeviceGuard {
  public:
    explicit CudaDeviceGuard(int target_device) {
        const cudaError_t get_status = cudaGetDevice(&previous_device_);
        if (get_status != cudaSuccess)
            throw std::runtime_error("failed to query current CUDA device");
        if (previous_device_ != target_device) {
            const cudaError_t set_status = cudaSetDevice(target_device);
            if (set_status != cudaSuccess)
                throw std::runtime_error("failed to select capsule CUDA device");
            restore_ = true;
        }
    }
    ~CudaDeviceGuard() {
        if (restore_)
            (void)cudaSetDevice(previous_device_);
    }
    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

  private:
    int previous_device_{-1};
    bool restore_{false};
};
#endif

struct EdgeLlmHandle {
    int32_t max_cache_length{0};
    int32_t vocab_size{0};
#ifndef TRTMC_QWEN_EDGE_FAKE_RUNTIME
    cudaStream_t stream{nullptr};
    int device{-1};
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime;
    ~EdgeLlmHandle() {
        int previous_device = -1;
        bool restore_device = false;
        if (device >= 0 && cudaGetDevice(&previous_device) == cudaSuccess &&
            previous_device != device && cudaSetDevice(device) == cudaSuccess) {
            restore_device = true;
        }
        runtime.reset();
        if (stream != nullptr)
            (void)cudaStreamDestroy(stream);
        if (restore_device)
            (void)cudaSetDevice(previous_device);
    }
#endif
};

#ifdef TRTMC_QWEN_EDGE_FAKE_RUNTIME
void require_fake_plugin_payload() {
    const fs::path plugin = adjacent_plugin_path();
    std::error_code plugin_error;
    const auto plugin_status = fs::symlink_status(plugin, plugin_error);
    if (plugin_error || fs::is_symlink(plugin_status) || !fs::is_regular_file(plugin_status))
        throw std::runtime_error("fake capsule contract requires adjacent plugin payload");
}
#else
cudaDeviceProp query_active_cuda_device(EdgeLlmHandle& handle) {
    int visible_devices = 0;
    if (cudaGetDeviceCount(&visible_devices) != cudaSuccess || visible_devices < 1)
        throw std::runtime_error("Edge-LLM capsule requires at least one visible CUDA GPU");
    if (cudaGetDevice(&handle.device) != cudaSuccess)
        throw std::runtime_error("failed to query capsule CUDA device");
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, handle.device) != cudaSuccess)
        throw std::runtime_error("failed to query capsule CUDA device properties");
    return properties;
}

std::pair<int32_t, int32_t> parse_compute_capability(const std::string& architecture) {
    if (architecture.size() < 4 || architecture.rfind("sm", 0) != 0 ||
        !std::all_of(architecture.begin() + 2, architecture.end(),
                     [](char character) { return character >= '0' && character <= '9'; })) {
        throw std::runtime_error(
            "implementation metadata target gpu_architecture must use canonical smNN form");
    }
    int32_t capability = 0;
    for (auto character = architecture.begin() + 2; character != architecture.end(); ++character) {
        if (capability > (std::numeric_limits<int32_t>::max() - (*character - '0')) / 10) {
            throw std::runtime_error(
                "implementation metadata target gpu_architecture is outside int32 range");
        }
        capability = capability * 10 + (*character - '0');
    }
    const int32_t major = capability / 10;
    const int32_t minor = capability % 10;
    if (major <= 0 || architecture != "sm" + std::to_string(major) + std::to_string(minor)) {
        throw std::runtime_error(
            "implementation metadata target gpu_architecture must use canonical smNN form");
    }
    return {major, minor};
}

void require_supported_target(const cudaDeviceProp& properties, const CapsuleMetadata& metadata) {
    const auto [expected_major, expected_minor] =
        parse_compute_capability(metadata.gpu_architecture);
    const std::string gpu_name(properties.name);
    const auto memory_mib = properties.totalGlobalMem / (1024ULL * 1024ULL);
    if (properties.major != expected_major || properties.minor != expected_minor ||
        gpu_name != metadata.gpu_name ||
        memory_mib < static_cast<std::uint64_t>(metadata.gpu_memory_mib)) {
        throw std::runtime_error(
            "active CUDA device does not match the bundle's qualified deployment target");
    }
}

void create_edge_runtime(EdgeLlmHandle& handle, const fs::path& engine_directory);
#endif

void validate_load_options(
    const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1& request) {
    if (request.load_options == nullptr)
        return;
    const auto& options = *request.load_options;
    if (options.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "Qwen Edge-LLM does not support LoadOptions::kv_cache_size_bytes");
    }
    if (!options.config_path.empty() || !options.set_tokens.empty()) {
        throw std::invalid_argument(
            "Qwen Edge-LLM does not support LoadOptions runtime configuration overrides");
    }
}

std::unique_ptr<EdgeLlmHandle>
initialize_edge_handle(const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1& request) {
    const CapsuleMetadata metadata = validate_metadata(request);
    validate_load_options(request);
    const fs::path artifact_root = require_directory(
        required_c_string(request.artifact_path, "artifact_path"), "capsule artifact root");
    const fs::path engine_directory =
        require_directory(artifact_root / "engine.dir", "capsule engine.dir");
    // Fail before plugin initialization, CUDA allocation, or Edge-LLM runtime
    // construction if the deployed dependency libraries are unsupported.
    require_supported_tensorrt_runtime();
    require_supported_cuda_runtime();
    auto handle = std::make_unique<EdgeLlmHandle>();
    handle->max_cache_length = metadata.max_cache_length;
    handle->vocab_size = metadata.vocab_size;
#ifdef TRTMC_QWEN_EDGE_FAKE_RUNTIME
    require_fake_plugin_payload();
    (void)engine_directory;
#else
    const cudaDeviceProp properties = query_active_cuda_device(*handle);
    require_supported_target(properties, metadata);
    ensure_edge_plugin_loaded();
    create_edge_runtime(*handle, engine_directory);
#endif
    return handle;
}

#ifndef TRTMC_QWEN_EDGE_FAKE_RUNTIME
void create_edge_runtime(EdgeLlmHandle& handle, const fs::path& engine_directory) {
    if (cudaStreamCreate(&handle.stream) != cudaSuccess)
        throw std::runtime_error("failed to create capsule CUDA stream");
    handle.runtime = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
        engine_directory.string(), std::string{}, std::unordered_map<std::string, std::string>{},
        handle.stream);
    // Match Edge-LLM's own long-lived inference entry point. Graph capture is
    // part of this adapter's qualified performance path.
    if (!handle.runtime->captureDecodingCUDAGraph(handle.stream))
        throw std::runtime_error("Qwen Edge-LLM decoding CUDA graph capture failed");
}
#endif

struct EdgeLlmResponse {
    std::string text;
    std::vector<int32_t> token_ids;
};

struct CapsuleGenerationRequest {
    const char* prompt{nullptr};
    std::size_t prompt_size{0};
    int32_t max_new_tokens{0};
    float temperature{0.0F};
    int32_t top_k{0};
    float top_p{0.0F};
    bool use_chat_template{false};
    bool enable_thinking{true};
};

bool valid_sampling_numbers(const CapsuleGenerationRequest& request) noexcept {
    return std::isfinite(request.temperature) && request.temperature >= 0.0F &&
           std::isfinite(request.top_p) && request.top_p >= 0.0F && request.top_p <= 1.0F &&
           request.top_k >= 0;
}

bool valid_sampling_limits(const EdgeLlmHandle& handle,
                           const CapsuleGenerationRequest& request) noexcept {
    if (request.max_new_tokens <= 0 || request.max_new_tokens > handle.max_cache_length)
        return false;
    const int32_t max_top_k = std::min(handle.vocab_size, kMaxTopK);
    return request.top_k <= max_top_k;
}

void validate_sampling_request(const EdgeLlmHandle& handle,
                               const CapsuleGenerationRequest& request) {
    if (!valid_sampling_numbers(request) || !valid_sampling_limits(handle, request))
        throw std::invalid_argument("invalid Qwen Edge-LLM sampling request");
}

#ifndef TRTMC_QWEN_EDGE_FAKE_RUNTIME
trt_edgellm::rt::LLMGenerationRequest make_edge_request(const CapsuleGenerationRequest& request) {
    trt_edgellm::rt::LLMGenerationRequest edge_request;
    trt_edgellm::rt::LLMGenerationRequest::Request item;
    trt_edgellm::rt::Message message;
    message.role = "user";
    const std::string prompt = request.prompt == nullptr
                                   ? std::string{}
                                   : std::string(request.prompt, request.prompt_size);
    message.contents.push_back({"text", prompt});
    item.messages.push_back(std::move(message));
    edge_request.requests.push_back(std::move(item));
    edge_request.temperature = request.temperature;
    edge_request.topP = request.top_p;
    edge_request.topK = request.top_k;
    edge_request.maxGenerateLength = request.max_new_tokens;
    edge_request.applyChatTemplate = request.use_chat_template;
    edge_request.addGenerationPrompt = true;
    edge_request.enableThinking = request.enable_thinking;
    return edge_request;
}
#endif

EdgeLlmResponse execute_one(EdgeLlmHandle& handle, const CapsuleGenerationRequest& request) {
    EdgeLlmResponse result;
#ifdef TRTMC_QWEN_EDGE_FAKE_RUNTIME
    (void)handle;
    if (request.top_k < 0)
        throw std::logic_error("fake Edge-LLM received a negative topK");
    const std::string prompt = request.prompt == nullptr
                                   ? std::string{}
                                   : std::string(request.prompt, request.prompt_size);
    if (prompt == "inspect-sampling") {
        result.text =
            "fake-sampling:" + std::to_string(request.top_k) + ":" + std::to_string(request.top_p);
        return result;
    }
    result.text = "fake:" + prompt;
    const std::size_t count =
        std::min<std::size_t>(prompt.size(), static_cast<std::size_t>(request.max_new_tokens));
    result.token_ids.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        result.token_ids.push_back(static_cast<unsigned char>(prompt[index]));
#else
    CudaDeviceGuard device_guard(handle.device);
    auto edge_request = make_edge_request(request);
    trt_edgellm::rt::LLMGenerationResponse edge_response;
    if (!handle.runtime->handleRequest(edge_request, edge_response, handle.stream))
        throw std::runtime_error("TensorRT Edge-LLM handleRequest returned false");
    if (edge_response.outputTexts.size() != 1 || edge_response.outputIds.size() != 1)
        throw std::runtime_error("TensorRT Edge-LLM returned an invalid response batch");
    result.text = std::move(edge_response.outputTexts.front());
    result.token_ids = std::move(edge_response.outputIds.front());
#endif
    return result;
}

bool uses_default_generation_scalars(const trtmc::GenerateConfig& config) noexcept {
    return config.num_samples == 1 && config.min_p == 0.0F && config.seed == -1 &&
           config.guidance_scale == -1.0F && config.cfg_scale < 0.0F && config.num_steps == -1 &&
           config.sde_gamma < 0.0F;
}

bool uses_no_diffusion_payload(const trtmc::GenerateConfig& config) noexcept {
    return config.initial_latents.empty() && config.condition_latents.empty() &&
           config.condition_mask.empty() && config.sampling_steps.empty() &&
           config.sde_noises.empty() && config.negative_prompt.empty();
}

bool uses_default_output_shape(const trtmc::GenerateConfig& config) noexcept {
    return config.height <= 0 && config.width <= 0;
}

bool uses_supported_text_mode(const trtmc::GenerateConfig& config) noexcept {
    const bool supported_mode = config.text_generation_mode.empty() ||
                                config.text_generation_mode == "auto" ||
                                config.text_generation_mode == "ar";
    return config.eos_token_id == -1 && supported_mode && config.block_length <= 0 &&
           config.confidence_threshold < 0.0F && config.lora_adapter_id.empty();
}

bool uses_default_stop_options(const trtmc::GenerateConfig& config) noexcept {
    return config.tail_frames == 0 && !config.stop_on_boxed_answer &&
           config.stop_check_interval == 16;
}

void validate_generate_config(const trtmc::GenerateConfig& config) {
    if (!uses_default_generation_scalars(config) || !uses_no_diffusion_payload(config) ||
        !uses_default_output_shape(config) || !uses_supported_text_mode(config) ||
        !uses_default_stop_options(config)) {
        throw std::invalid_argument(
            "Qwen Edge-LLM does not support one or more non-default GenerateConfig fields");
    }
}

struct EdgeSamplingParameters {
    int32_t top_k;
    float top_p;
};

bool has_invalid_sampling_range(const trtmc::GenerateConfig& config) noexcept {
    return !std::isfinite(config.temperature) || config.temperature < 0.0F ||
           !std::isfinite(config.top_p) || config.top_p < 0.0F || config.top_p > 1.0F;
}

bool uses_nucleus_sampling(const trtmc::GenerateConfig& config) noexcept {
    return config.top_p > 0.0F && config.top_p < 1.0F;
}

bool uses_greedy_sampling(const trtmc::GenerateConfig& config, bool nucleus) noexcept {
    return config.temperature <= 0.0F || config.top_p <= 0.0F || (config.top_k == 1 && !nucleus);
}

void validate_edge_sampling_support(const trtmc::GenerateConfig& config, bool nucleus) {
    constexpr float epsilon = 1.0e-6F;
    constexpr float edge_minimum_sampling_temperature = 1.0e-3F;
    if (nucleus && config.top_p >= 1.0F - epsilon) {
        throw std::invalid_argument(
            "TensorRT Edge-LLM 0.9 cannot preserve top-p values within 1e-6 of 1.0");
    }
    if (config.temperature < edge_minimum_sampling_temperature) {
        throw std::invalid_argument(
            "TensorRT Edge-LLM 0.9 cannot preserve sampling below temperature 0.001");
    }
    if (config.top_k <= 0 && !nucleus) {
        throw std::invalid_argument(
            "TensorRT Edge-LLM 0.9 cannot express unbounded full-vocabulary sampling");
    }
}

EdgeSamplingParameters edge_sampling_parameters(const trtmc::GenerateConfig& config) {
    if (has_invalid_sampling_range(config)) {
        throw std::invalid_argument("invalid Qwen Edge-LLM sampling parameters");
    }
    const bool nucleus = uses_nucleus_sampling(config);
    if (uses_greedy_sampling(config, nucleus))
        return {1, 1.0F};
    validate_edge_sampling_support(config, nucleus);
    return {config.top_k <= 1 ? 0 : config.top_k, config.top_p};
}

CapsuleGenerationRequest capsule_request(const std::string& prompt,
                                         const trtmc::GenerateConfig& config) {
    validate_generate_config(config);
    const EdgeSamplingParameters sampling = edge_sampling_parameters(config);
    return CapsuleGenerationRequest{
        prompt.data(),  prompt.size(),  config.max_new_tokens,    config.temperature,
        sampling.top_k, sampling.top_p, config.use_chat_template, config.enable_thinking};
}

class QwenEdgeLlmPipeline final : public trtmc::IPipeline {
  public:
    explicit QwenEdgeLlmPipeline(
        const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1& request)
        : model_id_(required_c_string(request.model_id, "model_id")),
          handle_(initialize_edge_handle(request)) {}

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::GenerateConfig& config) override {
        const CapsuleGenerationRequest request = capsule_request(prompt, config);
        validate_sampling_request(*handle_, request);
        std::lock_guard<std::mutex> lock(mutex_);
        EdgeLlmResponse response = execute_one(*handle_, request);
        // The pinned Edge-LLM API does not expose a trustworthy prefill/decode split.
        // Report both metrics as unavailable instead of mislabeling wall time.
        return trtmc::TextResult(std::move(response.text), std::move(response.token_ids), 0.0, 0.0);
    }

    const char* pipeline_type() const override { return "QwenTextGenerationPipeline"; }
    const char* model_id() const override { return model_id_.c_str(); }

  private:
    std::string model_id_;
    std::unique_ptr<EdgeLlmHandle> handle_;
    std::mutex mutex_;
};

trtmc::IPipeline*
create_pipeline(const trtmc::internal::OptimizedRuntimePipelineCreateRequestV1* request,
                char* error, std::size_t error_capacity) noexcept {
    if (request == nullptr ||
        request->abi_version != trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1 ||
        request->struct_size < sizeof(trtmc::internal::OptimizedRuntimePipelineCreateRequestV1)) {
        set_error(error, error_capacity, "invalid optimized-runtime pipeline create request");
        return nullptr;
    }
    try {
        return new QwenEdgeLlmPipeline(*request);
    } catch (const std::exception& exception) {
        set_error(error, error_capacity, exception.what());
        return nullptr;
    } catch (...) {
        set_error(error, error_capacity, "unknown capsule initialization failure");
        return nullptr;
    }
}

const trtmc::internal::OptimizedRuntimeFactoryV1 kFactoryV1 = {
    trtmc::internal::kOptimizedRuntimeFactoryAbiVersionV1,
    sizeof(trtmc::internal::OptimizedRuntimeFactoryV1),
    kImplementationId,
    "tensorrt-edge-llm",
    kEdgeVersion,
    kEdgeCommit,
    &create_pipeline,
};

} // namespace

extern "C" const trtmc::internal::OptimizedRuntimeFactoryV1*
trtmc_get_optimized_runtime_factory_v1() noexcept {
    return &kFactoryV1;
}
