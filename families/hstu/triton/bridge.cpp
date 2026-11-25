/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// In-process trusted host-pointer bridge. No model policy or TensorRT topology.
#include "families/hstu/runtime/history_cache.h"
#include "families/hstu/runtime/prototype_task.h"
#include "trtmc/bundle.h"
#include "trtmc/runtime/family_loader.h"

#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
using Json = nlohmann::json;
namespace {
thread_local std::string error, reply;
struct Handle {
    int device{};
    std::unique_ptr<trtmc::ITask> task;
    trtmc::IStructuredRanking* ranking{};
    trtmc::hstu::IHistoryCache* history{};
    std::vector<trtmc::FeatureTensor> outputs;
    std::vector<std::uint8_t> snapshot;
    std::mutex execute;
};
trtmc::DType dtype(const std::string& name) {
    if (name == "float32")
        return trtmc::DType::kFloat32;
    if (name == "int32")
        return trtmc::DType::kInt32;
    if (name == "int64")
        return trtmc::DType::kInt64;
    throw std::invalid_argument("unsupported public tensor dtype: " + name);
}
const char* dtype(trtmc::DType value) {
    if (value == trtmc::DType::kFloat32)
        return "float32";
    if (value == trtmc::DType::kInt32)
        return "int32";
    if (value == trtmc::DType::kInt64)
        return "int64";
    throw std::invalid_argument("unsupported output dtype");
}
std::vector<trtmc::FeatureTensor> read_tensors(const Json& specs) {
    if (!specs.is_array() || specs.size() > 1024)
        throw std::invalid_argument("invalid tensor descriptor list");
    std::vector<trtmc::FeatureTensor> inputs;
    for (const auto& spec : specs) {
        trtmc::FeatureTensor tensor{spec.at("name").get<std::string>(),
                                    dtype(spec.at("dtype").get<std::string>()),
                                    spec.at("shape").get<std::vector<std::int64_t>>(),
                                    {}};
        if (tensor.shape.empty())
            throw std::invalid_argument("empty input shape");
        std::size_t bytes = trtmc::dtype_size(tensor.dtype);
        for (auto dim : tensor.shape) {
            if (dim <= 0 || std::uint64_t(dim) > SIZE_MAX / bytes)
                throw std::invalid_argument("invalid tensor shape");
            bytes *= dim;
        }
        if (bytes != spec.at("bytes").get<std::size_t>() || bytes > 1024ull * 1024 * 1024)
            throw std::invalid_argument("wrong or excessive tensor byte count");
        const auto* data = reinterpret_cast<const void*>(spec.at("pointer").get<std::uintptr_t>());
        if (!data)
            throw std::invalid_argument("null input pointer");
        tensor.bytes.resize(bytes);
        std::memcpy(tensor.bytes.data(), data, bytes);
        inputs.push_back(std::move(tensor));
    }
    return inputs;
}
Json output_descriptors(const Handle& handle) {
    auto result = Json::array();
    for (const auto& tensor : handle.outputs)
        result.push_back({{"name", tensor.name},
                          {"dtype", dtype(tensor.dtype)},
                          {"shape", tensor.shape},
                          {"bytes", tensor.bytes.size()},
                          {"pointer", reinterpret_cast<std::uintptr_t>(tensor.bytes.data())}});
    return result;
}
} // namespace
extern "C" {
const char* hstu_runtime_metadata(const char* path) noexcept {
    try {
        trtmc::BundleReader reader(path);
        if (reader.info().family != "hstu" || reader.info().task != "structured_ranking")
            throw std::invalid_argument("expected a DLRMv3 structured-ranking bundle");
        const auto* section = reader.find_section("runtime.json");
        if (!section || section->length > 1024 * 1024)
            throw std::invalid_argument("missing or excessive runtime metadata");
        const auto data = reader.read_section("runtime.json");
        reply.assign(data.begin(), data.end());
        return reply.c_str();
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}
const char* hstu_last_error() noexcept {
    return error.c_str();
}
void* hstu_create(const char* configuration) noexcept {
    try {
        const auto config = Json::parse(configuration);
        if (config.contains("kv_cache_size_bytes")) {
            const auto& size = config.at("kv_cache_size_bytes");
            if (!size.is_number_integer() || size <= 0)
                throw std::invalid_argument("kv_cache_size_bytes must be a positive integer");
        }
        if (config.contains("kv_cache_memory_ratio")) {
            const auto& value = config.at("kv_cache_memory_ratio");
            if (!value.is_number() || !std::isfinite(value.get<double>()) || value <= 0 ||
                value > 1)
                throw std::invalid_argument("kv_cache_memory_ratio must be finite and in (0, 1]");
        }
        if (config.contains("kv_cache_memory_ratio") && config.contains("kv_cache_size_bytes"))
            throw std::invalid_argument("KV byte cap and memory ratio are mutually exclusive");
        if (cudaSetDevice(config.value("device", 0)) != cudaSuccess)
            throw std::runtime_error("cannot select CUDA device");
        auto handle = std::make_unique<Handle>();
        handle->device = config.value("device", 0);
        handle->task = trtmc::load_task(config.at("bundle").get<std::string>(),
                                        config.at("runtime_root").get<std::string>(), 0);
        handle->ranking = dynamic_cast<trtmc::IStructuredRanking*>(handle->task.get());
        if (!handle->ranking)
            throw std::invalid_argument("bundle does not implement structured ranking");
        handle->history = dynamic_cast<trtmc::hstu::IHistoryCache*>(handle->task.get());
        if (!handle->history)
            throw std::invalid_argument("bundle lacks the family cache configuration interface");
        handle->history->configure_cache_budget(
            config.value("kv_cache_size_bytes", std::uint64_t(0)),
            config.value("kv_cache_memory_ratio", 0.0));
        if (config.value("kv_cache_hydration_enabled", false)) {
            if (!handle->history || !handle->history->history_supported())
                throw std::invalid_argument("bundle does not support history restoration");
            const auto count = config.value("kv_cache_max_entries", Json(32));
            if (!count.is_number_integer() || count.get<std::int64_t>() <= 0)
                throw std::invalid_argument("kv_cache_max_entries must be a positive integer");
            const auto ttl = config.value("kv_cache_ttl_seconds", 300.0);
            handle->history->configure_history(
                count.get<std::size_t>(), ttl,
                config.value("kv_cache_lease_seconds", std::min(30.0, ttl)));
        } else if (config.contains("kv_cache_max_entries") ||
                   config.contains("kv_cache_ttl_seconds") ||
                   config.contains("kv_cache_lease_seconds")) {
            throw std::invalid_argument(
                "history cache settings require kv_cache_hydration_enabled");
        }
        return handle.release();
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}
void hstu_destroy(void* pointer) noexcept {
    auto* handle = static_cast<Handle*>(pointer);
    if (handle)
        cudaSetDevice(handle->device);
    delete handle;
}
const char* hstu_score(void* pointer, const char* specification) noexcept {
    try {
        auto& handle = *static_cast<Handle*>(pointer);
        std::lock_guard<std::mutex> lock(handle.execute);
        if (cudaSetDevice(handle.device) != cudaSuccess)
            throw std::runtime_error("cannot select native task CUDA device");
        const auto inputs = read_tensors(Json::parse(specification));
        handle.outputs = handle.ranking->score(inputs);
        reply = output_descriptors(handle).dump();
        return reply.c_str();
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}
const char* hstu_cache_stats(void* pointer) noexcept {
    try {
        const auto& handle = *static_cast<Handle*>(pointer);
        const auto s = handle.ranking->cache_stats();
        Json result{{"supported", s.supported},
                    {"budget_bytes", s.budget_bytes},
                    {"reserved_bytes", s.reserved_bytes},
                    {"used_bytes", s.used_bytes},
                    {"peak_bytes", s.peak_bytes},
                    {"active_sequences", s.active_sequences},
                    {"allocation_failures", s.allocation_failures}};
        if (handle.history && handle.history->history_supported())
            result.update(handle.history->history_stats());
        reply = result.dump();
        return reply.c_str();
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}
const char* hstu_history(void* pointer, const char* command) noexcept {
    try {
        auto& handle = *static_cast<Handle*>(pointer);
        std::lock_guard<std::mutex> lock(handle.execute);
        if (cudaSetDevice(handle.device) != cudaSuccess)
            throw std::runtime_error("cannot select history CUDA device");
        if (!handle.history || !handle.history->history_supported())
            throw std::invalid_argument("bundle does not support history restoration");
        const auto request = Json::parse(command);
        const std::string operation = request.at("operation");
        Json result;
        if (operation == "export") {
            handle.snapshot = handle.history->export_history(
                read_tensors(request.at("inputs")),
                request.at("revisions").get<std::vector<std::string>>());
            result = {{"pointer", reinterpret_cast<std::uintptr_t>(handle.snapshot.data())},
                      {"bytes", handle.snapshot.size()}};
        } else if (operation == "restore") {
            const auto size = request.at("bytes").get<std::size_t>();
            const auto* data =
                reinterpret_cast<const std::uint8_t*>(request.at("pointer").get<std::uintptr_t>());
            if (!data || size < 16 || size > 1024ull * 1024 * 1024)
                throw std::invalid_argument("invalid snapshot buffer");
            const std::vector<std::uint8_t> snapshot(data, data + size);
            result = {{"handles", handle.history->restore_history(snapshot)}};
        } else if (operation == "score") {
            handle.outputs = handle.history->score_with_history(
                request.at("handles").get<std::vector<std::string>>(),
                request.at("revisions").get<std::vector<std::string>>(),
                read_tensors(request.at("inputs")));
            result = {{"outputs", output_descriptors(handle)}};
        } else if (operation == "info") {
            result = handle.history->history_info(request.at("handle"));
        } else if (operation == "release") {
            handle.history->release_history(request.at("handles").get<std::vector<std::string>>());
            result = {{"released", true}};
        } else
            throw std::invalid_argument("unknown history operation");
        reply = result.dump();
        return reply.c_str();
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}
}
