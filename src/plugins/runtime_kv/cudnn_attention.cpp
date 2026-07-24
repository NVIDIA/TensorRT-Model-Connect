/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "plugins/runtime_kv/cudnn_attention.h"

#ifndef TRTMC_RUNTIME_KV_CUDNN_SDPA
#define TRTMC_RUNTIME_KV_CUDNN_SDPA 0
#endif

#if TRTMC_RUNTIME_KV_CUDNN_SDPA

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

namespace trtmc::runtime_kv {
namespace {

constexpr int64_t kQueryUid = 1;
constexpr int64_t kKeyUid = 2;
constexpr int64_t kValueUid = 3;
constexpr int64_t kOutputUid = 4;
constexpr int64_t kSequenceQueryUid = 5;
constexpr int64_t kSequenceKvUid = 6;
constexpr int64_t kLogSumExpUid = 7;
constexpr std::size_t kWorkspaceAlignment = 256;
constexpr std::size_t kMinimumCudnnVersion = 92000;

std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
    if (alignment == 0 || value > static_cast<std::size_t>(-1) - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

struct GraphKey {
    int32_t device{};
    int32_t num_query_heads{};
    int32_t num_kv_heads{};
    int32_t head_dim{};
    int32_t chunk_limit{};
    int32_t padded_query_rows{};
    int32_t padded_kv_rows{};
    bool token_major_kv{};
    bool causal{};

    bool operator==(const GraphKey& other) const noexcept {
        return device == other.device && num_query_heads == other.num_query_heads &&
               num_kv_heads == other.num_kv_heads && head_dim == other.head_dim &&
               chunk_limit == other.chunk_limit && padded_query_rows == other.padded_query_rows &&
               padded_kv_rows == other.padded_kv_rows && token_major_kv == other.token_major_kv &&
               causal == other.causal;
    }
};

struct GraphKeyHash {
    std::size_t operator()(const GraphKey& key) const noexcept {
        std::size_t result = 0;
        const auto mix = [&result](int32_t value) {
            result ^= std::hash<int32_t>{}(value) + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        };
        mix(key.device);
        mix(key.num_query_heads);
        mix(key.num_kv_heads);
        mix(key.head_dim);
        mix(key.chunk_limit);
        mix(key.padded_query_rows);
        mix(key.padded_kv_rows);
        mix(key.token_major_kv ? 1 : 0);
        mix(key.causal ? 1 : 0);
        return result;
    }
};

struct PreparedGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    std::size_t workspace_bytes{0};
    std::string plan_name;
};

std::mutex& graph_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<GraphKey, std::weak_ptr<PreparedGraph>, GraphKeyHash>& graph_cache() {
    static std::unordered_map<GraphKey, std::weak_ptr<PreparedGraph>, GraphKeyHash> cache;
    return cache;
}

std::shared_ptr<PreparedGraph> build_graph(cudnnHandle_t handle, const GraphKey& key) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    const int64_t hq = key.num_query_heads;
    const int64_t hkv = key.num_kv_heads;
    const int64_t d = key.head_dim;
    const int64_t sq = key.padded_query_rows;
    const int64_t skv = key.padded_kv_rows;

    auto query = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("query")
                                   .set_uid(kQueryUid)
                                   .set_dim({1, hq, sq, d})
                                   .set_stride({hq * sq * d, sq * d, d, 1}));

    const auto kv_stride = key.token_major_kv ? std::vector<int64_t>{skv * hkv * d, d, hkv * d, 1}
                                              : std::vector<int64_t>{hkv * skv * d, skv * d, d, 1};
    auto key_cache = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name(key.token_major_kv ? "key_history_token_major" : "key_current_head_major")
            .set_uid(kKeyUid)
            .set_dim({1, hkv, skv, d})
            .set_stride(kv_stride));
    auto value_cache = graph->tensor(
        fe::graph::Tensor_attributes()
            .set_name(key.token_major_kv ? "value_history_token_major" : "value_current_head_major")
            .set_uid(kValueUid)
            .set_dim({1, hkv, skv, d})
            .set_stride(kv_stride));

    auto sequence_query = graph->tensor(fe::graph::Tensor_attributes()
                                            .set_name("sequence_length_q")
                                            .set_uid(kSequenceQueryUid)
                                            .set_dim({1, 1, 1, 1})
                                            .set_stride({1, 1, 1, 1})
                                            .set_data_type(fe::DataType_t::INT32));
    auto sequence_kv = graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("sequence_length_kv")
                                         .set_uid(kSequenceKvUid)
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::INT32));

    auto attributes = fe::graph::SDPA_attributes()
                          .set_name(key.causal ? "trtmc_runtime_kv_current_sdpa"
                                               : "trtmc_runtime_kv_history_sdpa")
                          .set_generate_stats(true)
                          .set_attn_scale(1.0F / std::sqrt(static_cast<float>(d)))
                          .set_padding_mask(true)
                          .set_seq_len_q(sequence_query)
                          .set_seq_len_kv(sequence_kv);
    if (key.causal) {
        attributes.set_diagonal_alignment(fe::DiagonalAlignment_t::BOTTOM_RIGHT)
            .set_diagonal_band_right_bound(0);
    }

    auto [output, stats] = graph->sdpa(query, key_cache, value_cache, attributes);
    if (stats == nullptr) {
        throw std::runtime_error("cuDNN inference SDPA did not emit log-sum-exp");
    }
    output->set_output(true)
        .set_uid(kOutputUid)
        .set_dim({1, hq, sq, d})
        .set_stride({hq * sq * d, sq * d, d, 1});
    stats->set_output(true)
        .set_name("log_sum_exp")
        .set_uid(kLogSumExpUid)
        .set_dim({1, hq, sq, 1})
        .set_stride({hq * sq, sq, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT);

    auto status = graph->validate();
    if (!status.is_good()) {
        throw std::runtime_error("cuDNN SDPA graph validation failed: " + status.get_message());
    }
    status = graph->build(handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        throw std::runtime_error("cuDNN SDPA graph build failed: " + status.get_message());
    }
    int64_t workspace_bytes = 0;
    status = graph->get_workspace_size(workspace_bytes);
    if (!status.is_good() || workspace_bytes < 0) {
        throw std::runtime_error("cuDNN SDPA workspace query failed: " + status.get_message());
    }
    if (static_cast<std::uint64_t>(workspace_bytes) > kCudnnAttentionPlanWorkspaceLimit) {
        throw std::runtime_error("cuDNN SDPA plan exceeds the qualified workspace bound");
    }
    std::string plan_name;
    status = graph->get_plan_name(plan_name);
    if (!status.is_good() || plan_name.empty()) {
        throw std::runtime_error("cuDNN SDPA selected-plan identity is unavailable: " +
                                 status.get_message());
    }

    auto prepared = std::make_shared<PreparedGraph>();
    prepared->graph = std::move(graph);
    prepared->workspace_bytes = static_cast<std::size_t>(workspace_bytes);
    prepared->plan_name = std::move(plan_name);
    std::fprintf(stderr,
                 "[trtmc.runtime_kv.plan] schema=1 device=%d role=%s hq=%d hkv=%d d=%d C=%d "
                 "Sq=%d T=%d stats=lse heur=A plan=%s workspace_bytes=%zu cudnn_version=%zu\n",
                 key.device, key.causal ? "current" : "history", key.num_query_heads,
                 key.num_kv_heads, key.head_dim, key.chunk_limit, key.padded_query_rows,
                 key.padded_kv_rows, prepared->plan_name.c_str(), prepared->workspace_bytes,
                 cudnnGetVersion());
    return prepared;
}

std::shared_ptr<PreparedGraph> acquire_graph(cudnnHandle_t handle, const GraphKey& key) {
    std::lock_guard<std::mutex> lock(graph_cache_mutex());
    auto& cached = graph_cache()[key];
    if (auto existing = cached.lock()) {
        return existing;
    }
    auto prepared = build_graph(handle, key);
    cached = prepared;
    return prepared;
}

} // namespace

struct CudnnAttentionExecutor::Impl {
    explicit Impl(CudnnAttentionConfig input_config) : config(input_config) {
        if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
            throw std::runtime_error("unable to resolve CUDA device for cuDNN SDPA");
        }
        if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS || handle == nullptr) {
            handle = nullptr;
            throw std::runtime_error("unable to create cuDNN SDPA handle");
        }
        reset_variant_pack(history_variant_pack);
        reset_variant_pack(current_variant_pack);
    }

    ~Impl() {
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }

    static void reset_variant_pack(std::unordered_map<int64_t, void*>& pack) {
        pack.clear();
        pack.reserve(64);
        pack.emplace(kQueryUid, nullptr);
        pack.emplace(kKeyUid, nullptr);
        pack.emplace(kValueUid, nullptr);
        pack.emplace(kOutputUid, nullptr);
        pack.emplace(kSequenceQueryUid, nullptr);
        pack.emplace(kSequenceKvUid, nullptr);
        pack.emplace(kLogSumExpUid, nullptr);
    }

    CudnnAttentionConfig config;
    int32_t device{-1};
    cudnnHandle_t handle{nullptr};
    GraphKey history_key{};
    GraphKey current_key{};
    std::shared_ptr<PreparedGraph> history_prepared;
    std::shared_ptr<PreparedGraph> current_prepared;
    std::unordered_map<int64_t, void*> history_variant_pack;
    std::unordered_map<int64_t, void*> current_variant_pack;
};

bool native_cudnn_attention_available() noexcept {
    return cudnnGetVersion() >= kMinimumCudnnVersion;
}

std::size_t cudnn_attention_workspace_size(const CudnnAttentionConfig& config,
                                           int32_t padded_query_rows) noexcept {
    if (config.num_query_heads <= 0 || config.num_kv_heads <= 0 || config.head_dim <= 0 ||
        config.chunk_limit <= 0 || padded_query_rows <= 0 ||
        padded_query_rows > config.chunk_limit) {
        return 0;
    }
    std::size_t offset =
        align_up(sizeof(int32_t) * kCudnnAttentionControlScalarCount, kWorkspaceAlignment);
    if (offset == 0) {
        return 0;
    }

    const auto append_region = [&offset](std::uint64_t elements, std::size_t element_bytes) {
        if (elements > static_cast<std::uint64_t>(static_cast<std::size_t>(-1)) / element_bytes) {
            return false;
        }
        const auto bytes = static_cast<std::size_t>(elements) * element_bytes;
        if (offset > static_cast<std::size_t>(-1) - bytes) {
            return false;
        }
        offset = align_up(offset + bytes, kWorkspaceAlignment);
        return offset != 0;
    };
    const auto query_elements = static_cast<std::uint64_t>(config.num_query_heads) *
                                static_cast<std::uint64_t>(padded_query_rows) *
                                static_cast<std::uint64_t>(config.head_dim);
    const auto kv_elements = static_cast<std::uint64_t>(config.num_kv_heads) *
                             static_cast<std::uint64_t>(padded_query_rows) *
                             static_cast<std::uint64_t>(config.head_dim);
    const auto stats_elements = static_cast<std::uint64_t>(config.num_query_heads) *
                                static_cast<std::uint64_t>(padded_query_rows);

    // Packed Q/current K/current V, history/current contexts, and two FP32
    // log-sum-exp arrays. None scale with history length.
    if (!append_region(query_elements, sizeof(std::uint16_t)) ||
        !append_region(kv_elements, sizeof(std::uint16_t)) ||
        !append_region(kv_elements, sizeof(std::uint16_t)) ||
        !append_region(query_elements, sizeof(std::uint16_t)) ||
        !append_region(query_elements, sizeof(std::uint16_t))) {
        return 0;
    }
    for (int copy = 0; copy < 2; ++copy) {
        if (!append_region(stats_elements, sizeof(float))) {
            return 0;
        }
    }
    if (offset > static_cast<std::size_t>(-1) - kCudnnAttentionPlanWorkspaceLimit) {
        return 0;
    }
    return offset + kCudnnAttentionPlanWorkspaceLimit;
}

CudnnAttentionExecutor::CudnnAttentionExecutor(CudnnAttentionConfig config)
    : impl_(std::make_unique<Impl>(config)) {}

CudnnAttentionExecutor::~CudnnAttentionExecutor() = default;

bool CudnnAttentionExecutor::prepare(int32_t padded_kv_rows, int32_t padded_query_rows) noexcept {
    return prepare_history(padded_kv_rows, padded_query_rows) && prepare_current(padded_query_rows);
}

bool CudnnAttentionExecutor::prepare_history(int32_t padded_kv_rows,
                                             int32_t padded_query_rows) noexcept {
    try {
        if (!impl_ || !native_cudnn_attention_available() || padded_kv_rows <= 0 ||
            padded_query_rows <= 0 || padded_query_rows > impl_->config.chunk_limit) {
            return false;
        }
        int current_device = -1;
        if (cudaGetDevice(&current_device) != cudaSuccess || current_device != impl_->device) {
            return false;
        }
        GraphKey const next_history{
            impl_->device,
            impl_->config.num_query_heads,
            impl_->config.num_kv_heads,
            impl_->config.head_dim,
            impl_->config.chunk_limit,
            padded_query_rows,
            padded_kv_rows,
            true,
            false,
        };
        if (!impl_->history_prepared || !(impl_->history_key == next_history)) {
            impl_->history_prepared = acquire_graph(impl_->handle, next_history);
            impl_->history_key = next_history;
            // Graph::execute may add pass-by-value/intermediate UIDs to the
            // caller-owned map. Those entries belong to one fixed graph
            // shape and must not leak across a P/role transition.
            Impl::reset_variant_pack(impl_->history_variant_pack);
        }
        return impl_->history_prepared != nullptr &&
               impl_->history_prepared->workspace_bytes <= kCudnnAttentionPlanWorkspaceLimit;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.runtime_kv] cuDNN SDPA history plan preparation failed: %s\n",
                     error.what());
        return false;
    }
}

bool CudnnAttentionExecutor::prepare_current(int32_t padded_query_rows) noexcept {
    try {
        if (!impl_ || !native_cudnn_attention_available() || padded_query_rows <= 0 ||
            padded_query_rows > impl_->config.chunk_limit) {
            return false;
        }
        int current_device = -1;
        if (cudaGetDevice(&current_device) != cudaSuccess || current_device != impl_->device) {
            return false;
        }
        GraphKey const next_current{
            impl_->device,
            impl_->config.num_query_heads,
            impl_->config.num_kv_heads,
            impl_->config.head_dim,
            impl_->config.chunk_limit,
            padded_query_rows,
            padded_query_rows,
            false,
            true,
        };
        if (!impl_->current_prepared || !(impl_->current_key == next_current)) {
            impl_->current_prepared = acquire_graph(impl_->handle, next_current);
            impl_->current_key = next_current;
            Impl::reset_variant_pack(impl_->current_variant_pack);
        }
        return impl_->current_prepared != nullptr &&
               impl_->current_prepared->workspace_bytes <= kCudnnAttentionPlanWorkspaceLimit;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[trtmc.runtime_kv] cuDNN SDPA current plan preparation failed: %s\n",
                     error.what());
        return false;
    }
}

bool CudnnAttentionExecutor::execute_history(
    const void* query, const void* history_k, const void* history_v, void* history_context,
    void* history_log_sum_exp, const int32_t* sequence_length_q,
    const int32_t* sequence_length_history, void* plan_workspace,
    std::size_t plan_workspace_capacity, cudaStream_t stream) noexcept {
    try {
        if (!impl_ || !impl_->history_prepared || query == nullptr || history_k == nullptr ||
            history_v == nullptr || history_context == nullptr || history_log_sum_exp == nullptr ||
            sequence_length_q == nullptr || sequence_length_history == nullptr ||
            (impl_->history_prepared->workspace_bytes > 0 && plan_workspace == nullptr) ||
            impl_->history_prepared->workspace_bytes > plan_workspace_capacity ||
            cudnnSetStream(impl_->handle, stream) != CUDNN_STATUS_SUCCESS) {
            return false;
        }
        auto& pack = impl_->history_variant_pack;
        pack[kQueryUid] = const_cast<void*>(query);
        pack[kKeyUid] = const_cast<void*>(history_k);
        pack[kValueUid] = const_cast<void*>(history_v);
        pack[kOutputUid] = history_context;
        pack[kLogSumExpUid] = history_log_sum_exp;
        pack[kSequenceQueryUid] = const_cast<int32_t*>(sequence_length_q);
        pack[kSequenceKvUid] = const_cast<int32_t*>(sequence_length_history);
        return impl_->history_prepared->graph->execute(impl_->handle, pack, plan_workspace)
            .is_good();
    } catch (...) {
        return false;
    }
}

bool CudnnAttentionExecutor::execute_current(
    const void* query, const void* current_k, const void* current_v, void* current_context,
    void* current_log_sum_exp, const int32_t* sequence_length_q,
    const int32_t* sequence_length_current, void* plan_workspace,
    std::size_t plan_workspace_capacity, cudaStream_t stream) noexcept {
    try {
        if (!impl_ || !impl_->current_prepared || query == nullptr || current_k == nullptr ||
            current_v == nullptr || current_context == nullptr || current_log_sum_exp == nullptr ||
            sequence_length_q == nullptr || sequence_length_current == nullptr ||
            (impl_->current_prepared->workspace_bytes > 0 && plan_workspace == nullptr) ||
            impl_->current_prepared->workspace_bytes > plan_workspace_capacity ||
            cudnnSetStream(impl_->handle, stream) != CUDNN_STATUS_SUCCESS) {
            return false;
        }
        auto& pack = impl_->current_variant_pack;
        pack[kQueryUid] = const_cast<void*>(query);
        pack[kKeyUid] = const_cast<void*>(current_k);
        pack[kValueUid] = const_cast<void*>(current_v);
        pack[kOutputUid] = current_context;
        pack[kLogSumExpUid] = current_log_sum_exp;
        pack[kSequenceQueryUid] = const_cast<int32_t*>(sequence_length_q);
        pack[kSequenceKvUid] = const_cast<int32_t*>(sequence_length_current);
        return impl_->current_prepared->graph->execute(impl_->handle, pack, plan_workspace)
            .is_good();
    } catch (...) {
        return false;
    }
}

bool CudnnAttentionExecutor::execute_segmented(
    const void* query, const void* history_k, const void* history_v, const void* current_k,
    const void* current_v, void* history_context, void* current_context, void* history_log_sum_exp,
    void* current_log_sum_exp, const int32_t* sequence_length_q,
    const int32_t* sequence_length_history, const int32_t* sequence_length_current,
    void* plan_workspace, std::size_t plan_workspace_capacity, cudaStream_t stream) noexcept {
    try {
        if (!impl_ || !impl_->history_prepared || !impl_->current_prepared || query == nullptr ||
            history_k == nullptr || history_v == nullptr || current_k == nullptr ||
            current_v == nullptr || history_context == nullptr || current_context == nullptr ||
            history_log_sum_exp == nullptr || current_log_sum_exp == nullptr ||
            sequence_length_q == nullptr || sequence_length_history == nullptr ||
            sequence_length_current == nullptr ||
            ((impl_->history_prepared->workspace_bytes > 0 ||
              impl_->current_prepared->workspace_bytes > 0) &&
             plan_workspace == nullptr) ||
            std::max(impl_->history_prepared->workspace_bytes,
                     impl_->current_prepared->workspace_bytes) > plan_workspace_capacity ||
            cudnnSetStream(impl_->handle, stream) != CUDNN_STATUS_SUCCESS) {
            return false;
        }

        auto& history = impl_->history_variant_pack;
        history[kQueryUid] = const_cast<void*>(query);
        history[kKeyUid] = const_cast<void*>(history_k);
        history[kValueUid] = const_cast<void*>(history_v);
        history[kOutputUid] = history_context;
        history[kLogSumExpUid] = history_log_sum_exp;
        history[kSequenceQueryUid] = const_cast<int32_t*>(sequence_length_q);
        history[kSequenceKvUid] = const_cast<int32_t*>(sequence_length_history);
        auto status =
            impl_->history_prepared->graph->execute(impl_->handle, history, plan_workspace);
        if (!status.is_good()) {
            return false;
        }

        auto& current = impl_->current_variant_pack;
        current[kQueryUid] = const_cast<void*>(query);
        current[kKeyUid] = const_cast<void*>(current_k);
        current[kValueUid] = const_cast<void*>(current_v);
        current[kOutputUid] = current_context;
        current[kLogSumExpUid] = current_log_sum_exp;
        current[kSequenceQueryUid] = const_cast<int32_t*>(sequence_length_q);
        current[kSequenceKvUid] = const_cast<int32_t*>(sequence_length_current);
        status = impl_->current_prepared->graph->execute(impl_->handle, current, plan_workspace);
        return status.is_good();
    } catch (...) {
        return false;
    }
}

std::unique_ptr<CudnnAttentionExecutor>
make_cudnn_attention_executor(CudnnAttentionConfig config) noexcept {
    try {
        if (!native_cudnn_attention_available()) {
            return nullptr;
        }
        return std::make_unique<CudnnAttentionExecutor>(config);
    } catch (...) {
        return nullptr;
    }
}

} // namespace trtmc::runtime_kv

#else

namespace trtmc::runtime_kv {

bool native_cudnn_attention_available() noexcept {
    return false;
}

std::size_t cudnn_attention_workspace_size(const CudnnAttentionConfig&, int32_t) noexcept {
    return 0;
}

struct CudnnAttentionExecutor::Impl {};

CudnnAttentionExecutor::CudnnAttentionExecutor(CudnnAttentionConfig)
    : impl_(std::make_unique<Impl>()) {}

CudnnAttentionExecutor::~CudnnAttentionExecutor() = default;

bool CudnnAttentionExecutor::prepare(int32_t, int32_t) noexcept {
    return false;
}

bool CudnnAttentionExecutor::prepare_history(int32_t, int32_t) noexcept {
    return false;
}

bool CudnnAttentionExecutor::prepare_current(int32_t) noexcept {
    return false;
}

bool CudnnAttentionExecutor::execute_history(const void*, const void*, const void*, void*, void*,
                                             const int32_t*, const int32_t*, void*, std::size_t,
                                             cudaStream_t) noexcept {
    return false;
}

bool CudnnAttentionExecutor::execute_current(const void*, const void*, const void*, void*, void*,
                                             const int32_t*, const int32_t*, void*, std::size_t,
                                             cudaStream_t) noexcept {
    return false;
}

bool CudnnAttentionExecutor::execute_segmented(const void*, const void*, const void*, const void*,
                                               const void*, void*, void*, void*, void*,
                                               const int32_t*, const int32_t*, const int32_t*,
                                               void*, std::size_t, cudaStream_t) noexcept {
    return false;
}

std::unique_ptr<CudnnAttentionExecutor>
make_cudnn_attention_executor(CudnnAttentionConfig) noexcept {
    return nullptr;
}

} // namespace trtmc::runtime_kv

#endif
