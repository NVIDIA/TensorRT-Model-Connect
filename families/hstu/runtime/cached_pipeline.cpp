/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/cached_pipeline.h"

#include "families/hstu/runtime/cache_policy.h"
#include "families/hstu/runtime/paged_cache.h"
#include "families/hstu/runtime/request.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

namespace trtmc::hstu {
namespace {

using Json = nlohmann::json;
constexpr const char* kFormat = "hstu.native_history.v2";
constexpr const char* kLegacyFormat = "hstu.native_history.v1";

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("hstu ") + operation + ": " +
                                 cudaGetErrorString(result));
}

std::string cache_name(int layer, const char* kind) {
    return "cache_" + std::to_string(layer) + "_" + kind;
}

HistorySignature parse_signature(const HistoryCacheValue& value, std::size_t max_tokens) {
    if (value.schema_version != kHistoryCacheInterfaceVersion)
        throw std::invalid_argument("incompatible history snapshot format");
    if (value.format == kFormat)
        return decode_history_signature(value.metadata, max_tokens);
    if (value.format != kLegacyFormat)
        throw std::invalid_argument("incompatible history snapshot format");
    // Existing persisted snapshots remain readable across the metadata upgrade.
    auto json = Json::parse(value.metadata.begin(), value.metadata.end());
    HistorySignature result;
    result.model_namespace = json.at("model_namespace").get<std::string>();
    result.feature_version = json.at("feature_version").get<std::string>();
    result.history_revision = json.at("history_revision").get<std::string>();
    result.token_rows = json.at("tokens").get<std::vector<std::int32_t>>();
    result.position_ids = json.at("positions").get<std::vector<std::int32_t>>();
    result.time_ids = json.at("times").get<std::vector<std::int32_t>>();
    result.effective_scale = json.at("scale").get<float>();
    result.contextual_length = json.at("contextual_length").get<std::int32_t>();
    return result;
}

const HistoryCacheTensor& tensor(const HistoryCacheValue& value, std::string_view name) {
    const auto found = std::find_if(value.tensors.begin(), value.tensors.end(),
                                    [&](const auto& item) { return item.name == name; });
    if (found == value.tensors.end())
        throw std::invalid_argument("missing history tensor " + std::string(name));
    return *found;
}

const HistoryCacheTensor& kv_tensor(const HistoryCacheValue& value, int layer, int kind) {
    if (value.format == kFormat)
        return tensor(value, "history_kv");
    if (value.format == kLegacyFormat)
        return tensor(value, cache_name(layer, kind ? "v" : "k"));
    throw std::invalid_argument("incompatible history snapshot format");
}

std::size_t shape_bytes(const std::vector<std::int64_t>& shape, DType dtype) {
    auto bytes = dtype_size(dtype);
    if (!bytes)
        throw std::invalid_argument("invalid history tensor dtype");
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::size_t>(dimension) > std::numeric_limits<std::size_t>::max() / bytes)
            throw std::invalid_argument("invalid history tensor byte extent");
        bytes *= static_cast<std::size_t>(dimension);
    }
    return bytes;
}

void validate_device_kv(const HistoryCacheTensor& item, const std::vector<std::int64_t>& shape,
                        std::size_t bytes, DType dtype, int device) {
    if (!item.host_data.empty() && item.host_data.size() != bytes)
        throw std::invalid_argument("invalid historical K/V host mirror");
    if (!item.device->ok() || item.device->shape() != shape || item.device->dtype() != dtype ||
        item.device->nbytes() != bytes)
        throw std::invalid_argument("invalid historical K/V device tensor");
    cudaPointerAttributes attributes{};
    cuda_check(cudaPointerGetAttributes(&attributes, item.device->data()), "cache pointer");
    if (attributes.device != device || attributes.type != cudaMemoryTypeDevice)
        throw std::invalid_argument("history snapshot belongs to another device");
}

void validate_kv_tensor(const HistoryCacheTensor& item, const std::vector<std::int64_t>& shape,
                        DType dtype, int device) {
    const auto bytes = shape_bytes(shape, dtype);
    if (item.shape != shape || item.dtype != dtype)
        throw std::invalid_argument("invalid historical K/V shape or dtype");
    if (item.device)
        validate_device_kv(item, shape, bytes, dtype, device);
    else if (item.host_data.size() != bytes)
        throw std::invalid_argument("invalid historical K/V byte length");
}

void validate_snapshot(const HistoryCacheValue& value, const RuntimeConfig& config,
                       std::size_t length, DType dtype, int device) {
    if (value.format != kFormat && value.format != kLegacyFormat)
        throw std::invalid_argument("incompatible history snapshot format");
    const auto tensor_count = value.format == kFormat
                                  ? std::size_t{2}
                                  : 2 * static_cast<std::size_t>(config.num_layers) + 1;
    if (length == 0 || length > static_cast<std::size_t>(config.max_sequence_length) ||
        value.tensors.size() != tensor_count)
        throw std::invalid_argument("invalid history snapshot dimensions");
    const auto& embeddings = tensor(value, "embeddings");
    const std::vector<std::int64_t> embedding_shape{static_cast<std::int64_t>(length),
                                                    config.hidden_size};
    if (embeddings.shape != embedding_shape || embeddings.dtype != DType::kFloat32 ||
        embeddings.device ||
        embeddings.host_data.size() != shape_bytes(embedding_shape, DType::kFloat32))
        throw std::invalid_argument("invalid history embedding snapshot");
    if (value.format == kFormat) {
        validate_kv_tensor(tensor(value, "history_kv"),
                           {config.num_layers, 2, config.num_heads,
                            static_cast<std::int64_t>(length), config.head_dim},
                           dtype, device);
        return;
    }
    for (int layer = 0; layer < config.num_layers; ++layer) {
        for (int kind = 0; kind < 2; ++kind)
            validate_kv_tensor(
                kv_tensor(value, layer, kind),
                {config.num_heads, static_cast<std::int64_t>(length), config.head_dim}, dtype,
                device);
    }
}

const float* output(const TensorMap& values, const char* name, std::int64_t batch,
                    std::int64_t rows, std::int64_t width, bool compact = false) {
    const auto found = values.find(name);
    const auto shape = compact ? std::vector<std::int64_t>{rows, width}
                               : std::vector<std::int64_t>{batch, rows, width};
    if (found == values.end() || found->second.dtype != DType::kFloat32 ||
        found->second.shape != shape || !found->second.data)
        throw std::runtime_error(std::string("hstu malformed output ") + name);
    return static_cast<const float*>(found->second.data);
}

void finite(const std::vector<float>& values) {
    if (!all_finite(values))
        throw std::runtime_error("hstu returned nonfinite output");
}

struct Frame {
    Sequence sequence;
    HistorySignature signature;
    HistoryCache::Lease lease;
    CacheReuse reuse;
    std::vector<std::int32_t> positions, times;
    std::vector<float> embeddings;
    bool keyed{false};
    bool device_snapshot{false};
    bool direct_workspace{false};
};

struct LocalSessionState {
    HistorySignature signature;
    std::vector<float> embeddings;
    bool valid{false};
};

struct RestoredPageSnapshot {
    std::weak_ptr<const HistoryCacheValue> value;
    std::string owner;
};

struct QueryBatch {
    std::size_t rows{0}, candidates{1};
    std::vector<std::int32_t> tokens, positions, times, write_indices, active_lengths;
    std::vector<std::int32_t> packed_rows, update_lengths, candidate_tokens;
    AttentionMask mask;
    float scale{1.0F};
    std::int32_t empty_row{0};
    // Destroy the lease before host input vectors: failure must drain their H2D.
    PagedCacheArena::Lease page_lease;
};

} // namespace

struct CachedPipeline::Impl {
    RuntimeConfig config;
    std::shared_ptr<HistoryCache> cache;
    std::mutex mutex;
    int device{0};
    DType dtype{DType::kFloat32};
    std::unique_ptr<ITrtModule> engine, candidates, prefill;
    std::vector<std::unique_ptr<DeviceTensor>> workspace;
    std::unique_ptr<PagedCacheArena> pages;
    std::vector<RestoredPageSnapshot> restored_snapshots;

    Impl(std::unique_ptr<ITrtModule> model, std::unique_ptr<ITrtModule> lookup, RuntimeConfig cfg,
         std::shared_ptr<HistoryCache> history_cache, std::unique_ptr<ITrtModule> prefill_model)
        : config(std::move(cfg)), cache(std::move(history_cache)), engine(std::move(model)),
          candidates(std::move(lookup)), prefill(std::move(prefill_model)) {
        if (!config.enable_history_cache || !engine || !engine->ok())
            throw std::invalid_argument("hstu cached runtime requires a native KV engine");
        cuda_check(cudaGetDevice(&device), "get device");
        dtype = engine->tensor_dtype(paged() ? "cache_0_pages" : "cache_0_k");
        validate_executor(*engine);
        if (paged()) {
            if (prefill || config.mode != "ranking")
                throw std::invalid_argument("hstu paged attention uses one ranking engine");
            pages = make_pages(config.max_batch_size, std::numeric_limits<std::size_t>::max());
            restored_snapshots.resize(config.max_batch_size);
            return;
        }
        if (prefill) {
            if (!prefill->ok())
                throw std::invalid_argument("hstu prefill requires a valid native KV engine");
            validate_executor(*prefill);
            for (int layer = 0; layer < config.num_layers; ++layer)
                for (const auto* kind : {"k", "v"}) {
                    const auto name = cache_name(layer, kind);
                    if (prefill->tensor_shape(name) != engine->tensor_shape(name))
                        throw std::invalid_argument(
                            "hstu prefill cache capacity does not match engine.plan");
                }
        }
        for (int layer = 0; layer < config.num_layers; ++layer) {
            for (int kind = 0; kind < 2; ++kind) {
                workspace.push_back(std::make_unique<DeviceTensor>(
                    std::vector<std::int64_t>{config.max_batch_size, config.num_heads,
                                              config.max_sequence_length, config.head_dim},
                    dtype, engine->stream()));
                if (!workspace.back()->ok())
                    throw std::runtime_error("hstu KV workspace allocation failed");
            }
        }
        if (config.mode == "retrieval" && (!candidates || !candidates->ok()))
            throw std::invalid_argument("hstu cached retrieval requires candidate.plan");
    }

    bool paged() const { return engine->has_input("attention_metadata"); }

    std::unique_ptr<PagedCacheArena> make_pages(std::int32_t slots, std::size_t budget) const {
        PagedCacheGeometry geometry{config.num_layers,
                                    config.num_heads,
                                    config.head_dim,
                                    slots,
                                    config.max_sequence_length,
                                    128,
                                    dtype,
                                    budget};
        return std::make_unique<PagedCacheArena>(
            PagedCacheArena::create_cuda(geometry, engine->stream()));
    }

    void validate_executor(ITrtModule& executor) const {
        if (executor.has_input("attention_metadata")) {
            const auto metadata_shape = executor.tensor_shape("attention_metadata");
            if (dtype != DType::kBFloat16 || config.num_heads != 4 || config.head_dim != 64 ||
                config.scaling_seqlen != 1024 || config.max_sequence_length > 1024 ||
                !config.is_causal || config.target_group_size != 1 || config.time_buckets ||
                executor.tensor_shape("token_ids").size() != 1 ||
                executor.tensor_dtype("attention_metadata") != DType::kInt32 ||
                metadata_shape.size() != 3 || metadata_shape[0] != 5 || metadata_shape[2] != 8)
                throw std::invalid_argument("hstu paged attention contract mismatch");
            for (int layer = 0; layer < config.num_layers; ++layer) {
                const auto suffix = std::to_string(layer) + "_pages";
                const auto shape = executor.tensor_shape("cache_" + suffix);
                if (!executor.has_input("cache_" + suffix) ||
                    !executor.has_output("present_" + suffix) ||
                    executor.tensor_dtype("cache_" + suffix) != dtype || shape.size() != 4 ||
                    shape[1] != 2 || shape[2] != 128 ||
                    shape[3] != config.num_heads * config.head_dim)
                    throw std::invalid_argument("hstu page cache binding missing");
            }
            for (const auto* name :
                 {"page_update_rows", "page_update_lengths", "page_write_indices"})
                if (!executor.has_input(name) || executor.tensor_dtype(name) != DType::kInt32)
                    throw std::invalid_argument("hstu page update metadata missing");
            return;
        }
        if (executor.has_input("cache_update_rows") != executor.has_input("cache_update_lengths"))
            throw std::invalid_argument("hstu packed update inputs must be declared together");
        const auto* mask_name = attention_input_name(executor);
        if (!executor.has_input(mask_name) || executor.tensor_shape(mask_name).size() != 4 ||
            executor.tensor_dtype(mask_name) !=
                (prepared_attention(executor) ? dtype : DType::kFloat32))
            throw std::invalid_argument("hstu native attention input contract mismatch");
        if (!prepared_attention(executor) && !executor.has_input("scaling_seqlen"))
            throw std::invalid_argument("hstu legacy attention requires scaling_seqlen");
        for (int layer = 0; layer < config.num_layers; ++layer)
            for (const auto* kind : {"k", "v"}) {
                const auto input_name = cache_name(layer, kind);
                const auto output_name = "present_" + std::to_string(layer) + "_" + kind;
                if (!executor.has_input(input_name) || !executor.has_output(output_name) ||
                    executor.tensor_dtype(input_name) != dtype)
                    throw std::invalid_argument("hstu native KV engine contract mismatch");
            }
    }

    ITrtModule& select_executor(const std::vector<Frame>& frames) const {
        if (prefill && std::all_of(frames.begin(), frames.end(), [](const auto& frame) {
                return frame.reuse.reused_tokens == 0;
            }))
            return *prefill;
        return *engine;
    }

    Frame prepare_frame(const RecommendationSequence& request, float scale,
                        const LocalSessionState* local = nullptr) {
        Frame frame;
        frame.sequence = assemble(request, config);
        const auto history = static_cast<std::size_t>(frame.sequence.history_end);
        if (config.position_buckets) {
            frame.positions.resize(frame.sequence.tokens.size());
            fill_positions(frame.sequence, config, frame.positions.data());
        }
        if (config.time_buckets) {
            frame.times.resize(frame.sequence.tokens.size());
            fill_times(request, frame.sequence, config, frame.times.data());
        } else if (!request.token_timestamps.empty()) {
            throw std::invalid_argument("hstu bundle does not use timestamp embeddings");
        }
        auto& signature = frame.signature;
        signature.model_namespace = config.cache_artifact_id;
        signature.feature_version = request.cache.feature_version;
        signature.history_revision = request.cache.history_epoch;
        signature.token_rows.assign(frame.sequence.tokens.begin(),
                                    frame.sequence.tokens.begin() + history);
        if (!frame.positions.empty())
            signature.position_ids.assign(frame.positions.begin(),
                                          frame.positions.begin() + history);
        if (!frame.times.empty())
            signature.time_ids.assign(frame.times.begin(), frame.times.begin() + history);
        signature.effective_scale = scale;
        signature.contextual_length = frame.sequence.contextual_length;
        frame.embeddings.resize(frame.sequence.tokens.size() * config.hidden_size);
        if (local) {
            frame.direct_workspace = true;
            frame.reuse = plan_cache_reuse(
                local->valid ? &local->signature : nullptr, signature,
                {config.is_causal, config.mode == "ranking", config.disable_contextual_mask});
            if (frame.reuse.reused_tokens)
                std::copy(local->embeddings.begin(), local->embeddings.end(),
                          frame.embeddings.begin());
            return frame;
        }
        frame.keyed = cache && !request.cache.subject_id.empty();
        if (!frame.keyed)
            return frame;
        if (request.cache.feature_version.empty() || request.cache.history_epoch.empty())
            throw std::invalid_argument(
                "hstu cached requests require feature_version and history_epoch");
        frame.lease = cache->lookup({config.cache_artifact_id, request.cache.feature_version,
                                     request.cache.subject_id, request.cache.history_epoch});
        if (!frame.lease.value)
            return frame;
        try {
            const auto previous = parse_signature(*frame.lease.value, config.max_sequence_length);
            validate_snapshot(*frame.lease.value, config, previous.token_rows.size(), dtype,
                              device);
            frame.reuse = plan_cache_reuse(
                &previous, signature,
                {config.is_causal, config.mode == "ranking", config.disable_contextual_mask});
            if (frame.reuse.reused_tokens) {
                const auto& stored = tensor(*frame.lease.value, "embeddings");
                std::memcpy(frame.embeddings.data(), stored.host_data.data(),
                            stored.host_data.size());
                frame.device_snapshot = bool(kv_tensor(*frame.lease.value, 0, 0).device);
            }
        } catch (const std::exception&) {
            frame.reuse = {CacheReuseKind::kRecompute, CacheReuseReason::kInvalidSignature, 0};
        }
        return frame;
    }

    QueryBatch prepare_batch(const RecommendationRequest& request, const std::vector<Frame>& frames,
                             float scale, ITrtModule& executor) {
        QueryBatch data;
        const auto batch = frames.size();
        std::size_t logical_length = 0;
        data.scale = scale;
        for (const auto& frame : frames) {
            logical_length = std::max(logical_length, frame.sequence.tokens.size());
            data.rows =
                std::max(data.rows, frame.sequence.tokens.size() - frame.reuse.reused_tokens);
            data.candidates =
                std::max(data.candidates, static_cast<std::size_t>(frame.sequence.candidates));
        }
        data.tokens.resize(batch * data.rows);
        data.positions.resize(batch * data.rows);
        data.times.resize(batch * data.rows);
        data.mask =
            make_attention_mask(batch, data.rows, attention_key_width(executor, logical_length),
                                executor.tensor_dtype(attention_input_name(executor)),
                                prepared_attention(executor) ? 1.0F / scale : 1.0F,
                                executor.has_input("attention_weights_transposed"));
        data.candidate_tokens.resize(batch * data.candidates);
        const bool packed_updates = executor.has_input("cache_update_rows");
        if (packed_updates)
            data.update_lengths.push_back(0);
        for (std::size_t sample = 0; sample < batch; ++sample) {
            const auto& frame = frames[sample];
            const auto prefix = frame.reuse.reused_tokens;
            const auto length = frame.sequence.tokens.size();
            data.write_indices.push_back(static_cast<std::int32_t>(prefix));
            data.active_lengths.push_back(static_cast<std::int32_t>(length));
            for (std::size_t index = prefix; index < length; ++index) {
                const auto row = sample * data.rows + index - prefix;
                data.tokens[row] = frame.sequence.tokens[index];
                if (!frame.positions.empty())
                    data.positions[row] = frame.positions[index];
                if (!frame.times.empty())
                    data.times[row] = frame.times[index];
                if (packed_updates)
                    data.packed_rows.push_back(static_cast<std::int32_t>(row));
            }
            fill_attention_mask(data.mask, sample, frame.sequence, config, prefix);
            if (packed_updates)
                data.update_lengths.push_back(static_cast<std::int32_t>(data.packed_rows.size()));
            if (config.mode == "retrieval") {
                const auto* item = find_role(config, "item");
                for (std::size_t index = 0;
                     index < request.sequences[sample].candidate_item_ids.size(); ++index)
                    data.candidate_tokens[sample * data.candidates + index] =
                        lookup(*item, request.sequences[sample].candidate_item_ids[index]);
            }
        }
        return data;
    }

    HistoryCacheTensor compact_history(const HistoryCacheValue& value, ITrtModule& executor) const {
        if (value.format == kFormat)
            return tensor(value, "history_kv");
        const auto length = kv_tensor(value, 0, 0).shape.at(1);
        HistoryCacheTensor result;
        result.name = "history_kv";
        result.shape = {config.num_layers, 2, config.num_heads, length, config.head_dim};
        result.dtype = dtype;
        (void)shape_bytes(result.shape, dtype);
        auto buffer = std::make_shared<DeviceTensor>(result.shape, dtype, nullptr);
        if (!buffer->ok())
            throw std::runtime_error("hstu legacy history staging allocation failed");
        const auto bytes = shape_bytes({config.num_heads, length, config.head_dim}, dtype);
        try {
            for (int layer = 0; layer < config.num_layers; ++layer)
                for (int kind = 0; kind < 2; ++kind) {
                    const auto& saved = kv_tensor(value, layer, kind);
                    const auto* source =
                        saved.device ? saved.device->data() : saved.host_data.data();
                    auto* target = static_cast<std::uint8_t*>(buffer->data()) +
                                   (2 * static_cast<std::size_t>(layer) + kind) * bytes;
                    cuda_check(cudaMemcpyAsync(target, source, bytes,
                                               saved.device ? cudaMemcpyDeviceToDevice
                                                            : cudaMemcpyHostToDevice,
                                               executor.stream()),
                               "stage legacy history");
                }
            executor.sync();
        } catch (...) {
            executor.sync();
            throw;
        }
        result.device = std::move(buffer);
        return result;
    }

    QueryBatch prepare_paged_batch(const RecommendationRequest& request,
                                   const std::vector<Frame>& frames, bool session,
                                   ITrtModule& executor) {
        std::vector<PagedCacheRequest> admitted;
        for (std::size_t sample = 0; sample < frames.size(); ++sample) {
            const auto& frame = frames[sample];
            const auto slot = static_cast<std::int32_t>(sample);
            const auto owner = session ? std::string("session")
                                       : "request:" + request.sequences[sample].cache.subject_id;
            if (!frame.direct_workspace) {
                auto& previous = restored_snapshots.at(sample);
                const bool exact = frame.reuse.kind == CacheReuseKind::kExactHistory;
                const bool retained = !session && exact && previous.owner == owner &&
                                      previous.value.lock() == frame.lease.value &&
                                      frame.lease.value;
                if (!retained)
                    previous = {};
                if (frame.reuse.reused_tokens) {
                    if (!retained) {
                        auto compact = compact_history(*frame.lease.value, executor);
                        pages->initialize_history(
                            slot, owner,
                            parse_signature(*frame.lease.value, config.max_sequence_length),
                            compact);
                    }
                    if (!session && exact)
                        previous = {frame.lease.value, owner};
                } else {
                    pages->invalidate(slot);
                }
            }
            admitted.push_back({slot, owner, frame.signature,
                                static_cast<std::int32_t>(frame.sequence.tokens.size())});
        }
        QueryBatch data;
        data.page_lease = pages->prepare(
            admitted, {config.is_causal, config.mode == "ranking", config.disable_contextual_mask});
        const auto& plan = data.page_lease.plan();
        data.rows = static_cast<std::size_t>(plan.total_queries());
        data.tokens.reserve(data.rows);
        data.positions.reserve(data.rows);
        for (std::size_t sample = 0; sample < frames.size(); ++sample) {
            const auto& frame = frames[sample];
            const auto prefix = frame.reuse.reused_tokens;
            if (plan.reused_tokens[sample] != static_cast<std::int32_t>(prefix))
                throw std::runtime_error("hstu history and page ownership disagree");
            data.tokens.insert(data.tokens.end(), frame.sequence.tokens.begin() + prefix,
                               frame.sequence.tokens.end());
            if (!frame.positions.empty())
                data.positions.insert(data.positions.end(), frame.positions.begin() + prefix,
                                      frame.positions.end());
        }
        return data;
    }

    TensorMap execute_paged(QueryBatch& data, ITrtModule& executor) {
        if (!data.rows)
            return {};
        const auto& plan = data.page_lease.plan();
        const auto count = static_cast<std::int64_t>(plan.binding_pages());
        const auto width = config.num_heads * config.head_dim;
        for (int layer = 0; layer < config.num_layers; ++layer) {
            const auto suffix = std::to_string(layer) + "_pages";
            // Keep the arena's full layer stride; only the TensorRT view is shortened.
            auto* pointer = data.page_lease.layer_data(layer);
            executor.bind_external("cache_" + suffix, pointer, {count, 2, 128, width});
            if (executor.device_ptr("present_" + suffix) != pointer)
                throw std::runtime_error("hstu native page output did not alias its input");
        }
        const auto rows = static_cast<std::int64_t>(data.rows);
        auto vector_input = [&](const std::vector<std::int32_t>& value, std::int64_t length) {
            return Tensor{value.empty() ? &data.empty_row : const_cast<std::int32_t*>(value.data()),
                          {length},
                          DType::kInt32};
        };
        TensorMap input{
            {"token_ids", {data.tokens.data(), {rows}, DType::kInt32}},
            {"page_write_indices", vector_input(plan.page_write_indices, count)},
            {"page_update_rows",
             vector_input(plan.page_update_rows,
                          static_cast<std::int64_t>(plan.page_update_rows.size()))},
            {"page_update_lengths", vector_input(plan.page_update_lengths, count + 1)},
            {"attention_metadata",
             {const_cast<std::int32_t*>(plan.packed_attention_metadata.data()),
              {5, static_cast<std::int64_t>(plan.active_requests.size() + 1), 8},
              DType::kInt32}},
        };
        if (config.position_buckets)
            input["position_ids"] = {data.positions.data(), {rows}, DType::kInt32};
        try {
            return executor.forward(input);
        } catch (...) {
            executor.sync();
            throw;
        }
    }

    void bind_workspace(const std::vector<Frame>& frames, ITrtModule& executor) {
        const auto capacity = static_cast<std::size_t>(config.max_sequence_length);
        const auto pitch = capacity * config.head_dim * dtype_size(dtype);
        for (int layer = 0; layer < config.num_layers; ++layer) {
            for (int kind = 0; kind < 2; ++kind) {
                auto& buffer = *workspace[2 * layer + kind];
                const auto name = cache_name(layer, kind ? "v" : "k");
                for (std::size_t sample = 0; sample < frames.size(); ++sample) {
                    const auto& frame = frames[sample];
                    if (!frame.reuse.reused_tokens || frame.direct_workspace)
                        continue;
                    const auto& saved = kv_tensor(*frame.lease.value, layer, kind);
                    const auto width =
                        frame.reuse.reused_tokens * config.head_dim * dtype_size(dtype);
                    auto* destination = static_cast<std::uint8_t*>(buffer.data()) +
                                        sample * config.num_heads * pitch;
                    const auto* source = static_cast<const std::uint8_t*>(
                        saved.device ? saved.device->data() : saved.host_data.data());
                    if (frame.lease.value->format == kFormat)
                        source +=
                            (2 * static_cast<std::size_t>(layer) + kind) * config.num_heads * width;
                    cuda_check(cudaMemcpy2DAsync(
                                   destination, pitch, source, width, width, config.num_heads,
                                   saved.device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                   executor.stream()),
                               "restore history KV");
                }
                executor.bind_external(name, buffer.data(),
                                       {static_cast<std::int64_t>(frames.size()), config.num_heads,
                                        config.max_sequence_length, config.head_dim});
                const auto present = "present_" + std::to_string(layer) + (kind ? "_v" : "_k");
                if (executor.device_ptr(present) != buffer.data())
                    throw std::runtime_error("hstu native KV output did not alias its input");
            }
        }
    }

    TensorMap execute(QueryBatch& data, const std::vector<Frame>& frames, ITrtModule& executor) {
        const auto batch = static_cast<std::int64_t>(frames.size());
        const auto rows = static_cast<std::int64_t>(data.rows);
        TensorMap candidate_input{{"candidate_token_ids",
                                   {data.candidate_tokens.data(),
                                    {batch, static_cast<std::int64_t>(data.candidates)},
                                    DType::kInt32}}};
        if (rows == 0) {
            if (config.mode != "retrieval")
                return {};
            try {
                return candidates->forward(candidate_input);
            } catch (...) {
                candidates->sync();
                throw;
            }
        }
        bind_workspace(frames, executor);
        TensorMap input = {
            {"token_ids", {data.tokens.data(), {batch, rows}, DType::kInt32}},
            {attention_input_name(executor), data.mask.tensor()},
            {"cache_write_indices", {data.write_indices.data(), {batch}, DType::kInt32}},
            {"cache_active_lengths", {data.active_lengths.data(), {batch}, DType::kInt32}},
        };
        if (executor.has_input("cache_update_rows")) {
            input["cache_update_rows"] = {data.packed_rows.empty() ? &data.empty_row
                                                                   : data.packed_rows.data(),
                                          {static_cast<std::int64_t>(data.packed_rows.size())},
                                          DType::kInt32};
            input["cache_update_lengths"] = {
                data.update_lengths.data(), {batch + 1}, DType::kInt32};
        }
        if (executor.has_input("scaling_seqlen"))
            input["scaling_seqlen"] = {&data.scale, {1}, DType::kFloat32};
        if (config.position_buckets)
            input["position_ids"] = {data.positions.data(), {batch, rows}, DType::kInt32};
        if (config.time_buckets)
            input["time_ids"] = {data.times.data(), {batch, rows}, DType::kInt32};
        if (config.mode == "retrieval")
            input.insert(candidate_input.begin(), candidate_input.end());
        try {
            return executor.forward(input);
        } catch (...) {
            // Input storage belongs to QueryBatch and must outlive pending copies.
            executor.sync();
            throw;
        }
    }

    HistoryCacheValue snapshot(const Frame& frame, std::size_t sample, ITrtModule& executor) {
        HistoryCacheValue value;
        value.format = kFormat;
        value.metadata = encode_history_signature(frame.signature, config.max_sequence_length);
        const auto length = static_cast<std::size_t>(frame.sequence.history_end);
        HistoryCacheTensor embeddings;
        embeddings.name = "embeddings";
        embeddings.shape = {static_cast<std::int64_t>(length), config.hidden_size};
        embeddings.dtype = DType::kFloat32;
        embeddings.host_data.resize(shape_bytes(embeddings.shape, embeddings.dtype));
        std::memcpy(embeddings.host_data.data(), frame.embeddings.data(),
                    embeddings.host_data.size());
        value.tensors.push_back(std::move(embeddings));
        if (pages) {
            value.tensors.push_back(pages->copy_history(static_cast<std::int32_t>(sample)));
            return value;
        }
        const auto pitch = static_cast<std::size_t>(config.max_sequence_length) * config.head_dim *
                           dtype_size(dtype);
        const auto width = length * config.head_dim * dtype_size(dtype);
        HistoryCacheTensor saved;
        saved.name = "history_kv";
        saved.shape = {config.num_layers, 2, config.num_heads, static_cast<std::int64_t>(length),
                       config.head_dim};
        saved.dtype = dtype;
        // Validate the complete byte product before the single GPU allocation.
        (void)shape_bytes(saved.shape, dtype);
        auto buffer = std::make_shared<DeviceTensor>(saved.shape, dtype, nullptr);
        if (!buffer->ok())
            throw std::runtime_error("hstu history snapshot allocation failed");
        try {
            for (int layer = 0; layer < config.num_layers; ++layer) {
                for (int kind = 0; kind < 2; ++kind) {
                    const auto slot = 2 * static_cast<std::size_t>(layer) + kind;
                    auto* destination = static_cast<std::uint8_t*>(buffer->data()) +
                                        slot * config.num_heads * width;
                    const auto* source = static_cast<const std::uint8_t*>(workspace[slot]->data()) +
                                         sample * config.num_heads * pitch;
                    cuda_check(cudaMemcpy2DAsync(destination, width, source, pitch, width,
                                                 config.num_heads, cudaMemcpyDeviceToDevice,
                                                 executor.stream()),
                               "commit history KV");
                }
            }
            executor.sync();
        } catch (...) {
            // Keep the one destination allocation alive while pending copies
            // finish, including a failure partway through the layer/kind loop.
            executor.sync();
            throw;
        }
        saved.device = std::move(buffer);
        value.tensors.push_back(std::move(saved));
        return value;
    }

    RecommendationResult run(const RecommendationRequest& request,
                             LocalSessionState* local = nullptr, bool seed = false) {
        if (request.sequences.empty() ||
            request.sequences.size() > static_cast<std::size_t>(config.max_batch_size))
            throw std::invalid_argument("hstu batch size is outside the built profile");
        int current_device = -1;
        cuda_check(cudaGetDevice(&current_device), "get device");
        if (current_device != device)
            throw std::invalid_argument("hstu task used on a different CUDA device");
        std::size_t longest = 0;
        for (const auto& item : request.sequences)
            longest = std::max(longest, sequence_length(item, config.mode == "ranking"));
        const auto scale = config.scaling_seqlen > 0 ? static_cast<float>(config.scaling_seqlen)
                                                     : static_cast<float>(longest);
        std::vector<Frame> frames;
        for (const auto& item : request.sequences)
            frames.push_back(prepare_frame(item, scale, seed ? nullptr : local));
        auto& executor = select_executor(frames);
        try {
            return run_prepared(request, frames, scale, local, seed, executor);
        } catch (...) {
            executor.sync();
            for (auto& snapshot : restored_snapshots)
                snapshot = {};
            throw;
        }
    }

    RecommendationResult run_prepared(const RecommendationRequest& request,
                                      std::vector<Frame>& frames, float scale,
                                      LocalSessionState* local, bool seed, ITrtModule& executor) {
        auto data = paged() ? prepare_paged_batch(request, frames, local != nullptr, executor)
                            : prepare_batch(request, frames, scale, executor);
        if (!paged() && seed && !data.rows)
            bind_workspace(frames, executor);
        const auto outputs =
            paged() ? execute_paged(data, executor) : execute(data, frames, executor);
        const auto batch = static_cast<std::int64_t>(frames.size());
        const auto logit_rows = data.rows;
        const auto* encoded =
            data.rows ? output(outputs, "embeddings", batch, data.rows, config.hidden_size, paged())
                      : nullptr;
        const auto* logits =
            data.rows && config.mode == "ranking"
                ? output(outputs, "logits", batch, logit_rows, config.output_dim, paged())
                : nullptr;
        const auto* items = config.mode == "retrieval" ? output(outputs, "item_embeddings", batch,
                                                                data.candidates, config.hidden_size)
                                                       : nullptr;
        RecommendationResult result;
        for (std::size_t sample = 0; sample < frames.size(); ++sample) {
            auto& frame = frames[sample];
            const auto prefix = frame.reuse.reused_tokens;
            const auto count = frame.sequence.tokens.size() - prefix;
            const auto row_begin =
                paged() ? static_cast<std::size_t>(data.page_lease.plan().query_offsets[sample])
                        : sample * data.rows;
            if (count)
                std::copy_n(encoded + row_begin * config.hidden_size, count * config.hidden_size,
                            frame.embeddings.begin() + prefix * config.hidden_size);
            RecommendationSequenceResult row;
            row.candidate_item_ids = request.sequences[sample].candidate_item_ids;
            row.num_candidates = frame.sequence.candidates;
            row.embedding_dim = config.hidden_size;
            row.output_dim = config.output_dim;
            row.sequence_length = static_cast<std::int32_t>(frame.sequence.tokens.size());
            row.sequence_embeddings = frame.embeddings;
            if (config.mode == "ranking") {
                auto begin =
                    frame.embeddings.begin() + frame.sequence.history_end * config.hidden_size;
                row.embeddings.assign(begin, frame.embeddings.end());
                if (row.num_candidates) {
                    const auto offset = frame.sequence.history_end - prefix;
                    const auto* start = logits + (row_begin + offset) * config.output_dim;
                    row.logits.assign(start, start + row.num_candidates * config.output_dim);
                }
            } else {
                const auto* candidate = items + sample * data.candidates * config.hidden_size;
                row.embeddings.assign(candidate,
                                      candidate + row.num_candidates * config.hidden_size);
                const auto* query =
                    frame.embeddings.data() + frame.sequence.query_position * config.hidden_size;
                for (int item = 0; item < row.num_candidates; ++item) {
                    float score = 0.0F;
                    for (int column = 0; column < config.hidden_size; ++column)
                        score += query[column] * candidate[item * config.hidden_size + column];
                    row.scores.push_back(score);
                }
            }
            finite(row.sequence_embeddings);
            finite(row.embeddings);
            finite(row.logits);
            finite(row.scores);
            row.cache.source = local          ? "request_local"
                               : !frame.keyed ? "disabled"
                               : frame.lease.source == HistoryCacheSource::kStorage ? "storage"
                               : frame.lease.value                                  ? "memory"
                                                                                    : "miss";
            row.cache.reason =
                (local || frame.keyed) ? cache_reuse_reason_name(frame.reuse.reason) : "disabled";
            row.cache.history_tokens = frame.sequence.history_end;
            row.cache.reused_history_tokens = static_cast<std::int32_t>(prefix);
            row.cache.computed_tokens = static_cast<std::int32_t>(count);
            result.sequences.push_back(std::move(row));
        }
        if (paged())
            pages->commit(data.page_lease);
        if (local) {
            const auto& frame = frames.front();
            local->signature = frame.signature;
            local->embeddings.assign(frame.embeddings.begin(),
                                     frame.embeddings.begin() +
                                         frame.sequence.history_end * config.hidden_size);
            local->valid = true;
            executor.sync();
            return result;
        }
        // Publish only after all logical outputs have completed and passed validation.
        for (std::size_t sample = 0; sample < frames.size(); ++sample) {
            auto& frame = frames[sample];
            if (!frame.keyed || request.sequences[sample].cache.read_only ||
                !frame.sequence.history_end || (!config.is_causal && config.mode == "ranking"))
                continue;
            if ((paged() || frame.device_snapshot) &&
                frame.reuse.kind == CacheReuseKind::kExactHistory)
                continue;
            if (!data.rows)
                continue;
            try {
                result.sequences[sample].cache.published =
                    cache->publish(frame.lease, snapshot(frame, sample, executor));
            } catch (const std::exception&) {
                executor.sync();
                // A cache is an optimization. Keep the validated model result
                // when optional snapshot allocation/storage cannot complete.
                result.sequences[sample].cache.reason += ":publication_failed";
            }
        }
        return result;
    }

    RecommendationResult run_local(const RecommendationRequest& request, LocalSessionState& local,
                                   std::vector<std::unique_ptr<DeviceTensor>>& buffers,
                                   std::unique_ptr<PagedCacheArena>& local_pages,
                                   bool seed = false) {
        struct Restore {
            std::vector<std::unique_ptr<DeviceTensor>>& target;
            std::vector<std::unique_ptr<DeviceTensor>>& previous;
            std::unique_ptr<PagedCacheArena>& arena;
            std::unique_ptr<PagedCacheArena>& previous_arena;
            ~Restore() {
                target.swap(previous);
                arena.swap(previous_arena);
            }
        } restore{workspace, buffers, pages, local_pages};
        workspace.swap(buffers);
        pages.swap(local_pages);
        for (auto& snapshot : restored_snapshots)
            snapshot = {};
        try {
            return run(request, &local, seed);
        } catch (...) {
            // Native KV writes are in place. A failed execution cannot retain
            // a committed prefix until the next successful full recomputation.
            local.valid = false;
            if (pages) {
                try {
                    pages->invalidate(0);
                } catch (...) {
                    // Preserve the original execution error. Failed CUDA state
                    // cannot be reused; prepare will fail until the device recovers.
                }
            }
            // run() drained the selected executor before restoring the buffers.
            throw;
        }
    }

    struct Session final : IRecommendationSession {
        std::shared_ptr<Impl> runtime;
        LocalSessionState state;
        std::vector<std::unique_ptr<DeviceTensor>> buffers;
        std::unique_ptr<PagedCacheArena> pages;
        RecommendationSequence history;
        std::size_t budget;
        mutable std::mutex history_mutex;

        Session(std::shared_ptr<Impl> owner, RecommendationSequence initial, std::size_t bytes)
            : runtime(std::move(owner)), history(std::move(initial)), budget(bytes) {
            if (!history.candidate_item_ids.empty())
                throw std::invalid_argument("a recommendation session starts with history only");
            int current_device = -1;
            cuda_check(cudaGetDevice(&current_device), "get session device");
            if (current_device != runtime->device)
                throw std::invalid_argument("hstu session used on a different CUDA device");
            if (runtime->paged()) {
                pages = runtime->make_pages(1, bytes);
                return;
            }
            const auto& cfg = runtime->config;
            const auto tensor_bytes = static_cast<std::size_t>(cfg.num_heads) *
                                      cfg.max_sequence_length * cfg.head_dim *
                                      dtype_size(runtime->dtype);
            if (tensor_bytes > bytes / (2 * static_cast<std::size_t>(cfg.num_layers)))
                throw std::invalid_argument("session KV budget is smaller than the built capacity");
            for (int index = 0; index < 2 * cfg.num_layers; ++index) {
                buffers.push_back(std::make_unique<DeviceTensor>(
                    std::vector<std::int64_t>{1, cfg.num_heads, cfg.max_sequence_length,
                                              cfg.head_dim},
                    runtime->dtype, nullptr));
                if (!buffers.back()->ok())
                    throw std::runtime_error("session KV allocation failed");
            }
        }

        RecommendationSequenceResult score(const std::vector<std::int64_t>& ids,
                                           const std::vector<std::int64_t>& timestamps) override {
            std::lock_guard<std::mutex> history_guard(history_mutex);
            auto input = history;
            input.candidate_item_ids = ids;
            if (runtime->config.time_buckets && runtime->config.mode == "ranking") {
                if (timestamps.size() != ids.size())
                    throw std::invalid_argument(
                        "session scoring requires one timestamp per candidate");
                input.token_timestamps.insert(input.token_timestamps.end(), timestamps.begin(),
                                              timestamps.end());
            } else if (!timestamps.empty()) {
                throw std::invalid_argument("this session does not use candidate timestamps");
            }
            std::lock_guard<std::mutex> runtime_guard(runtime->mutex);
            return runtime->run_local({{std::move(input)}}, state, buffers, pages)
                .sequences.front();
        }

        void append(const RecommendationHistoryAppend& update) override {
            std::lock_guard<std::mutex> guard(history_mutex);
            auto next = history;
            next.history_item_ids.insert(next.history_item_ids.end(), update.item_ids.begin(),
                                         update.item_ids.end());
            next.history_action_ids.insert(next.history_action_ids.end(), update.action_ids.begin(),
                                           update.action_ids.end());
            if (runtime->config.time_buckets) {
                if (update.token_timestamps.size() !=
                    update.item_ids.size() + update.action_ids.size())
                    throw std::invalid_argument(
                        "session append requires timestamps for every new token");
                next.token_timestamps.insert(next.token_timestamps.end(),
                                             update.token_timestamps.begin(),
                                             update.token_timestamps.end());
            } else if (!update.token_timestamps.empty()) {
                throw std::invalid_argument("this session does not use timestamps");
            }
            // Validate every new ID, length and item/action relationship before mutation.
            (void)assemble(next, runtime->config);
            history = std::move(next);
        }

        std::unique_ptr<IRecommendationSession> branch() const override {
            std::lock_guard<std::mutex> guard(history_mutex);
            auto child = std::make_unique<Session>(runtime, history, budget);
            std::lock_guard<std::mutex> runtime_guard(runtime->mutex);
            if (state.valid) {
                if (pages) {
                    if (!state.signature.token_rows.empty()) {
                        auto history = pages->copy_history(0);
                        child->pages->initialize_history(0, "session", state.signature, history);
                    }
                    child->state = state;
                    return child;
                }
                const auto& cfg = runtime->config;
                const auto pitch = static_cast<std::size_t>(cfg.max_sequence_length) *
                                   cfg.head_dim * dtype_size(runtime->dtype);
                const auto width =
                    state.signature.token_rows.size() * cfg.head_dim * dtype_size(runtime->dtype);
                for (std::size_t index = 0; index < buffers.size(); ++index)
                    cuda_check(cudaMemcpy2DAsync(child->buffers[index]->data(), pitch,
                                                 buffers[index]->data(), pitch, width,
                                                 cfg.num_heads, cudaMemcpyDeviceToDevice,
                                                 runtime->engine->stream()),
                               "fork session history KV");
                runtime->engine->sync();
                child->state = state;
            }
            return child;
        }
    };
};

CachedPipeline::CachedPipeline(std::unique_ptr<ITrtModule> engine,
                               std::unique_ptr<ITrtModule> candidate_engine, RuntimeConfig config,
                               std::shared_ptr<HistoryCache> cache,
                               std::unique_ptr<ITrtModule> prefill_engine)
    : impl_(std::make_shared<Impl>(std::move(engine), std::move(candidate_engine),
                                   std::move(config), std::move(cache),
                                   std::move(prefill_engine))) {}
CachedPipeline::~CachedPipeline() = default;
RecommendationResult CachedPipeline::recommend(const RecommendationRequest& request) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->run(request);
}
void CachedPipeline::set_history_cache(std::shared_ptr<HistoryCache> cache) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->cache = std::move(cache);
    for (auto& snapshot : impl_->restored_snapshots)
        snapshot = {};
}
std::string CachedPipeline::history_cache_artifact_id() const {
    return impl_->config.cache_artifact_id;
}

std::unique_ptr<IRecommendationSession>
CachedPipeline::create_recommendation_session(const RecommendationSequence& initial_history,
                                              std::size_t max_cache_bytes) {
    if (!initial_history.candidate_item_ids.empty())
        throw std::invalid_argument("a recommendation session starts with history only");
    std::lock_guard<std::mutex> guard(impl_->mutex);
    auto session = std::make_unique<Impl::Session>(impl_, initial_history, max_cache_bytes);
    // Read persistent ContextKV once into request-local buffers. Subsequent
    // appends and branch scores mutate only these buffers, without snapshots,
    // cache-manager lookups, or publication under the user's persistent key.
    (void)impl_->run_local({{initial_history}}, session->state, session->buffers, session->pages,
                           true);
    return session;
}

} // namespace trtmc::hstu
