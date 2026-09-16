/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/cached_pipeline.h"

#include "families/hstu/runtime/cache_policy.h"
#include "families/hstu/runtime/request.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

namespace trtmc::hstu {
namespace {

using Json = nlohmann::json;
constexpr const char* kFormat = "hstu.native_history.v1";

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("hstu ") + operation + ": " +
                                 cudaGetErrorString(result));
}

std::string cache_name(int layer, const char* kind) {
    return "cache_" + std::to_string(layer) + "_" + kind;
}

Json signature_json(const HistorySignature& value) {
    return {{"model_namespace", value.model_namespace},
            {"feature_version", value.feature_version},
            {"history_revision", value.history_revision},
            {"tokens", value.token_rows},
            {"positions", value.position_ids},
            {"times", value.time_ids},
            {"scale", value.effective_scale},
            {"contextual_length", value.contextual_length}};
}

HistorySignature parse_signature(const HistoryCacheValue& value) {
    if (value.schema_version != kHistoryCacheInterfaceVersion || value.format != kFormat)
        throw std::invalid_argument("incompatible history snapshot format");
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

void validate_snapshot(const HistoryCacheValue& value, const RuntimeConfig& config,
                       std::size_t length, DType dtype, int device) {
    if (length == 0 || length > static_cast<std::size_t>(config.max_sequence_length) ||
        value.tensors.size() != static_cast<std::size_t>(2 * config.num_layers + 1))
        throw std::invalid_argument("invalid history snapshot dimensions");
    const auto& embeddings = tensor(value, "embeddings");
    const std::vector<std::int64_t> embedding_shape{static_cast<std::int64_t>(length),
                                                    config.hidden_size};
    if (embeddings.shape != embedding_shape || embeddings.dtype != DType::kFloat32 ||
        embeddings.device ||
        embeddings.host_data.size() != length * config.hidden_size * sizeof(float))
        throw std::invalid_argument("invalid history embedding snapshot");
    for (int layer = 0; layer < config.num_layers; ++layer) {
        for (const auto* kind : {"k", "v"}) {
            const auto& item = tensor(value, cache_name(layer, kind));
            const std::vector<std::int64_t> shape{
                config.num_heads, static_cast<std::int64_t>(length), config.head_dim};
            const auto bytes = config.num_heads * length * config.head_dim * dtype_size(dtype);
            if (item.shape != shape || item.dtype != dtype)
                throw std::invalid_argument("invalid historical K/V shape or dtype");
            if (item.device) {
                cudaPointerAttributes attributes{};
                cuda_check(cudaPointerGetAttributes(&attributes, item.device->data()),
                           "cache pointer");
                if (attributes.device != device || item.device->shape() != shape ||
                    item.device->dtype() != dtype || item.device->nbytes() != bytes)
                    throw std::invalid_argument("history snapshot belongs to another device");
            } else if (item.host_data.size() != bytes) {
                throw std::invalid_argument("invalid historical K/V byte length");
            }
        }
    }
}

const float* output(const TensorMap& values, const char* name, std::int64_t batch,
                    std::int64_t rows, std::int64_t width) {
    const auto found = values.find(name);
    if (found == values.end() || found->second.dtype != DType::kFloat32 ||
        found->second.shape != std::vector<std::int64_t>{batch, rows, width} || !found->second.data)
        throw std::runtime_error(std::string("hstu malformed output ") + name);
    return static_cast<const float*>(found->second.data);
}

void finite(const std::vector<float>& values) {
    if (!std::all_of(values.begin(), values.end(),
                     [](float value) { return std::isfinite(value); }))
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

struct QueryBatch {
    std::size_t rows{0}, candidates{1};
    std::vector<std::int32_t> tokens, positions, times, write_indices, active_lengths;
    std::vector<std::int32_t> packed_rows, update_lengths, candidate_tokens;
    std::vector<float> mask;
    float scale{1.0F};
    std::int32_t empty_row{0};
};

} // namespace

struct CachedPipeline::Impl {
    RuntimeConfig config;
    std::shared_ptr<HistoryCache> cache;
    std::mutex mutex;
    int device{0};
    DType dtype{DType::kFloat32};
    std::vector<std::unique_ptr<DeviceTensor>> workspace;
    std::unique_ptr<ITrtModule> engine, candidates;

    Impl(std::unique_ptr<ITrtModule> model, std::unique_ptr<ITrtModule> lookup, RuntimeConfig cfg,
         std::shared_ptr<HistoryCache> history_cache)
        : config(std::move(cfg)), cache(std::move(history_cache)), engine(std::move(model)),
          candidates(std::move(lookup)) {
        if (!config.enable_history_cache || !engine || !engine->ok())
            throw std::invalid_argument("hstu cached runtime requires a native KV engine");
        cuda_check(cudaGetDevice(&device), "get device");
        dtype = engine->tensor_dtype("cache_0_k");
        for (int layer = 0; layer < config.num_layers; ++layer) {
            for (const auto* kind : {"k", "v"}) {
                const auto input_name = cache_name(layer, kind);
                const auto output_name = "present_" + std::to_string(layer) + "_" + kind;
                if (!engine->has_input(input_name) || !engine->has_output(output_name) ||
                    engine->tensor_dtype(input_name) != dtype)
                    throw std::invalid_argument("hstu native KV engine contract mismatch");
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
            const auto previous = parse_signature(*frame.lease.value);
            validate_snapshot(*frame.lease.value, config, previous.token_rows.size(), dtype,
                              device);
            frame.reuse = plan_cache_reuse(
                &previous, signature,
                {config.is_causal, config.mode == "ranking", config.disable_contextual_mask});
            if (frame.reuse.reused_tokens) {
                const auto& stored = tensor(*frame.lease.value, "embeddings");
                std::memcpy(frame.embeddings.data(), stored.host_data.data(),
                            stored.host_data.size());
                frame.device_snapshot = bool(tensor(*frame.lease.value, "cache_0_k").device);
            }
        } catch (const std::exception&) {
            frame.reuse = {CacheReuseKind::kRecompute, CacheReuseReason::kInvalidSignature, 0};
        }
        return frame;
    }

    QueryBatch prepare_batch(const RecommendationRequest& request, const std::vector<Frame>& frames,
                             float scale) {
        QueryBatch data;
        const auto batch = frames.size();
        const auto capacity = static_cast<std::size_t>(config.max_sequence_length);
        data.scale = scale;
        for (const auto& frame : frames) {
            data.rows =
                std::max(data.rows, frame.sequence.tokens.size() - frame.reuse.reused_tokens);
            data.candidates =
                std::max(data.candidates, static_cast<std::size_t>(frame.sequence.candidates));
        }
        data.tokens.resize(batch * data.rows);
        data.positions.resize(batch * data.rows);
        data.times.resize(batch * data.rows);
        data.mask.resize(batch * data.rows * capacity, 0.0F);
        data.candidate_tokens.resize(batch * data.candidates);
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
                data.packed_rows.push_back(static_cast<std::int32_t>(row));
                for (std::size_t key = 0; key < length; ++key)
                    data.mask[row * capacity + key] =
                        attention_allowed(static_cast<std::int32_t>(index),
                                          static_cast<std::int32_t>(key), frame.sequence, config)
                            ? 1.0F
                            : 0.0F;
            }
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

    void bind_workspace(const std::vector<Frame>& frames) {
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
                    const auto& saved = tensor(*frame.lease.value, name);
                    const auto width =
                        frame.reuse.reused_tokens * config.head_dim * dtype_size(dtype);
                    auto* destination = static_cast<std::uint8_t*>(buffer.data()) +
                                        sample * config.num_heads * pitch;
                    const void* source =
                        saved.device ? saved.device->data() : saved.host_data.data();
                    cuda_check(cudaMemcpy2DAsync(
                                   destination, pitch, source, width, width, config.num_heads,
                                   saved.device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                                   engine->stream()),
                               "restore history KV");
                }
                engine->bind_external(name, buffer.data(),
                                      {static_cast<std::int64_t>(frames.size()), config.num_heads,
                                       config.max_sequence_length, config.head_dim});
                const auto present = "present_" + std::to_string(layer) + (kind ? "_v" : "_k");
                if (engine->device_ptr(present) != buffer.data())
                    throw std::runtime_error("hstu native KV output did not alias its input");
            }
        }
    }

    TensorMap execute(QueryBatch& data, const std::vector<Frame>& frames) {
        const auto batch = static_cast<std::int64_t>(frames.size());
        const auto rows = static_cast<std::int64_t>(data.rows);
        TensorMap candidate_input{{"candidate_token_ids",
                                   {data.candidate_tokens.data(),
                                    {batch, static_cast<std::int64_t>(data.candidates)},
                                    DType::kInt32}}};
        if (rows == 0)
            return config.mode == "retrieval" ? candidates->forward(candidate_input) : TensorMap{};
        bind_workspace(frames);
        TensorMap input = {
            {"token_ids", {data.tokens.data(), {batch, rows}, DType::kInt32}},
            {"attention_mask",
             {data.mask.data(), {batch, 1, rows, config.max_sequence_length}, DType::kFloat32}},
            {"scaling_seqlen", {&data.scale, {1}, DType::kFloat32}},
            {"cache_write_indices", {data.write_indices.data(), {batch}, DType::kInt32}},
            {"cache_active_lengths", {data.active_lengths.data(), {batch}, DType::kInt32}},
            {"cache_update_rows",
             {data.packed_rows.empty() ? &data.empty_row : data.packed_rows.data(),
              {static_cast<std::int64_t>(data.packed_rows.size())},
              DType::kInt32}},
            {"cache_update_lengths", {data.update_lengths.data(), {batch + 1}, DType::kInt32}},
        };
        if (config.position_buckets)
            input["position_ids"] = {data.positions.data(), {batch, rows}, DType::kInt32};
        if (config.time_buckets)
            input["time_ids"] = {data.times.data(), {batch, rows}, DType::kInt32};
        if (config.mode == "retrieval")
            input.insert(candidate_input.begin(), candidate_input.end());
        return engine->forward(input);
    }

    HistoryCacheValue snapshot(const Frame& frame, std::size_t sample) {
        HistoryCacheValue value;
        value.format = kFormat;
        const auto metadata = signature_json(frame.signature).dump();
        value.metadata.assign(metadata.begin(), metadata.end());
        const auto length = static_cast<std::size_t>(frame.sequence.history_end);
        HistoryCacheTensor embeddings;
        embeddings.name = "embeddings";
        embeddings.shape = {static_cast<std::int64_t>(length), config.hidden_size};
        embeddings.dtype = DType::kFloat32;
        embeddings.host_data.resize(length * config.hidden_size * sizeof(float));
        std::memcpy(embeddings.host_data.data(), frame.embeddings.data(),
                    embeddings.host_data.size());
        value.tensors.push_back(std::move(embeddings));
        const auto pitch = static_cast<std::size_t>(config.max_sequence_length) * config.head_dim *
                           dtype_size(dtype);
        const auto width = length * config.head_dim * dtype_size(dtype);
        for (int layer = 0; layer < config.num_layers; ++layer) {
            for (int kind = 0; kind < 2; ++kind) {
                HistoryCacheTensor saved;
                saved.name = cache_name(layer, kind ? "v" : "k");
                saved.shape = {config.num_heads, static_cast<std::int64_t>(length),
                               config.head_dim};
                saved.dtype = dtype;
                auto buffer = std::make_shared<DeviceTensor>(saved.shape, dtype, nullptr);
                if (!buffer->ok())
                    throw std::runtime_error("hstu history snapshot allocation failed");
                auto* source =
                    static_cast<const std::uint8_t*>(workspace[2 * layer + kind]->data()) +
                    sample * config.num_heads * pitch;
                cuda_check(cudaMemcpy2DAsync(buffer->data(), width, source, pitch, width,
                                             config.num_heads, cudaMemcpyDeviceToDevice,
                                             engine->stream()),
                           "commit history KV");
                saved.device = std::move(buffer);
                value.tensors.push_back(std::move(saved));
            }
        }
        engine->sync();
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
        auto data = prepare_batch(request, frames, scale);
        if (seed && !data.rows)
            bind_workspace(frames);
        const auto outputs = execute(data, frames);
        const auto batch = static_cast<std::int64_t>(frames.size());
        const auto* encoded =
            data.rows ? output(outputs, "embeddings", batch, data.rows, config.hidden_size)
                      : nullptr;
        const auto* logits = data.rows && config.mode == "ranking"
                                 ? output(outputs, "logits", batch, data.rows, config.output_dim)
                                 : nullptr;
        const auto* items = config.mode == "retrieval" ? output(outputs, "item_embeddings", batch,
                                                                data.candidates, config.hidden_size)
                                                       : nullptr;
        RecommendationResult result;
        for (std::size_t sample = 0; sample < frames.size(); ++sample) {
            auto& frame = frames[sample];
            const auto prefix = frame.reuse.reused_tokens;
            const auto count = frame.sequence.tokens.size() - prefix;
            if (count)
                std::copy_n(encoded + sample * data.rows * config.hidden_size,
                            count * config.hidden_size,
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
                    const auto* start =
                        logits + (sample * data.rows + frame.sequence.history_end - prefix) *
                                     config.output_dim;
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
        if (local) {
            const auto& frame = frames.front();
            local->signature = frame.signature;
            local->embeddings.assign(frame.embeddings.begin(),
                                     frame.embeddings.begin() +
                                         frame.sequence.history_end * config.hidden_size);
            local->valid = true;
            engine->sync();
            return result;
        }
        // Publish only after all logical outputs have completed and passed validation.
        for (std::size_t sample = 0; sample < frames.size(); ++sample) {
            auto& frame = frames[sample];
            if (!frame.keyed || request.sequences[sample].cache.read_only ||
                !frame.sequence.history_end || (!config.is_causal && config.mode == "ranking"))
                continue;
            if (frame.device_snapshot && frame.reuse.kind == CacheReuseKind::kExactHistory)
                continue;
            if (!data.rows)
                continue;
            try {
                result.sequences[sample].cache.published =
                    cache->publish(frame.lease, snapshot(frame, sample));
            } catch (const std::exception&) {
                // A cache is an optimization. Keep the validated model result
                // when optional snapshot allocation/storage cannot complete.
                result.sequences[sample].cache.reason += ":publication_failed";
            }
        }
        return result;
    }

    RecommendationResult run_local(const RecommendationRequest& request, LocalSessionState& local,
                                   std::vector<std::unique_ptr<DeviceTensor>>& buffers,
                                   bool seed = false) {
        struct Restore {
            std::vector<std::unique_ptr<DeviceTensor>>& target;
            std::vector<std::unique_ptr<DeviceTensor>>& previous;
            ~Restore() { target.swap(previous); }
        } restore{workspace, buffers};
        workspace.swap(buffers);
        try {
            return run(request, &local, seed);
        } catch (...) {
            // Native KV writes are in place. A failed execution cannot retain
            // a committed prefix until the next successful full recomputation.
            local.valid = false;
            engine->sync();
            throw;
        }
    }

    struct Session final : IRecommendationSession {
        std::shared_ptr<Impl> runtime;
        LocalSessionState state;
        std::vector<std::unique_ptr<DeviceTensor>> buffers;
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
            return runtime->run_local({{std::move(input)}}, state, buffers).sequences.front();
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
                               std::shared_ptr<HistoryCache> cache)
    : impl_(std::make_shared<Impl>(std::move(engine), std::move(candidate_engine),
                                   std::move(config), std::move(cache))) {}
CachedPipeline::~CachedPipeline() = default;
RecommendationResult CachedPipeline::recommend(const RecommendationRequest& request) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->run(request);
}
void CachedPipeline::set_history_cache(std::shared_ptr<HistoryCache> cache) {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->cache = std::move(cache);
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
    (void)impl_->run_local({{initial_history}}, session->state, session->buffers, true);
    return session;
}

} // namespace trtmc::hstu
