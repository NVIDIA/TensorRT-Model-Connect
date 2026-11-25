/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hstu/runtime/staged_pipeline.h"

#include "families/hstu/runtime/history_snapshot.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>

namespace trtmc::hstu {
namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace
StagedPipeline::StagedPipeline(std::unique_ptr<ITrtModule> prepare,
                               std::unique_ptr<ITrtModule> step, nlohmann::json config,
                               std::uint64_t cache_bytes, double cache_ratio,
                               std::unique_ptr<ITrtModule> request, std::string identity)
    : prepare_(std::move(prepare)), step_(std::move(step)), request_(std::move(request)),
      config_(std::move(config)), bundle_identity_(std::move(identity)) {
    if (!prepare_ || !step_ || !prepare_->ok() || !step_->ok() ||
        prepare_->stream() != step_->stream() || config_.at("decode_steps") != 1 ||
        config_.at("candidates") <= 0)
        throw std::invalid_argument("invalid staged ranking contract");
    if (prepare_->input_info().size() != config_.at("stages").at(0).at("inputs").size())
        throw std::invalid_argument("staged feature input count mismatch");
    if (!request_ || !request_->ok() || request_->stream() != step_->stream() ||
        bundle_identity_.empty())
        throw std::invalid_argument("invalid history-cache modules");
    // Count only the immutable projected history K/V, never workspace or slate state.
    for (const auto& spec : config_.at("stages").at(0).at("outputs")) {
        const std::string name = spec.at("name");
        if (name != "history_k_b" && name != "history_v_b")
            continue;
        auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
        shape.at(0) = 1;
        bytes_per_sequence_ += checked_bytes(shape, parse_dtype(spec.at("dtype")));
    }
    if (!bytes_per_sequence_)
        throw std::invalid_argument("staged history K/V schema is missing");
    if (request_) {
        for (std::size_t i = 0; i < stage_validation_.size(); ++i)
            stage_validation_[i] = {{"inputs", config_.at("stages").at(i).at("inputs")},
                                    {"min_batch_size", config_.value("min_batch_size", 1)},
                                    {"max_batch_size", config_.at("max_batch_size")}};
        for (const auto& spec : config_.at("stages").at(0).at("outputs"))
            if (spec.at("name") != "history_k_b" && spec.at("name") != "history_v_b") {
                auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
                shape.at(0) = 1;
                mask_bytes_per_sequence_ += checked_bytes(shape, parse_dtype(spec.at("dtype")));
            }
    }
    check(cudaStreamSynchronize(step_->stream()));
    stats_.supported = true;
    stats_.reserved_bytes = bytes_per_sequence_ * config_.at("max_batch_size").get<std::uint64_t>();
    scratch_bytes_ = stats_.reserved_bytes;
    configure_cache_budget(cache_bytes, cache_ratio);
    std::random_device random;
    generation_ = std::to_string(random()) + "-" + std::to_string(random()) + "-" +
                  std::to_string(random()) + "-" + std::to_string(random()) + "-";
    history_stats_ = {{"hydration_supported", bool(request_)},
                      {"hydration_enabled", false},
                      {"resident_entries", 0},
                      {"resident_bytes", 0},
                      {"restoring_bytes", 0},
                      {"context_device_bytes",
                       scratch_bytes_ + mask_bytes_per_sequence_ *
                                            config_.at("max_batch_size").get<std::uint64_t>()}};
    for (const auto* key :
         {"restores_total", "restore_failures_total", "restored_bytes_total",
          "deduplicated_rows_total", "hits_total", "misses_total", "evictions_total",
          "history_prepare_calls_total", "request_prepare_calls_total", "decode_step_calls_total",
          "cached_scores_total", "history_pack_bytes_total"})
        history_stats_[key] = std::uint64_t(0);
    history_stats_["restore_seconds_total"] = 0.0;
}
void StagedPipeline::configure_cache_budget(std::uint64_t bytes, double ratio) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    if (!std::isfinite(ratio) || ratio < 0 || ratio > 1 || (bytes && ratio))
        throw std::invalid_argument("specify positive KV bytes or a ratio in (0, 1], exclusively");
    if (history_enabled_ || !entries_.empty())
        throw std::invalid_argument("configure cache budget before enabling restoration");
    std::size_t free_bytes{}, total_bytes{};
    check(cudaMemGetInfo(&free_bytes, &total_bytes));
    const auto budget = ratio ? std::uint64_t(ratio * (free_bytes + scratch_bytes_))
                              : (bytes ? bytes : scratch_bytes_);
    if (budget < scratch_bytes_)
        throw std::invalid_argument(
            "KV cache budget is smaller than the complete engine profile allocation");
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.budget_bytes = budget;
}
CacheStats StagedPipeline::cache_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}
void StagedPipeline::set_active(std::int64_t batch) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.active_sequences = batch;
    stats_.used_bytes = bytes_per_sequence_ * batch;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.used_bytes);
}
std::vector<FeatureTensor> StagedPipeline::score(const std::vector<FeatureTensor>& inputs) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    const auto bound = validate_inputs(inputs, config_);
    const auto batch = inputs.front().shape.front();
    const auto stream = step_->stream();
    set_active(batch);
    struct ActiveGuard {
        StagedPipeline* owner;
        cudaStream_t stream;
        ~ActiveGuard() {
            cudaStreamSynchronize(stream);
            owner->set_active(0);
        }
    } guard{this, stream};
    TensorMap history, request;
    for (const auto& spec : config_.at("stages").at(0).at("inputs"))
        history.emplace(spec.at("name"), bound.at(spec.at("name")));
    for (const auto& spec : config_.at("stages").at(1).at("inputs"))
        request.emplace(spec.at("name"), bound.at(spec.at("name")));
    prepare_->forward_async(history);
    increment("history_prepare_calls_total");
    request_->forward_async(request);
    increment("request_prepare_calls_total");
    bind_context(*request_, config_.at("stages").at(1).at("outputs"), batch);
    bind_context(*prepare_, config_.at("stages").at(0).at("outputs"), batch);
    return run_step(batch);
}
void StagedPipeline::bind_context(ITrtModule& module, const nlohmann::json& specs,
                                  std::int64_t batch) {
    for (const auto& spec : specs) {
        auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
        shape.at(0) = batch;
        step_->bind_external(spec.at("name"), module.device_ptr(spec.at("name")), shape);
    }
}
std::vector<FeatureTensor> StagedPipeline::run_step(std::int64_t batch) {
    const auto stream = step_->stream();
    step_->forward_device_async({});
    increment("decode_step_calls_total");
    std::vector<FeatureTensor> outputs;
    outputs.reserve(config_.at("outputs").size());
    for (const auto& spec : config_.at("outputs")) {
        auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
        shape.at(0) = batch;
        FeatureTensor output{spec.at("name"), parse_dtype(spec.at("dtype")), shape, {}};
        output.bytes.resize(checked_bytes(shape, output.dtype));
        outputs.push_back(std::move(output));
    }
    struct CopyFence {
        cudaStream_t stream;
        bool pending{true};
        ~CopyFence() {
            if (pending)
                cudaStreamSynchronize(stream);
        }
    } fence{stream};
    for (auto& output : outputs)
        check(cudaMemcpyAsync(output.bytes.data(), step_->device_ptr(output.name),
                              output.bytes.size(), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    fence.pending = false;
    return outputs;
}

void StagedPipeline::increment(const char* key, std::uint64_t amount) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    history_stats_[key] = history_stats_.at(key).get<std::uint64_t>() + amount;
}
nlohmann::json StagedPipeline::history_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return history_stats_;
}
nlohmann::json StagedPipeline::cache_snapshot() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    auto result = history_stats_;
    result.update({{"supported", stats_.supported},
                   {"budget_bytes", stats_.budget_bytes},
                   {"reserved_bytes", stats_.reserved_bytes},
                   {"used_bytes", stats_.used_bytes},
                   {"peak_bytes", stats_.peak_bytes},
                   {"active_sequences", stats_.active_sequences},
                   {"allocation_failures", stats_.allocation_failures}});
    return result;
}
void StagedPipeline::require_history(bool enabled) const {
    if (!request_)
        throw std::invalid_argument(
            "bundle has no history-cache plans; rebuild with staged history-cache plans");
    if (enabled && !history_enabled_)
        throw std::invalid_argument("history cache is not enabled");
}
TensorMap StagedPipeline::validate_stage(const std::vector<FeatureTensor>& inputs,
                                         std::size_t index) const {
    return validate_inputs(inputs, stage_validation_.at(index));
}
void StagedPipeline::configure_history(std::size_t max_entries, double ttl, double lease) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    require_history(false);
    if (history_enabled_)
        throw std::invalid_argument("history cache is already configured");
    if (!max_entries || max_entries > 1000000 || !std::isfinite(ttl) || ttl <= 0 ||
        ttl > 31536000 || !std::isfinite(lease) || lease < 0 || lease > ttl)
        throw std::invalid_argument("invalid history entry/TTL/lease limits");
    if (stats_.budget_bytes - scratch_bytes_ < bytes_per_sequence_)
        throw std::invalid_argument(
            "KV budget has no capacity for a restored history entry beyond scratch");
    max_entries_ = max_entries;
    ttl_seconds_ = ttl;
    lease_seconds_ = lease;
    history_enabled_ = true;
    std::lock_guard<std::mutex> lock(stats_mutex_);
    history_stats_["hydration_enabled"] = true;
}
void StagedPipeline::refresh_residency(std::uint64_t pending_kv, std::uint64_t pending_mask) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.reserved_bytes = scratch_bytes_ + entries_.size() * bytes_per_sequence_ + pending_kv;
    history_stats_["resident_entries"] = entries_.size();
    history_stats_["resident_bytes"] = entries_.size() * bytes_per_sequence_;
    history_stats_["restoring_bytes"] = pending_kv;
    history_stats_["context_device_bytes"] =
        stats_.reserved_bytes +
        mask_bytes_per_sequence_ *
            (entries_.size() + config_.at("max_batch_size").get<std::uint64_t>()) +
        pending_mask;
}
void StagedPipeline::expire_entries() {
    const auto now = HistoryClock::now();
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.expires <= now) {
            it = entries_.erase(it);
            increment("evictions_total");
        } else
            ++it;
    }
    refresh_residency();
}
StagedPipeline::Entry& StagedPipeline::find_entry(const std::string& handle) {
    const auto found = entries_.find(handle);
    if (found == entries_.end() || found->second.expires <= HistoryClock::now())
        throw std::invalid_argument(
            "history handle is missing, expired, released, or belongs to another instance");
    return found->second;
}
std::vector<std::uint8_t>
StagedPipeline::export_history(const std::vector<FeatureTensor>& inputs,
                               const std::vector<std::string>& revisions) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    require_history(false);
    const auto bound = validate_stage(inputs, 0);
    const auto batch = inputs.front().shape.front();
    if (revisions.size() != std::size_t(batch))
        throw std::invalid_argument("one history revision is required per row");
    for (const auto& revision : revisions)
        if (revision.empty() || revision.size() > 256)
            throw std::invalid_argument("invalid history revision");
    HistorySnapshot snapshot;
    auto& meta = snapshot.metadata;
    meta = {{"bundle_id", bundle_identity_},
            {"batch", batch},
            {"revisions", revisions},
            {"history_key", nlohmann::json::array()}};
    for (std::int64_t row = 0; row < batch; ++row) {
        HistoryHash hash;
        for (const auto& spec : config_.at("stages").at(0).at("inputs")) {
            const std::string name = spec.at("name");
            const auto& tensor = bound.at(name);
            const auto schema = spec.dump();
            hash.update(schema.data(), schema.size());
            const auto bytes = tensor.nbytes() / std::size_t(batch);
            hash.update(static_cast<const std::uint8_t*>(tensor.data) + row * bytes, bytes);
        }
        meta["history_key"].push_back(hash.finish());
    }
    prepare_->forward_async(bound);
    increment("history_prepare_calls_total");
    check(cudaStreamSynchronize(prepare_->stream()));
    for (const auto& spec : config_.at("stages").at(0).at("outputs")) {
        auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
        shape.at(0) = batch;
        FeatureTensor tensor{spec.at("name"), parse_dtype(spec.at("dtype")), shape, {}};
        tensor.bytes.resize(checked_bytes(shape, tensor.dtype));
        check(cudaMemcpy(tensor.bytes.data(), prepare_->device_ptr(tensor.name),
                         tensor.bytes.size(), cudaMemcpyDeviceToHost));
        snapshot.tensors.push_back(std::move(tensor));
    }
    auto encoded = encode_history_snapshot(std::move(snapshot));
    // Apply the same finite/schema gate to generated and externally restored snapshots.
    decode_history_snapshot(encoded, bundle_identity_, config_.at("stages").at(0).at("outputs"),
                            config_.at("max_batch_size"));
    return encoded;
}
std::vector<std::string> StagedPipeline::restore_history(const std::vector<std::uint8_t>& bytes) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    require_history();
    const auto start = Clock::now();
    try {
        const auto snapshot = decode_history_snapshot(bytes, bundle_identity_,
                                                      config_.at("stages").at(0).at("outputs"),
                                                      config_.at("max_batch_size"));
        const auto batch = snapshot.metadata.at("batch").get<std::size_t>();
        expire_entries();
        const auto now = HistoryClock::now();
        const auto duration = [](double seconds) {
            return std::chrono::duration_cast<HistoryClock::duration>(
                std::chrono::duration<double>(seconds));
        };
        std::vector<std::string> keys(batch), result(batch);
        std::map<std::string, std::size_t> unique_rows;
        std::set<std::string> retained;
        std::map<std::string, std::string> existing;
        for (const auto& [id, entry] : entries_)
            existing.emplace(entry.key, id);
        for (std::size_t row = 0; row < batch; ++row) {
            HistoryHash hash;
            const auto metadata =
                nlohmann::json{{"bundle", bundle_identity_},
                               {"revision", snapshot.metadata.at("revisions").at(row)},
                               {"history", snapshot.metadata.at("history_key").at(row)}}
                    .dump();
            hash.update(metadata.data(), metadata.size());
            for (const auto& tensor : snapshot.tensors) {
                const auto n = tensor.bytes.size() / batch;
                hash.update(tensor.bytes.data() + row * n, n);
            }
            keys[row] = hash.finish();
            if (existing.count(keys[row]))
                retained.insert(existing.at(keys[row]));
            else
                unique_rows.emplace(keys[row], row);
        }
        const auto capacity = std::min<std::uint64_t>(
            max_entries_, (stats_.budget_bytes - scratch_bytes_) / bytes_per_sequence_);
        std::vector<std::string> victims;
        for (const auto& [id, entry] : entries_)
            if (!retained.count(id) && entry.leased_until <= now)
                victims.push_back(id);
        std::sort(victims.begin(), victims.end(), [&](const auto& a, const auto& b) {
            return entries_.at(a).touched < entries_.at(b).touched;
        });
        const auto required = entries_.size() + unique_rows.size();
        const auto remove = required > capacity ? required - capacity : 0;
        if (remove > victims.size()) {
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.allocation_failures;
            }
            throw std::invalid_argument(
                "history cache capacity exhausted; entries are leased or request exceeds capacity");
        }
        // Eviction is committed admission policy; later copy failure does not resurrect victims.
        for (std::size_t i = 0; i < remove; ++i) {
            entries_.erase(victims[i]);
            increment("evictions_total");
        }
        refresh_residency();
        std::map<std::string, Entry> pending;
        std::map<std::string, std::string> ids = existing;
        std::uint64_t pending_kv = 0, pending_mask = 0;
        try {
            for (const auto& [key, row] : unique_rows) {
                Entry entry;
                entry.id = generation_ + std::to_string(++next_handle_);
                entry.key = key;
                entry.revision = snapshot.metadata.at("revisions").at(row);
                entry.feature_digest = snapshot.metadata.at("history_key").at(row);
                entry.expires = now + duration(ttl_seconds_);
                entry.leased_until = now + duration(lease_seconds_);
                entry.touched = now;
                for (const auto& tensor : snapshot.tensors) {
                    auto shape = tensor.shape;
                    shape.at(0) = 1;
                    DeviceTensor device(shape, tensor.dtype, step_->stream());
                    if (!device.ok())
                        throw std::runtime_error("history device allocation failed");
                    ((tensor.name == "history_k_b" || tensor.name == "history_v_b")
                         ? pending_kv
                         : pending_mask) += device.nbytes();
                    refresh_residency(pending_kv, pending_mask);
                    const auto n = tensor.bytes.size() / batch;
                    check(cudaMemcpyAsync(device.data(), tensor.bytes.data() + row * n, n,
                                          cudaMemcpyHostToDevice, step_->stream()));
                    entry.tensors.push_back(std::move(device));
                }
                ids[key] = entry.id;
                pending.emplace(entry.id, std::move(entry));
            }
            check(cudaStreamSynchronize(step_->stream()));
        } catch (...) {
            cudaStreamSynchronize(step_->stream());
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                ++stats_.allocation_failures;
            }
            throw;
        }
        for (std::size_t row = 0; row < batch; ++row)
            result[row] = ids.at(keys[row]);
        const auto imported = pending.size();
        entries_.merge(pending);
        const auto ready_now = HistoryClock::now();
        for (const auto& id : result) {
            auto& entry = entries_.at(id);
            entry.expires = ready_now + duration(ttl_seconds_);
            entry.leased_until = ready_now + duration(lease_seconds_);
            entry.touched = ready_now;
        }
        increment("restores_total");
        increment("deduplicated_rows_total", batch - imported);
        increment("restored_bytes_total",
                  imported * (bytes_per_sequence_ + mask_bytes_per_sequence_));
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            history_stats_["restore_seconds_total"] =
                history_stats_.at("restore_seconds_total").get<double>() + elapsed(start) / 1000;
        }
        refresh_residency();
        return result;
    } catch (...) {
        refresh_residency();
        increment("restore_failures_total");
        throw;
    }
}
std::vector<FeatureTensor>
StagedPipeline::score_with_history(const std::vector<std::string>& handles,
                                   const std::vector<std::string>& revisions,
                                   const std::vector<FeatureTensor>& inputs) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    require_history();
    const auto bound = validate_stage(inputs, 1);
    const auto batch = inputs.front().shape.front();
    if (handles.size() != std::size_t(batch) || revisions.size() != handles.size())
        throw std::invalid_argument("one history handle/revision is required per request row");
    std::vector<Entry*> selected;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        try {
            auto& entry = find_entry(handles[i]);
            if (entry.revision != revisions[i])
                throw std::invalid_argument("history revision mismatch");
            selected.push_back(&entry);
        } catch (...) {
            increment("misses_total");
            throw;
        }
    }
    increment("hits_total", handles.size());
    const auto stream = step_->stream();
    set_active(batch);
    struct Guard {
        StagedPipeline* self;
        cudaStream_t stream;
        ~Guard() {
            cudaStreamSynchronize(stream);
            self->set_active(0);
        }
    } guard{this, stream};
    const auto& specs = config_.at("stages").at(0).at("outputs");
    for (std::size_t tensor = 0; tensor < specs.size(); ++tensor) {
        if (batch == 1) {
            const auto& value = selected.front()->tensors.at(tensor);
            step_->bind_external(specs.at(tensor).at("name"),
                                 selected.front()->tensors.at(tensor).data(), value.shape());
            continue;
        }
        auto* destination =
            static_cast<std::uint8_t*>(prepare_->device_ptr(specs.at(tensor).at("name")));
        for (std::size_t row = 0; row < selected.size(); ++row) {
            const auto& value = selected[row]->tensors.at(tensor);
            check(cudaMemcpyAsync(destination + row * value.nbytes(), value.data(), value.nbytes(),
                                  cudaMemcpyDeviceToDevice, stream));
            increment("history_pack_bytes_total", value.nbytes());
        }
    }
    if (batch != 1)
        bind_context(*prepare_, specs, batch);
    request_->forward_async(bound);
    increment("request_prepare_calls_total");
    bind_context(*request_, config_.at("stages").at(1).at("outputs"), batch);
    auto outputs = run_step(batch);
    const auto now = HistoryClock::now();
    for (auto* entry : selected) {
        entry->touched = now;
        entry->leased_until = now;
    }
    increment("cached_scores_total");
    return outputs;
}
nlohmann::json StagedPipeline::history_info(const std::string& handle) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    require_history();
    expire_entries();
    const auto& entry = find_entry(handle);
    const auto now = HistoryClock::now();
    return {{"handle", handle},
            {"ready", true},
            {"revision", entry.revision},
            {"history_key", entry.feature_digest},
            {"bundle_id", bundle_identity_},
            {"kv_bytes", bytes_per_sequence_},
            {"mask_bytes", mask_bytes_per_sequence_},
            {"leased", entry.leased_until > now},
            {"expires_in_seconds", std::chrono::duration<double>(entry.expires - now).count()}};
}
void StagedPipeline::release_history(const std::vector<std::string>& handles) {
    std::lock_guard<std::mutex> execution(execute_mutex_);
    require_history();
    for (const auto& id : handles)
        entries_.erase(id);
    refresh_residency();
}
} // namespace trtmc::hstu
