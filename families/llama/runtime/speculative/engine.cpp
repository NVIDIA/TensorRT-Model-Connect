/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace trtmc::llama::speculative {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument("speculative ABI: " + message);
}

} // namespace

Contract Contract::parse(const nlohmann::json& value) {
    const bool linear =
        value.at("version") == 1 && value.at("cache_layout") == "batch_heads_capacity_dim" &&
        value.at("cache_update") == "aliased_contiguous_append" &&
        value.value("alias_contract", "engine_required_alias") == "engine_required_alias";
    require(linear && value.at("precision") == "fp16",
            "unsupported version, precision or state layout");
    const auto role = value.at("role").get<std::string>();
    require(role == "target" || role == "draft", "invalid engine role");
    Contract c{value.at("layers"),    value.at("hidden_size"),   value.at("kv_heads"),
               value.at("head_dim"),  value.at("vocab_size"),    value.at("capacity"),
               value.at("max_query"), value.at("feature_width"), role == "draft"};
    require(c.layers > 0 && c.hidden > 0 && c.heads > 0 && c.dim > 0 && c.vocab > 0 &&
                c.capacity > 0 && c.max_query > 0 && c.max_query <= c.capacity &&
                c.feature_width > 0,
            "invalid dimensions");
    require(!c.draft || c.vocab >= 2, "draft selection needs two tokens");
    if (value.contains("execution_profiles"))
        require(value.at("execution_profiles").is_array(), "execution profiles must be an array");
    if (value.contains("execution_profiles") && !value.at("execution_profiles").empty()) {
        const auto& profiles = value.at("execution_profiles");
        require(profiles.size() == 2, "expected prefill/decode profiles");
        for (int i = 0; i < 2; ++i) {
            require(profiles[i].at("phase") == (i == 0 ? "prefill" : "decode"),
                    "execution profile phase order mismatch");
            for (const auto* name : {"query", "logits"}) {
                const auto& bounds = profiles[i].at(name);
                require(bounds.is_array() && bounds.size() == 3,
                        "expected MIN/OPT/MAX profile bounds");
                for (const auto& bound : bounds)
                    require(bound.is_number_integer(), "profile bounds must be integers");
            }
            ExecutionProfile p{profiles[i].at("query").get<std::array<int, 3>>(),
                               profiles[i].at("logits").get<std::array<int, 3>>()};
            for (const auto& bounds : {p.query, p.logits})
                require(bounds[0] == 1 && bounds[1] >= 1 && bounds[1] <= bounds[2] &&
                            bounds[2] <= c.max_query,
                        "invalid execution profile bounds");
            for (int j = 0; j < 3; ++j)
                require(p.logits[j] <= p.query[j], "selected rows exceed query profile");
            if (i == 0 || c.draft)
                require(p.logits == std::array<int, 3>{1, 1, 1},
                        "prefill/draft must select one logit row");
            c.profiles.push_back(p);
        }
        require(std::max(c.profiles[0].query[2], c.profiles[1].query[2]) == c.max_query,
                "manifest query bound does not match profiles");
    }
    return c;
}

Engine::Engine(std::unique_ptr<ITrtModule> module, Contract contract,
               std::unique_ptr<ITrtModule> selection, std::unique_ptr<ITrtModule> prefill)
    : module_(std::move(module)), prefill_(std::move(prefill)), selection_(std::move(selection)),
      contract_(contract) {
    require(module_ && module_->ok(), "invalid execution module");
    const auto& c = contract_;
    require(selection_ != nullptr, "missing GPU selection module");
    {
        const int max_rows = c.profiles.empty()
                                 ? c.max_query
                                 : std::max(c.profiles[0].logits[2], c.profiles[1].logits[2]);
        const int columns = c.draft ? 3 : 2;
        require(selection_->ok() && selection_->optimization_profile_count() == 1 &&
                    selection_->has_input("logits") &&
                    selection_->tensor_dtype("logits") == DType::kFloat32 &&
                    selection_->has_output("selection") &&
                    selection_->tensor_dtype("selection") == DType::kInt32 &&
                    selection_->tensor_shape("selection") ==
                        std::vector<std::int64_t>({max_rows, columns}) &&
                    selection_->input_profile_shape("logits", 0, ProfileShapeSelector::kMin) ==
                        std::vector<std::int64_t>({1, c.vocab}) &&
                    selection_->input_profile_shape("logits", 0, ProfileShapeSelector::kMax) ==
                        std::vector<std::int64_t>({max_rows, c.vocab}),
                "invalid selection engine contract");
        void* host = nullptr;
        checked(cudaMallocHost(&host, std::size_t(max_rows) * columns * sizeof(std::int32_t)));
        selection_host_ = std::shared_ptr<void>(host, [](void* p) { cudaFreeHost(p); });
        cudaEvent_t ready = nullptr;
        checked(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        selection_ready_ = std::shared_ptr<void>(
            ready, [](void* p) { cudaEventDestroy(static_cast<cudaEvent_t>(p)); });
    }
    require(c.profiles.empty() == (prefill_ == nullptr), "execution module/profile mismatch");
    if (prefill_)
        require(prefill_->ok() && prefill_->profile_idx() == 0 && module_->profile_idx() == 1 &&
                    prefill_->stream() == module_->stream(),
                "prefill/decode must use matching contexts on one stream");
    // Validate both contexts against the compiler's complete row contract.
    for (const auto* module : {module_.get(), prefill_.get()}) {
        if (!module)
            continue;
        const int profile = module->profile_idx();
        require(module->optimization_profile_count() == (c.profiles.empty() ? 1 : 2),
                "unexpected optimization profile count");
        const ExecutionProfile bounds =
            c.profiles.empty() ? ExecutionProfile{{1, std::min(16, c.max_query), c.max_query},
                                                  {1, std::min(16, c.max_query), c.max_query}}
                               : c.profiles.at(profile);
        auto check_rows = [&](const char* name, std::vector<std::int64_t> shape, int axis,
                              const std::array<int, 3>& rows) {
            constexpr ProfileShapeSelector selectors[]{
                ProfileShapeSelector::kMin, ProfileShapeSelector::kOpt, ProfileShapeSelector::kMax};
            for (int i = 0; i < 3; ++i) {
                shape[axis] = rows[i];
                require(module->input_profile_shape(name, profile, selectors[i]) == shape,
                        std::string("profile mismatch: ") + name);
            }
        };
        for (const auto* name : {"token_id", "position_id"})
            check_rows(name, {0}, 0, bounds.query);
        check_rows("logits_indices", {0}, 0, bounds.logits);
        check_rows("attention_mask", {0, c.capacity}, 0, bounds.query);
        if (c.draft) {
            check_rows("target_features", {0, c.feature_width}, 0, bounds.query);
            check_rows("draft_features", {0, c.hidden}, 0, bounds.query);
        }
    }
    for (const auto* name :
         {"token_id", "position_id", "logits_indices", "attention_mask", "key_value_lengths"}) {
        require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kInt32,
                std::string("missing or mistyped input ") + name);
    }
    require(module_->has_output("logits") && module_->tensor_dtype("logits") == DType::kFloat32,
            "missing FP32 logits");
    require(module_->has_output("features") && module_->tensor_dtype("features") == DType::kFloat16,
            "missing FP16 features");
    for (const auto* name : {"key_value_lengths"})
        require(module_->tensor_shape(name) == std::vector<std::int64_t>{1},
                "invalid state scalar shape");
    require(module_->has_input("cache_write_indices") &&
                module_->tensor_dtype("cache_write_indices") == DType::kInt32 &&
                module_->tensor_shape("cache_write_indices") == std::vector<std::int64_t>{1},
            "invalid linear write index");
    for (const auto* name : {"logits", "features"}) {
        const auto output_shape = module_->tensor_shape(name);
        const int width = name[0] == 'l' ? c.vocab : (c.draft ? c.hidden : c.feature_width);
        require(output_shape.size() == 2 && output_shape.back() == width,
                std::string("output row contract mismatch: ") + name);
    }
    if (c.draft) {
        for (const auto* name : {"target_features", "draft_features"}) {
            require(module_->has_input(name) && module_->tensor_dtype(name) == DType::kFloat16,
                    std::string("conditioning contract mismatch: ") + name);
        }
        target_input_ =
            DeviceTensor({c.max_query, c.feature_width}, DType::kFloat16, module_->stream());
        draft_input_ = DeviceTensor({c.max_query, c.hidden}, DType::kFloat16, module_->stream());
        require(target_input_.ok() && draft_input_.ok(), "conditioning allocation failed");
    }
    const std::vector<std::int64_t> shape{1, c.heads, c.capacity, c.dim};
    for (int layer = 0; layer < c.layers; ++layer) {
        keys_.push_back(DeviceTensor::zeros(shape, DType::kFloat16, module_->stream()));
        values_.push_back(DeviceTensor::zeros(shape, DType::kFloat16, module_->stream()));
        for (const auto* kind : {"k", "v"}) {
            const auto suffix = std::string(kind) + "_" + std::to_string(layer);
            const auto input = "cache_" + suffix;
            const auto output = "present_" + suffix;
            require(module_->has_input(input) && module_->has_output(output) &&
                        module_->tensor_shape(input) == shape &&
                        module_->tensor_shape(output) == shape &&
                        module_->tensor_dtype(input) == DType::kFloat16 &&
                        module_->tensor_dtype(output) == DType::kFloat16,
                    "cache tensor does not match manifest: " + input);
            auto& tensor = kind[0] == 'k' ? keys_.back() : values_.back();
            require(tensor.ok(), "KV allocation failed");
            module_->bind_external(input, tensor.data());
            if (prefill_)
                prefill_->bind_external(input, tensor.data());
            // Output bindings must propagate through TensorRT's engine alias
            // metadata; equal external bindings alone are insufficient.
            require(module_->device_ptr(output) == tensor.data(),
                    "required cache allocation binding is missing: " + output);
            if (prefill_)
                require(prefill_->device_ptr(output) == tensor.data(),
                        "prefill cache allocation binding is missing: " + output);
        }
    }
    module_->sync();
}

StepResult Engine::run(Phase phase, const std::vector<std::int32_t>& tokens, int start,
                       const std::vector<std::int32_t>& parents, bool all_logits,
                       FeatureView target_features, FeatureView draft_features) {
    const auto& c = contract_;
    if (device_bound_) {
        // Restore owned maximum-sized inputs before a host call. Borrowed
        // device-policy buffers can be smaller than a subsequent prefill.
        for (const auto* name : {"token_id", "position_id", "logits_indices", "attention_mask",
                                 "key_value_lengths", "cache_write_indices"}) {
            auto it = device_inputs_.find(name);
            if (it == device_inputs_.end()) {
                std::vector<std::int64_t> shape{c.max_query};
                if (std::string(name) == "attention_mask")
                    shape.push_back(c.capacity);
                else if (std::string(name) == "key_value_lengths" ||
                         std::string(name) == "cache_write_indices")
                    shape = {1};
                it = device_inputs_.emplace(name, DeviceTensor(shape, DType::kInt32, stream()))
                         .first;
                require(it->second.ok(), "host input allocation failed");
            }
            module_->bind_external(name, it->second.data());
        }
        device_bound_ = false;
    }
    const int rows = static_cast<int>(tokens.size());
    require(rows > 0 && rows <= c.query_limit(phase) && start >= 0 && start <= c.capacity - rows,
            "query exceeds engine profile or cache bounds");
    require(parents.size() == tokens.size(), "parent count mismatch");
    auto& positions = positions_;
    auto& mask = mask_;
    positions.resize(rows);
    mask.assign(static_cast<std::size_t>(rows) * c.capacity, 0);
    for (int row = 0; row < rows; ++row) {
        require(parents[row] >= -1 && parents[row] < row, "parents must precede children");
        positions[row] = parents[row] < 0 ? start : positions[parents[row]] + 1;
        auto* visible = mask.data() + static_cast<std::size_t>(row) * c.capacity;
        std::fill(visible, visible + start, 1);
        for (int ancestor = row; ancestor >= 0; ancestor = parents[ancestor])
            visible[start + ancestor] = 1;
    }
    auto& selected = selected_;
    selected.clear();
    if (all_logits) {
        for (int row = 0; row < rows; ++row)
            selected.push_back(row);
    } else {
        selected.push_back(rows - 1);
    }
    std::int32_t write_start = start, length = start + rows;
    TensorMap inputs{
        {"token_id", {const_cast<std::int32_t*>(tokens.data()), {rows}, DType::kInt32}},
        {"position_id", {positions.data(), {rows}, DType::kInt32}},
        {"logits_indices",
         {selected.data(), {static_cast<std::int64_t>(selected.size())}, DType::kInt32}},
        {"attention_mask", {mask.data(), {rows, c.capacity}, DType::kInt32}},
        {"key_value_lengths", {&length, {1}, DType::kInt32}},
    };
    inputs["cache_write_indices"] = {&write_start, {1}, DType::kInt32};
    auto& module = phase == Phase::kPrefill && prefill_ ? prefill_ : module_;
    if (c.draft) {
        auto stage = [&](const char* name, FeatureView source, DeviceTensor& destination,
                         int width) {
            const auto bytes = std::size_t(rows) * width * sizeof(std::uint16_t);
            if (!source.data) {
                require(source.rows == 0 && source.width == 0, "invalid empty conditioning");
                checked(cudaMemsetAsync(destination.data(), 0, bytes, module->stream()));
            } else {
                require(source.width == width && source.rows == rows,
                        "draft conditioning shape mismatch");
                checked(cudaMemcpyAsync(destination.data(), source.data, bytes,
                                        cudaMemcpyDeviceToDevice, module->stream()));
            }
            module->bind_external(name, destination.data(), {rows, width});
        };
        stage("target_features", target_features, target_input_, c.feature_width);
        stage("draft_features", draft_features, draft_input_, c.hidden);
    }
    if (!c.profiles.empty()) {
        const auto& bounds = c.profiles[phase == Phase::kPrefill ? 0 : 1].logits;
        require(selected.size() >= std::size_t(bounds[0]) &&
                    selected.size() <= std::size_t(bounds[2]),
                "selected logits exceed phase profile");
    }
    module->forward_async(inputs);
    StepResult result;
    result.vocab = c.vocab;
    result.ranks = c.draft ? 2 : 1;
    // Target/draft and selector contexts may use different streams. The
    // event orders the borrowed logits; the final sync completes both.
    auto ready = static_cast<cudaEvent_t>(selection_ready_.get());
    checked(cudaEventRecord(ready, module->stream()));
    checked(cudaStreamWaitEvent(selection_->stream(), ready, 0));
    selection_->bind_external("logits", module->device_ptr("logits"),
                              {static_cast<std::int64_t>(selected.size()), c.vocab});
    selection_->forward_device_async({});
    result.selection.resize(selected.size() * (result.ranks + 1));
    checked(cudaMemcpyAsync(selection_host_.get(), selection_->device_ptr("selection"),
                            result.selection.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                            selection_->stream()));
    checked(cudaStreamSynchronize(selection_->stream()));
    std::copy_n(static_cast<const std::int32_t*>(selection_host_.get()), result.selection.size(),
                result.selection.begin());
    result.features = {static_cast<const std::uint16_t*>(module->device_ptr("features")), rows,
                       c.draft ? c.hidden : c.feature_width};
    return result;
}

DeviceStep Engine::device_result(FeatureView features) const {
    require(selection_ != nullptr, "device policy requires a GPU selector");
    return {static_cast<const std::int32_t*>(selection_->device_ptr("selection")), features,
            selection_->stream()};
}

DeviceStep Engine::run_device(const std::int32_t* tokens, int rows, ITrtModule& metadata,
                              const std::int32_t* start, bool all_logits,
                              FeatureView target_features, FeatureView draft_features) {
    const auto& c = contract_;
    require(selection_ && tokens && start && rows > 0 && rows <= c.query_limit(Phase::kDecode),
            "invalid device invocation");
    // The caller orders policy production and protects these borrowed inputs
    // through the returned completion stream, including before template reuse.
    auto bind = [&](const char* name, const void* source, std::vector<std::int64_t> shape) {
        module_->bind_external(name, const_cast<void*>(source), shape);
    };
    bind("token_id", tokens, {rows});
    bind("position_id", metadata.device_ptr("position_id"), {rows});
    bind("logits_indices", metadata.device_ptr(all_logits ? "logits_all" : "logits_last"),
         {all_logits ? rows : 1});
    bind("attention_mask", metadata.device_ptr("attention_mask"), {rows, c.capacity});
    bind("key_value_lengths", metadata.device_ptr("key_value_lengths"), {1});
    bind("cache_write_indices", start, {1});
    device_bound_ = true;
    if (c.draft) {
        auto condition = [&](const char* name, FeatureView source, DeviceTensor& destination,
                             int width) {
            const auto bytes = std::size_t(rows) * width * sizeof(std::uint16_t);
            if (source.data) {
                require(source.rows == rows && source.width == width,
                        "device conditioning shape mismatch");
                checked(cudaMemcpyAsync(destination.data(), source.data, bytes,
                                        cudaMemcpyDeviceToDevice, stream()));
            } else {
                checked(cudaMemsetAsync(destination.data(), 0, bytes, stream()));
            }
            module_->bind_external(name, destination.data(), {rows, width});
        };
        condition("target_features", target_features, target_input_, c.feature_width);
        condition("draft_features", draft_features, draft_input_, c.hidden);
    }
    module_->forward_device_async({});
    auto event = static_cast<cudaEvent_t>(selection_ready_.get());
    checked(cudaEventRecord(event, stream()));
    checked(cudaStreamWaitEvent(selection_->stream(), event, 0));
    selection_->bind_external("logits", module_->device_ptr("logits"),
                              {all_logits ? rows : 1, c.vocab});
    selection_->forward_device_async({});
    return device_result({static_cast<const std::uint16_t*>(module_->device_ptr("features")), rows,
                          c.draft ? c.hidden : c.feature_width});
}

void Engine::commit_device(int start, const std::int32_t* device_start, const std::int32_t* path,
                           int length, ITrtModule& gather) {
    const auto& c = contract_;
    require(start >= 0 && length > 0 && start + length <= c.capacity &&
                length <= c.query_limit(Phase::kDecode),
            "invalid device commit span");
    gather.bind_external("start", const_cast<std::int32_t*>(device_start), {1});
    gather.bind_external("path", const_cast<std::int32_t*>(path), {length});
    const auto element_row = std::size_t(c.dim) * sizeof(std::uint16_t);
    for (int layer = 0; layer < c.layers; ++layer) {
        gather.bind_external("k", keys_[layer].data());
        gather.bind_external("v", values_[layer].data());
        gather.forward_device_async({});
        // Gather the whole path before overwriting any source, then copy the
        // accepted span across all KV heads in one call per K/V tensor.
        for (const auto* name : {"k_out", "v_out"}) {
            auto& cache = name[0] == 'k' ? keys_[layer] : values_[layer];
            checked(cudaMemcpy2DAsync(static_cast<char*>(cache.data()) + start * element_row,
                                      c.capacity * element_row, gather.device_ptr(name),
                                      length * element_row, length * element_row, c.heads,
                                      cudaMemcpyDeviceToDevice, gather.stream()));
        }
    }
}

void Engine::reset() {
    // No byte clearing: subsequent inputs expose only the current valid prefix.
    module_->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
    selection_->reset_execution_context();
}

FeatureView FeatureView::slice(int first, int count) const {
    require(data && width > 0 && count > 0 && first >= 0 && first <= rows - count,
            "invalid device feature slice");
    return {data + std::size_t(first) * width, count, width};
}

std::int32_t StepResult::token(int row, int rank) const {
    require(row >= 0 && rank >= 0 && rank < 2 && vocab > rank, "invalid selection index");
    require(ranks >= 1 && ranks <= 2 && rank < ranks && selection.size() % (ranks + 1) == 0 &&
                std::size_t(row) < selection.size() / (ranks + 1),
            "invalid compact selection");
    const auto offset = std::size_t(row) * (ranks + 1);
    require(selection[offset + ranks] == 1, "non-finite logits");
    const auto id = selection[offset + rank];
    require(id >= 0 && id < vocab, "selected token outside vocabulary");
    return id;
}
} // namespace trtmc::llama::speculative
