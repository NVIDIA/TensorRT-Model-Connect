/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/llama/runtime/speculative/engine.h"

#include <algorithm>
#include <cmath>
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

template <typename Select>
std::vector<std::int32_t> walk_greedy_path(const std::vector<std::int32_t>& tokens,
                                           const std::vector<std::int32_t>& parents,
                                           Select select) {
    require(!tokens.empty() && parents.size() == tokens.size() && parents[0] == -1,
            "invalid verification rows");
    for (std::size_t row = 1; row < parents.size(); ++row)
        require(parents[row] >= 0 && parents[row] < static_cast<int>(row), "invalid tree");
    std::vector<std::int32_t> path{0};
    for (;;) {
        const auto parent = path.back();
        const auto next = select(parent);
        auto found = tokens.size();
        for (std::size_t row = parent + 1; row < tokens.size(); ++row) {
            if (parents[row] == parent && tokens[row] == next) {
                found = row;
                break;
            }
        }
        if (found == tokens.size())
            return path;
        path.push_back(static_cast<std::int32_t>(found));
    }
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
    const auto selection = value.value("greedy_selection", "host");
    require(selection == "host" || selection == "device_v1", "unsupported greedy selection");
    c.device_selection = selection == "device_v1";
    require(!c.device_selection || !c.draft || c.vocab >= 2, "draft selection needs two tokens");
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
               std::unique_ptr<ITrtModule> prefill, std::unique_ptr<ITrtModule> selection)
    : module_(std::move(module)), prefill_(std::move(prefill)), selection_(std::move(selection)),
      contract_(contract) {
    require(module_ && module_->ok(), "invalid execution module");
    const auto& c = contract_;
    require(c.device_selection == (selection_ != nullptr), "selection capability/module mismatch");
    if (selection_) {
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
    commit_scratch_ = DeviceTensor({c.heads, c.query_limit(Phase::kDecode), c.dim}, DType::kFloat16,
                                   module_->stream());
    require(commit_scratch_.ok(), "commit scratch allocation failed");
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
                       FeatureView target_features, FeatureView draft_features,
                       const std::vector<std::int32_t>& feature_rows) {
    const auto& c = contract_;
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
        auto stage = [&](const char* name, FeatureView source, DeviceTensor& destination, int width,
                         const std::vector<std::int32_t>& gather) {
            const auto bytes = std::size_t(rows) * width * sizeof(std::uint16_t);
            require(gather.empty() || (source.data && gather.size() == tokens.size()),
                    "conditioning gather shape mismatch");
            if (!source.data) {
                require(source.rows == 0 && source.width == 0, "invalid empty conditioning");
                checked(cudaMemsetAsync(destination.data(), 0, bytes, module->stream()));
            } else {
                require(source.width == width &&
                            (gather.empty() ? source.rows == rows : source.rows > 0),
                        "draft conditioning shape mismatch");
                if (gather.empty()) {
                    checked(cudaMemcpyAsync(destination.data(), source.data, bytes,
                                            cudaMemcpyDeviceToDevice, module->stream()));
                } else {
                    for (int row = 0; row < rows; ++row) {
                        const auto view = source.slice(gather[row], 1);
                        checked(cudaMemcpyAsync(static_cast<std::uint16_t*>(destination.data()) +
                                                    std::size_t(row) * width,
                                                view.data,
                                                std::size_t(width) * sizeof(std::uint16_t),
                                                cudaMemcpyDeviceToDevice, module->stream()));
                    }
                }
            }
            module->bind_external(name, destination.data(), {rows, width});
        };
        stage("target_features", target_features, target_input_, c.feature_width, feature_rows);
        stage("draft_features", draft_features, draft_input_, c.hidden, {});
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
    if (selection_) {
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
                                result.selection.size() * sizeof(std::int32_t),
                                cudaMemcpyDeviceToHost, selection_->stream()));
        checked(cudaStreamSynchronize(selection_->stream()));
        std::copy_n(static_cast<const std::int32_t*>(selection_host_.get()),
                    result.selection.size(), result.selection.begin());
    } else {
        result.logits.resize(selected.size() * c.vocab);
        checked(cudaMemcpyAsync(result.logits.data(), module->device_ptr("logits"),
                                result.logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                module->stream()));
        checked(cudaStreamSynchronize(module->stream()));
    }
    result.features = {static_cast<const std::uint16_t*>(module->device_ptr("features")), rows,
                       c.draft ? c.hidden : c.feature_width};
    return result;
}

void Engine::commit(int start, const std::vector<std::int32_t>& rows) {
    const auto& c = contract_;
    require(start >= 0 && !rows.empty() && start + static_cast<int>(rows.size()) <= c.capacity,
            "invalid commit destination");
    for (auto row : rows)
        require(row >= 0 && row < c.max_query && start + row < c.capacity, "invalid commit source");
    require(rows.size() <= std::size_t(c.query_limit(Phase::kDecode)),
            "commit exceeds scratch capacity");
    auto& scratch = commit_scratch_;
    const auto width = static_cast<std::size_t>(c.dim) * sizeof(std::uint16_t);
    for (auto* storage : {&keys_, &values_}) {
        for (auto& cache : *storage) {
            for (std::size_t row = 0; row < rows.size(); ++row) {
                checked(cudaMemcpy2DAsync(
                    static_cast<char*>(scratch.data()) + row * width, rows.size() * width,
                    static_cast<char*>(cache.data()) + std::size_t(start + rows[row]) * width,
                    c.capacity * width, width, c.heads, cudaMemcpyDeviceToDevice,
                    module_->stream()));
            }
            for (std::size_t row = 0; row < rows.size(); ++row)
                checked(cudaMemcpy2DAsync(
                    static_cast<char*>(cache.data()) + (std::size_t(start) + row) * width,
                    c.capacity * width, static_cast<char*>(scratch.data()) + row * width,
                    rows.size() * width, width, c.heads, cudaMemcpyDeviceToDevice,
                    module_->stream()));
        }
    }
    module_->sync();
}

void Engine::reset() {
    // No byte clearing: subsequent inputs expose only the current valid prefix.
    module_->reset_execution_context();
    if (prefill_)
        prefill_->reset_execution_context();
    if (selection_)
        selection_->reset_execution_context();
}

FeatureView FeatureView::slice(int first, int count) const {
    require(data && width > 0 && count > 0 && first >= 0 && first <= rows - count,
            "invalid device feature slice");
    return {data + std::size_t(first) * width, count, width};
}

std::int32_t StepResult::token(int row, int rank) const {
    require(row >= 0 && rank >= 0 && rank < 2 && vocab > rank, "invalid selection index");
    if (!selection.empty()) {
        require(ranks >= 1 && ranks <= 2 && rank < ranks && selection.size() % (ranks + 1) == 0 &&
                    std::size_t(row) < selection.size() / (ranks + 1),
                "invalid compact selection");
        const auto offset = std::size_t(row) * (ranks + 1);
        require(selection[offset + ranks] == 1, "non-finite logits");
        const auto id = selection[offset + rank];
        require(id >= 0 && id < vocab, "selected token outside vocabulary");
        return id;
    }
    require(logits.size() % vocab == 0 && std::size_t(row) < logits.size() / vocab,
            "invalid host selection row");
    const auto* values = logits.data() + std::size_t(row) * vocab;
    const int best = argmax(values, vocab);
    if (rank == 0)
        return best;
    int second = best == 0 ? 1 : 0;
    for (int index = 0; index < vocab; ++index)
        if (index != best && values[index] > values[second])
            second = index;
    return second;
}

std::vector<std::int32_t> greedy_path(const std::vector<std::int32_t>& tokens,
                                      const std::vector<std::int32_t>& parents,
                                      const StepResult& result) {
    require(result.vocab > 0 &&
                (result.selection.empty()
                     ? result.logits.size() == tokens.size() * result.vocab
                     : result.ranks >= 1 && result.ranks <= 2 &&
                           result.selection.size() == tokens.size() * (result.ranks + 1)),
            "verification selection row mismatch");
    return walk_greedy_path(tokens, parents, [&](int row) { return result.token(row); });
}

std::int32_t argmax(const float* values, int count) {
    require(count > 0, "empty logits");
    // Construct the error message only on failure, not once per finite logit.
    for (int index = 0; index < count; ++index)
        if (!std::isfinite(values[index]))
            require(false, "non-finite logits");
    return static_cast<std::int32_t>(std::max_element(values, values + count) - values);
}

std::vector<std::int32_t> greedy_path(const std::vector<std::int32_t>& tokens,
                                      const std::vector<std::int32_t>& parents,
                                      const std::vector<float>& logits, int vocab) {
    require(vocab > 0 && logits.size() == tokens.size() * static_cast<std::size_t>(vocab),
            "invalid verification rows");
    return walk_greedy_path(tokens, parents, [&](int row) {
        return argmax(logits.data() + std::size_t(row) * vocab, vocab);
    });
}
} // namespace trtmc::llama::speculative
