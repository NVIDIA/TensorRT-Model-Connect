/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/recurrent.h"

#include "api_internal.h"
#include "trtmc/recurrent.h"

#include <algorithm>
#include <set>

struct trtmc_recurrent_state {
    // Family state/leases are destroyed before releasing the loaded model/DSO.
    std::shared_ptr<trtmc::api::ModelState> owner;
    std::unique_ptr<trtmc::internal::IRecurrentState> implementation;
    bool poisoned{false};
};

namespace trtmc::api {
namespace {
using Memory = internal::RecurrentMemory;
void output_check(bool value, const char* message) {
    if (!value)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}
std::unique_lock<std::mutex> model_lock(const std::shared_ptr<ModelState>& owner,
                                        bool idle = true) {
    std::unique_lock<std::mutex> lock(model_mutex(owner), std::try_to_lock);
    if (!lock.owns_lock())
        throw ApiFailure{TRTMC_BUSY, "model is executing another operation"};
    if (idle)
        require_model_idle(owner);
    return lock;
}
trtmc_recurrent_state& checked_state(trtmc_recurrent_state* state) {
    require(state && state->implementation, "recurrent state is null");
    return *state;
}
const trtmc_recurrent_state& checked_state(const trtmc_recurrent_state* state) {
    require(state && state->implementation, "recurrent state is null");
    return *state;
}
void require_usable(const trtmc_recurrent_state& state) {
    require(!state.poisoned, "recurrent state is poisoned; reset or assign before reuse");
    require(state.implementation->info().context_valid,
            "family reports stale recurrent state context");
}
template <class Interface>
Interface& family_interface(const std::shared_ptr<ModelState>& owner) {
    auto* family = dynamic_cast<Interface*>(&require_family(owner, Interface::kTask));
    if (!family)
        throw ApiFailure{TRTMC_UNSUPPORTED, "family has not implemented the recurrent Task"};
    return *family;
}
Memory memory(std::uint32_t value, bool default_host = false) {
    if (default_host && value == 0)
        return Memory::Host;
    require(value == TRTMC_RECURRENT_HOST || value == TRTMC_RECURRENT_CUDA,
            "unknown recurrent memory target");
    return static_cast<Memory>(value);
}
std::size_t scalar_size(std::uint32_t value) {
    require(value >= TRTMC_RECURRENT_FLOAT32 && value <= TRTMC_RECURRENT_BFLOAT16,
            "unknown recurrent scalar type");
    return value == TRTMC_RECURRENT_FLOAT32 ? 4U : 2U;
}
internal::RecurrentArrayView array_input(const trtmc_recurrent_array_v1& input,
                                         bool empty_rows = false) {
    require(input.columns > 0 && (empty_rows || input.rows > 0),
            "recurrent array axes must be positive");
    const auto columns = checked_size(input.columns, 1);
    checked_size(input.rows, columns);
    const auto elements = input.rows * input.columns;
    const auto width = scalar_size(input.buffer.scalar);
    checked_size(elements, width);
    require(input.buffer.byte_size == elements * width,
            "recurrent array byte size disagrees with its axes/dtype");
    require(input.buffer.data || input.buffer.byte_size == 0, "recurrent array data is null");
    const auto domain = memory(input.buffer.memory);
    require(domain == Memory::Host ? input.buffer.device_ordinal == -1
                                   : input.buffer.device_ordinal >= 0,
            "recurrent array memory domain/device ordinal disagree");
    return {{input.buffer.data, input.buffer.byte_size,
             static_cast<internal::RecurrentScalar>(input.buffer.scalar), domain,
             input.buffer.device_ordinal},
            input.rows,
            input.columns};
}
trtmc_recurrent_array_v1 array_output(const internal::RecurrentArray& input, Memory target,
                                      bool empty_rows = false) {
    const auto& value = input.view;
    trtmc_recurrent_array_v1 out{
        {value.buffer.data, value.buffer.byte_size, static_cast<std::uint32_t>(value.buffer.scalar),
         static_cast<std::uint32_t>(value.buffer.memory), value.buffer.device_ordinal},
        value.rows,
        value.columns};
    try {
        (void)array_input(out, empty_rows);
    } catch (const ApiFailure& error) {
        throw ApiFailure{TRTMC_INTERNAL_ERROR, error.message};
    }
    output_check(value.buffer.memory == target,
                 "family changed the requested output memory target");
    output_check(input.lease || value.buffer.byte_size == 0,
                 "nonempty recurrent result lacks an owned lease");
    return out;
}
internal::RecurrentTokensInput tokens_input(const trtmc_recurrent_tokens_input_v1& input) {
    const auto ids = checked_span(input.token_ids.data, input.token_ids.size);
    const auto mask = checked_span(input.input_mask, input.mask_count);
    require(!ids.empty(), "recurrent forward requires supplied tokens");
    require(mask.empty() || mask.size() == ids.size(),
            "recurrent input mask length differs from token count");
    for (const auto value : mask)
        require(value <= 1, "recurrent mask values must be zero or one");
    return {ids, mask};
}
internal::RecurrentEmbeddingsInput
embeddings_input(const trtmc_recurrent_embeddings_input_v1& input) {
    const auto values = array_input(input.embeddings);
    const auto mask = checked_span(input.input_mask, input.mask_count);
    require(mask.empty() || mask.size() == values.rows,
            "embedding mask length differs from token rows");
    for (const auto value : mask)
        require(value <= 1, "embedding mask must contain zero or one");
    return {values, mask};
}
std::uint64_t sequence_length(const internal::RecurrentTokensInput& input) {
    return input.token_ids.size();
}
std::uint64_t sequence_length(const internal::RecurrentEmbeddingsInput& input) {
    return input.embeddings.rows;
}
internal::RecurrentLogitRows logit_rows(const trtmc_recurrent_logit_rows_v1& input,
                                        std::uint64_t tokens) {
    require(input.kind <= TRTMC_RECURRENT_ROWS_INDICES, "unknown recurrent logit-row selector");
    const auto indices = checked_span(input.indices, input.index_count);
    if (input.kind == TRTMC_RECURRENT_ROWS_ALL)
        require(input.last_count == 0 && indices.empty(), "All row selection has extra operands");
    else if (input.kind == TRTMC_RECURRENT_ROWS_LAST)
        require(input.last_count > 0 && indices.empty(),
                "Last row selection requires a positive count only");
    else {
        require(input.last_count == 0, "Indices row selection cannot also supply Last count");
        for (const auto index : indices)
            require(index < tokens, "selected logit row is outside supplied inputs");
    }
    return {static_cast<internal::RecurrentLogitRowsKind>(input.kind), input.last_count, indices};
}
internal::RecurrentTokensLogitsRequest
request_input(const trtmc_recurrent_tokens_logits_request_v1& input) {
    const auto tokens = tokens_input(input.input);
    return {tokens, logit_rows(input.rows, tokens.token_ids.size()),
            memory(input.output_memory, true)};
}
internal::RecurrentEmbeddingsLogitsRequest
request_input(const trtmc_recurrent_embeddings_logits_request_v1& input) {
    const auto values = embeddings_input(input.input);
    return {values, logit_rows(input.rows, values.embeddings.rows),
            memory(input.output_memory, true)};
}
internal::RecurrentTokensHiddenRequest
request_input(const trtmc_recurrent_tokens_hidden_request_v1& input) {
    return {tokens_input(input.input), memory(input.output_memory, true)};
}
internal::RecurrentEmbeddingsHiddenRequest
request_input(const trtmc_recurrent_embeddings_hidden_request_v1& input) {
    return {embeddings_input(input.input), memory(input.output_memory, true)};
}
struct TraceViews {
    explicit TraceViews(const std::optional<std::vector<internal::RecurrentTraceItem>>& input,
                        std::uint64_t tokens, Memory target) {
        if (!input)
            return;
        for (const auto& item : *input) {
            const auto stage = static_cast<std::uint32_t>(item.stage);
            output_check(stage >= TRTMC_RECURRENT_POST_BLOCK &&
                             stage <= TRTMC_RECURRENT_MIXER_CONTRIBUTION,
                         "unknown recurrent trace stage");
            output_check(stage == TRTMC_RECURRENT_FINAL_NORMALIZATION ? item.block_index == -1
                                                                      : item.block_index >= 0,
                         "trace stage and block index disagree");
            output_check(item.values.view.rows == tokens,
                         "recurrent trace omitted supplied token rows");
            items.push_back({stage, item.block_index, array_output(item.values, target)});
        }
        view = {1, items.data(), items.size()};
    }
    std::vector<trtmc_recurrent_trace_item_v1> items;
    trtmc_recurrent_trace_v1 view{};
};
struct LogitsStorage final : ResultStorage {
    template <class Request>
    LogitsStorage(std::shared_ptr<ModelState> model, internal::RecurrentLogitsResult input,
                  const Request& request)
        : owner(std::move(model)), value(std::move(input)),
          trace(value.trace, sequence_length(request.input), request.output_memory) {
        const auto tokens = sequence_length(request.input);
        const auto& rows = request.rows;
        const auto count = rows.kind == internal::RecurrentLogitRowsKind::All
                               ? tokens
                               : (rows.kind == internal::RecurrentLogitRowsKind::Last
                                      ? std::min<std::uint64_t>(rows.last_count, tokens)
                                      : rows.indices.size());
        output_check(value.token_positions.size() == count && value.logits.view.rows == count &&
                         !value.vocabulary_id.empty(),
                     "logit result lacks selected row/vocabulary metadata");
        for (std::size_t i = 0; i < count; ++i) {
            const auto expected =
                rows.kind == internal::RecurrentLogitRowsKind::All
                    ? i
                    : (rows.kind == internal::RecurrentLogitRowsKind::Last ? tokens - count + i
                                                                           : rows.indices[i]);
            output_check(value.token_positions[i] == expected,
                         "family changed selected logit-row order");
        }
        view = {array_output(value.logits, request.output_memory, true),
                value.token_positions.data(), value.token_positions.size(),
                borrowed_string(value.vocabulary_id), trace.view};
    }
    std::shared_ptr<ModelState> owner;
    internal::RecurrentLogitsResult value;
    TraceViews trace;
    trtmc_recurrent_logits_view_v1 view{};
};
struct HiddenStorage final : ResultStorage {
    template <class Request>
    HiddenStorage(std::shared_ptr<ModelState> model, internal::RecurrentHiddenResult input,
                  const Request& request)
        : owner(std::move(model)), value(std::move(input)),
          trace(value.trace, sequence_length(request.input), request.output_memory) {
        output_check(value.hidden.view.rows == sequence_length(request.input),
                     "hidden result omitted supplied token rows");
        view = {array_output(value.hidden, request.output_memory), trace.view};
    }
    std::shared_ptr<ModelState> owner;
    internal::RecurrentHiddenResult value;
    TraceViews trace;
    trtmc_recurrent_hidden_view_v1 view{};
};
template <class Storage>
struct BatchStorage final : ResultStorage {
    template <class Value, class Request>
    BatchStorage(const std::shared_ptr<ModelState>& owner, std::vector<Value> values,
                 const std::vector<internal::RecurrentBatchItem<Request>>& requests) {
        output_check(values.size() == requests.size(), "native recurrent batch changed item count");
        for (std::size_t i = 0; i < values.size(); ++i)
            items.push_back(
                std::make_unique<Storage>(owner, std::move(values[i]), requests[i].request));
    }
    std::vector<std::unique_ptr<Storage>> items;
};
template <class Storage>
trtmc_status TRTMC_CALL batch_count(const trtmc_result* result, std::uint64_t* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out, "batch count output is null");
        *out = require_result<BatchStorage<Storage>>(result).items.size();
    });
}
template <class Storage, class View>
trtmc_status TRTMC_CALL batch_item(const trtmc_result* result, std::uint64_t index, View* out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "batch item output is null");
        const auto& items = require_result<BatchStorage<Storage>>(result).items;
        require(index < items.size(), "recurrent batch item index is out of range");
        *out = items[static_cast<std::size_t>(index)]->view;
    });
}
struct MambaStorage final : ResultStorage {
    MambaStorage(std::shared_ptr<ModelState> model, internal::Mamba1StateSnapshot input,
                 Memory target)
        : owner(std::move(model)), value(std::move(input)) {
        output_check(!value.layers.empty(), "Mamba state snapshot has no layers");
        for (const auto& layer : value.layers) {
            output_check(layer.convolution.view.rows == layer.ssm.view.rows,
                         "Mamba state widths disagree");
            layers.push_back({array_output(layer.convolution, target),
                              array_output(layer.ssm, target), layer.has_previous_state ? 1U : 0U});
        }
        view = {layers.data(), layers.size(), value.tokens_seen ? 1U : 0U,
                value.tokens_seen.value_or(0)};
    }
    std::shared_ptr<ModelState> owner;
    internal::Mamba1StateSnapshot value;
    std::vector<trtmc_mamba1_layer_state_v1> layers;
    trtmc_mamba1_state_view_v1 view{};
};
struct RwkvStorage final : ResultStorage {
    RwkvStorage(std::shared_ptr<ModelState> model, internal::Rwkv4StateSnapshot input,
                Memory target)
        : owner(std::move(model)), value(std::move(input)) {
        view = {array_output(value.ffn_previous, target),
                array_output(value.attention_previous, target),
                array_output(value.wkv_numerator, target),
                array_output(value.wkv_denominator, target),
                array_output(value.wkv_running_max, target),
                value.tokens_seen ? 1U : 0U,
                value.tokens_seen.value_or(0)};
        const auto rows = view.ffn_previous.rows, columns = view.ffn_previous.columns;
        for (const auto* item : {&view.attention_previous, &view.wkv_numerator,
                                 &view.wkv_denominator, &view.wkv_running_max})
            output_check(item->rows == rows && item->columns == columns,
                         "RWKV4 state component axes disagree");
    }
    std::shared_ptr<ModelState> owner;
    internal::Rwkv4StateSnapshot value;
    trtmc_rwkv4_state_view_v1 view{};
};
template <class Storage, class View>
trtmc_status TRTMC_CALL result_view(const trtmc_result* result, View* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "recurrent result view output is null");
        *out = require_result<Storage>(result).view;
    });
}
template <class Interface>
trtmc_status TRTMC_CALL create_state(trtmc_model* model, trtmc_recurrent_state** out,
                                     trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "state output is null");
        auto owner = model_owner(model);
        auto lock = model_lock(owner);
        auto implementation = family_interface<Interface>(owner).create_recurrent_state();
        output_check(implementation != nullptr, "family returned no recurrent state");
        *out = new trtmc_recurrent_state{std::move(owner), std::move(implementation), false};
    });
}
trtmc_status TRTMC_CALL clone_state(const trtmc_recurrent_state* input, trtmc_recurrent_state** out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "clone output is null");
        const auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        require_usable(state);
        auto clone = state.implementation->clone();
        if (clone.get() == state.implementation.get()) {
            clone.release();
            throw ApiFailure{TRTMC_INTERNAL_ERROR, "family clone reused the original state object"};
        }
        output_check(clone != nullptr, "family returned no cloned state");
        *out = new trtmc_recurrent_state{state.owner, std::move(clone), false};
    });
}
trtmc_status TRTMC_CALL reset_state(trtmc_recurrent_state* input, trtmc_error** error) noexcept {
    return guarded(error, [&] {
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        state.poisoned = true;
        state.implementation->reset();
        state.poisoned = false;
    });
}
trtmc_status TRTMC_CALL state_info(const trtmc_recurrent_state* input,
                                   trtmc_recurrent_state_info_v1* out,
                                   trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out, "state info output is null");
        const auto& state = checked_state(input);
        auto lock = model_lock(state.owner, false);
        const auto info = state.implementation->info();
        *out = {state.poisoned ? 1U : 0U,     info.context_valid ? 1U : 0U,
                info.initialized ? 1U : 0U,   info.tokens_seen ? 1U : 0U,
                info.tokens_seen.value_or(0), info.device_memory_bytes};
    });
}
void TRTMC_CALL release_state(trtmc_recurrent_state* state) noexcept {
    if (!state)
        return;
    {
        const std::lock_guard<std::mutex> lock(model_mutex(state->owner));
        state->implementation.reset();
    }
    delete state;
}
template <class Exchange>
Exchange& require_exchange(Exchange* value) {
    if (!value)
        throw ApiFailure{TRTMC_UNSUPPORTED,
                         "state does not expose the requested typed interchange"};
    return *value;
}
trtmc_status TRTMC_CALL assign_mamba(trtmc_recurrent_state* input,
                                     const trtmc_mamba1_state_view_v1* view,
                                     trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(view, "Mamba state input is null");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        auto& exchange = require_exchange(state.implementation->mamba1_exchange());
        require(view->has_tokens_seen <= 1, "state position presence must be zero or one");
        const auto supplied = checked_span(view->layers, view->layer_count);
        require(!supplied.empty(), "Mamba state requires layers");
        std::vector<internal::Mamba1LayerStateView> layers;
        for (const auto& layer : supplied) {
            require(layer.has_previous_state <= 1, "Mamba previous-state flag must be zero or one");
            auto conv = array_input(layer.convolution), ssm = array_input(layer.ssm);
            require(conv.rows == ssm.rows, "Mamba state widths disagree");
            layers.push_back({conv, ssm, layer.has_previous_state != 0});
        }
        internal::Mamba1StateView value{
            {layers.data(), layers.size()},
            view->has_tokens_seen ? std::optional<std::uint64_t>{view->tokens_seen} : std::nullopt};
        const bool previous_poison = state.poisoned;
        state.poisoned = true;
        try {
            exchange.assign(value);
        } catch (const std::invalid_argument&) {
            state.poisoned = previous_poison;
            throw;
        } catch (const internal::UnsupportedTask&) {
            state.poisoned = previous_poison;
            throw;
        }
        state.poisoned = false;
    });
}
trtmc_status TRTMC_CALL snapshot_mamba(trtmc_recurrent_state* input, std::uint32_t target,
                                       trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "Mamba snapshot output is null");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        require_usable(state);
        auto& exchange = require_exchange(state.implementation->mamba1_exchange());
        const auto domain = memory(target, true);
        *out = make_result<MambaStorage>(state.owner, exchange.snapshot(domain), domain);
    });
}
trtmc_status TRTMC_CALL assign_rwkv(trtmc_recurrent_state* input,
                                    const trtmc_rwkv4_state_view_v1* view,
                                    trtmc_error** error) noexcept {
    return guarded(error, [&] {
        require(view, "RWKV state input is null");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        auto& exchange = require_exchange(state.implementation->rwkv4_exchange());
        require(view->has_tokens_seen <= 1, "state position presence must be zero or one");
        internal::Rwkv4StateView value{
            array_input(view->ffn_previous),
            array_input(view->attention_previous),
            array_input(view->wkv_numerator),
            array_input(view->wkv_denominator),
            array_input(view->wkv_running_max),
            view->has_tokens_seen ? std::optional<std::uint64_t>{view->tokens_seen} : std::nullopt};
        const auto rows = value.ffn_previous.rows, columns = value.ffn_previous.columns;
        for (const auto* item : {&value.attention_previous, &value.wkv_numerator,
                                 &value.wkv_denominator, &value.wkv_running_max})
            require(item->rows == rows && item->columns == columns,
                    "RWKV4 state component axes disagree");
        const bool previous_poison = state.poisoned;
        state.poisoned = true;
        try {
            exchange.assign(value);
        } catch (const std::invalid_argument&) {
            state.poisoned = previous_poison;
            throw;
        } catch (const internal::UnsupportedTask&) {
            state.poisoned = previous_poison;
            throw;
        }
        state.poisoned = false;
    });
}
trtmc_status TRTMC_CALL snapshot_rwkv(trtmc_recurrent_state* input, std::uint32_t target,
                                      trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "RWKV snapshot output is null");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        require_usable(state);
        auto& exchange = require_exchange(state.implementation->rwkv4_exchange());
        const auto domain = memory(target, true);
        *out = make_result<RwkvStorage>(state.owner, exchange.snapshot(domain), domain);
    });
}
const trtmc_mamba1_state_exchange_api_v1 mamba_api{
    {1, 0, sizeof(mamba_api)},
    assign_mamba,
    snapshot_mamba,
    result_view<MambaStorage, trtmc_mamba1_state_view_v1>};
const trtmc_rwkv4_state_exchange_api_v1 rwkv_api{
    {1, 0, sizeof(rwkv_api)},
    assign_rwkv,
    snapshot_rwkv,
    result_view<RwkvStorage, trtmc_rwkv4_state_view_v1>};
trtmc_status TRTMC_CALL get_mamba(trtmc_recurrent_state* input, std::uint32_t major,
                                  std::uint32_t minor,
                                  const trtmc_mamba1_state_exchange_api_v1** out,
                                  trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "Mamba exchange output is null");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner, false);
        if (major != 1 || minor != 0)
            throw ApiFailure{TRTMC_VERSION_MISMATCH, "Mamba exchange version unavailable"};
        (void)require_exchange(state.implementation->mamba1_exchange());
        *out = &mamba_api;
    });
}
trtmc_status TRTMC_CALL get_rwkv(trtmc_recurrent_state* input, std::uint32_t major,
                                 std::uint32_t minor, const trtmc_rwkv4_state_exchange_api_v1** out,
                                 trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "RWKV exchange output is null");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner, false);
        if (major != 1 || minor != 0)
            throw ApiFailure{TRTMC_VERSION_MISMATCH, "RWKV exchange version unavailable"};
        (void)require_exchange(state.implementation->rwkv4_exchange());
        *out = &rwkv_api;
    });
}
const trtmc_recurrent_state_api_v1 state_api{{1, 0, sizeof(state_api)},
                                             clone_state,
                                             reset_state,
                                             state_info,
                                             get_mamba,
                                             get_rwkv,
                                             release_state};
template <class Interface, class Storage, class Request, class Invoke>
trtmc_status forward_one(trtmc_recurrent_state* input, const Request* request,
                         const trtmc_config_view_v1* config, trtmc_result** out,
                         trtmc_error** error, Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request && out, "recurrent request and result output are required");
        auto& state = checked_state(input);
        auto lock = model_lock(state.owner);
        require_usable(state);
        const auto value = request_input(*request);
        const ConvertedConfig options(config);
        auto& family = family_interface<Interface>(state.owner);
        // Mutation can precede execution/packing errors. Never silently retry.
        state.poisoned = true;
        auto result = [&] {
            try {
                return invoke(family, *state.implementation, value, options.view());
            } catch (const std::invalid_argument&) {
                state.poisoned = false;
                throw;
            } catch (const internal::UnsupportedTask&) {
                state.poisoned = false;
                throw;
            }
        }();
        *out = make_result<Storage>(state.owner, std::move(result), value);
        state.poisoned = false;
    });
}
#define RECURRENT_FORWARD(Function, Interface, Request, Storage, Method)                           \
    trtmc_status TRTMC_CALL Function(trtmc_recurrent_state* state, const Request* request,         \
                                     const trtmc_config_view_v1* config, trtmc_result** out,       \
                                     trtmc_error** error) noexcept {                               \
        return forward_one<internal::Interface, Storage>(                                          \
            state, request, config, out, error,                                                    \
            [](auto& family, auto& input, const auto& value, auto options) {                       \
                return family.Method(input, value, options);                                       \
            });                                                                                    \
    }
RECURRENT_FORWARD(forward_tokens_logits, IRecurrentTokensToLogits,
                  trtmc_recurrent_tokens_logits_request_v1, LogitsStorage, forward_tokens_logits)
RECURRENT_FORWARD(forward_embeddings_logits, IRecurrentEmbeddingsToLogits,
                  trtmc_recurrent_embeddings_logits_request_v1, LogitsStorage,
                  forward_embeddings_logits)
RECURRENT_FORWARD(forward_tokens_hidden, IRecurrentTokensToHiddenStates,
                  trtmc_recurrent_tokens_hidden_request_v1, HiddenStorage, forward_tokens_hidden)
RECURRENT_FORWARD(forward_embeddings_hidden, IRecurrentEmbeddingsToHiddenStates,
                  trtmc_recurrent_embeddings_hidden_request_v1, HiddenStorage,
                  forward_embeddings_hidden)
#undef RECURRENT_FORWARD
template <class Interface, class Storage, class WireItem, class Invoke>
trtmc_status forward_batch(const WireItem* input, std::uint64_t count, trtmc_result** out,
                           trtmc_error** error, Invoke invoke) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(out, "batch result output is null");
        const auto source = checked_span(input, count);
        require(!source.empty(), "recurrent batch requires items");
        auto owner = checked_state(source[0].state).owner;
        auto lock = model_lock(owner);
        auto& family = family_interface<Interface>(owner);
        using Request = decltype(request_input(source[0].request));
        std::vector<ConvertedConfig> configs;
        configs.reserve(source.size());
        std::vector<internal::RecurrentBatchItem<Request>> items;
        items.reserve(source.size());
        std::set<trtmc_recurrent_state*> states;
        for (const auto& item : source) {
            auto& state = checked_state(item.state);
            require(state.owner == owner, "batch states belong to different loaded models");
            require(states.insert(item.state).second,
                    "batch items cannot alias the same recurrent state");
            require_usable(state);
            const auto request = request_input(item.request);
            configs.emplace_back(&item.config);
            items.push_back({state.implementation.get(), request, configs.back().view()});
        }
        for (auto* state : states)
            state->poisoned = true;
        auto results = [&] {
            try {
                return invoke(family, Span<const internal::RecurrentBatchItem<Request>>{
                                          items.data(), items.size()});
            } catch (const std::invalid_argument&) {
                for (auto* state : states)
                    state->poisoned = false;
                throw;
            } catch (const internal::UnsupportedTask&) {
                for (auto* state : states)
                    state->poisoned = false;
                throw;
            }
        }();
        *out = make_result<BatchStorage<Storage>>(owner, std::move(results), items);
        for (auto* state : states)
            state->poisoned = false;
    });
}
#define RECURRENT_BATCH_FORWARD(Function, Interface, Item, Storage, Method)                        \
    trtmc_status TRTMC_CALL Function(const Item* items, std::uint64_t count, trtmc_result** out,   \
                                     trtmc_error** error) noexcept {                               \
        return forward_batch<internal::Interface, Storage>(                                        \
            items, count, out, error,                                                              \
            [](auto& family, auto values) { return family.Method(values); });                      \
    }
RECURRENT_BATCH_FORWARD(batch_tokens_logits, IBatchRecurrentTokensToLogits,
                        trtmc_batch_recurrent_tokens_to_logits_item_v1, LogitsStorage,
                        forward_batch_tokens_logits)
RECURRENT_BATCH_FORWARD(batch_embeddings_logits, IBatchRecurrentEmbeddingsToLogits,
                        trtmc_batch_recurrent_embeddings_to_logits_item_v1, LogitsStorage,
                        forward_batch_embeddings_logits)
RECURRENT_BATCH_FORWARD(batch_tokens_hidden, IBatchRecurrentTokensToHiddenStates,
                        trtmc_batch_recurrent_tokens_to_hidden_states_item_v1, HiddenStorage,
                        forward_batch_tokens_hidden)
RECURRENT_BATCH_FORWARD(batch_embeddings_hidden, IBatchRecurrentEmbeddingsToHiddenStates,
                        trtmc_batch_recurrent_embeddings_to_hidden_states_item_v1, HiddenStorage,
                        forward_batch_embeddings_hidden)
#undef RECURRENT_BATCH_FORWARD
const trtmc_recurrent_tokens_to_logits_api_v1 tokens_logits_api{
    {1, 0, sizeof(tokens_logits_api)},
    create_state<internal::IRecurrentTokensToLogits>,
    forward_tokens_logits,
    result_view<LogitsStorage, trtmc_recurrent_logits_view_v1>,
    &state_api};
const trtmc_recurrent_embeddings_to_logits_api_v1 embeddings_logits_api{
    {1, 0, sizeof(embeddings_logits_api)},
    create_state<internal::IRecurrentEmbeddingsToLogits>,
    forward_embeddings_logits,
    result_view<LogitsStorage, trtmc_recurrent_logits_view_v1>,
    &state_api};
const trtmc_recurrent_tokens_to_hidden_states_api_v1 tokens_hidden_api{
    {1, 0, sizeof(tokens_hidden_api)},
    create_state<internal::IRecurrentTokensToHiddenStates>,
    forward_tokens_hidden,
    result_view<HiddenStorage, trtmc_recurrent_hidden_view_v1>,
    &state_api};
const trtmc_recurrent_embeddings_to_hidden_states_api_v1 embeddings_hidden_api{
    {1, 0, sizeof(embeddings_hidden_api)},
    create_state<internal::IRecurrentEmbeddingsToHiddenStates>,
    forward_embeddings_hidden,
    result_view<HiddenStorage, trtmc_recurrent_hidden_view_v1>,
    &state_api};
#define RECURRENT_BATCH_TABLE(Name, Type, Interface, Function, Storage, View)                      \
    const Type Name{{1, 0, sizeof(Name)}, create_state<internal::Interface>, Function,             \
                    batch_count<Storage>, batch_item<Storage, View>,         &state_api};
RECURRENT_BATCH_TABLE(batch_tokens_logits_api, trtmc_batch_recurrent_tokens_to_logits_api_v1,
                      IBatchRecurrentTokensToLogits, batch_tokens_logits, LogitsStorage,
                      trtmc_recurrent_logits_view_v1)
RECURRENT_BATCH_TABLE(batch_embeddings_logits_api,
                      trtmc_batch_recurrent_embeddings_to_logits_api_v1,
                      IBatchRecurrentEmbeddingsToLogits, batch_embeddings_logits, LogitsStorage,
                      trtmc_recurrent_logits_view_v1)
RECURRENT_BATCH_TABLE(batch_tokens_hidden_api, trtmc_batch_recurrent_tokens_to_hidden_states_api_v1,
                      IBatchRecurrentTokensToHiddenStates, batch_tokens_hidden, HiddenStorage,
                      trtmc_recurrent_hidden_view_v1)
RECURRENT_BATCH_TABLE(batch_embeddings_hidden_api,
                      trtmc_batch_recurrent_embeddings_to_hidden_states_api_v1,
                      IBatchRecurrentEmbeddingsToHiddenStates, batch_embeddings_hidden,
                      HiddenStorage, trtmc_recurrent_hidden_view_v1)
#undef RECURRENT_BATCH_TABLE
} // namespace
Span<const TaskBinding> recurrent_task_bindings() noexcept {
#define B(Interface, Table)                                                                        \
    {                                                                                              \
        internal::Interface::kTask, 1, 0, &Table.header, implements<internal::Interface>           \
    }
    static const TaskBinding bindings[] = {
        B(IRecurrentTokensToLogits, tokens_logits_api),
        B(IRecurrentEmbeddingsToLogits, embeddings_logits_api),
        B(IRecurrentTokensToHiddenStates, tokens_hidden_api),
        B(IRecurrentEmbeddingsToHiddenStates, embeddings_hidden_api),
        B(IBatchRecurrentTokensToLogits, batch_tokens_logits_api),
        B(IBatchRecurrentEmbeddingsToLogits, batch_embeddings_logits_api),
        B(IBatchRecurrentTokensToHiddenStates, batch_tokens_hidden_api),
        B(IBatchRecurrentEmbeddingsToHiddenStates, batch_embeddings_hidden_api)};
#undef B
    return bindings;
}
} // namespace trtmc::api
