/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/core.hpp"
#include "trtmc/recurrent.h"

namespace trtmc {
enum class RecurrentScalar : std::uint32_t {
    Float32 = TRTMC_RECURRENT_FLOAT32,
    Float16 = TRTMC_RECURRENT_FLOAT16,
    BFloat16 = TRTMC_RECURRENT_BFLOAT16
};
enum class RecurrentMemory : std::uint32_t {
    Host = TRTMC_RECURRENT_HOST,
    Cuda = TRTMC_RECURRENT_CUDA
};
class RecurrentArrayView {
  public:
    explicit RecurrentArrayView(trtmc_recurrent_array_v1 value = {}) : value_(value) {}
    static RecurrentArrayView host_f32(Span<const float> values, std::uint64_t rows,
                                       std::uint64_t columns) {
        return RecurrentArrayView{{{values.data(), values.size() * sizeof(float),
                                    TRTMC_RECURRENT_FLOAT32, TRTMC_RECURRENT_HOST, -1},
                                   rows,
                                   columns}};
    }
    const trtmc_recurrent_array_v1& c_view() const noexcept { return value_; }
    std::uint64_t rows() const noexcept { return value_.rows; }
    std::uint64_t columns() const noexcept { return value_.columns; }
    RecurrentMemory memory() const noexcept {
        return static_cast<RecurrentMemory>(value_.buffer.memory);
    }
    RecurrentScalar scalar() const noexcept {
        return static_cast<RecurrentScalar>(value_.buffer.scalar);
    }
    Span<const float> host_floats() const {
        if (memory() != RecurrentMemory::Host || scalar() != RecurrentScalar::Float32)
            throw Error(TRTMC_UNSUPPORTED, "array is not host float32; no implicit copy or cast");
        if (value_.buffer.byte_size % sizeof(float) ||
            (!value_.buffer.data && value_.buffer.byte_size))
            throw Error(TRTMC_INVALID_ARGUMENT, "invalid host float32 buffer");
        return {static_cast<const float*>(value_.buffer.data),
                static_cast<std::size_t>(value_.buffer.byte_size / sizeof(float))};
    }

  private:
    trtmc_recurrent_array_v1 value_;
};
struct RecurrentTokensInput {
    std::vector<std::int32_t> token_ids;
    std::vector<std::uint8_t> input_mask{};
    trtmc_recurrent_tokens_input_v1 c_view() const noexcept {
        return {{token_ids.data(), token_ids.size()}, input_mask.data(), input_mask.size()};
    }
};
struct RecurrentEmbeddingsInput {
    RecurrentArrayView embeddings;
    std::vector<std::uint8_t> input_mask{};
    trtmc_recurrent_embeddings_input_v1 c_view() const noexcept {
        return {embeddings.c_view(), input_mask.data(), input_mask.size()};
    }
};
enum class RecurrentLogitRowsKind : std::uint32_t {
    All = TRTMC_RECURRENT_ROWS_ALL,
    Last = TRTMC_RECURRENT_ROWS_LAST,
    Indices = TRTMC_RECURRENT_ROWS_INDICES
};
struct RecurrentLogitRows {
    RecurrentLogitRowsKind kind{RecurrentLogitRowsKind::All};
    std::uint64_t last_count{0};
    std::vector<std::uint64_t> indices{};
    static RecurrentLogitRows last(std::uint64_t count) {
        return {RecurrentLogitRowsKind::Last, count, {}};
    }
    static RecurrentLogitRows select(std::vector<std::uint64_t> rows) {
        return {RecurrentLogitRowsKind::Indices, 0, std::move(rows)};
    }
    trtmc_recurrent_logit_rows_v1 c_view() const noexcept {
        return {static_cast<std::uint32_t>(kind), last_count, indices.data(), indices.size()};
    }
};
struct RecurrentTokensLogitsRequest {
    RecurrentTokensInput input;
    RecurrentLogitRows rows{};
    RecurrentMemory output_memory{RecurrentMemory::Host};
    trtmc_recurrent_tokens_logits_request_v1 c_view() const noexcept {
        return {input.c_view(), rows.c_view(), static_cast<std::uint32_t>(output_memory)};
    }
};
struct RecurrentEmbeddingsLogitsRequest {
    RecurrentEmbeddingsInput input;
    RecurrentLogitRows rows{};
    RecurrentMemory output_memory{RecurrentMemory::Host};
    trtmc_recurrent_embeddings_logits_request_v1 c_view() const noexcept {
        return {input.c_view(), rows.c_view(), static_cast<std::uint32_t>(output_memory)};
    }
};
struct RecurrentTokensHiddenRequest {
    RecurrentTokensInput input;
    RecurrentMemory output_memory{RecurrentMemory::Host};
    trtmc_recurrent_tokens_hidden_request_v1 c_view() const noexcept {
        return {input.c_view(), static_cast<std::uint32_t>(output_memory)};
    }
};
struct RecurrentEmbeddingsHiddenRequest {
    RecurrentEmbeddingsInput input;
    RecurrentMemory output_memory{RecurrentMemory::Host};
    trtmc_recurrent_embeddings_hidden_request_v1 c_view() const noexcept {
        return {input.c_view(), static_cast<std::uint32_t>(output_memory)};
    }
};
class RecurrentLogitsView {
  public:
    explicit RecurrentLogitsView(trtmc_recurrent_logits_view_v1 value) : value_(value) {}
    RecurrentArrayView logits() const { return RecurrentArrayView{value_.logits}; }
    Span<const std::uint64_t> token_positions() const {
        return {value_.token_positions, static_cast<std::size_t>(value_.position_count)};
    }
    std::string_view vocabulary_id() const { return detail::string_view(value_.vocabulary_id); }
    const trtmc_recurrent_logits_view_v1& c_view() const noexcept { return value_; }

  private:
    trtmc_recurrent_logits_view_v1 value_;
};
class RecurrentHiddenView {
  public:
    explicit RecurrentHiddenView(trtmc_recurrent_hidden_view_v1 value) : value_(value) {}
    RecurrentArrayView hidden() const { return RecurrentArrayView{value_.hidden}; }
    const trtmc_recurrent_hidden_view_v1& c_view() const noexcept { return value_; }

  private:
    trtmc_recurrent_hidden_view_v1 value_;
};
class RecurrentLogitsResult : public detail::ViewResult<trtmc_recurrent_logits_view_v1> {
  public:
    using ViewResult::ViewResult;
    RecurrentArrayView logits() const { return RecurrentArrayView{view().logits}; }
    std::string_view vocabulary_id() const { return detail::string_view(view().vocabulary_id); }
    Span<const std::uint64_t> token_positions() const {
        const auto value = view();
        return {value.token_positions, static_cast<std::size_t>(value.position_count)};
    }
};
class RecurrentHiddenResult : public detail::ViewResult<trtmc_recurrent_hidden_view_v1> {
  public:
    using ViewResult::ViewResult;
    RecurrentArrayView hidden() const { return RecurrentArrayView{view().hidden}; }
};
struct RecurrentStateInfo {
    bool poisoned, context_valid, initialized;
    std::optional<std::uint64_t> tokens_seen;
    std::uint64_t device_memory_bytes;
    bool valid() const noexcept { return !poisoned && context_valid; }
};
namespace detail {
template <class Traits>
class RecurrentTask;
template <class Traits>
class RecurrentBatchTask;
struct RecurrentStateOwner {
    RecurrentStateOwner(std::shared_ptr<ModelState> model, const trtmc_recurrent_state_api_v1* api)
        : model(std::move(model)), api(api) {}
    ~RecurrentStateOwner() {
        if (handle)
            api->release(handle);
    }
    std::shared_ptr<ModelState> model;
    const trtmc_recurrent_state_api_v1* api;
    trtmc_recurrent_state* handle{nullptr};
};
inline void recurrent_open(const std::shared_ptr<RecurrentStateOwner>& owner) {
    if (!owner || !owner->handle)
        throw Error(TRTMC_INVALID_ARGUMENT, "recurrent state is closed");
}
inline void validate_recurrent_state_api(const trtmc_recurrent_state_api_v1* api) {
    if (!api || api->header.major != 1 || api->header.minor != 0 ||
        api->header.byte_size < sizeof(*api))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible recurrent state API");
}
} // namespace detail
class Mamba1StateExchange;
class Rwkv4StateExchange;
class RecurrentState {
  public:
    RecurrentState(const RecurrentState&) = delete;
    RecurrentState& operator=(const RecurrentState&) = delete;
    RecurrentState(RecurrentState&&) noexcept = default;
    RecurrentState& operator=(RecurrentState&&) noexcept = default;
    RecurrentState clone() const {
        detail::recurrent_open(owner_);
        auto copy = std::make_shared<detail::RecurrentStateOwner>(owner_->model, owner_->api);
        trtmc_error* error = nullptr;
        const auto status = owner_->api->clone(owner_->handle, &copy->handle, &error);
        detail::check(owner_->model->api, status, error);
        return RecurrentState(std::move(copy));
    }
    void reset() const {
        detail::recurrent_open(owner_);
        trtmc_error* error = nullptr;
        const auto status = owner_->api->reset(owner_->handle, &error);
        detail::check(owner_->model->api, status, error);
    }
    RecurrentStateInfo info() const {
        detail::recurrent_open(owner_);
        trtmc_recurrent_state_info_v1 value{};
        trtmc_error* error = nullptr;
        const auto status = owner_->api->info(owner_->handle, &value, &error);
        detail::check(owner_->model->api, status, error);
        return {value.poisoned != 0, value.context_valid != 0, value.initialized != 0,
                value.has_tokens_seen ? std::optional<std::uint64_t>{value.tokens_seen}
                                      : std::nullopt,
                value.device_memory_bytes};
    }
    bool supports_mamba1_exchange() const {
        detail::recurrent_open(owner_);
        const trtmc_mamba1_state_exchange_api_v1* api = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->get_mamba1_exchange(owner_->handle, 1, 0, &api, &error);
        if (status == TRTMC_UNSUPPORTED) {
            owner_->model->api.error_release(error);
            return false;
        }
        detail::check(owner_->model->api, status, error);
        return true;
    }
    bool supports_rwkv4_exchange() const {
        detail::recurrent_open(owner_);
        const trtmc_rwkv4_state_exchange_api_v1* api = nullptr;
        trtmc_error* error = nullptr;
        const auto status = owner_->api->get_rwkv4_exchange(owner_->handle, 1, 0, &api, &error);
        if (status == TRTMC_UNSUPPORTED) {
            owner_->model->api.error_release(error);
            return false;
        }
        detail::check(owner_->model->api, status, error);
        return true;
    }
    Mamba1StateExchange mamba1_exchange() const;
    Rwkv4StateExchange rwkv4_exchange() const;
    void close() noexcept {
        if (owner_ && owner_->handle)
            owner_->api->release(std::exchange(owner_->handle, nullptr));
        owner_.reset();
    }

  private:
    template <class Traits>
    friend class detail::RecurrentTask;
    template <class Traits>
    friend class detail::RecurrentBatchTask;
    explicit RecurrentState(std::shared_ptr<detail::RecurrentStateOwner> owner)
        : owner_(std::move(owner)) {}
    std::shared_ptr<detail::RecurrentStateOwner> owner_;
};
template <class Result>
struct RecurrentOutcome {
    RecurrentState state;
    Result output;
};
namespace detail {
template <class Traits>
class RecurrentTask {
  public:
    using Request = typename Traits::Request;
    using Result = typename Traits::Result;
    using Table = typename Traits::Table;
    static constexpr std::string_view kTask = Traits::id;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible recurrent Task table");
        detail::validate_recurrent_state_api(reinterpret_cast<const Table*>(table)->state_api);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    RecurrentState create_state() const {
        auto owner = std::make_shared<detail::RecurrentStateOwner>(model_, api_->state_api);
        trtmc_error* error = nullptr;
        const auto status = api_->create_state(model_->handle, &owner->handle, &error);
        detail::check(model_->api, status, error);
        return RecurrentState(std::move(owner));
    }
    Result forward(RecurrentState& state, const Request& request, const Config& config = {}) const {
        detail::recurrent_open(state.owner_);
        if (state.owner_->model != model_)
            throw Error(TRTMC_INVALID_ARGUMENT, "state belongs to another loaded model");
        const auto input = request.c_view();
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->forward(state.owner_->handle, &input, &options, &raw, &error);
        detail::ResultOwner result(model_, raw);
        detail::check(model_->api, status, error);
        return Result(std::move(result), api_->result_view);
    }
    RecurrentOutcome<Result> forward(const Request& request, const Config& config = {}) const {
        auto state = create_state();
        auto result = forward(state, request, config);
        return {std::move(state), std::move(result)};
    }

  private:
    friend class trtmc::Model;
    RecurrentTask(std::shared_ptr<detail::ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)), api_(reinterpret_cast<const Table*>(table)) {}
    std::shared_ptr<detail::ModelState> model_;
    const Table* api_;
};
struct TokensLogitsTraits {
    static constexpr std::string_view id = TRTMC_TASK_RECURRENT_TOKENS_TO_LOGITS;
    using Request = RecurrentTokensLogitsRequest;
    using Result = RecurrentLogitsResult;
    using Table = trtmc_recurrent_tokens_to_logits_api_v1;
};
struct EmbeddingsLogitsTraits {
    static constexpr std::string_view id = TRTMC_TASK_RECURRENT_EMBEDDINGS_TO_LOGITS;
    using Request = RecurrentEmbeddingsLogitsRequest;
    using Result = RecurrentLogitsResult;
    using Table = trtmc_recurrent_embeddings_to_logits_api_v1;
};
struct TokensHiddenTraits {
    static constexpr std::string_view id = TRTMC_TASK_RECURRENT_TOKENS_TO_HIDDEN_STATES;
    using Request = RecurrentTokensHiddenRequest;
    using Result = RecurrentHiddenResult;
    using Table = trtmc_recurrent_tokens_to_hidden_states_api_v1;
};
struct EmbeddingsHiddenTraits {
    static constexpr std::string_view id = TRTMC_TASK_RECURRENT_EMBEDDINGS_TO_HIDDEN_STATES;
    using Request = RecurrentEmbeddingsHiddenRequest;
    using Result = RecurrentHiddenResult;
    using Table = trtmc_recurrent_embeddings_to_hidden_states_api_v1;
};
} // namespace detail
using RecurrentTokensToLogits = detail::RecurrentTask<detail::TokensLogitsTraits>;
using RecurrentEmbeddingsToLogits = detail::RecurrentTask<detail::EmbeddingsLogitsTraits>;
using RecurrentTokensToHiddenStates = detail::RecurrentTask<detail::TokensHiddenTraits>;
using RecurrentEmbeddingsToHiddenStates = detail::RecurrentTask<detail::EmbeddingsHiddenTraits>;
template <class Request>
struct RecurrentBatchItem {
    RecurrentState* state;
    Request request;
    Config config{};
};
namespace detail {
template <class Table, class View, class ItemView>
class RecurrentBatchResult {
  public:
    RecurrentBatchResult(ResultOwner owner, const Table* table)
        : owner_(std::move(owner)), table_(table) {}
    RecurrentBatchResult(const RecurrentBatchResult&) = delete;
    RecurrentBatchResult& operator=(const RecurrentBatchResult&) = delete;
    RecurrentBatchResult(RecurrentBatchResult&&) noexcept = default;
    RecurrentBatchResult& operator=(RecurrentBatchResult&&) noexcept = default;
    std::uint64_t size() const {
        std::uint64_t count = 0;
        trtmc_error* error = nullptr;
        const auto status = table_->result_count(owner_.get(), &count, &error);
        check(owner_.api(), status, error);
        return count;
    }
    // Borrowed item view; this result owns every row's data.
    ItemView at(std::uint64_t index) const {
        View view{};
        trtmc_error* error = nullptr;
        const auto status = table_->result_item(owner_.get(), index, &view, &error);
        check(owner_.api(), status, error);
        return ItemView{view};
    }

  private:
    ResultOwner owner_;
    const Table* table_;
};
template <class Traits>
class RecurrentBatchTask {
  public:
    static constexpr std::string_view kTask = Traits::id;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Table = typename Traits::Table;
    using WireItem = typename Traits::WireItem;
    using Result = RecurrentBatchResult<Table, typename Traits::View, typename Traits::ItemView>;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible recurrent batch Task");
        validate_recurrent_state_api(reinterpret_cast<const Table*>(table)->state_api);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    RecurrentState create_state() const {
        auto owner = std::make_shared<RecurrentStateOwner>(model_, api_->state_api);
        trtmc_error* error = nullptr;
        const auto status = api_->create_state(model_->handle, &owner->handle, &error);
        check(model_->api, status, error);
        return RecurrentState(std::move(owner));
    }
    Result forward(const std::vector<trtmc::RecurrentBatchItem<Request>>& items) const {
        std::vector<Config::CEntries> configs;
        configs.reserve(items.size());
        std::vector<WireItem> wire;
        wire.reserve(items.size());
        for (const auto& item : items) {
            if (!item.state)
                throw Error(TRTMC_INVALID_ARGUMENT, "batch item state is null");
            recurrent_open(item.state->owner_);
            if (item.state->owner_->model != model_)
                throw Error(TRTMC_INVALID_ARGUMENT, "batch state belongs to another loaded model");
            configs.push_back(item.config.c_entries());
            wire.push_back(
                {item.state->owner_->handle, item.request.c_view(), configs.back().view()});
        }
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->forward(wire.data(), wire.size(), &raw, &error);
        ResultOwner owner(model_, raw);
        check(model_->api, status, error);
        return Result(std::move(owner), api_);
    }

  private:
    friend class trtmc::Model;
    RecurrentBatchTask(std::shared_ptr<ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)), api_(reinterpret_cast<const Table*>(table)) {}
    std::shared_ptr<ModelState> model_;
    const Table* api_;
};
#define TRTMC_RECURRENT_BATCH_TRAITS(Name, Id, RequestType, TableType, WireType, ViewType,         \
                                     ItemType)                                                     \
    struct Name {                                                                                  \
        static constexpr std::string_view id = Id;                                                 \
        using Request = RequestType;                                                               \
        using Table = TableType;                                                                   \
        using WireItem = WireType;                                                                 \
        using View = ViewType;                                                                     \
        using ItemView = ItemType;                                                                 \
    };
TRTMC_RECURRENT_BATCH_TRAITS(BatchTokensLogitsTraits, TRTMC_TASK_BATCH_RECURRENT_TOKENS_TO_LOGITS,
                             RecurrentTokensLogitsRequest,
                             trtmc_batch_recurrent_tokens_to_logits_api_v1,
                             trtmc_batch_recurrent_tokens_to_logits_item_v1,
                             trtmc_recurrent_logits_view_v1, RecurrentLogitsView)
TRTMC_RECURRENT_BATCH_TRAITS(BatchEmbeddingsLogitsTraits,
                             TRTMC_TASK_BATCH_RECURRENT_EMBEDDINGS_TO_LOGITS,
                             RecurrentEmbeddingsLogitsRequest,
                             trtmc_batch_recurrent_embeddings_to_logits_api_v1,
                             trtmc_batch_recurrent_embeddings_to_logits_item_v1,
                             trtmc_recurrent_logits_view_v1, RecurrentLogitsView)
TRTMC_RECURRENT_BATCH_TRAITS(BatchTokensHiddenTraits,
                             TRTMC_TASK_BATCH_RECURRENT_TOKENS_TO_HIDDEN_STATES,
                             RecurrentTokensHiddenRequest,
                             trtmc_batch_recurrent_tokens_to_hidden_states_api_v1,
                             trtmc_batch_recurrent_tokens_to_hidden_states_item_v1,
                             trtmc_recurrent_hidden_view_v1, RecurrentHiddenView)
TRTMC_RECURRENT_BATCH_TRAITS(BatchEmbeddingsHiddenTraits,
                             TRTMC_TASK_BATCH_RECURRENT_EMBEDDINGS_TO_HIDDEN_STATES,
                             RecurrentEmbeddingsHiddenRequest,
                             trtmc_batch_recurrent_embeddings_to_hidden_states_api_v1,
                             trtmc_batch_recurrent_embeddings_to_hidden_states_item_v1,
                             trtmc_recurrent_hidden_view_v1, RecurrentHiddenView)
#undef TRTMC_RECURRENT_BATCH_TRAITS
template <class Api>
void validate_recurrent_exchange(const Api* api) {
    if (!api || api->header.major != 1 || api->header.minor != 0 ||
        api->header.byte_size < sizeof(*api))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible typed recurrent exchange");
}
} // namespace detail
using BatchRecurrentTokensToLogits = detail::RecurrentBatchTask<detail::BatchTokensLogitsTraits>;
using BatchRecurrentEmbeddingsToLogits =
    detail::RecurrentBatchTask<detail::BatchEmbeddingsLogitsTraits>;
using BatchRecurrentTokensToHiddenStates =
    detail::RecurrentBatchTask<detail::BatchTokensHiddenTraits>;
using BatchRecurrentEmbeddingsToHiddenStates =
    detail::RecurrentBatchTask<detail::BatchEmbeddingsHiddenTraits>;
struct Mamba1LayerStateView {
    RecurrentArrayView convolution, ssm;
    bool has_previous_state{false};
};
struct Mamba1StateView {
    std::vector<Mamba1LayerStateView> layers;
    std::optional<std::uint64_t> tokens_seen{};
};
struct Rwkv4StateView {
    RecurrentArrayView ffn_previous, attention_previous, wkv_numerator, wkv_denominator,
        wkv_running_max;
    std::optional<std::uint64_t> tokens_seen{};
};
class Mamba1StateSnapshot : public detail::ViewResult<trtmc_mamba1_state_view_v1> {
  public:
    using ViewResult::ViewResult;
    // Copies descriptors only. Keep this snapshot alive until assign returns.
    Mamba1StateView input() const {
        const auto value = view();
        Mamba1StateView out;
        if (value.has_tokens_seen)
            out.tokens_seen = value.tokens_seen;
        for (std::uint64_t i = 0; i < value.layer_count; ++i)
            out.layers.push_back({RecurrentArrayView{value.layers[i].convolution},
                                  RecurrentArrayView{value.layers[i].ssm},
                                  value.layers[i].has_previous_state != 0});
        return out;
    }
};
class Rwkv4StateSnapshot : public detail::ViewResult<trtmc_rwkv4_state_view_v1> {
  public:
    using ViewResult::ViewResult;
    Rwkv4StateView input() const {
        const auto value = view();
        return {RecurrentArrayView{value.ffn_previous},
                RecurrentArrayView{value.attention_previous},
                RecurrentArrayView{value.wkv_numerator},
                RecurrentArrayView{value.wkv_denominator},
                RecurrentArrayView{value.wkv_running_max},
                value.has_tokens_seen ? std::optional<std::uint64_t>{value.tokens_seen}
                                      : std::nullopt};
    }
};
class Mamba1StateExchange {
  public:
    void assign(const Mamba1StateView& input) const {
        detail::recurrent_open(owner_);
        std::vector<trtmc_mamba1_layer_state_v1> layers;
        for (const auto& layer : input.layers)
            layers.push_back({layer.convolution.c_view(), layer.ssm.c_view(),
                              layer.has_previous_state ? 1U : 0U});
        const trtmc_mamba1_state_view_v1 wire{layers.data(), layers.size(),
                                              input.tokens_seen ? 1U : 0U,
                                              input.tokens_seen.value_or(0)};
        trtmc_error* error = nullptr;
        const auto status = api_->assign(owner_->handle, &wire, &error);
        detail::check(owner_->model->api, status, error);
    }
    void assign(const Mamba1StateSnapshot& input) const { assign(input.input()); }
    Mamba1StateSnapshot snapshot(RecurrentMemory target = RecurrentMemory::Host) const {
        detail::recurrent_open(owner_);
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            api_->snapshot(owner_->handle, static_cast<std::uint32_t>(target), &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return Mamba1StateSnapshot(std::move(result), api_->result_view);
    }

  private:
    friend class RecurrentState;
    Mamba1StateExchange(std::shared_ptr<detail::RecurrentStateOwner> owner,
                        const trtmc_mamba1_state_exchange_api_v1* api)
        : owner_(std::move(owner)), api_(api) {}
    std::shared_ptr<detail::RecurrentStateOwner> owner_;
    const trtmc_mamba1_state_exchange_api_v1* api_;
};
class Rwkv4StateExchange {
  public:
    void assign(const Rwkv4StateView& input) const {
        detail::recurrent_open(owner_);
        const trtmc_rwkv4_state_view_v1 wire{
            input.ffn_previous.c_view(),    input.attention_previous.c_view(),
            input.wkv_numerator.c_view(),   input.wkv_denominator.c_view(),
            input.wkv_running_max.c_view(), input.tokens_seen ? 1U : 0U,
            input.tokens_seen.value_or(0)};
        trtmc_error* error = nullptr;
        const auto status = api_->assign(owner_->handle, &wire, &error);
        detail::check(owner_->model->api, status, error);
    }
    void assign(const Rwkv4StateSnapshot& input) const { assign(input.input()); }
    Rwkv4StateSnapshot snapshot(RecurrentMemory target = RecurrentMemory::Host) const {
        detail::recurrent_open(owner_);
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status =
            api_->snapshot(owner_->handle, static_cast<std::uint32_t>(target), &raw, &error);
        detail::ResultOwner result(owner_->model, raw);
        detail::check(owner_->model->api, status, error);
        return Rwkv4StateSnapshot(std::move(result), api_->result_view);
    }

  private:
    friend class RecurrentState;
    Rwkv4StateExchange(std::shared_ptr<detail::RecurrentStateOwner> owner,
                       const trtmc_rwkv4_state_exchange_api_v1* api)
        : owner_(std::move(owner)), api_(api) {}
    std::shared_ptr<detail::RecurrentStateOwner> owner_;
    const trtmc_rwkv4_state_exchange_api_v1* api_;
};
inline Mamba1StateExchange RecurrentState::mamba1_exchange() const {
    detail::recurrent_open(owner_);
    const trtmc_mamba1_state_exchange_api_v1* api = nullptr;
    trtmc_error* error = nullptr;
    const auto status = owner_->api->get_mamba1_exchange(owner_->handle, 1, 0, &api, &error);
    detail::check(owner_->model->api, status, error);
    detail::validate_recurrent_exchange(api);
    return Mamba1StateExchange(owner_, api);
}
inline Rwkv4StateExchange RecurrentState::rwkv4_exchange() const {
    detail::recurrent_open(owner_);
    const trtmc_rwkv4_state_exchange_api_v1* api = nullptr;
    trtmc_error* error = nullptr;
    const auto status = owner_->api->get_rwkv4_exchange(owner_->handle, 1, 0, &api, &error);
    detail::check(owner_->model->api, status, error);
    detail::validate_recurrent_exchange(api);
    return Rwkv4StateExchange(owner_, api);
}
} // namespace trtmc
