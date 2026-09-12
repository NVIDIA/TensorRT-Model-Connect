/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/internal/config.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trtmc::internal {
// Stateful operations reserve std::invalid_argument (including ConfigError)
// and UnsupportedTask for preflight
// rejection BEFORE touching any state. They preserve the previous state.
// Once mutation begins, throw an execution exception instead; failure then
// poisons affected states until a successful reset or typed assign. Batch
// preflight covers ALL items before mutation, with no partial-success result.
enum class RecurrentScalar : std::uint32_t { Float32 = 1, Float16 = 2, BFloat16 = 3 };
enum class RecurrentMemory : std::uint32_t { Host = 1, Cuda = 2 };
struct RecurrentBufferView {
    const void* data{nullptr};
    std::uint64_t byte_size{0};
    RecurrentScalar scalar{RecurrentScalar::Float32};
    RecurrentMemory memory{RecurrentMemory::Host};
    std::int32_t device_ordinal{-1};
};
// Named roles below define axes; every array is contiguous row-major.
struct RecurrentArrayView {
    RecurrentBufferView buffer;
    std::uint64_t rows{0}, columns{0};
};
struct RecurrentArray {
    RecurrentArrayView view;
    std::shared_ptr<void> lease; // Immutable owned snapshot, not reusable engine output.
};
struct RecurrentTokensInput {
    Span<const std::int32_t> token_ids;    // Exactly one sequence, no automatic tokens.
    Span<const std::uint8_t> input_mask{}; // Family semantics, never a shared skip-step loop.
};
struct RecurrentEmbeddingsInput {
    RecurrentArrayView embeddings; // [T,H], exactly one sequence.
    Span<const std::uint8_t> input_mask{};
};
enum class RecurrentLogitRowsKind : std::uint32_t { All = 0, Last = 1, Indices = 2 };
struct RecurrentLogitRows {
    RecurrentLogitRowsKind kind{RecurrentLogitRowsKind::All};
    std::uint64_t last_count{0};
    Span<const std::uint64_t> indices{};
};
struct RecurrentTokensLogitsRequest {
    RecurrentTokensInput input;
    RecurrentLogitRows rows{};
    RecurrentMemory output_memory{RecurrentMemory::Host};
};
struct RecurrentEmbeddingsLogitsRequest {
    RecurrentEmbeddingsInput input;
    RecurrentLogitRows rows{};
    RecurrentMemory output_memory{RecurrentMemory::Host};
};
struct RecurrentTokensHiddenRequest {
    RecurrentTokensInput input;
    RecurrentMemory output_memory{RecurrentMemory::Host};
};
struct RecurrentEmbeddingsHiddenRequest {
    RecurrentEmbeddingsInput input;
    RecurrentMemory output_memory{RecurrentMemory::Host};
};
enum class RecurrentTraceStage : std::uint32_t {
    PostBlock = 1,
    FinalNormalization = 2,
    MixerContribution = 3
};
struct RecurrentTraceItem {
    RecurrentTraceStage stage;
    std::int64_t block_index{-1}; // -1 only for final normalization.
    RecurrentArray values;        // [T,H]; not an attention-probability matrix.
};
struct RecurrentLogitsResult {
    RecurrentArray logits;                      // [K,V], unnormalized vocabulary scores.
    std::vector<std::uint64_t> token_positions; // Input-relative rows, exact requested order.
    std::string vocabulary_id;
    std::optional<std::vector<RecurrentTraceItem>> trace{};
};
struct RecurrentHiddenResult {
    RecurrentArray hidden; // [T,H], final normalized hidden values, not trained embeddings.
    std::optional<std::vector<RecurrentTraceItem>> trace{};
};
struct RecurrentStateInfo {
    bool context_valid{true}; // Family effective-weight/config validity; no shared hash.
    bool initialized{false};
    std::optional<std::uint64_t> tokens_seen{};
    std::uint64_t device_memory_bytes{0};
};
struct Mamba1LayerStateView {
    RecurrentArrayView convolution; // [D,K], oldest to newest.
    RecurrentArrayView ssm;         // [D,N].
    bool has_previous_state{false};
};
struct Mamba1StateView {
    Span<const Mamba1LayerStateView> layers;
    std::optional<std::uint64_t> tokens_seen{};
};
struct Mamba1LayerState {
    RecurrentArray convolution, ssm;
    bool has_previous_state{false};
};
struct Mamba1StateSnapshot {
    std::vector<Mamba1LayerState> layers;
    std::optional<std::uint64_t> tokens_seen{};
};
struct Rwkv4StateView {
    RecurrentArrayView ffn_previous, attention_previous, wkv_numerator, wkv_denominator,
        wkv_running_max;
    // Every array is [channel,L] for ONE sequence, not [B,channel,L].
    std::optional<std::uint64_t> tokens_seen{};
};
struct Rwkv4StateSnapshot {
    RecurrentArray ffn_previous, attention_previous, wkv_numerator, wkv_denominator,
        wkv_running_max;
    std::optional<std::uint64_t> tokens_seen{};
};
class IMamba1StateExchange {
  public:
    virtual ~IMamba1StateExchange() = default;
    virtual void assign(const Mamba1StateView&) = 0;
    virtual Mamba1StateSnapshot snapshot(RecurrentMemory) const = 0;
};
class IRwkv4StateExchange {
  public:
    virtual ~IRwkv4StateExchange() = default;
    virtual void assign(const Rwkv4StateView&) = 0;
    virtual Rwkv4StateSnapshot snapshot(RecurrentMemory) const = 0;
};
class IRecurrentState {
  public:
    virtual ~IRecurrentState() = default;
    virtual std::unique_ptr<IRecurrentState> clone() const = 0;
    virtual void reset() = 0;
    virtual RecurrentStateInfo info() const = 0;
    virtual IMamba1StateExchange* mamba1_exchange() noexcept { return nullptr; }
    virtual IRwkv4StateExchange* rwkv4_exchange() noexcept { return nullptr; }
};
template <class Request>
struct RecurrentBatchItem {
    IRecurrentState* state; // A distinct one-sequence state for each item.
    Request request;
    ConfigView config;
};
#define TRTMC_RECURRENT_INTERFACE(Name, Id, Method, Request, Result)                               \
    class I##Name {                                                                                \
      public:                                                                                      \
        using TaskInterface = I##Name;                                                             \
        static constexpr std::string_view kTask = Id;                                              \
        virtual ~I##Name() = default;                                                              \
        virtual std::unique_ptr<IRecurrentState> create_recurrent_state() = 0;                     \
        virtual Result Method(IRecurrentState&, const Request&, ConfigView) = 0;                   \
    };
TRTMC_RECURRENT_INTERFACE(RecurrentTokensToLogits, "recurrent_tokens_to_logits",
                          forward_tokens_logits, RecurrentTokensLogitsRequest,
                          RecurrentLogitsResult)
TRTMC_RECURRENT_INTERFACE(RecurrentEmbeddingsToLogits, "recurrent_embeddings_to_logits",
                          forward_embeddings_logits, RecurrentEmbeddingsLogitsRequest,
                          RecurrentLogitsResult)
TRTMC_RECURRENT_INTERFACE(RecurrentTokensToHiddenStates, "recurrent_tokens_to_hidden_states",
                          forward_tokens_hidden, RecurrentTokensHiddenRequest,
                          RecurrentHiddenResult)
TRTMC_RECURRENT_INTERFACE(RecurrentEmbeddingsToHiddenStates,
                          "recurrent_embeddings_to_hidden_states", forward_embeddings_hidden,
                          RecurrentEmbeddingsHiddenRequest, RecurrentHiddenResult)
#undef TRTMC_RECURRENT_INTERFACE
// Preflight every item before mutation. One family-native batch invocation;
// no shared padding or serial request loop and no partial-success result.
#define TRTMC_RECURRENT_BATCH_INTERFACE(Name, Id, Method, Request, Result)                         \
    class I##Name {                                                                                \
      public:                                                                                      \
        using TaskInterface = I##Name;                                                             \
        static constexpr std::string_view kTask = Id;                                              \
        virtual ~I##Name() = default;                                                              \
        virtual std::unique_ptr<IRecurrentState> create_recurrent_state() = 0;                     \
        virtual std::vector<Result> Method(Span<const RecurrentBatchItem<Request>>) = 0;           \
    };
TRTMC_RECURRENT_BATCH_INTERFACE(BatchRecurrentTokensToLogits, "batch_recurrent_tokens_to_logits",
                                forward_batch_tokens_logits, RecurrentTokensLogitsRequest,
                                RecurrentLogitsResult)
TRTMC_RECURRENT_BATCH_INTERFACE(BatchRecurrentEmbeddingsToLogits,
                                "batch_recurrent_embeddings_to_logits",
                                forward_batch_embeddings_logits, RecurrentEmbeddingsLogitsRequest,
                                RecurrentLogitsResult)
TRTMC_RECURRENT_BATCH_INTERFACE(BatchRecurrentTokensToHiddenStates,
                                "batch_recurrent_tokens_to_hidden_states",
                                forward_batch_tokens_hidden, RecurrentTokensHiddenRequest,
                                RecurrentHiddenResult)
TRTMC_RECURRENT_BATCH_INTERFACE(BatchRecurrentEmbeddingsToHiddenStates,
                                "batch_recurrent_embeddings_to_hidden_states",
                                forward_batch_embeddings_hidden, RecurrentEmbeddingsHiddenRequest,
                                RecurrentHiddenResult)
#undef TRTMC_RECURRENT_BATCH_INTERFACE
} // namespace trtmc::internal
