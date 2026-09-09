/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/recurrent.h"
#include "trtmc/internal/stream.h"
#include "trtmc/runtime/family_factory.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>

namespace {
using namespace trtmc::internal;
using trtmc::Span;
std::atomic<int> live_models{0}, waiting{0}, batch_calls{0};
std::mutex wait_mutex;
std::condition_variable wait_cv;
bool release_waiters{false};
using AllocationHook = void (*)();
std::atomic<AllocationHook> allocation_hook{nullptr};
struct Context {
    std::uint64_t revision{0};
    std::string schema;
};
RecurrentArray owned_array(std::vector<float> values, std::uint64_t rows, std::uint64_t columns);
std::vector<float> copy_array(const RecurrentArrayView& array, std::uint64_t rows,
                              std::uint64_t columns) {
    if (array.buffer.memory != RecurrentMemory::Host ||
        array.buffer.scalar != RecurrentScalar::Float32)
        throw UnsupportedTask("fixture interchange requires host FP32");
    if (array.rows != rows || array.columns != columns)
        throw std::invalid_argument("fixture interchange shape mismatch");
    const auto* values = static_cast<const float*>(array.buffer.data);
    return {values, values + rows * columns};
}
class State final : public IRecurrentState {
    struct MambaExchange final : IMamba1StateExchange {
        explicit MambaExchange(State& state) : state(state) {}
        void assign(const Mamba1StateView& input) override { state.assign(input); }
        Mamba1StateSnapshot snapshot(RecurrentMemory target) const override {
            return state.mamba_snapshot(target);
        }
        State& state;
    };
    struct RwkvExchange final : IRwkv4StateExchange {
        explicit RwkvExchange(State& state) : state(state) {}
        void assign(const Rwkv4StateView& input) override { state.assign(input); }
        Rwkv4StateSnapshot snapshot(RecurrentMemory target) const override {
            return state.rwkv_snapshot(target);
        }
        State& state;
    };

  public:
    explicit State(std::shared_ptr<Context> context) : context(std::move(context)) { reset(); }
    std::unique_ptr<IRecurrentState> clone() const override {
        auto result = std::make_unique<State>(context);
        result->revision = revision;
        result->position = position;
        result->value = value;
        result->position_known = position_known;
        result->initialized = initialized;
        result->conv = conv;
        result->ssm = ssm;
        for (std::size_t i = 0; i < 5; ++i)
            result->rwkv[i] = rwkv[i];
        return result;
    }
    void reset() override {
        value = 0;
        position = 0;
        position_known = true;
        initialized = false;
        revision = context->revision;
        conv.assign(6, 0);
        ssm.assign(8, 0);
        for (auto& part : rwkv)
            part.assign(4, 0);
        rwkv[4].assign(4, -1.0e30F);
    }
    RecurrentStateInfo info() const override {
        return {revision == context->revision, initialized,
                position_known ? std::optional<std::uint64_t>{position} : std::nullopt, 0};
    }
    IMamba1StateExchange* mamba1_exchange() noexcept override {
        return context->schema == "mamba" ? &mamba_exchange_ : nullptr;
    }
    IRwkv4StateExchange* rwkv4_exchange() noexcept override {
        return context->schema == "rwkv" ? &rwkv_exchange_ : nullptr;
    }
    void assign(const Mamba1StateView& input) {
        if (input.layers.size() != 1)
            throw std::invalid_argument("fixture Mamba has one layer");
        auto next_conv = copy_array(input.layers[0].convolution, 2, 3);
        auto next_ssm = copy_array(input.layers[0].ssm, 2, 4);
        conv = std::move(next_conv);
        ssm = std::move(next_ssm);
        value = conv[0];
        initialized = input.layers[0].has_previous_state;
        position_known = input.tokens_seen.has_value();
        position = input.tokens_seen.value_or(0);
        revision = context->revision;
    }
    void assign(const Rwkv4StateView& input) {
        std::vector<float> next[]{
            copy_array(input.ffn_previous, 2, 2), copy_array(input.attention_previous, 2, 2),
            copy_array(input.wkv_numerator, 2, 2), copy_array(input.wkv_denominator, 2, 2),
            copy_array(input.wkv_running_max, 2, 2)};
        for (std::size_t i = 0; i < 5; ++i)
            rwkv[i] = std::move(next[i]);
        value = rwkv[0][0];
        initialized = true;
        position_known = input.tokens_seen.has_value();
        position = input.tokens_seen.value_or(0);
        revision = context->revision;
    }
    // Distinct typed return types require distinct overrides through local bases.
    Mamba1StateSnapshot mamba_snapshot(RecurrentMemory target) const {
        if (target != RecurrentMemory::Host)
            throw UnsupportedTask("fixture has no CUDA state allocator");
        return {{{owned_array(conv, 2, 3), owned_array(ssm, 2, 4), initialized}},
                position_known ? std::optional<std::uint64_t>{position} : std::nullopt};
    }
    Rwkv4StateSnapshot rwkv_snapshot(RecurrentMemory target) const {
        if (target != RecurrentMemory::Host)
            throw UnsupportedTask("fixture has no CUDA state allocator");
        return {owned_array(rwkv[0], 2, 2),
                owned_array(rwkv[1], 2, 2),
                owned_array(rwkv[2], 2, 2),
                owned_array(rwkv[3], 2, 2),
                owned_array(rwkv[4], 2, 2),
                position_known ? std::optional<std::uint64_t>{position} : std::nullopt};
    }
    void append(float token) {
        value += token;
        ++position;
        initialized = true;
        conv[0] = ssm[0] = rwkv[0][0] = value;
    }
    std::shared_ptr<Context> context;
    std::uint64_t revision{0}, position{0};
    float value{0};
    bool position_known{true}, initialized{false};
    std::vector<float> conv, ssm, rwkv[5];

  private:
    MambaExchange mamba_exchange_{*this};
    RwkvExchange rwkv_exchange_{*this};
};
RecurrentArray owned_array(std::vector<float> values, std::uint64_t rows, std::uint64_t columns) {
    auto owner = std::make_shared<std::vector<float>>(std::move(values));
    return {{{owner->data(), owner->size() * sizeof(float), RecurrentScalar::Float32,
              RecurrentMemory::Host, -1},
             rows,
             columns},
            owner};
}
class IdleStream final : public ITextStream {
  public:
    std::optional<TextStreamEvent> next(std::int64_t) override {
        if (cancelled.load())
            return TextStreamEvent{StreamEventKind::Cancelled, {}, {}, std::nullopt};
        return TextStreamEvent{StreamEventKind::Complete, {}, {}, TextResult{"done", {1}}};
    }
    void cancel() noexcept override { cancelled.store(true); }

  private:
    std::atomic<bool> cancelled{false};
};
class Model final : public IModel,
                    public IRecurrentTokensToLogits,
                    public IRecurrentEmbeddingsToLogits,
                    public IRecurrentTokensToHiddenStates,
                    public IRecurrentEmbeddingsToHiddenStates,
                    public IBatchRecurrentTokensToLogits,
                    public IBatchRecurrentEmbeddingsToLogits,
                    public IBatchRecurrentTokensToHiddenStates,
                    public IBatchRecurrentEmbeddingsToHiddenStates,
                    public ITextContinuation,
                    public IStreamingTextContinuation,
                    public trtmc::ILoraAdapterManager {
  public:
    explicit Model(std::string mode)
        : mode_(std::move(mode)), context_(std::make_shared<Context>()) {
        context_->schema = mode_;
        ++live_models;
    }
    ~Model() override { --live_models; }
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInfo> task_info() const override {
        if (mode_ == "disabled")
            return {{ITextContinuation::kTask, 1, 0}};
        return {{IRecurrentTokensToLogits::kTask, 1, 0},
                {IRecurrentEmbeddingsToLogits::kTask, 1, 0},
                {IRecurrentTokensToHiddenStates::kTask, 1, 0},
                {IRecurrentEmbeddingsToHiddenStates::kTask, 1, 0},
                {IBatchRecurrentTokensToLogits::kTask, 1, 0},
                {IBatchRecurrentEmbeddingsToLogits::kTask, 1, 0},
                {IBatchRecurrentTokensToHiddenStates::kTask, 1, 0},
                {IBatchRecurrentEmbeddingsToHiddenStates::kTask, 1, 0},
                {ITextContinuation::kTask, 1, 0},
                {IStreamingTextContinuation::kTask, 1, 0}};
    }
    std::vector<ConfigField> config_fields(std::string_view task) const override {
        if (task == ITextContinuation::kTask || task == IStreamingTextContinuation::kTask)
            return {};
        return {{"failure", ConfigKind::String, ConfigValue{std::string_view{}},
                 "Synthetic failure mode"},
                {"wait", ConfigKind::Bool, ConfigValue{false}, "Wait for test release"},
                {"output_hidden_states", ConfigKind::Bool, ConfigValue{false},
                 "Return same-evaluation trace"}};
    }
    std::unique_ptr<IRecurrentState> create_recurrent_state() override {
        return std::make_unique<State>(context_);
    }
    RecurrentLogitsResult forward_tokens_logits(IRecurrentState& input,
                                                const RecurrentTokensLogitsRequest& request,
                                                ConfigView config) override {
        return logits(execute(input, request.input, request.output_memory, config), request.rows);
    }
    RecurrentLogitsResult forward_embeddings_logits(IRecurrentState& input,
                                                    const RecurrentEmbeddingsLogitsRequest& request,
                                                    ConfigView config) override {
        return logits(execute(input, request.input, request.output_memory, config), request.rows);
    }
    RecurrentHiddenResult forward_tokens_hidden(IRecurrentState& input,
                                                const RecurrentTokensHiddenRequest& request,
                                                ConfigView config) override {
        return hidden(execute(input, request.input, request.output_memory, config));
    }
    RecurrentHiddenResult forward_embeddings_hidden(IRecurrentState& input,
                                                    const RecurrentEmbeddingsHiddenRequest& request,
                                                    ConfigView config) override {
        return hidden(execute(input, request.input, request.output_memory, config));
    }
    std::vector<RecurrentLogitsResult> forward_batch_tokens_logits(
        Span<const RecurrentBatchItem<RecurrentTokensLogitsRequest>> items) override {
        return batch<RecurrentLogitsResult>(items, [](const auto& output, const auto& request) {
            return logits(output, request.rows);
        });
    }
    std::vector<RecurrentLogitsResult> forward_batch_embeddings_logits(
        Span<const RecurrentBatchItem<RecurrentEmbeddingsLogitsRequest>> items) override {
        return batch<RecurrentLogitsResult>(items, [](const auto& output, const auto& request) {
            return logits(output, request.rows);
        });
    }
    std::vector<RecurrentHiddenResult> forward_batch_tokens_hidden(
        Span<const RecurrentBatchItem<RecurrentTokensHiddenRequest>> items) override {
        return batch<RecurrentHiddenResult>(
            items, [](const auto& output, const auto&) { return hidden(output); });
    }
    std::vector<RecurrentHiddenResult> forward_batch_embeddings_hidden(
        Span<const RecurrentBatchItem<RecurrentEmbeddingsHiddenRequest>> items) override {
        return batch<RecurrentHiddenResult>(
            items, [](const auto& output, const auto&) { return hidden(output); });
    }

  private:
    struct Evaluation {
        std::vector<float> totals;
        std::string_view fault;
        bool trace;
    };
    static std::vector<float> values(const RecurrentTokensInput& input) {
        std::vector<float> result;
        for (const auto token : input.token_ids) {
            if (token < 0 || token >= 16)
                throw std::invalid_argument("fixture token outside vocabulary");
            result.push_back(static_cast<float>(token));
        }
        return result;
    }
    static std::vector<float> values(const RecurrentEmbeddingsInput& input) {
        const auto data = copy_array(input.embeddings, input.embeddings.rows, 2);
        std::vector<float> result;
        for (std::size_t i = 0; i < data.size(); i += 2)
            result.push_back(data[i] + data[i + 1]);
        return result;
    }
    template <class Input>
    void preflight(IRecurrentState& input, const Input& request, RecurrentMemory target,
                   ConfigView config) const {
        auto* state = dynamic_cast<State*>(&input);
        if (!state || state->context != context_)
            throw std::invalid_argument("foreign family state");
        const auto options = parse(config);
        const auto fault = config_value_as<std::string_view>(options[0]);
        if (target != RecurrentMemory::Host && fault != "wrong_target")
            throw UnsupportedTask("fixture has no CUDA output allocator");
        (void)values(request);
        for (const auto value : request.input_mask)
            if (value == 0)
                throw UnsupportedTask("fixture does not implement input masking");
    }
    template <class Input>
    Evaluation execute(IRecurrentState& input, const Input& request, RecurrentMemory target,
                       ConfigView config) const {
        preflight(input, request, target, config);
        auto& state = dynamic_cast<State&>(input);
        const auto options = parse(config);
        const auto fault = config_value_as<std::string_view>(options[0]);
        std::vector<float> totals;
        for (const auto token : values(request)) {
            state.append(token);
            totals.push_back(state.value);
        }
        if (fault == "after_mutation")
            throw std::runtime_error("fixture failed after state mutation");
        if (config_value_as<bool>(options[1])) {
            std::unique_lock<std::mutex> lock(wait_mutex);
            release_waiters = false;
            ++waiting;
            wait_cv.wait(lock, [] { return release_waiters; });
            --waiting;
        }
        return {std::move(totals), fault, config_value_as<bool>(options[2])};
    }
    template <class Result, class Request, class Pack>
    std::vector<Result> batch(Span<const RecurrentBatchItem<Request>> items, Pack pack) const {
        ++batch_calls;
        for (const auto& item : items)
            preflight(*item.state, item.request.input, item.request.output_memory, item.config);
        // Synthetic family-native implementation, never a shared Task loop.
        std::vector<Result> result;
        for (const auto& item : items)
            result.push_back(pack(
                execute(*item.state, item.request.input, item.request.output_memory, item.config),
                item.request));
        if (!items.empty() &&
            config_value_as<std::string_view>(parse(items[0].config)[0]) == "bad_batch_count")
            result.pop_back();
        return result;
    }
    static std::optional<std::vector<RecurrentTraceItem>> trace(const Evaluation& output) {
        if (!output.trace)
            return std::nullopt;
        std::vector<float> hidden;
        for (const auto value : output.totals) {
            hidden.push_back(value);
            hidden.push_back(-value);
        }
        return std::vector<RecurrentTraceItem>{
            {RecurrentTraceStage::FinalNormalization, -1,
             owned_array(std::move(hidden), output.totals.size(), 2)}};
    }
    static void allocation_failure(std::string_view fault) {
        if (fault == "packing_oom") {
            const auto hook = allocation_hook.load();
            if (!hook)
                throw std::logic_error("allocation hook is missing");
            hook();
        }
    }
    static RecurrentHiddenResult hidden(const Evaluation& output) {
        std::vector<float> values;
        for (const auto value : output.totals) {
            values.push_back(value);
            values.push_back(-value);
        }
        RecurrentHiddenResult result{owned_array(std::move(values), output.totals.size(), 2),
                                     trace(output)};
        if (output.fault == "bad_hidden_shape")
            ++result.hidden.view.rows;
        if (output.fault == "no_lease")
            result.hidden.lease.reset();
        allocation_failure(output.fault);
        return result;
    }
    static RecurrentLogitsResult logits(const Evaluation& output, const RecurrentLogitRows& rows) {
        RecurrentLogitsResult result;
        result.vocabulary_id = "fixture-vocab-16";
        const auto count = output.totals.size();
        if (rows.kind == RecurrentLogitRowsKind::Indices)
            result.token_positions.assign(rows.indices.begin(), rows.indices.end());
        else {
            const auto kept = rows.kind == RecurrentLogitRowsKind::All
                                  ? count
                                  : std::min<std::uint64_t>(count, rows.last_count);
            for (std::size_t i = count - kept; i < count; ++i)
                result.token_positions.push_back(i);
        }
        std::vector<float> logits;
        for (const auto row : result.token_positions)
            for (int column = 0; column < 16; ++column)
                logits.push_back(output.totals[row] + static_cast<float>(column));
        result.logits = owned_array(std::move(logits), result.token_positions.size(), 16);
        result.trace = trace(output);
        if (output.fault == "bad_positions" && !result.token_positions.empty())
            ++result.token_positions[0];
        if (output.fault == "no_lease")
            result.logits.lease.reset();
        allocation_failure(output.fault);
        return result;
    }

  public:
    TextResult run(const TextContinuationRequest&, ConfigView config) override {
        if (!config.empty())
            throw ConfigError("sync fixture takes no config");
        return {"sync", {1}};
    }
    std::unique_ptr<ITextStream> start(const TextContinuationRequest&, ConfigView config) override {
        if (!config.empty())
            throw ConfigError("idle stream takes no config");
        return std::make_unique<IdleStream>();
    }
    trtmc::ILoraAdapterManager* lora_adapters() noexcept override { return this; }
    void load_lora_adapter(const std::string& id, const std::string&) override {
        adapters_.push_back(id);
        ++context_->revision;
    }
    void unload_lora_adapter(const std::string&) override {
        adapters_.clear();
        ++context_->revision;
    }
    std::vector<std::string> loaded_lora_adapters() const override { return adapters_; }

  private:
    std::vector<ConfigValue> parse(ConfigView config) const {
        const auto fields = config_fields(IRecurrentTokensToLogits::kTask);
        std::vector<ConfigValue> values;
        for (const auto& field : fields)
            values.push_back(*field.default_value);
        std::vector<bool> seen(fields.size(), false);
        for (const auto& entry : config) {
            auto found = std::find_if(fields.begin(), fields.end(),
                                      [&](const auto& field) { return field.name == entry.name; });
            if (found == fields.end())
                throw ConfigError("unknown recurrent fixture option");
            const auto index = static_cast<std::size_t>(found - fields.begin());
            if (seen[index] || config_kind(entry.value) != found->kind)
                throw ConfigError("duplicate or mistyped recurrent option");
            seen[index] = true;
            values[index] = entry.value;
        }
        return values;
    }
    std::string mode_;
    std::shared_ptr<Context> context_;
    std::vector<std::string> adapters_;
};
} // namespace
extern "C" int trtmc_test_recurrent_live_models() {
    return live_models.load();
}
extern "C" int trtmc_test_recurrent_waiting() {
    return waiting.load();
}
extern "C" int trtmc_test_recurrent_batch_calls() {
    return batch_calls.load();
}
extern "C" void trtmc_test_recurrent_release_waiters() {
    const std::lock_guard<std::mutex> lock(wait_mutex);
    release_waiters = true;
    wait_cv.notify_all();
}
extern "C" void trtmc_test_recurrent_allocation_hook(AllocationHook hook) {
    allocation_hook.store(hook);
}
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new Model(context.reader.info().task);
}
