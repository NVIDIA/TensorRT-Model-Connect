/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/control.hpp"
#include "trtmc/recurrent.hpp"
#include "trtmc/stream.hpp"

#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <new>
#include <thread>

namespace {
thread_local bool fail_next_allocation = false;
void arm_failure() {
    fail_next_allocation = true;
}
} // namespace
void* operator new(std::size_t size) {
    if (std::exchange(fail_next_allocation, false))
        throw std::bad_alloc();
    if (void* result = std::malloc(size ? size : 1))
        return result;
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept {
    std::free(value);
}
void operator delete(void* value, std::size_t) noexcept {
    std::free(value);
}
namespace {
int failures = 0;
void check(bool value, const char* name) {
    if (!value) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}
template <class F>
bool fails(trtmc_status code, F fn) {
    try {
        fn();
    } catch (const trtmc::Error& error) {
        return error.code() == code;
    }
    return false;
}
void bundle(const std::filesystem::path& path, const std::string& mode = "enabled") {
    const char magic[]{'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header = R"({"format":1,"family":"recurrent_fixture","task":")" + mode +
                               R"(","backend":"fake","sections":{}})";
    std::ofstream out(path, std::ios::binary);
    out.write(magic, 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((header.size() >> shift) & 255));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
}
void extended(trtmc::Model& model, const std::filesystem::path& root,
              const trtmc::LoadOptions& options) {
    const float data[]{1, 0, 2, 0};
    const auto embeddings = trtmc::RecurrentArrayView::host_f32({data, 4}, 2, 2);
    auto tokens_logits = model.task<trtmc::RecurrentTokensToLogits>();
    auto embeddings_logits = model.task<trtmc::RecurrentEmbeddingsToLogits>();
    auto tokens_hidden = model.task<trtmc::RecurrentTokensToHiddenStates>();
    auto embeddings_hidden = model.task<trtmc::RecurrentEmbeddingsToHiddenStates>();
    auto state = tokens_logits.create_state();
    auto logits = embeddings_logits.forward(state, {{embeddings}});
    check(logits.logits().host_floats()[16] == 3,
          "embeddings are typed input rather than token or config reinterpretation");
    state.reset();
    auto hidden = tokens_hidden.forward(state, {{{1, 2}}}, {{"output_hidden_states", true}});
    check(hidden.hidden().rows() == 2 && hidden.hidden().columns() == 2 &&
              hidden.hidden().host_floats()[2] == 3 && hidden.hidden().host_floats()[3] == -3 &&
              hidden.view().trace.has_trace,
          "tokens-to-hidden has distinct typed normalized output and same-forward trace");
    state.reset();
    check(embeddings_hidden.forward(state, {{embeddings}}).hidden().host_floats()[2] == 3,
          "embeddings-to-hidden preserves complete supplied sequence");
    auto wrong = embeddings.c_view();
    wrong.buffer.byte_size = 1;
    check(fails(TRTMC_INVALID_ARGUMENT,
                [&] {
                    (void)embeddings_hidden.forward(state, {{trtmc::RecurrentArrayView{wrong}}});
                }) &&
              state.info().valid() && state.info().tokens_seen == 2,
          "typed embedding byte/shape error occurs before family mutation");
    for (bool unsupported : {false, true}) {
        check(fails(unsupported ? TRTMC_UNSUPPORTED : TRTMC_INVALID_CONFIG,
                    [&] {
                        if (unsupported)
                            (void)tokens_logits.forward(state,
                                                        {{{1}}, {}, trtmc::RecurrentMemory::Cuda});
                        else
                            (void)tokens_logits.forward(state, {{{1}}}, {{"unknown", true}});
                    }) &&
                  state.info().valid() && state.info().tokens_seen == 2,
              "explicit family preflight rejection preserves prefix state");
    }
    check(tokens_logits.forward(state, {{{4}}}).logits().host_floats()[0] == 7,
          "same prefix can continue after family preflight rejection");
    check(fails(TRTMC_INVALID_ARGUMENT, [&] { (void)tokens_logits.forward(state, {{{99}}}); }) &&
              state.info().valid() && state.info().tokens_seen == 3,
          "family token-vocabulary preflight rejection preserves prefix");
    check(fails(TRTMC_INTERNAL_ERROR,
                [&] {
                    (void)tokens_logits.forward(state, {{{1}}, {}, trtmc::RecurrentMemory::Cuda},
                                                {{"failure", "wrong_target"}});
                }) &&
              state.info().poisoned,
          "family returning host for explicit device target fails rather than silently copying");
    state.reset();
    check(fails(TRTMC_INTERNAL_ERROR,
                [&] {
                    (void)tokens_hidden.forward(state, {{{1}}}, {{"failure", "bad_hidden_shape"}});
                }) &&
              state.info().poisoned,
          "invalid hidden output poisons mutated state");

    auto bt = model.task<trtmc::BatchRecurrentTokensToLogits>();
    auto be = model.task<trtmc::BatchRecurrentEmbeddingsToLogits>();
    auto bth = model.task<trtmc::BatchRecurrentTokensToHiddenStates>();
    auto beh = model.task<trtmc::BatchRecurrentEmbeddingsToHiddenStates>();
    auto a = bt.create_state(), b = be.create_state();
    void* fixture =
        dlopen((root / "libtrtmc_model_recurrent_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!fixture)
        throw std::runtime_error("batch fixture unavailable");
    using CountCalls = int (*)();
    const auto count_calls =
        reinterpret_cast<CountCalls>(dlsym(fixture, "trtmc_test_recurrent_batch_calls"));
    if (!count_calls)
        throw std::runtime_error("batch call probe unavailable");
    const auto before_batch = count_calls();
    auto batch = bt.forward({{&a, {{{1, 2}}}}, {&b, {{{4}}}}});
    check(count_calls() == before_batch + 1,
          "one C invocation calls one family-native batch virtual");
    dlclose(fixture);
    check(batch.size() == 2 && batch.at(0).logits().rows() == 2 &&
              batch.at(0).logits().host_floats()[16] == 3 &&
              batch.at(1).logits().host_floats()[0] == 4 && a.info().tokens_seen == 2 &&
              b.info().tokens_seen == 1,
          "native batch owns distinct sequence states and ordered ragged outputs");
    check(fails(TRTMC_INVALID_ARGUMENT, [&] { (void)batch.at(2); }), "batch item bounds checked");
    check(
        fails(TRTMC_INVALID_ARGUMENT, [&] { (void)bt.forward({{&a, {{{1}}}}, {&a, {{{2}}}}}); }) &&
            a.info().valid() && a.info().tokens_seen == 2,
        "batch state alias rejected before mutation");
    check(fails(TRTMC_INVALID_CONFIG,
                [&] { (void)bt.forward({{&a, {{{1}}}}, {&b, {{{2}}}, {{"unknown", true}}}}); }) &&
              a.info().valid() && b.info().valid() && a.info().tokens_seen == 2 &&
              b.info().tokens_seen == 1,
          "all-item family preflight rejection preserves both prefixes");
    check(fails(TRTMC_UNSUPPORTED,
                [&] {
                    (void)bt.forward(
                        {{&a, {{{1}}}}, {&b, {{{2}}, {}, trtmc::RecurrentMemory::Cuda}}});
                }) &&
              a.info().valid() && b.info().valid() && a.info().tokens_seen == 2,
          "unsupported batch composition does not poison otherwise valid states");
    auto continued = bt.forward({{&a, {{{4}}}}, {&b, {{{3}}}}});
    check(continued.at(0).logits().host_floats()[0] == 7 &&
              continued.at(1).logits().host_floats()[0] == 7,
          "native batch continues original prefixes after rejected request");
    check(
        fails(TRTMC_INVALID_ARGUMENT, [&] { (void)bt.forward({{&a, {{{1}}}}, {&b, {{{99}}}}}); }) &&
            a.info().valid() && b.info().valid() && a.info().tokens_seen == 3 &&
            b.info().tokens_seen == 2,
        "family validates every batch token input before changing any state");
    for (const char* fault : {"after_mutation", "bad_batch_count", "no_lease"}) {
        check(fails(TRTMC_INTERNAL_ERROR,
                    [&] {
                        (void)bt.forward({{&a, {{{1}}}, {{"failure", fault}}}, {&b, {{{2}}}}});
                    }) &&
                  a.info().poisoned && b.info().poisoned,
              "execution or packing failure poisons every submitted batch state");
        a.reset();
        b.reset();
    }
    auto eb = be.forward(
        {{&a, {{embeddings}}}, {&b, {{embeddings}, trtmc::RecurrentLogitRows::last(1)}}});
    check(eb.at(0).logits().rows() == 2 && eb.at(1).logits().rows() == 1 &&
              eb.at(1).token_positions()[0] == 1,
          "native embeddings batch preserves per-item row policy");
    a.reset();
    b.reset();
    auto th = bth.forward({{&a, {{{1}}}}, {&b, {{{2, 3}}}}});
    check(th.at(1).hidden().rows() == 2 && th.at(1).hidden().host_floats()[2] == 5,
          "native token-hidden batch carries per-item target axes");
    a.reset();
    b.reset();
    auto eh = beh.forward({{&a, {{embeddings}}}, {&b, {{embeddings}}}});
    check(eh.at(0).hidden().host_floats()[2] == 3 && eh.at(1).hidden().host_floats()[3] == -3,
          "native embeddings-hidden batch invokes its distinct Task");
    a.close();
    b.close();
    check(batch.at(0).logits().host_floats()[16] == 3, "owned batch results survive states");

    for (const std::string schema : {"mamba", "rwkv"}) {
        const auto path = root / ("recurrent-cpp-" + schema + ".bundle");
        bundle(path, schema);
        auto owner = trtmc::Model::load(path.string(), options);
        auto forward = owner.task<trtmc::RecurrentTokensToLogits>();
        auto source = forward.create_state();
        (void)forward.forward(source, {{{1, 2}}});
        auto destination = forward.create_state();
        if (schema == "mamba") {
            check(source.supports_mamba1_exchange() && !source.supports_rwkv4_exchange(),
                  "Mamba interchange is one explicit family getter");
            auto exchange = destination.mamba1_exchange();
            auto saved = source.mamba1_exchange().snapshot();
            exchange.assign(saved);
            check(saved.view().layers[0].convolution.rows == 2 &&
                      saved.view().layers[0].convolution.columns == 3 &&
                      saved.view().layers[0].ssm.columns == 4 &&
                      forward.forward(destination, {{{4}}}).logits().host_floats()[0] == 7,
                  "Mamba typed layer axes and assign replay preserve prefix");
            auto input = saved.input();
            auto bad = input.layers[0].convolution.c_view();
            bad.buffer.scalar = TRTMC_RECURRENT_FLOAT16;
            bad.buffer.byte_size /= 2;
            input.layers[0].convolution = trtmc::RecurrentArrayView{bad};
            check(fails(TRTMC_UNSUPPORTED, [&] { exchange.assign(input); }) &&
                      destination.info().valid() && destination.info().tokens_seen == 3,
                  "valid state survives unsupported family dtype assign");
            check(fails(TRTMC_INTERNAL_ERROR,
                        [&] {
                            (void)forward.forward(destination, {{{1}}},
                                                  {{"failure", "after_mutation"}});
                        }),
                  "Mamba state mutation fault is observable");
            check(fails(TRTMC_UNSUPPORTED, [&] { exchange.assign(input); }) &&
                      destination.info().poisoned,
                  "rejected typed assign cannot unpoison a previously poisoned state");
            exchange.assign(saved);
            check(destination.info().valid() && destination.info().tokens_seen == 2,
                  "successful typed assign recovers poison without reset");
            auto unknown = saved.input();
            unknown.tokens_seen.reset();
            exchange.assign(unknown);
            (void)forward.forward(destination, {{{1}}});
            check(!destination.info().tokens_seen, "imported unknown position stays unknown");
            check(fails(TRTMC_UNSUPPORTED,
                        [&] { (void)exchange.snapshot(trtmc::RecurrentMemory::Cuda); }) &&
                      destination.info().valid(),
                  "snapshot refuses unsupported target without poison");
            auto copy = source.clone();
            (void)forward.forward(copy, {{{5}}});
            check(saved.input().layers[0].convolution.host_floats()[0] == 3 &&
                      source.mamba1_exchange()
                              .snapshot()
                              .input()
                              .layers[0]
                              .convolution.host_floats()[0] == 3,
                  "cloned Mamba arrays and saved snapshots do not alias mutable state");
            std::vector<float> custom_conv(6, 11), custom_ssm(8, 22);
            exchange.assign(trtmc::Mamba1StateView{
                {{trtmc::RecurrentArrayView::host_f32({custom_conv.data(), custom_conv.size()}, 2,
                                                      3),
                  trtmc::RecurrentArrayView::host_f32({custom_ssm.data(), custom_ssm.size()}, 2, 4),
                  true}},
                20});
            custom_conv[0] = 99;
            custom_ssm[0] = 88;
            auto custom = exchange.snapshot();
            check(custom.input().layers[0].convolution.host_floats()[0] == 11 &&
                      custom.input().layers[0].ssm.host_floats()[0] == 22 &&
                      custom.view().tokens_seen == 20 &&
                      forward.forward(destination, {{{1}}}).logits().host_floats()[0] == 12,
                  "custom Mamba fields are copied before return and affect continuation");
            destination.close();
            check(fails(TRTMC_INVALID_ARGUMENT, [&] { exchange.assign(saved); }),
                  "explicit state close invalidates retained exchange proxy");
        } else {
            check(source.supports_rwkv4_exchange() && !source.supports_mamba1_exchange(),
                  "RWKV interchange is independent from Mamba schema");
            auto exchange = destination.rwkv4_exchange();
            auto saved = source.rwkv4_exchange().snapshot();
            check(saved.input().wkv_running_max.host_floats()[0] == -1.0e30F &&
                      saved.view().ffn_previous.rows == 2 && saved.view().ffn_previous.columns == 2,
                  "RWKV named arrays preserve channel/layer axes and family initialization");
            exchange.assign(saved);
            check(forward.forward(destination, {{{4}}}).logits().host_floats()[0] == 7,
                  "RWKV typed assign continues saved state");
            auto bad = saved.input();
            auto buffer = bad.ffn_previous.c_view();
            buffer.buffer.scalar = TRTMC_RECURRENT_BFLOAT16;
            buffer.buffer.byte_size /= 2;
            bad.ffn_previous = trtmc::RecurrentArrayView{buffer};
            check(fails(TRTMC_UNSUPPORTED, [&] { exchange.assign(bad); }) &&
                      destination.info().valid(),
                  "RWKV rejected assign preserves valid prefix");
            check(fails(TRTMC_INTERNAL_ERROR,
                        [&] {
                            (void)forward.forward(destination, {{{1}}},
                                                  {{"failure", "after_mutation"}});
                        }) &&
                      destination.info().poisoned,
                  "RWKV delegated mutation poisons state");
            check(fails(TRTMC_UNSUPPORTED, [&] { exchange.assign(bad); }) &&
                      destination.info().poisoned,
                  "RWKV rejected assign preserves poison");
            exchange.assign(saved);
            auto copy = destination.clone();
            (void)forward.forward(copy, {{{5}}});
            check(destination.info().valid() && saved.input().ffn_previous.host_floats()[0] == 3 &&
                      exchange.snapshot().input().ffn_previous.host_floats()[0] == 3,
                  "RWKV successful assign recovers and clone/snapshot are immutable");
            std::vector<float> values[5];
            for (std::size_t i = 0; i < 5; ++i)
                values[i].assign(4, static_cast<float>(10 + i));
            auto array = [&](std::size_t i) {
                return trtmc::RecurrentArrayView::host_f32({values[i].data(), values[i].size()}, 2,
                                                           2);
            };
            exchange.assign(
                trtmc::Rwkv4StateView{array(0), array(1), array(2), array(3), array(4), 30});
            values[0][0] = 99;
            const auto custom = exchange.snapshot();
            check(custom.input().ffn_previous.host_floats()[0] == 10 &&
                      custom.input().attention_previous.host_floats()[0] == 11 &&
                      custom.input().wkv_numerator.host_floats()[0] == 12 &&
                      custom.input().wkv_denominator.host_floats()[0] == 13 &&
                      custom.input().wkv_running_max.host_floats()[0] == 14 &&
                      custom.view().tokens_seen == 30 &&
                      forward.forward(destination, {{{1}}}).logits().host_floats()[0] == 11,
                  "five named RWKV inputs stay distinct and copied, not a hidden tensor bag");
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    const std::filesystem::path root(argv[1]), path = root / "recurrent-cpp.bundle";
    bundle(path);
    try {
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        auto model = trtmc::Model::load(path.string(), options);
        auto task = model.task<trtmc::RecurrentTokensToLogits>();
        extended(model, root, options);
        auto state = task.create_state();
        auto idle = task.create_state();
        check(state.info().valid() && !state.info().initialized && state.info().tokens_seen == 0,
              "family creates fresh state without a Config scope");
        check(model.task<trtmc::TextContinuation>().run({"hi"}).text() == "sync",
              "multiple idle states do not reserve model execution");
        auto first = task.forward(state, {{{1, 2}}});
        check(first.logits().rows() == 2 && first.logits().columns() == 16 &&
                  first.logits().host_floats()[0] == 1 && first.logits().host_floats()[16] == 3 &&
                  state.info().tokens_seen == 2,
              "forward returns every supplied token row and mutates receiver state");
        auto branch = state.clone();
        auto suffix = task.forward(state, {{{4}}});
        auto other = task.forward(branch, {{{7}}});
        check(suffix.logits().host_floats()[0] == 7 && other.logits().host_floats()[0] == 10 &&
                  first.logits().host_floats()[16] == 3,
              "clone branches independently and output snapshots remain owned");
        auto selected =
            task.forward(state, {{{1, 2, 3}}, trtmc::RecurrentLogitRows::select({2, 0, 2})},
                         {{"output_hidden_states", true}});
        check(selected.token_positions().size() == 3 && selected.token_positions()[0] == 2 &&
                  selected.logits().host_floats()[0] == 13 &&
                  selected.logits().host_floats()[16] == 8 && selected.view().trace.has_trace &&
                  selected.view().trace.items[0].values.rows == 3,
              "row selector preserves repeats/order and trace comes from same forward");
        check(fails(TRTMC_INVALID_ARGUMENT,
                    [&] {
                        (void)task.forward(
                            state, {{{1}}, {trtmc::RecurrentLogitRowsKind::Indices, 0, {5}}});
                    }) &&
                  state.info().valid(),
              "pre-call row error leaves state usable");
        auto active = model.task<trtmc::StreamingTextContinuation>().start({"hold"});
        check(fails(TRTMC_BUSY, [&] { (void)task.forward(state, {{{1}}}); }) &&
                  fails(TRTMC_BUSY, [&] { (void)state.clone(); }),
              "active session excludes recurrent work");
        active.close();
        check(state.info().valid(), "active-session BUSY does not poison idle state");
        auto second_model = trtmc::Model::load(path.string(), options);
        check(fails(TRTMC_INVALID_ARGUMENT,
                    [&] {
                        (void)second_model.task<trtmc::RecurrentTokensToLogits>().forward(state,
                                                                                          {{{1}}});
                    }),
              "C++ task rejects state from another loaded model");
        check(!state.supports_mamba1_exchange() && !state.supports_rwkv4_exchange(),
              "unimplemented state interchange is not guessed from family name");
        void* library =
            dlopen((root / "libtrtmc_model_recurrent_fixture.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error("fixture library unavailable");
        using Probe = int (*)();
        using Release = void (*)();
        using Hook = void (*)(void (*)());
        const auto waiting =
            reinterpret_cast<Probe>(dlsym(library, "trtmc_test_recurrent_waiting"));
        const auto release =
            reinterpret_cast<Release>(dlsym(library, "trtmc_test_recurrent_release_waiters"));
        const auto set_hook =
            reinterpret_cast<Hook>(dlsym(library, "trtmc_test_recurrent_allocation_hook"));
        if (!waiting || !release || !set_hook)
            throw std::runtime_error("missing recurrent probes");
        auto pending = std::async(std::launch::async,
                                  [&] { return task.forward(state, {{{1}}}, {{"wait", true}}); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!waiting() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        check(waiting() != 0 && fails(TRTMC_BUSY, [&] { (void)task.forward(idle, {{{1}}}); }),
              "active forward excludes another state's operation");
        release();
        (void)pending.get();
        for (const std::string fault :
             {"after_mutation", "bad_positions", "no_lease", "packing_oom"}) {
            state.reset();
            if (fault == "packing_oom")
                set_hook(arm_failure);
            check(fails(fault == "packing_oom" ? TRTMC_OUT_OF_MEMORY : TRTMC_INTERNAL_ERROR,
                        [&] { (void)task.forward(state, {{{1, 2}}}, {{"failure", fault}}); }),
                  "delegated or packing failure is reported");
            check(state.info().poisoned &&
                      fails(TRTMC_INVALID_ARGUMENT, [&] { (void)state.clone(); }),
                  "failed mutation is poisoned rather than silently retried");
            state.reset();
            check(state.info().valid() &&
                      task.forward(state, {{{2}}}).logits().host_floats()[0] == 2,
                  "family reset recovers poisoned state");
        }
        set_hook(nullptr);
        dlclose(library);
        model.lora_adapters().load("changed", "unused");
        check(!state.info().context_valid && !state.info().poisoned &&
                  fails(TRTMC_INVALID_ARGUMENT, [&] { (void)task.forward(state, {{{1}}}); }),
              "effective-weight context invalidation is family-owned, without a hash");
        state.reset();
        check(state.info().valid(), "reset binds current family context");
        auto outcome = task.forward({{{3}}});
        check(outcome.output.logits().host_floats()[0] == 3 &&
                  outcome.state.info().tokens_seen == 1,
              "first-forward convenience creates state through the existing C API");
        state.close();
        check(first.logits().host_floats()[0] == 1, "result survives state release");
    } catch (const std::exception& error) {
        std::cerr << "Unexpected: " << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
