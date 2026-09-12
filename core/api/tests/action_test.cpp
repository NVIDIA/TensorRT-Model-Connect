/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/action.hpp"

#include <chrono>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
template <class Function>
void rejects(Function function, trtmc_status expected, const char* message) {
    bool matched = false;
    try {
        function();
    } catch (const trtmc::Error& error) {
        matched = error.code() == expected;
    }
    check(matched, message);
}
void bundle(const std::filesystem::path& path, const std::string& mode) {
    const std::string header =
        "{\"format\":1,\"family\":\"action_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.put(static_cast<char>((static_cast<uint64_t>(header.size()) >> shift) & 255U));
    out.write(header.data(), header.size());
    out.write("PLAN", 4);
}
trtmc::Model load(const std::filesystem::path& root, const std::string& mode) {
    auto path = root / ("action_cpp_" + mode + ".bundle");
    bundle(path, mode);
    trtmc::LoadOptions options;
    options.runtime_root = root.string();
    return trtmc::Model::load(path.string(), options);
}
void exercise_released_model(const std::filesystem::path& root) {
    using namespace trtmc;
    const auto path = root / "action_released_model.bundle";
    bundle(path, "all");
    const auto runtime_root = root.string();
    const auto& core = detail::core_api();
    const trtmc_load_options_v1 options{sizeof(options), detail::c_string(runtime_root), 0, {}, 0};
    trtmc_error* error = nullptr;
    auto checked = [&](trtmc_status status) {
        auto* current = std::exchange(error, nullptr);
        detail::check(core, status, current);
    };
    trtmc_model* raw_model = nullptr;
    checked(core.model_load(detail::c_string(path.string()), &options, &raw_model, &error));
    std::unique_ptr<trtmc_model, decltype(core.model_release)> model(raw_model, core.model_release);
    const trtmc_api_header* table = nullptr;
    checked(core.model_get_task_api(raw_model, detail::c_string(ImageStateActionQueue::kTask), 1, 0,
                                    &table, &error));
    ImageStateActionQueue::validate_table(table);
    const auto* queue = reinterpret_cast<const trtmc_image_state_action_queue_api_v1*>(table);
    trtmc_image_state_action_session* raw_session = nullptr;
    checked(queue->create(raw_model, nullptr, &raw_session, &error));
    std::unique_ptr<trtmc_image_state_action_session, decltype(queue->release)> session(
        raw_session, queue->release);
    model.reset(); // The continuation must never access this released public handle.

    const float pixels[]{0.5F, 0.25F, 0.125F}, state[]{2, 4};
    const ImageStateObservation observation{ImageInput({pixels, 3}, 1, 1), {state, 2}};
    const auto input = detail::action_observation(observation);
    const Config invalid{{"tag", 1}};
    const auto entries = invalid.c_entries();
    const auto config = entries.view();
    trtmc_result* output = nullptr;
    const auto status = queue->act(raw_session, &input, &config, &output, &error);
    check(status == TRTMC_INVALID_CONFIG && output == nullptr,
          "session validates Config through its owner after public model release");
    core.error_release(std::exchange(error, nullptr));
    core.result_release(std::exchange(output, nullptr));
    checked(queue->act(raw_session, &input, nullptr, &output, &error));
    std::unique_ptr<trtmc_result, decltype(core.result_release)> result(output,
                                                                        core.result_release);
    trtmc_action_step_view_v1 view{};
    checked(queue->result_view(output, &view, &error));
    check(view.inference_ms == 1,
          "rejected Config does not consume the released-model session's first chunk");
}
void exercise(const std::filesystem::path& root) {
    using namespace trtmc;
    auto model = load(root, "all");
    check(model.tasks().size() == 2,
          "stateless chunk and queue factory are distinct advertised Tasks");
    const float pixels[] = {0.5F, 0.25F, 0.125F};
    float state[] = {2, 4};
    const ImageInput image({pixels, 3}, 1, 1);
    const ImageStateObservation observation{image, {state, 2}};
    const auto predictor = model.task<ImageStateToActionChunk>();
    for (const Config& invalid :
         std::vector<Config>{{{"unknown", 1}}, {{"tag", 1}}, {{"tag", "a"}, {"tag", "b"}}})
        rejects([&] { (void)predictor.run({observation}, invalid); }, TRTMC_INVALID_CONFIG,
                "action prediction rejects invalid Config before consuming an inference");
    auto first_chunk = predictor.run({observation});
    check(first_chunk.actions().rows == 2 && first_chunk.actions().columns == 2 &&
              first_chunk.actions().values[0] == 2.5F && first_chunk.actions().values[1] == -3 &&
              first_chunk.actions().values[3] == 8 && !first_chunk.within_training_bounds() &&
              first_chunk.inference_ms() == 101,
          "stateless prediction retains ordered unnormalized/out-of-range actions, flag and "
          "measured timing");
    const auto schema = first_chunk.view().actions.schema;
    check(trtmc::detail::string_view(schema.domain) == "fixture.default" &&
              schema.component_names.size == 2 &&
              trtmc::detail::string_view(schema.component_names.data[1]) == "right" &&
              schema.units.size == 0 &&
              trtmc::detail::string_view(schema.normalization) == "unnormalized" &&
              first_chunk.view().actions.timestamps_seconds.size == 0,
          "action schema retains ordering and honest unspecified units/timing");
    auto second_chunk = predictor.run({observation});
    check(second_chunk.inference_ms() == 102, "stateless calls are independent predictions");
    const auto factory = model.task<ImageStateActionQueue>();
    rejects([&] { (void)factory.create({{"unknown", true}}); }, TRTMC_INVALID_CONFIG,
            "queue creation rejects undeclared Config without reserving the model");
    Config config{{"tag", "copied"}};
    auto session = factory.create(config);
    config = Config{{"tag", "changed"}};
    rejects([&] { predictor.run({observation}); }, TRTMC_BUSY,
            "live queue reservation prevents a stateless call from touching its state");
    rejects([&] { factory.create(); }, TRTMC_BUSY, "second live execution reservation is rejected");
    check(model.supports<ImageStateToActionChunk>(),
          "metadata remains queryable during session lifetime");
    auto first = session.act(observation);
    check(first.values().size() == 2 && first.values()[0] == 2.5F && first.values()[1] == -3 &&
              first.started_new_chunk() && first.within_training_bounds() &&
              first.inference_ms() == 1,
          "first act refills one family chunk, unaffected by prior stateless predictions");
    check(trtmc::detail::string_view(first.view().action.schema.domain) == "fixture.copied",
          "family copies retained creation config before caller mutation");
    state[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { session.act(observation); }, TRTMC_INVALID_ARGUMENT,
            "family checks state even while an action is already queued");
    state[0] = 20;
    state[1] = 40;
    auto queued = session.act(observation);
    check(queued.values()[0] == 4 && queued.values()[1] == 8 && !queued.started_new_chunk() &&
              !queued.within_training_bounds() && queued.inference_ms() == 0,
          "queued action retains old prediction and no additional inference or clipping");
    rejects([&] { session.act(observation, {{"tag", "ignored"}}); }, TRTMC_INVALID_CONFIG,
            "unsupported per-act config is not silently ignored");
    auto refilled = session.act(observation);
    check(refilled.values()[0] == 20.5F && refilled.started_new_chunk() &&
              refilled.inference_ms() == 2,
          "refill uses current observation only after previous chunk is exhausted");
    session.reset();
    state[0] = 3;
    state[1] = 5;
    auto after_reset = session.act(observation);
    check(after_reset.values()[0] == 3.5F && after_reset.started_new_chunk() &&
              after_reset.inference_ms() == 3,
          "reset discards queued action and delegates a fresh family prediction");
    session.close();
    check(first.values()[0] == 2.5F && queued.values()[1] == 8,
          "step results own snapshots across reset and session close");
    rejects([&] { session.act(observation); }, TRTMC_INVALID_ARGUMENT,
            "closed C++ session fails explicitly");
    auto after_session = predictor.run({observation});
    check(after_session.inference_ms() == 103,
          "BUSY stateless call never entered the family or consumed queue work");

    const auto library = root / "libtrtmc_model_action_fixture.so";
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (!handle)
        throw std::runtime_error("cannot access loaded action fixture synchronization");
    const auto block =
        reinterpret_cast<void (*)()>(dlsym(handle, "trtmc_action_fixture_block_next"));
    const auto entered = reinterpret_cast<int (*)()>(dlsym(handle, "trtmc_action_fixture_entered"));
    const auto unblock =
        reinterpret_cast<void (*)()>(dlsym(handle, "trtmc_action_fixture_unblock"));
    if (!block || !entered || !unblock)
        throw std::runtime_error("missing action fixture synchronization");
    auto concurrent = factory.create();
    block();
    auto pending = std::async(std::launch::async, [&] { return concurrent.act(observation); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!entered() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    check(entered(), "deterministic fixture entered its blocked family act");
    rejects([&] { concurrent.reset(); }, TRTMC_BUSY,
            "overlapping reset is BUSY, not queued behind active execution");
    rejects([&] { concurrent.act(observation); }, TRTMC_BUSY,
            "competing act is BUSY without a second family call");
    unblock();
    auto completed = pending.get();
    check(completed.started_new_chunk() && completed.inference_ms() == 1,
          "only the original concurrent act consumes a chunk");
    concurrent.close();
    dlclose(handle);

    auto moved = factory.create();
    auto destination = std::move(moved);
    rejects([&] { moved.reset(); }, TRTMC_INVALID_ARGUMENT, "moved-from session is closed");
    destination.close();
    for (const std::string mode : {"none", "missing"}) {
        auto absent = load(root, mode);
        check(absent.tasks().empty() && !absent.supports<ImageStateActionQueue>(),
              "unbound model variants do not expose DSO-wide action interfaces");
    }
    auto malformed = load(root, "bad_chunk");
    rejects([&] { malformed.task<ImageStateToActionChunk>().run({observation}); },
            TRTMC_INTERNAL_ERROR, "malformed family chunk shape does not escape C ABI");
    auto wrong_step = load(root, "bad_step").task<ImageStateActionQueue>().create();
    rejects([&] { wrong_step.act(observation); }, TRTMC_INTERNAL_ERROR,
            "queue cannot report multiple actions as one step");
    wrong_step.close();
    auto empty_factory = load(root, "null_queue");
    rejects([&] { empty_factory.task<ImageStateActionQueue>().create(); }, TRTMC_INTERNAL_ERROR,
            "null family session is not an advertised successful creation");
    check(empty_factory.task<ImageStateToActionChunk>().run({observation}).actions().rows == 2,
          "failed creation releases model reservation without a stuck BUSY state");
    auto retained = [&] {
        auto local = load(root, "all");
        auto own = local.task<ImageStateActionQueue>().create();
        return own.act(observation);
    }();
    check(retained.values()[0] == 3.5F, "owned step survives all local model/session wrappers");
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    try {
        exercise(argv[1]);
        exercise_released_model(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return failures ? 1 : 0;
}
