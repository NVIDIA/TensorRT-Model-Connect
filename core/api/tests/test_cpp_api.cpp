/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

// A protocol fixture, not a TensorRT engine or a model accuracy claim. The
// consumer includes only public SDK headers and loads the real family DSO.
void write_bundle(const std::filesystem::path& path, const std::string& mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const std::string header =
        "{\"format\":1,\"family\":\"api_fixture\",\"task\":\"" + mode +
        "\",\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}";
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
}

void test_model_and_tasks(const std::string& bundle, const std::string& disabled,
                          const trtmc::LoadOptions& options) {
    auto model = trtmc::Model::load(bundle, options);
    check(model.info().family == "api_fixture", "model metadata crosses the C ABI");
    check(model.tasks().size() == 1 && model.tasks()[0].id == trtmc::TextContinuation::kTask,
          "typed task is discoverable");
    check(model.supports<trtmc::TextContinuation>(), "supported task is available");
    auto no_text = trtmc::Model::load(disabled, options);
    check(no_text.tasks().empty() && !no_text.supports<trtmc::TextContinuation>(),
          "same family can advertise a different loaded-bundle capability");
    bool rejected = false;
    try {
        (void)no_text.task<trtmc::TextContinuation>();
    } catch (const trtmc::Error& error) {
        rejected = error.code() == TRTMC_UNSUPPORTED;
    }
    check(rejected, "unsupported task access throws the C status");
}

void test_calls_and_ownership(const std::string& bundle, const trtmc::LoadOptions& options) {
    auto text = [&] {
        auto model = trtmc::Model::load(bundle, options);
        return model.task<trtmc::TextContinuation>();
    }();
    auto result = text.run({"Hello"});
    check(result.text() == "Hello!|eos", "task proxy retains the model after caller scope");
    check(result.token_ids().size() == 2 && result.token_ids()[1] == 12,
          "typed token result is readable");
    check(result.setup_ms() == 16 && result.prefill_ms() == 0.75 && result.decode_ms() == 4,
          "runtime load controls and generation timings are preserved");
    const auto segments = result.segments();
    check(segments.size() == 1 && segments[0].text == std::string_view("seg\0tail", 8) &&
              segments[0].token_ids[0] == 41,
          "segments preserve explicit lengths and typed token views");

    auto explicit_values = text.run(
        {"Hello"},
        {{"max_new_tokens", 0}, {"temperature", 0.0}, {"emit_eos", false}, {"suffix", ""}});
    check(explicit_values.text() == "Hello" && explicit_values.decode_ms() == 0 &&
              explicit_values.prefill_ms() == 0,
          "explicit false, zero and empty string survive all three layers");
    auto arrays =
        text.run({"Hello"}, {{"emit_eos", false},
                             {"suffix", ""},
                             {"labels", std::vector<std::string>{"first", "second"}},
                             {"token_biases", std::vector<std::int64_t>{9007199254740993LL}},
                             {"schedule", std::vector<double>{0.5, 0.25}}});
    check(arrays.text() == "Hello|first|second|9007199254740993" && arrays.setup_ms() == 16.5,
          "all homogeneous list types reach the family without precision loss");
    auto tokens = text.run({std::vector<std::int32_t>{7, 8, 9}});
    check(tokens.text() == "tokens!|eos" && tokens.token_ids().size() == 3 &&
              tokens.token_ids()[2] == 9,
          "tokenized prefix is a typed representation, not a config parameter");

    auto moved = std::move(result);
    check(moved.text() == "Hello!|eos" && result.text().empty(),
          "result move transfers the handle and clears moved-from views");
    moved = std::move(explicit_values);
    check(moved.text() == "Hello" && explicit_values.text().empty(),
          "result move assignment releases the prior result");

    for (const trtmc::Config& config : {
             trtmc::Config{{"temperature", "0.9"}},
             trtmc::Config{{"temperature", 0.5}, {"temperature", 0.75}},
             trtmc::Config{{"unknown", true}},
             trtmc::Config{{"max_new_tokens", -1}},
         }) {
        bool rejected = false;
        try {
            (void)text.run({"Hello"}, config);
        } catch (const trtmc::Error& error) {
            rejected = error.code() == TRTMC_INVALID_CONFIG && error.what()[0] != '\0';
        }
        check(rejected, "family config errors cross C and become owned C++ errors");
    }
}

void test_metadata_ownership(const std::string& bundle, const trtmc::LoadOptions& options) {
    const auto fields = [&] {
        auto model = trtmc::Model::load(bundle, options);
        return model.task<trtmc::TextContinuation>().config_fields();
    }();
    check(fields.size() == 8 && fields[0].name == "max_new_tokens" &&
              fields[0].default_value->get<std::int64_t>() == 4,
          "config metadata owns defaults and names after model release");
    check(fields[2].default_value->get<bool>() &&
              fields[3].default_value->get<std::string>() == "!" &&
              fields[4].default_value->get<std::vector<std::int64_t>>().empty() &&
              !fields[7].default_value,
          "fixed and computed defaults remain distinguishable");
}

void test_instance_bindings(const std::string& bundle, trtmc::LoadOptions options) {
    options.kv_cache_size_bytes = 16;
    auto first = trtmc::Model::load(bundle, options).task<trtmc::TextContinuation>();
    options.kv_cache_size_bytes = 32;
    auto second = trtmc::Model::load(bundle, options).task<trtmc::TextContinuation>();
    check(first.run({"first"}).setup_ms() == 16 && second.run({"second"}).setup_ms() == 32 &&
              first.run({"again"}).setup_ms() == 16,
          "bindings belong to each loaded instance, not the first object in a static table");
}

void test_invalid_bindings(const std::filesystem::path& root, const trtmc::LoadOptions& options) {
    const std::pair<const char*, trtmc_status> cases[] = {
        {"unknown_binding", TRTMC_UNSUPPORTED},  {"wrong_version", TRTMC_VERSION_MISMATCH},
        {"null_binding", TRTMC_INTERNAL_ERROR},  {"duplicate_binding", TRTMC_INTERNAL_ERROR},
        {"empty_field", TRTMC_INTERNAL_ERROR},   {"duplicate_field", TRTMC_INTERNAL_ERROR},
        {"wrong_default", TRTMC_INTERNAL_ERROR},
    };
    for (const auto& item : cases) {
        const auto path = root / (std::string("invalid-") + item.first + ".bundle");
        write_bundle(path, item.first);
        bool rejected = false;
        try {
            (void)trtmc::Model::load(path.string(), options);
        } catch (const trtmc::Error& error) {
            rejected = error.code() == item.second;
        }
        std::filesystem::remove(path);
        check(rejected, item.first);
    }
}

void test_config_precedes_execution(const std::filesystem::path& root,
                                    const trtmc::LoadOptions& options) {
    const auto path = root / "config-preflight.bundle";
    write_bundle(path, "must_not_run");
    auto task = trtmc::Model::load(path.string(), options).task<trtmc::TextContinuation>();
    for (const trtmc::Config& config : {
             trtmc::Config{{"unknown", true}},
             trtmc::Config{{"temperature", "not-a-number"}},
             trtmc::Config{{"temperature", 0.5}, {"temperature", 0.6}},
         }) {
        bool rejected = false;
        try {
            (void)task.run({"Hello"}, config);
        } catch (const trtmc::Error& error) {
            rejected = error.code() == TRTMC_INVALID_CONFIG;
        }
        check(rejected, "Core rejects invalid config before the family execution trap");
    }
    bool executed = false;
    try {
        (void)task.run({"Hello"});
    } catch (const trtmc::Error& error) {
        executed = error.code() == TRTMC_INTERNAL_ERROR;
    }
    check(executed, "valid config reaches the deliberately throwing fixture");
    std::filesystem::remove(path);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_cpp_api RUNTIME_ROOT\n";
        return 2;
    }
    const auto root = std::filesystem::path(argv[1]);
    const auto bundle = root / "cpp-api-text.bundle";
    const auto disabled = root / "cpp-api-disabled.bundle";
    try {
        write_bundle(bundle, "text_generation");
        write_bundle(disabled, "disabled");
        trtmc::LoadOptions options;
        options.runtime_root = root.string();
        options.kv_cache_size_bytes = 16;
        test_model_and_tasks(bundle.string(), disabled.string(), options);
        test_calls_and_ownership(bundle.string(), options);
        test_metadata_ownership(bundle.string(), options);
        test_instance_bindings(bundle.string(), options);
        test_invalid_bindings(root, options);
        test_config_precedes_execution(root, options);
    } catch (const std::exception& error) {
        std::cerr << "Unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove(bundle);
    std::filesystem::remove(disabled);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures == 0 ? 0 : 1;
}
