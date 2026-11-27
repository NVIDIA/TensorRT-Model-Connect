/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "core/runtime/bundle/bundle_format.h"
#include "request_backend.h"
#include "tokenizer_fixture.h"
#include "trtmc/image.h"
#include "trtmc/internal/image.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/trtmc.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <thread>

namespace {
using trtmc::ideogram4::test::RequestBackend;
namespace internal = trtmc::internal;
int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class Error, class Function>
void expect_error(Function&& call, const char* message) {
    try {
        call();
        check(false, message);
    } catch (const Error& error) {
        check(std::string(error.what()).find(message) != std::string::npos, message);
    } catch (...) {
        check(false, message);
    }
}

void write_bundle(const std::filesystem::path& path) {
    trtmc::ideogram4::test::Fixture fixture;
    std::vector<float> times;
    for (int index = 20; index >= 0; --index)
        times.push_back(static_cast<float>(index) / 20);
    nlohmann::json config{{"height", 256},          {"width", 256},
                          {"text_length", 128},     {"sequence_layout", "packed_text_image"},
                          {"latent_channels", 128}, {"features", 53248},
                          {"precision", "bf16"},    {"sampler", "V4_DEFAULT_20"},
                          {"times", times},         {"guidance", std::vector<float>(20, 1)}};
    const std::vector<std::pair<std::string, std::string>> sections{
        {"runtime.json", config.dump()},
        {"tokenizer.json", trtmc::ideogram4::test::tokenizer_json(fixture)},
        {"text.plan", "text"},
        {"positive.plan", "positive"},
        {"negative.plan", "negative"},
        {"decoder.plan", "decoder"}};
    nlohmann::json header{
        {"format", 1}, {"family", "ideogram4"}, {"task", "text_to_image"}, {"backend", "trt"}};
    std::string payload;
    for (const auto& [name, value] : sections) {
        header["sections"][name] = {{"offset", payload.size()}, {"length", value.size()}};
        payload += value;
    }
    const auto serialized = header.dump();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    for (int shift = 0; shift < 64; shift += 8)
        file.put(static_cast<char>((static_cast<std::uint64_t>(serialized.size()) >> shift) & 255));
    file << serialized << payload;
    if (!file)
        throw std::runtime_error("cannot write request fixture");
}

trtmc_string_view borrowed(std::string_view value) {
    return {value.data(), value.size()};
}

void test_client_errors(const std::filesystem::path& bundle, const std::filesystem::path& runtime) {
    const trtmc_core_api_v1* core = nullptr;
    check(trtmc_get_api(1, 0, &core) == TRTMC_OK, "load core API");
    if (!core)
        return;
    trtmc_load_options_v1 options{};
    options.struct_size = sizeof(options);
    const auto root = runtime.string();
    options.runtime_root = borrowed(root);
    trtmc_model* model = nullptr;
    trtmc_error* error = nullptr;
    const auto filename = bundle.string();
    const auto status = core->model_load(borrowed(filename), &options, &model, &error);
    check(status == TRTMC_OK, "load real Ideogram4 family with the test backend");
    if (error) {
        const auto text = core->error_message(error);
        std::cerr.write(text.data, text.size);
        core->error_release(error);
    }
    if (!model)
        return;
    const trtmc_api_header* table = nullptr;
    check(core->model_get_task_api(model, borrowed(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &table,
                                   &error) == TRTMC_OK,
          "load image API");
    const auto* api = reinterpret_cast<const trtmc_text_to_image_api_v1*>(table);
    if (api) {
        const std::vector<std::string> prompts{
            "", "{", "not JSON", "[]", "null", "true", std::string(256 * 1024 + 1, 'x')};
        for (const auto& prompt : prompts) {
            trtmc_text_to_image_request_v1 input{borrowed(prompt), {nullptr, 0}};
            trtmc_result* result = nullptr;
            error = nullptr;
            check(api->run(model, &input, nullptr, &result, &error) == TRTMC_INVALID_ARGUMENT,
                  "malformed and oversized captions report TRTMC_INVALID_ARGUMENT");
            check(!result && error, "rejected request returns an error without an image");
            core->error_release(error);
            core->result_release(result);
        }
    }
    core->model_release(model);
}

void test_byte_budget(internal::ITextToImage& pipeline, RequestBackend& backend) {
    const std::string prompt = "{\"caption\":\"\xc3\xa9\"}";
    internal::ConfigEntry budget{"max_prompt_bytes", static_cast<std::int64_t>(prompt.size())};
    expect_error<std::runtime_error>([&] { pipeline.run({prompt, {}}, {&budget, 1}); },
                                     "injected engine stop");
    check(backend.state->calls == 1, "caption at the exact byte budget reaches the engine");
    budget.value = static_cast<std::int64_t>(prompt.size() - 1);
    expect_error<std::invalid_argument>([&] { pipeline.run({prompt, {}}, {&budget, 1}); },
                                        "exceeds max_prompt_bytes");
    for (const auto value : {std::int64_t{0}, std::int64_t{-1},
                             std::int64_t{std::numeric_limits<std::int32_t>::max()}}) {
        budget.value = value;
        expect_error<internal::ConfigError>([&] { pipeline.run({prompt, {}}, {&budget, 1}); },
                                            "max_prompt_bytes must fit");
    }
    check(backend.state->calls == 1, "rejected budgets do not run any engine");
}

void test_admission_while_busy(internal::ITextToImage& pipeline, RequestBackend& backend) {
    backend.state->block = true;
    backend.state->entered = false;
    bool stopped = false;
    std::thread active([&] {
        try {
            pipeline.run({"{}", {}}, {});
        } catch (const std::runtime_error& error) {
            stopped = std::string(error.what()) == "injected engine stop";
        }
    });
    {
        std::unique_lock<std::mutex> lock(backend.state->mutex);
        check(backend.state->changed.wait_for(lock, std::chrono::seconds(2),
                                              [&] { return backend.state->entered; }),
              "first request holds its engine transaction");
    }
    expect_error<std::invalid_argument>([&] { pipeline.run({"{", {}}, {}); },
                                        "structured JSON caption object");
    const auto overlong = "{\"caption\":\"" + std::string(256, 'a') + "\"}";
    expect_error<std::invalid_argument>([&] { pipeline.run({overlong, {}}, {}); },
                                        "framed prompt exceeds the bundle text profile");
    expect_error<std::invalid_argument>([&] { pipeline.run({"{}", {}}, {}); },
                                        "already has an active request");
    {
        std::lock_guard<std::mutex> lock(backend.state->mutex);
        backend.state->release = true;
    }
    backend.state->changed.notify_all();
    active.join();
    check(stopped, "engine execution stayed serialized until its transaction ended");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    char directory[] = "/tmp/trtmc-ideogram4-request-XXXXXX";
    if (!mkdtemp(directory))
        return 2;
    const auto root = std::filesystem::path(directory);
    const auto bundle = root / "request.bundle";
    write_bundle(bundle);
    std::filesystem::create_symlink(std::filesystem::canonical(argv[1]),
                                    root / "libtrtmc_model_ideogram4.so");
    std::filesystem::create_symlink(std::filesystem::canonical(argv[2]),
                                    root / "libtrtmc_backend_trt.so");
    test_client_errors(bundle, root);
    RequestBackend backend;
    const trtmc::BundleReader reader(bundle.string());
    std::unique_ptr<trtmc::ITask> family(trtmc_create_family({reader, backend}));
    auto* pipeline = dynamic_cast<internal::ITextToImage*>(family.get());
    check(pipeline != nullptr, "family implements text-to-image");
    if (pipeline) {
        test_byte_budget(*pipeline, backend);
        test_admission_while_busy(*pipeline, backend);
    }
    std::filesystem::remove_all(root);
    return failures ? 1 : 0;
}
