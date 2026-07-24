/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "test_helpers.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_factory.h"
#include "trtmc/runtime/pipeline_pool.h"
#include "utils/sha256.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef TRTMC_TEST_OPTIMIZED_PROVIDER_DSO
#error "TRTMC_TEST_OPTIMIZED_PROVIDER_DSO must be defined"
#endif

#ifndef TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_DSO
#error "TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_DSO must be defined"
#endif

#ifndef TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_ABI_DSO
#error "TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_ABI_DSO must be defined"
#endif

#ifndef TRTMC_TEST_OPTIMIZED_EMBEDDING_DSO
#error "TRTMC_TEST_OPTIMIZED_EMBEDDING_DSO must be defined"
#endif

namespace {

namespace fs = std::filesystem;

constexpr const char* kImplementationId = "example-optimized-runtime";
constexpr const char* kTextPipelineType = "FakeTextGenerationPipeline";
constexpr const char* kRuntimeLibrary = "libtrtmc_impl_example_optimized_runtime.so";
constexpr const char* kModelId = "Example/Optimized-Model";
constexpr const char* kEmbeddingImplementationId = "example-optimized-embedding";
constexpr const char* kEmbeddingRuntimeLibrary = "libtrtmc_impl_example_optimized_embedding.so";
constexpr const char* kEmbeddingModelId = "Example/Embedding-Model";
constexpr const char* kEmbeddingPipelineType = "FakeEmbeddingPipeline";
constexpr const char* kArtifact = "GENERIC-RUNTIME-DATA";
constexpr const char* kPrivateMetadata = R"({"capsule":"example-test"})";

struct RuntimeSpec {
    std::string implementation_id;
    std::string runtime_library;
    std::string model_id;
    std::string profile_id;
    std::string runtime_name;
    std::string runtime_version;
    std::string runtime_commit;
    fs::path dso;
};

RuntimeSpec text_spec(const fs::path& dso = TRTMC_TEST_OPTIMIZED_PROVIDER_DSO) {
    return RuntimeSpec{kImplementationId,
                       kRuntimeLibrary,
                       kModelId,
                       "generic-profile",
                       "test-optimized-runtime",
                       "test-runtime-1.0",
                       "test-runtime-commit",
                       dso};
}

RuntimeSpec embedding_spec() {
    return RuntimeSpec{kEmbeddingImplementationId, kEmbeddingRuntimeLibrary,
                       kEmbeddingModelId,          "embedding-profile",
                       "test-embedding-runtime",   "test-embedding-1.0",
                       "test-embedding-commit",    TRTMC_TEST_OPTIMIZED_EMBEDDING_DSO};
}

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::string replace_once(std::string value, const std::string& needle,
                         const std::string& replacement) {
    const std::size_t offset = value.find(needle);
    if (offset == std::string::npos)
        throw std::logic_error("test replacement needle not found");
    value.replace(offset, needle.size(), replacement);
    return value;
}

void update_tree_record(trtmc::internal::Sha256& tree, const std::string& kind,
                        const std::string& path) {
    constexpr char terminator = '\0';
    tree.update(kind);
    tree.update(&terminator, 1);
    tree.update(path);
    tree.update(&terminator, 1);
}

std::string read_binary(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to read fake optimized-runtime DSO");
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

std::vector<std::pair<std::string, std::string>> artifact_files(const RuntimeSpec& spec) {
    std::vector<std::pair<std::string, std::string>> files = {
        {"payload/runtime.data", kArtifact},
        {spec.runtime_library, read_binary(spec.dso)},
    };
    std::sort(files.begin(), files.end());
    return files;
}

std::string artifact_tree_hash(const RuntimeSpec& spec) {
    trtmc::internal::Sha256 tree;
    update_tree_record(tree, "directory", "payload");
    for (const auto& [path, contents] : artifact_files(spec)) {
        trtmc::internal::Sha256 file;
        file.update(contents);
        const auto digest = file.digest();
        update_tree_record(tree, "file", path);
        tree.update(std::to_string(contents.size()));
        constexpr char terminator = '\0';
        tree.update(&terminator, 1);
        tree.update(digest.data(), digest.size());
    }
    return tree.hex_digest();
}

std::string descriptor_json(const RuntimeSpec& spec, int factory_abi = 1,
                            const std::string& hash_override = {}) {
    const auto files = artifact_files(spec);
    std::uint64_t total_size = 0;
    for (const auto& file : files)
        total_size += file.second.size();
    const std::string hash = hash_override.empty() ? artifact_tree_hash(spec) : hash_override;
    std::ostringstream output;
    output << "{\"schema_version\":2,"
           << "\"implementation_id\":\"" << spec.implementation_id << "\","
           << "\"model_id\":\"" << spec.model_id << "\","
           << "\"profile_id\":\"" << spec.profile_id << "\","
           << "\"runtime_library\":\"" << spec.runtime_library << "\","
           << "\"factory_abi\":" << factory_abi << ','
           << "\"implementation_metadata_section\":\"implementation.json\","
           << "\"runtime\":{\"name\":\"" << spec.runtime_name << "\",\"version\":\""
           << spec.runtime_version << "\",\"commit\":\"" << spec.runtime_commit << "\"},"
           << "\"artifact\":{\"section_prefix\":\"optimized_runtime_artifacts\","
              "\"directories\":[\"payload\"],\"file_count\":"
           << files.size() << ",\"total_size\":" << total_size << ",\"tree_sha256\":\"" << hash
           << "\"}}";
    return output.str();
}

struct NamedSection {
    std::string name;
    std::string contents;
};

void write_u64(std::ofstream& output, std::uint64_t value) {
    unsigned char bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<unsigned char>((value >> (8 * index)) & 0xffU);
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_bundle(const fs::path& path, const RuntimeSpec& spec, const std::string& descriptor = {},
                  bool include_artifacts = true) {
    std::vector<NamedSection> sections = {
        {"optimized_runtime.json", descriptor.empty() ? descriptor_json(spec) : descriptor},
        {"implementation.json", kPrivateMetadata},
    };
    if (include_artifacts) {
        for (auto& [name, contents] : artifact_files(spec))
            sections.push_back({"optimized_runtime_artifacts/" + name, std::move(contents)});
    }

    std::ostringstream header;
    header << "{\"model_id\":\"" << spec.model_id
           << "\",\"model_type\":\"optimized_runtime\","
              "\"family\":\"optimized_runtime\",\"precision\":\"\","
              "\"vocab_size\":0,\"max_cache_length\":0,\"sections\":{";
    std::uint64_t offset = 0;
    for (std::size_t index = 0; index < sections.size(); ++index) {
        if (index != 0)
            header << ',';
        header << '"' << sections[index].name << "\":{\"offset\":" << offset
               << ",\"size\":" << sections[index].contents.size() << '}';
        offset += sections[index].contents.size();
    }
    header << "}}";

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create optimized-runtime test bundle");
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), sizeof(trtmc::kBundleMagic));
    const std::string header_text = header.str();
    write_u64(output, header_text.size());
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const auto& section : sections)
        output.write(section.contents.data(),
                     static_cast<std::streamsize>(section.contents.size()));
}

trtmc::LoadOptions load_options(const fs::path& cache) {
    trtmc::LoadOptions options;
    options.runtime_cache_path = cache.string();
    return options;
}

std::vector<std::string> read_lines(const fs::path& path) {
    std::ifstream input(path);
    std::vector<std::string> result;
    std::string line;
    while (std::getline(input, line))
        result.push_back(line);
    return result;
}

std::size_t count_line(const std::vector<std::string>& lines, const std::string& value) {
    return static_cast<std::size_t>(std::count(lines.begin(), lines.end(), value));
}

void test_model_owned_text_pipeline_and_eager_load() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "optimized.trtfb";
    const fs::path events = root / "events.txt";
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());
    trtmc_test::EnvVarGuard metadata_guard("TRTMC_FAKE_OPTIMIZED_EXPECT_METADATA",
                                           kPrivateMetadata);
    trtmc_test::EnvVarGuard artifact_guard("TRTMC_FAKE_OPTIMIZED_EXPECT_ARTIFACT", kArtifact);
    {
        auto options = load_options(root / "cache");
        options.cuda_graphs = true;
        auto pipeline = trtmc::load(bundle.string(), options);
        check(std::string(pipeline->model_id()) == kModelId, "factory exposes model-owned id");
        check(std::string(pipeline->pipeline_type()) == kTextPipelineType,
              "model-owned adapter preserves descriptive pipeline type semantics");
        const auto loaded_events = read_lines(events);
        check(count_line(loaded_events, "dlopen") == 1, "DSO loads during trtmc::load");
        check(count_line(loaded_events, "create") == 1, "pipeline initializes during load");
        check(count_line(loaded_events, "generate") == 0, "execution is not lazy to generate");

        trtmc::GenerateConfig config;
        config.max_new_tokens = 7;
        const auto result = pipeline->generate("hello", config);
        check(result.text == "optimized:hello", "model-owned DSO implements generate");
        check(result.token_ids == std::vector<int32_t>({41, 7}),
              "model-owned DSO owns result construction");
    }
    const auto final_events = read_lines(events);
    check(count_line(final_events, "destroy") == 1, "pipeline lifecycle is released");
    check(count_line(final_events, "dlclose") == 0, "DSO remains loaded after pipeline destroy");

    bool materialized = false;
    for (const auto& entry : fs::recursive_directory_iterator(root / "cache")) {
        if (entry.path().filename() == "runtime.data") {
            std::ifstream input(entry.path(), std::ios::binary);
            const std::string contents((std::istreambuf_iterator<char>(input)),
                                       std::istreambuf_iterator<char>());
            materialized = contents == kArtifact;
        }
    }
    check(materialized, "opaque artifact tree is materialized into runtime cache");
}

void test_legacy_load_overload_delegates_to_optimized_runtime() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "legacy-load.trtfb";
    const fs::path events = root / "events.txt";
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());

    auto pipeline = trtmc::load(bundle.string(), "", (root / "cache").string(), false);
    const auto result = pipeline->generate("legacy");
    check(result.text == "optimized:legacy",
          "legacy public C++ load overload delegates execution to the model-owned DSO");
    const auto loaded_events = read_lines(events);
    check(count_line(loaded_events, "create") == 1,
          "legacy public C++ load overload initializes the optimized runtime");
}

void test_c_abi_create_loads_optimized_runtime_bundle() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "c-abi-load.trtfb";
    const fs::path events = root / "events.txt";
    const std::string runtime_cache = (root / "cache").string();
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());

    TrtmcPipelineOptions options{};
    options.runtime_cache = runtime_cache.c_str();
    trtmc::IPipeline* pipeline = trtmc_create_pipeline_ex(bundle.string().c_str(), &options);
    if (pipeline == nullptr)
        std::cerr << "C ABI optimized-runtime load error: " << trtmc_last_error() << '\n';
    check(pipeline != nullptr, "C ABI creates a delegated pipeline");
    if (pipeline != nullptr)
        delete pipeline;

    const auto loaded_events = read_lines(events);
    check(count_line(loaded_events, "create") == 1,
          "C ABI delegates pipeline creation to the model-owned DSO");
}

void test_non_text_pipeline_uses_same_host() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "embedding.trtfb";
    write_bundle(bundle, embedding_spec());
    auto pipeline = trtmc::load(bundle.string(), load_options(root / "cache"));
    check(std::string(pipeline->pipeline_type()) == kEmbeddingPipelineType,
          "generic host does not reinterpret a non-text pipeline type as implementation identity");
    const auto result = pipeline->embed("four");
    check(result.dim == 3 && result.data == std::vector<float>({4.0F, 3.5F, -2.0F}),
          "non-text adapter implements existing embed operation without host changes");
    bool unsupported = false;
    try {
        (void)pipeline->generate("unused");
    } catch (const std::runtime_error&) {
        unsupported = true;
    }
    check(unsupported, "non-text adapter does not inherit host-side operation behavior");
}

void test_public_pipeline_pool_fails_before_loading_optimized_runtime() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "pool.trtfb";
    const fs::path events = root / "events.txt";
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());

    bool rejected = false;
    try {
        (void)trtmc::PipelineFactory::from_bundle_pool(bundle.string(), 2,
                                                       load_options(root / "cache"));
    } catch (const std::invalid_argument& error) {
        rejected =
            std::string(error.what()).find("delegated runtime owns batching") != std::string::npos;
    }
    check(rejected, "public pool API rejects optimized bundles with an ownership explanation");
    check(read_lines(events).empty(), "pool rejection does not load or initialize the runtime DSO");
}

void test_runtime_kv_policy_fails_before_loading_optimized_runtime() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "dynamic-kv-policy.trtfb";
    const fs::path events = root / "events.txt";
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());

    trtmc::LoadOptionsV2 options;
    options.runtime_cache_path = (root / "cache").string();
    options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kFraction;
    options.kv_cache_memory_fraction = 0.8;
    bool rejected = false;
    try {
        (void)trtmc::load(bundle.string(), options);
    } catch (const std::invalid_argument& error) {
        rejected =
            std::string(error.what()).find("does not declare runtime_memory contract version 1") !=
            std::string::npos;
    }
    check(rejected, "bundle contract rejects unsupported dynamic KV policy");
    check(read_lines(events).empty(),
          "dynamic KV policy rejection does not load or initialize the runtime DSO");
}

void test_concurrent_repeated_loads_share_published_cache_and_dso() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path bundle = root / "concurrent.trtfb";
    const fs::path events = root / "events.txt";
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());
    trtmc_test::EnvVarGuard artifact_guard("TRTMC_FAKE_OPTIMIZED_EXPECT_ARTIFACT", kArtifact);

    constexpr std::size_t kWorkers = 8;
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::unique_ptr<trtmc::IPipeline>> pipelines(kWorkers);
    std::vector<std::exception_ptr> errors(kWorkers);
    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (std::size_t index = 0; index < kWorkers; ++index) {
        workers.emplace_back([&, index] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            try {
                pipelines[index] = trtmc::load(bundle.string(), load_options(root / "cache"));
            } catch (...) {
                errors[index] = std::current_exception();
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kWorkers)
        std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers)
        worker.join();

    check(std::all_of(errors.begin(), errors.end(),
                      [](const auto& error) { return error == nullptr; }),
          "concurrent repeated loads all succeed");
    check(std::all_of(pipelines.begin(), pipelines.end(),
                      [](const auto& pipeline) { return pipeline != nullptr; }),
          "concurrent repeated loads return every pipeline");
    const auto loaded_events = read_lines(events);
    check(count_line(loaded_events, "dlopen") == 1,
          "concurrent repeated loads initialize one DSO identity");
    check(count_line(loaded_events, "create") == kWorkers,
          "concurrent repeated loads create every requested pipeline");

    pipelines.clear();
    const auto final_events = read_lines(events);
    check(count_line(final_events, "destroy") == kWorkers,
          "concurrent repeated pipelines are all destroyed");
    check(count_line(final_events, "dlclose") == 0,
          "deduplicated process-lifetime DSO reference remains loaded");
}

void test_descriptor_is_strict_and_fail_closed() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const RuntimeSpec spec = text_spec();
    const std::string descriptor = descriptor_json(spec);
    const fs::path unknown = root / "unknown.trtfb";
    const fs::path duplicate = root / "duplicate.trtfb";
    write_bundle(unknown, spec,
                 replace_once(descriptor, "{\"schema_version\":2",
                              "{\"unknown\":true,\"schema_version\":2"));
    write_bundle(duplicate, spec,
                 replace_once(descriptor, "\"implementation_id\":\"example-optimized-runtime\"",
                              "\"implementation_id\":\"example-optimized-runtime\","
                              "\"implementation_id\":\"example-optimized-runtime\""));
    for (const auto& bundle : {unknown, duplicate}) {
        bool threw = false;
        try {
            (void)trtmc::load(bundle.string(), load_options(root / "cache"));
        } catch (const std::runtime_error& error) {
            const std::string message(error.what());
            threw = message.find("unknown field") != std::string::npos ||
                    message.find("Duplicate JSON object key") != std::string::npos;
        }
        check(threw, "claimed optimized bundle rejects malformed dispatch metadata");
    }
}

void test_artifact_integrity_and_cache_tamper_fail_closed() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const RuntimeSpec spec = text_spec();
    const fs::path bad_bundle = root / "bad-hash.trtfb";
    write_bundle(bad_bundle, spec, descriptor_json(spec, 1, std::string(64, '0')));
    bool hash_threw = false;
    try {
        (void)trtmc::load(bad_bundle.string(), load_options(root / "bad-cache"));
    } catch (const std::runtime_error& error) {
        hash_threw = std::string(error.what()).find("SHA-256 mismatch") != std::string::npos;
    }
    check(hash_threw, "artifact hash mismatch fails before DSO loading");

    const fs::path bundle = root / "good.trtfb";
    const fs::path cache = root / "cache";
    write_bundle(bundle, spec);
    {
        auto pipeline = trtmc::load(bundle.string(), load_options(cache));
    }
    fs::path payload;
    for (const auto& entry : fs::recursive_directory_iterator(cache)) {
        if (entry.path().filename() == "runtime.data")
            payload = entry.path();
    }
    {
        std::ofstream(payload, std::ios::binary | std::ios::trunc) << "TAMPERED";
    }
    bool tamper_threw = false;
    try {
        (void)trtmc::load(bundle.string(), load_options(cache));
    } catch (const std::exception&) {
        tamper_threw = true;
    }
    check(tamper_threw, "tampered materialized artifact is rejected on cache reuse");
}

void test_exact_embedded_dso_identity() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    RuntimeSpec spec = text_spec(TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_DSO);
    const fs::path bundle = root / "wrong-identity.trtfb";
    write_bundle(bundle, spec);
    bool threw = false;
    try {
        (void)trtmc::load(bundle.string(), load_options(root / "cache"));
    } catch (const std::runtime_error& error) {
        threw =
            std::string(error.what()).find("implementation identity mismatch") != std::string::npos;
    }
    check(threw, "wrong embedded DSO identity fails closed");
}

void test_pipeline_abi_version_fails_before_create() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path events = root / "events.txt";
    const RuntimeSpec spec = text_spec(TRTMC_TEST_WRONG_OPTIMIZED_PROVIDER_ABI_DSO);
    const fs::path bundle = root / "wrong-pipeline-abi.trtfb";
    write_bundle(bundle, spec);
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());

    bool threw = false;
    try {
        (void)trtmc::load(bundle.string(), load_options(root / "cache"));
    } catch (const std::runtime_error& error) {
        threw =
            std::string(error.what()).find("IPipeline ABI version mismatch") != std::string::npos;
    }
    check(threw, "mismatched IPipeline ABI version fails closed");
    const auto events_before_return = read_lines(events);
    check(count_line(events_before_return, "dlopen") == 1,
          "mismatched IPipeline DSO is inspected exactly once");
    check(count_line(events_before_return, "create") == 0,
          "mismatched IPipeline DSO is rejected before create");
}

void test_toolchain_abi_fails_before_create() {
    trtmc_test::TempDirGuard temporary;
    const fs::path root(temporary.path());
    const fs::path events = root / "events.txt";
    const fs::path bundle = root / "wrong-toolchain-abi.trtfb";
    write_bundle(bundle, text_spec());
    trtmc_test::EnvVarGuard event_guard("TRTMC_FAKE_OPTIMIZED_EVENTS", events.c_str());
    trtmc_test::EnvVarGuard mismatch_guard("TRTMC_FAKE_OPTIMIZED_WRONG_TOOLCHAIN_ABI", "1");

    bool threw = false;
    try {
        (void)trtmc::load(bundle.string(), load_options(root / "cache"));
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("C++ toolchain ABI mismatch") != std::string::npos;
    }
    check(threw, "mismatched toolchain ABI fails closed");
    const auto events_before_return = read_lines(events);
    check(count_line(events_before_return, "dlopen") == 1,
          "mismatched toolchain DSO is inspected exactly once");
    check(count_line(events_before_return, "create") == 0,
          "mismatched toolchain DSO is rejected before create");
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--load-writer-bundle") {
        try {
            trtmc::LoadOptions options;
            options.runtime_cache_path = argv[3];
            auto pipeline = trtmc::load(argv[2], options);
            trtmc::GenerateConfig config;
            config.max_new_tokens = 9;
            const auto result = pipeline->generate("writer-contract", config);
            const bool valid = std::string(pipeline->model_id()) == kModelId &&
                               std::string(pipeline->pipeline_type()) == kTextPipelineType &&
                               result.text == "optimized:writer-contract" &&
                               result.token_ids == std::vector<int32_t>({41, 9});
            if (!valid)
                std::cerr << "Python-writer bundle produced unexpected runtime output\n";
            return valid ? 0 : 1;
        } catch (const std::exception& error) {
            std::cerr << "Python-writer bundle failed C++ load: " << error.what() << '\n';
            return 1;
        }
    }
    test_pipeline_abi_version_fails_before_create();
    test_exact_embedded_dso_identity();
    test_toolchain_abi_fails_before_create();
    test_model_owned_text_pipeline_and_eager_load();
    test_legacy_load_overload_delegates_to_optimized_runtime();
    test_c_abi_create_loads_optimized_runtime_bundle();
    test_concurrent_repeated_loads_share_published_cache_and_dso();
    test_public_pipeline_pool_fails_before_loading_optimized_runtime();
    test_runtime_kv_policy_fails_before_loading_optimized_runtime();
    test_non_text_pipeline_uses_same_host();
    test_descriptor_is_strict_and_fail_closed();
    test_artifact_integrity_and_cache_tamper_fail_closed();
    if (failures == 0)
        std::cout << "All generic optimized-runtime host tests passed\n";
    return failures == 0 ? 0 : 1;
}
