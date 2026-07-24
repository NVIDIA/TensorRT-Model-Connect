/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Qualification-only positive loader for UX-05.  It deliberately exercises
// the existing public C++ and versioned C-ABI construction surfaces without
// adding a product ABI.  The current C ABI returns IPipeline*, so the C-ABI
// path uses that documented handle to issue the same text request.

#include "trtmc/pipeline.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

struct Arguments {
    std::string surface;
    std::string bundle;
    std::string prompt{"Hello"};
    std::string hf_python;
    std::uint64_t kv_cache_bytes{0};
    std::uint64_t max_sequence_length{0};
    std::int32_t max_new_tokens{2};
    std::vector<std::string> backend_dirs;
    std::vector<std::string> model_plugin_dirs;
};

[[noreturn]] void usage_error(const std::string& message) {
    throw std::invalid_argument(message +
                                "\nusage: trtmc_dynamic_memory_surfaces --surface cpp|cabi "
                                "--bundle MODEL.trtfb --kv-cache-bytes N --max-sequence-length N "
                                "[--prompt TEXT] [--max-new-tokens N] [--hf-python PATH] "
                                "[--backend-dir DIR] [--model-plugin-dir DIR]");
}

std::uint64_t parse_u64(const std::string& text, const char* name) {
    std::size_t consumed = 0;
    const auto value = std::stoull(text, &consumed, 10);
    if (consumed != text.size())
        usage_error(std::string(name) + " must be an unsigned integer");
    return value;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments out;
    for (int index = 1; index < argc; ++index) {
        const std::string flag = argv[index];
        auto require_value = [&]() -> std::string {
            if (index + 1 >= argc)
                usage_error(flag + " requires a value");
            return argv[++index];
        };
        if (flag == "--surface")
            out.surface = require_value();
        else if (flag == "--bundle")
            out.bundle = require_value();
        else if (flag == "--prompt")
            out.prompt = require_value();
        else if (flag == "--hf-python")
            out.hf_python = require_value();
        else if (flag == "--kv-cache-bytes")
            out.kv_cache_bytes = parse_u64(require_value(), "--kv-cache-bytes");
        else if (flag == "--max-sequence-length")
            out.max_sequence_length = parse_u64(require_value(), "--max-sequence-length");
        else if (flag == "--max-new-tokens") {
            const auto value = parse_u64(require_value(), "--max-new-tokens");
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                usage_error("--max-new-tokens exceeds int32");
            }
            out.max_new_tokens = static_cast<std::int32_t>(value);
        } else if (flag == "--backend-dir")
            out.backend_dirs.push_back(require_value());
        else if (flag == "--model-plugin-dir")
            out.model_plugin_dirs.push_back(require_value());
        else
            usage_error("unknown argument: " + flag);
    }
    if (out.surface != "cpp" && out.surface != "cabi")
        usage_error("--surface must be cpp or cabi");
    if (out.bundle.empty() || out.kv_cache_bytes == 0 || out.max_sequence_length == 0) {
        usage_error("--bundle, --kv-cache-bytes, and --max-sequence-length are required");
    }
    return out;
}

std::string join_paths(const std::vector<std::string>& paths) {
    std::string joined;
    for (const auto& path : paths) {
        if (!joined.empty())
            joined.push_back(':');
        joined += path;
    }
    return joined;
}

trtmc::TextResult run_request(trtmc::IPipeline& pipeline, const Arguments& args) {
    trtmc::GenerateConfig config;
    config.max_new_tokens = args.max_new_tokens;
    config.temperature = 0.0F;
    config.top_k = 1;
    config.use_chat_template = false;
    return pipeline.generate(args.prompt, config);
}

json success(const char* surface, trtmc::IPipeline& pipeline, const trtmc::TextResult& result,
             const std::string& receipt_json) {
    if (receipt_json.empty())
        throw std::runtime_error("runtime-memory surface returned an empty receipt");
    return {
        {"status", "accepted"},
        {"surface", surface},
        {"model_id", pipeline.model_id()},
        {"pipeline_type", pipeline.pipeline_type()},
        {"generated_text", result.text},
        {"generated_token_ids", result.token_ids},
        {"runtime_memory_receipt", json::parse(receipt_json)},
    };
}

json run_cpp(const Arguments& args) {
    trtmc::LoadOptionsV2 options;
    options.hf_python = args.hf_python;
    options.kv_cache_memory_policy = trtmc::KvCacheMemoryPolicy::kBytes;
    options.kv_cache_memory_bytes = args.kv_cache_bytes;
    options.max_sequence_length = args.max_sequence_length;
    options.max_sequence_length_explicit = 1;
    options.backend_search_paths = args.backend_dirs;
    options.model_plugin_search_paths = args.model_plugin_dirs;
    auto pipeline = trtmc::load(args.bundle, options);
    auto result = run_request(*pipeline, args);
    auto* introspection = dynamic_cast<trtmc::IRuntimeMemoryIntrospectionV1*>(pipeline.get());
    if (introspection == nullptr || introspection->runtime_memory_api_version() != 1) {
        throw std::runtime_error("C++ surface did not expose runtime-memory introspection V1");
    }
    return success("cpp", *pipeline, result, introspection->runtime_memory_receipt_json());
}

json run_cabi(const Arguments& args) {
    if (!args.model_plugin_dirs.empty()) {
        const auto paths = join_paths(args.model_plugin_dirs);
        if (setenv("TRTMC_MODEL_PLUGIN_DIR", paths.c_str(), 1) != 0)
            throw std::runtime_error("failed to configure C-ABI model plugin path");
    }

    TrtmcPipelineOptionsV2 options;
    trtmc_pipeline_options_v2_init(&options);
    options.hf_python = args.hf_python.empty() ? nullptr : args.hf_python.c_str();
    options.kv_cache_memory_policy = TRTMC_KV_CACHE_MEMORY_BYTES;
    options.kv_cache_memory_bytes = args.kv_cache_bytes;
    options.max_sequence_length = args.max_sequence_length;
    options.max_sequence_length_explicit = 1;
    std::unique_ptr<trtmc::IPipeline> pipeline(
        trtmc_create_pipeline_v2(args.bundle.c_str(), &options));
    if (!pipeline) {
        throw std::runtime_error(std::string("C-ABI V2 create failed: ") + trtmc_last_error());
    }
    auto result = run_request(*pipeline, args);
    const char* receipt = trtmc_pipeline_runtime_memory_receipt_json(pipeline.get());
    if (receipt == nullptr) {
        throw std::runtime_error(std::string("C-ABI receipt failed: ") + trtmc_last_error());
    }
    return success("cabi", *pipeline, result, receipt);
}

} // namespace

int main(int argc, char** argv) {
    std::string surface = "unknown";
    try {
        const auto args = parse_arguments(argc, argv);
        surface = args.surface;
        const auto output = args.surface == "cpp" ? run_cpp(args) : run_cabi(args);
        std::cout << output.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cout << json{
                         {"status", "error"},
                         {"surface", surface},
                         {"message", error.what()},
                     }
                         .dump()
                  << '\n';
        return 1;
    }
}
