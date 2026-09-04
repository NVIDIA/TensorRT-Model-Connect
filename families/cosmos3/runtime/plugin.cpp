/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/cosmos3/runtime/distributed_runtime.h"
#include "families/cosmos3/runtime/pipeline.h"
#include "families/cosmos3/runtime/runtime_config.h"
#include "families/cosmos3/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::cosmos3 {
namespace {

const BundleSectionInfo& require_section(const BundleReader& reader, const char* name) {
    const auto* section = reader.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error(std::string("Cosmos3 bundle missing section '") + name + "'");
    return *section;
}

std::string read_text(const BundleReader& reader, const char* name) {
    const auto bytes = reader.read_section(name);
    if (bytes.empty())
        throw std::runtime_error(std::string("Cosmos3 bundle has empty section '") + name + "'");
    return {bytes.begin(), bytes.end()};
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleReader& reader) {
    const auto tokenizer_json = reader.read_section("tokenizer.json");
    (void)read_text(reader, "tokenizer_config.json");
    if (tokenizer_json.empty())
        throw std::runtime_error("Cosmos3 bundle has empty tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (tokenizer == nullptr)
        throw std::runtime_error("Cosmos3 tokenizer construction failed");
    constexpr std::array<std::pair<const char*, std::int32_t>, 3> expected = {{
        {"<|im_start|>", 151644},
        {"<|im_end|>", 151645},
        {"<|vision_start|>", 151652},
    }};
    for (const auto& [token, id] : expected) {
        if (tokenizer->id_for_token(token) != id)
            throw std::runtime_error(std::string("Cosmos3 tokenizer has an invalid ID for ") +
                                     token);
    }
    return tokenizer;
}

} // namespace

ITask* create(const FamilyContext& context) {
    if (context.reader.info().backend != "trt")
        throw std::runtime_error("Cosmos3 requires the TensorRT backend");
    for (const char* name : {"denoiser.plan", "vae.plan", "vae.first_frame.plan", "tokenizer.json",
                             "tokenizer_config.json", "runtime.json"}) {
        (void)require_section(context.reader, name);
    }

    const RuntimeConfig runtime = parse_runtime_config(read_text(context.reader, "runtime.json"));
    const DistributedRuntimeGroup group =
        initialize_context_parallel_group(runtime.context_parallel_size);
    auto tokenizer = load_tokenizer(context.reader);
    return new Cosmos3Pipeline(context.reader, context.backend, std::move(tokenizer), runtime,
                               group.communicator, group.owner, group.rank, group.world_size);
}

} // namespace trtmc::cosmos3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::cosmos3::create(context);
}
