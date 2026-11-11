/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/phonon/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::phonon {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* info = reader.find_section(name);
    if (!info || info->length == 0)
        throw std::runtime_error(std::string("Phonon missing bundle section: ") + name);
    return reader.read_section(name);
}
std::vector<float> floats(const BundleReader& reader, const char* name, std::size_t count) {
    const auto bytes = section(reader, name);
    if (bytes.size() != count * sizeof(float))
        throw std::runtime_error(std::string("Phonon invalid bundle asset: ") + name);
    std::vector<float> values(count);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    for (float v : values)
        if (!std::isfinite(v))
            throw std::runtime_error("Phonon bundle asset contains nonfinite data");
    return values;
}
} // namespace
} // namespace trtmc::phonon

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    using namespace trtmc::phonon;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("phonon does not support --kv-cache-size");
    if (context.reader.info().task != "text_to_speech")
        throw std::invalid_argument("Phonon requires text_to_speech");
    const auto data = section(context.reader, "runtime.json");
    const auto config = nlohmann::json::parse(data.begin(), data.end());
    const auto maximum = config.at("max_sequence_length").get<std::int64_t>();
    const auto voice_frames = config.at("voice_frames").get<std::int64_t>();
    if (voice_frames <= 0 || maximum <= voice_frames + 2 || maximum > 8192 ||
        config.at("sample_rate") != 24000 || config.at("samples_per_frame") != 1920 ||
        config.at("tensor_parallel_size") != 1)
        throw std::runtime_error("Phonon invalid runtime geometry");
    Assets assets{floats(context.reader, "text.embed", 4001 * 512),
                  floats(context.reader, "voice.embed", voice_frames * 512),
                  floats(context.reader, "bos.embed", 512),
                  floats(context.reader, "latent.proj", 512 * 32),
                  floats(context.reader, "condition.embed", 512),
                  maximum};
    auto load = [&](const char* name) {
        const auto plan = section(context.reader, name);
        auto module = context.backend.create_module(plan.data(), plan.size(), {});
        if (!module || !module->ok())
            throw std::runtime_error(std::string("Phonon cannot load engine: ") + name);
        module->set_timing_label(name);
        return module;
    };
    const auto tok = section(context.reader, "tokenizer.json");
    return new Pipeline(load("backbone.plan"), load("flow.plan"), load("codec.plan"),
                        std::move(assets), Tokenizer(std::string(tok.begin(), tok.end())));
}
