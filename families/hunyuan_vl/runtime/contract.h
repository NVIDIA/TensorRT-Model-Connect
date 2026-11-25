/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "media.h"
#include "trtmc/runtime/tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::hunyuan_vl {
// The bundle's runtime.json, validated before anything is allocated or bound.
struct Metadata {
    DType dtype{};
    int hidden{}, layers{}, kv_heads{}, head_dim{}, vocab{}, capacity{}, max_input{};
    int max_patches{}, edge{};
    std::vector<float> inverse_frequencies;
    std::vector<int> sections;
    ImageSettings image;
    int32_t bos{}, system_end{}, user{}, assistant{}, image_token{}, image_start{}, image_end{};
    std::vector<int32_t> eos;
};

namespace contract {
inline std::invalid_argument invalid(const std::string& what) {
    return std::invalid_argument("Invalid HunYuanVL bundle: " + what);
}

// An integer field (never a float or a boolean) within [low, high].
inline int64_t integer(const nlohmann::json& value, const std::string& name, int64_t low,
                       int64_t high) {
    if (!value.is_number_integer())
        throw invalid(name + " is not an integer");
    if (value.is_number_unsigned() ? value.get<uint64_t>() > uint64_t(high)
                                   : value.get<int64_t>() > high)
        throw invalid(name + " is out of range");
    const auto number = value.get<int64_t>();
    if (number < low)
        throw invalid(name + " is out of range");
    return number;
}

// A finite number that a float holds.
inline float real(const nlohmann::json& value, const std::string& name) {
    if (!value.is_number())
        throw invalid(name + " is not a number");
    const double number = value.get<double>();
    if (!std::isfinite(number) || std::abs(number) > double(std::numeric_limits<float>::max()))
        throw invalid(name + " is not a finite float");
    return static_cast<float>(number);
}

// A finite float rounded to bfloat16 (nearest, ties to even) as __float2bfloat16_rn packs
// activations, widened back; values past the largest bfloat16 become infinite.
inline float bfloat16_round(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    bits = (bits + 0x7FFFU + ((bits >> 16) & 1U)) & 0xFFFF0000U;
    std::memcpy(&value, &bits, sizeof bits);
    return value;
}

inline int32_t token(const nlohmann::json& value, const std::string& name, int vocab) {
    return static_cast<int32_t>(integer(value, name, 0, int64_t(vocab) - 1));
}
} // namespace contract

// Validates every field the runtime reads: integer types and ranges, RoPE sections that are
// positive and cover head_dim / 2, RoPE angles that stay finite over the cache, image settings
// whose folded float32 normalization stays finite over the uint8 range (computed as patchify
// does), token ids inside the vocabulary, and an embedding table of exactly vocab x hidden floats.
inline Metadata parse_metadata(const nlohmann::json& json, size_t embedding_bytes) {
    using contract::integer;
    using contract::invalid;
    using contract::real;
    if (!json.is_object())
        throw invalid("runtime.json is not an object");
    integer(json.at("format"), "format", 1, 1);
    Metadata m;
    const auto& precision = json.at("precision");
    if (precision == "bf16")
        m.dtype = DType::kBFloat16;
    else if (precision == "fp32")
        m.dtype = DType::kFloat32;
    else
        throw invalid("precision is neither bf16 nor fp32");
    m.hidden = int(integer(json.at("hidden_size"), "hidden_size", 1, 1 << 16));
    m.layers = int(integer(json.at("num_hidden_layers"), "num_hidden_layers", 1, 1 << 10));
    m.kv_heads = int(integer(json.at("num_key_value_heads"), "num_key_value_heads", 1, 1 << 10));
    m.head_dim = int(integer(json.at("head_dim"), "head_dim", 2, 1 << 12));
    if (m.head_dim % 2 != 0)
        throw invalid("head_dim is odd");
    m.vocab = int(integer(json.at("vocab_size"), "vocab_size", 1, 1 << 24));
    m.capacity = int(integer(json.at("max_cache_length"), "max_cache_length", 1, 1 << 22));
    m.max_input = int(integer(json.at("max_input_length"), "max_input_length", 1, m.capacity));
    // Every cache tensor holds kv_heads x capacity x head_dim values of at most 4 bytes.
    if (int64_t(m.kv_heads) * m.capacity * m.head_dim > (int64_t(1) << 36))
        throw invalid("the KV cache is too large");
    m.image.patch = int(integer(json.at("patch_size"), "patch_size", 1, 1 << 10));
    m.image.merge = int(integer(json.at("merge_size"), "merge_size", 2, 2));
    m.max_patches = int(integer(json.at("max_patches"), "max_patches", 4, 1 << 24));
    if (m.max_patches % 4 != 0)
        throw invalid("max_patches is not a multiple of the merge block");
    m.edge = int(integer(json.at("position_edge"), "position_edge", 2, 1 << 12));

    const auto& sections = json.at("rope_sections");
    if (!sections.is_array() || sections.size() < 3 || sections.size() > 8)
        throw invalid("rope_sections must hold 3 to 8 sections");
    int64_t half = 0;
    for (const auto& section : sections) {
        m.sections.push_back(int(integer(section, "rope_sections", 1, m.head_dim / 2)));
        half += m.sections.back();
    }
    if (half * 2 != m.head_dim)
        throw invalid("rope_sections do not cover head_dim / 2");
    const auto& frequencies = json.at("rope_inverse_frequencies");
    if (!frequencies.is_array() || frequencies.size() != size_t(m.head_dim / 2))
        throw invalid("rope_inverse_frequencies must hold head_dim / 2 values");
    for (const auto& frequency : frequencies) {
        m.inverse_frequencies.push_back(real(frequency, "rope_inverse_frequencies"));
        if (!(m.inverse_frequencies.back() >= std::numeric_limits<float>::min()))
            throw invalid("rope_inverse_frequencies must be positive normal floats");
    }
    // The largest angle the runtime computes, in float: the last cache position times the largest
    // frequency.
    const float fastest =
        *std::max_element(m.inverse_frequencies.begin(), m.inverse_frequencies.end());
    if (!std::isfinite(float(m.capacity - 1) * fastest))
        throw invalid("RoPE angles overflow float within the cache");

    const auto& image = json.at("image");
    m.image.min_pixels = integer(image.at("min_pixels"), "min_pixels", 1, int64_t(1) << 40);
    m.image.max_pixels =
        integer(image.at("max_pixels"), "max_pixels", m.image.min_pixels, int64_t(1) << 40);
    const auto& mean = image.at("mean");
    const auto& deviation = image.at("std");
    if (!mean.is_array() || mean.size() != 3 || !deviation.is_array() || deviation.size() != 3)
        throw invalid("image mean and std must hold three values");
    const auto rescale = real(image.at("rescale_factor"), "rescale_factor");
    if (!(rescale >= std::numeric_limits<float>::min()))
        throw invalid("rescale_factor must be a positive normal float");
    m.image.rescale_factor = image.at("rescale_factor").get<double>();
    const double inverse = 1.0 / m.image.rescale_factor;
    for (size_t c = 0; c < 3; ++c) {
        m.image.mean[c] = real(mean[c], "mean");
        m.image.std[c] = real(deviation[c], "std");
        const float folded_mean = float(m.image.mean[c] * inverse);
        const float folded_std = float(m.image.std[c] * inverse);
        const float low = (0.0F - folded_mean) / folded_std;
        const float high = (255.0F - folded_mean) / folded_std;
        if (!std::isfinite(inverse) || !(m.image.std[c] >= std::numeric_limits<float>::min()) ||
            !std::isfinite(folded_mean) || !std::isfinite(folded_std) ||
            !(folded_std >= std::numeric_limits<float>::min()) || !std::isfinite(low) ||
            !std::isfinite(high))
            throw invalid("image mean, std and rescale_factor overflow or underflow float32");
        // A bfloat16 bundle packs the pixels as bfloat16.
        if (m.dtype == DType::kBFloat16 && (!std::isfinite(contract::bfloat16_round(low)) ||
                                            !std::isfinite(contract::bfloat16_round(high))))
            throw invalid("image mean, std and rescale_factor overflow bfloat16");
    }

    const auto& tokens = json.at("tokens");
    m.bos = contract::token(tokens.at("bos"), "bos", m.vocab);
    m.system_end = contract::token(tokens.at("system_end"), "system_end", m.vocab);
    m.user = contract::token(tokens.at("user"), "user", m.vocab);
    m.assistant = contract::token(tokens.at("assistant"), "assistant", m.vocab);
    m.image_token = contract::token(tokens.at("image"), "image", m.vocab);
    m.image_start = contract::token(tokens.at("image_start"), "image_start", m.vocab);
    m.image_end = contract::token(tokens.at("image_end"), "image_end", m.vocab);
    const auto& eos = tokens.at("eos");
    if (!eos.is_array() || eos.empty() || eos.size() > 16)
        throw invalid("eos must hold 1 to 16 token ids");
    for (const auto& id : eos)
        m.eos.push_back(contract::token(id, "eos", m.vocab));

    if (uint64_t(embedding_bytes) != uint64_t(m.hidden) * uint64_t(m.vocab) * sizeof(float))
        throw invalid("the embedding table does not hold vocab_size x hidden_size floats");
    return m;
}

// One tensor of a loaded engine as the backend reports it: a dynamic input by the smallest and
// largest shapes of optimization profile 0; a static input or an output by its shape.
struct EngineTensor {
    std::string name;
    DType dtype{};
    bool input{}, dynamic{};
    std::vector<int64_t> shape, profile_min, profile_max;
};

// What the runtime binds: the dtype and shape pattern; for a dynamic input the smallest and largest
// shapes it feeds, which profile 0 must accept; for an output whose rows follow the input rows, its
// shape at the largest inputs, where the backend sizes the output buffer (so the runtime's
// per-step copies of fewer rows stay inside it).
struct TensorContract {
    std::string name;
    DType dtype{};
    std::vector<int64_t> shape;
    bool input{};
    std::vector<int64_t> fed_min, fed_max;
};

inline std::vector<TensorContract> decoder_contract(const Metadata& m) {
    const int64_t keys = int64_t(m.capacity) + m.max_input;
    std::vector<TensorContract> result{
        {"inputs_embeds", m.dtype, {-1, m.hidden}, true, {1, m.hidden}, {m.max_input, m.hidden}},
        {"rope_cos",
         DType::kFloat32,
         {-1, m.head_dim},
         true,
         {1, m.head_dim},
         {m.max_input, m.head_dim}},
        {"rope_sin",
         DType::kFloat32,
         {-1, m.head_dim},
         true,
         {1, m.head_dim},
         {m.max_input, m.head_dim}},
        {"attention_mask",
         DType::kFloat32,
         {1, -1, -1},
         true,
         {1, 1, m.capacity + 1},
         {1, m.max_input, keys}},
        {"logits", DType::kFloat32, {1, m.vocab}, false, {}, {}},
    };
    for (int i = 0; i < m.layers; ++i)
        for (const char* kind : {"k_", "v_"}) {
            result.push_back({std::string("cache_") + kind + std::to_string(i),
                              m.dtype,
                              {m.kv_heads, m.capacity, m.head_dim},
                              true,
                              {},
                              {}});
            result.push_back({std::string("present_") + kind + std::to_string(i),
                              m.dtype,
                              {m.kv_heads, -1, m.head_dim},
                              false,
                              {},
                              {m.kv_heads, m.max_input, m.head_dim}});
        }
    return result;
}

inline std::vector<TensorContract> visual_contract(const Metadata& m) {
    const int64_t width = 3 * int64_t(m.image.patch) * m.image.patch;
    const int64_t merge = int64_t(m.image.merge) * m.image.merge;
    return {
        {"pixel_values", m.dtype, {-1, width}, true, {merge, width}, {m.max_patches, width}},
        {"position_ids", DType::kInt32, {-1, 4}, true, {merge, 4}, {m.max_patches, 4}},
        {"position_lambdas", DType::kFloat32, {-1, 4}, true, {merge, 4}, {m.max_patches, 4}},
        // One row per merged cell, then the row separator and the begin and end markers.
        {"image_features",
         m.dtype,
         {-1, m.hidden},
         false,
         {},
         {m.max_patches / merge + 3, m.hidden}},
    };
}

// Requires exactly the contract's tensors with its dtypes and fixed dimensions, a fixed shape where
// the runtime binds its own buffer, and a profile 0 that accepts every shape the runtime feeds.
inline void check_engine(const std::string& engine, const std::vector<EngineTensor>& tensors,
                         const std::vector<TensorContract>& expected) {
    const auto invalid = [&](const std::string& name, const char* what) {
        return contract::invalid(engine + " tensor " + name + " " + what);
    };
    if (tensors.size() != expected.size())
        throw contract::invalid(engine + " has " + std::to_string(tensors.size()) +
                                " I/O tensors, the runtime binds " +
                                std::to_string(expected.size()));
    for (const auto& want : expected) {
        const auto found = std::find_if(tensors.begin(), tensors.end(), [&](const auto& tensor) {
            return tensor.name == want.name;
        });
        if (found == tensors.end() || found->input != want.input)
            throw invalid(want.name, "is missing");
        if (found->dtype != want.dtype)
            throw invalid(want.name, "has another dtype");
        const size_t rank = want.shape.size();
        if (want.input) {
            const bool fed = !want.fed_max.empty();
            if (found->dynamic != fed)
                throw invalid(want.name, fed ? "is static, the runtime feeds several shapes"
                                             : "is dynamic, the runtime binds a fixed buffer");
            if (!fed) {
                if (found->shape != want.shape)
                    throw invalid(want.name, "has another shape");
                continue;
            }
            if (found->profile_min.size() != rank || found->profile_max.size() != rank)
                throw invalid(want.name, "has no profile shapes");
            for (size_t d = 0; d < rank; ++d) {
                const bool accepted = want.shape[d] < 0
                                          ? found->profile_min[d] <= want.fed_min[d] &&
                                                found->profile_max[d] >= want.fed_max[d]
                                          : found->profile_min[d] == want.shape[d] &&
                                                found->profile_max[d] == want.shape[d];
                if (!accepted)
                    throw invalid(want.name, "has a profile that rejects shapes the runtime feeds");
            }
            continue;
        }
        // The backend reports outputs at the largest inputs.
        const auto& at_largest = want.fed_max.empty() ? want.shape : want.fed_max;
        if (found->shape.size() != rank)
            throw invalid(want.name, "has another rank");
        for (size_t d = 0; d < rank; ++d)
            if (at_largest[d] >= 0 && found->shape[d] != at_largest[d])
                throw invalid(want.name, "has another shape at the largest inputs");
    }
}
} // namespace trtmc::hunyuan_vl
