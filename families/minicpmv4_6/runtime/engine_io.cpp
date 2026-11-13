/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minicpmv4_6/runtime/engine_io.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::minicpmv4_6 {
namespace {

struct Expected {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
};

std::string indexed(const char* stem, std::int32_t index) {
    return std::string(stem) + "_" + std::to_string(index);
}

// Each tensor's dtype, rank and static dimensions. The module reports a tensor at its current
// shape (a dynamic input at its profile's opt shape until a launch sets another, an output at the
// shape that follows), so a dimension expected as -1 is not compared: such an input must be
// dynamic, and its profile is checked separately (prefill_rows, validate_vision_engine).
void check_side(const ITrtModule& module, const std::vector<Expected>& expected, bool input,
                const char* engine) {
    const auto actual = input ? module.input_info() : module.output_info();
    if (actual.size() != expected.size())
        throw std::runtime_error(std::string("MiniCPM-V 4.6 ") + engine + " engine has " +
                                 std::to_string(actual.size()) +
                                 (input ? " inputs, expected " : " outputs, expected ") +
                                 std::to_string(expected.size()));
    for (const auto& tensor : expected) {
        bool matches = input ? module.has_input(tensor.name) : module.has_output(tensor.name);
        matches = matches && module.tensor_dtype(tensor.name) == tensor.dtype;
        const auto shape = matches ? module.tensor_shape(tensor.name) : std::vector<int64_t>{};
        matches = matches && shape.size() == tensor.shape.size();
        bool dynamic = false;
        for (std::size_t i = 0; matches && i < shape.size(); ++i) {
            dynamic = dynamic || tensor.shape[i] < 0;
            matches = tensor.shape[i] < 0 || shape[i] == tensor.shape[i];
        }
        if (matches && input)
            matches = module.input_is_dynamic(tensor.name) == dynamic;
        if (!matches)
            throw std::runtime_error(std::string("MiniCPM-V 4.6 ") + engine + " engine tensor " +
                                     tensor.name + " disagrees with the bundle metadata");
    }
}

void check_positive(std::int64_t value, const char* what) {
    if (value <= 0)
        throw std::invalid_argument(std::string("MiniCPM-V 4.6 ") + what + " must be positive");
}

} // namespace

void validate_decoder_engine(const ITrtModule& module, const DecoderGeometry& geometry,
                             bool prefill, std::int32_t vocab_size, std::int32_t hidden_size) {
    for (const auto value : {geometry.attention_layers, geometry.deltanet_layers, geometry.kv_width,
                             geometry.conv_width, geometry.conv_kernel, geometry.deltanet_heads,
                             geometry.deltanet_head_dim, vocab_size, hidden_size})
        check_positive(value, "decoder geometry");
    if (geometry.capacity <= 1)
        throw std::invalid_argument("MiniCPM-V 4.6 decoder capacity must exceed one row");
    const char* engine = prefill ? "prefill" : "decoder";
    const int64_t rows = prefill ? -1 : 1;
    const DType cache = geometry.cache_dtype;
    const std::vector<int64_t> conv{geometry.conv_width, geometry.conv_kernel};
    const std::vector<int64_t> ssm{geometry.deltanet_heads, geometry.deltanet_head_dim,
                                   geometry.deltanet_head_dim};
    const std::vector<int64_t> kv_cache{geometry.capacity, geometry.kv_width};
    const std::vector<int64_t> kv_rows{rows, geometry.kv_width};

    std::vector<Expected> inputs{
        {"token_id", DType::kInt32, {rows}},
        {"position_id", DType::kInt32, {rows}},
        {"attention_mask",
         DType::kFloat32,
         {rows, prefill ? int64_t{-1} : int64_t{geometry.capacity} + 1}},
    };
    if (prefill) {
        inputs.push_back({"visual_embedding", cache, {-1, hidden_size}});
        inputs.push_back({"use_visual_embedding", DType::kInt32, {-1, 1}});
    }
    std::vector<Expected> outputs{{"logits", DType::kFloat32, {1, vocab_size}}};
    for (std::int32_t i = 0; i < geometry.deltanet_layers; ++i) {
        inputs.push_back({indexed("conv_state", i), DType::kFloat32, conv});
        inputs.push_back({indexed("ssm_state", i), DType::kFloat32, ssm});
        outputs.push_back({indexed("present_conv", i), DType::kFloat32, conv});
        outputs.push_back({indexed("present_ssm", i), DType::kFloat32, ssm});
    }
    for (std::int32_t i = 0; i < geometry.attention_layers; ++i) {
        inputs.push_back({indexed("cache_k", i), cache, kv_cache});
        inputs.push_back({indexed("cache_v", i), cache, kv_cache});
        outputs.push_back({indexed("present_k", i), cache, kv_rows});
        outputs.push_back({indexed("present_v", i), cache, kv_rows});
    }
    check_side(module, inputs, true, engine);
    check_side(module, outputs, false, engine);
}

void validate_vision_engine(const ITrtModule& module, DType feature_dtype, std::int32_t patch_width,
                            std::int32_t hidden_size, std::int32_t max_patches) {
    check_positive(patch_width, "patch width");
    check_positive(hidden_size, "hidden size");
    if (max_patches < 16 || max_patches % 16)
        throw std::invalid_argument("MiniCPM-V 4.6 vision patch capacity must be a multiple of 16");
    check_side(module,
               {{"patches", DType::kFloat32, {-1, patch_width}},
                {"position_ids", DType::kInt32, {-1}},
                {"window_index", DType::kInt32, {-1}},
                {"merge_index", DType::kInt32, {-1}}},
               true, "vision");
    check_side(module, {{"visual_features", feature_dtype, {-1, hidden_size}}}, false, "vision");
    // A slice has 16 (one 4 x 4 block) to max_patches patches and a quarter as many merged rows;
    // the profile must admit both ends.
    const auto profile = module.profile_idx();
    const struct {
        const char* name;
        std::vector<int64_t> smallest, largest;
    } inputs[] = {
        {"patches", {16, patch_width}, {max_patches, patch_width}},
        {"position_ids", {16}, {max_patches}},
        {"window_index", {16}, {max_patches}},
        {"merge_index", {4}, {max_patches / 4}},
    };
    for (const auto& input : inputs) {
        const auto minimum =
            module.input_profile_shape(input.name, profile, ProfileShapeSelector::kMin);
        bool admitted = minimum.size() == input.smallest.size() &&
                        module.input_profile_shape(input.name, profile,
                                                   ProfileShapeSelector::kMax) == input.largest;
        for (std::size_t i = 0; admitted && i < minimum.size(); ++i)
            admitted = minimum[i] <= input.smallest[i];
        if (!admitted)
            throw std::runtime_error(std::string("MiniCPM-V 4.6 vision engine profile of ") +
                                     input.name + " does not admit slices of 16 to " +
                                     std::to_string(max_patches) + " patches");
    }
}

std::int32_t prefill_rows(const ITrtModule& prefill, const DecoderGeometry& geometry,
                          std::int32_t hidden_size) {
    const std::int64_t capacity = geometry.capacity;
    const auto profile = prefill.profile_idx();
    const auto rows = prefill.input_profile_shape("token_id", profile, ProfileShapeSelector::kMax);
    if (rows.size() != 1 || rows[0] < 1 || rows[0] > capacity ||
        prefill.input_profile_shape("token_id", profile, ProfileShapeSelector::kMin) !=
            std::vector<int64_t>{1})
        throw std::runtime_error("MiniCPM-V 4.6 prefill engine has an invalid row profile");
    // A chunk of r rows binds these shapes (DecoderState::prepare, Decoder::prefill); the profile
    // must admit them for every r from one to the most rows.
    const int64_t most = rows[0];
    const struct {
        const char* name;
        std::vector<int64_t> one, all;
    } chunks[] = {
        {"position_id", {1}, {most}},
        {"attention_mask", {1, capacity + 1}, {most, capacity + most}},
        {"visual_embedding", {1, hidden_size}, {most, hidden_size}},
        {"use_visual_embedding", {1, 1}, {most, 1}},
    };
    for (const auto& chunk : chunks) {
        const auto minimum =
            prefill.input_profile_shape(chunk.name, profile, ProfileShapeSelector::kMin);
        const auto maximum =
            prefill.input_profile_shape(chunk.name, profile, ProfileShapeSelector::kMax);
        bool admitted = minimum.size() == chunk.one.size() && maximum.size() == chunk.all.size();
        for (std::size_t i = 0; admitted && i < chunk.one.size(); ++i)
            admitted = minimum[i] <= chunk.one[i] && maximum[i] >= chunk.all[i];
        if (!admitted)
            throw std::runtime_error(std::string("MiniCPM-V 4.6 prefill engine profile of ") +
                                     chunk.name + " does not admit chunks of 1 to " +
                                     std::to_string(most) + " rows");
    }
    // The module sizes each output for the profile's largest inputs and reports that shape at load;
    // commit_prefill copies up to `most` rows of present K/V from these buffers.
    for (std::int32_t i = 0; i < geometry.attention_layers; ++i)
        for (const char* stem : {"present_k", "present_v"}) {
            const auto name = indexed(stem, i);
            const auto shape = prefill.tensor_shape(name);
            if (shape.size() != 2 || shape[0] < most || shape[1] != geometry.kv_width)
                throw std::runtime_error("MiniCPM-V 4.6 prefill engine output " + name +
                                         " does not hold a chunk of " + std::to_string(most) +
                                         " rows");
        }
    return static_cast<std::int32_t>(most);
}

} // namespace trtmc::minicpmv4_6
