/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/contract.h"

#include "families/esmfold2/runtime/protein.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace trtmc::esmfold2 {
namespace {

using Shape = std::vector<std::int64_t>;

// kMask is TensorRT bool, which the backend's DType cannot name: its shapes are checked, not its
// type.
enum class Kind { kCompute, kFloat, kInt, kMask };

struct Input {
    std::string name;
    Kind kind;
    Shape min;
    Shape max;
    // Declared without a dynamic dimension (no profile); every other input is checked against its
    // profile, also when the profile's ends coincide (a bundle built for two tokens).
    bool fixed{false};
};

struct Output {
    std::string name;
    Kind kind;
    Shape shape;
};

struct Contract {
    std::vector<Input> inputs;
    std::vector<Output> outputs;
};

std::string text(const Shape& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < shape.size(); ++index)
        stream << (index ? ", " : "") << shape[index];
    stream << ']';
    return stream.str();
}

Contract contractOf(const std::string& plan, const BundleConfig& config) {
    const std::int64_t tokens = config.max_tokens;
    const std::int64_t atoms = config.max_atoms;
    const std::int64_t pair = config.pair_size;
    const std::int64_t single = config.single_inputs_size;
    const std::int64_t width = config.atom_hidden_size;
    const std::int64_t head = config.atom_head_dim;
    const std::int64_t layers = config.diffusion_layers;
    const std::int64_t heads = config.diffusion_heads;
    // Outputs as allocated for the inputs' largest profile: every request up to the maximum fits in
    // them.
    const Shape pair_out{1, tokens, tokens, pair};
    auto token = [&](std::string name, Kind kind = Kind::kInt) {
        return Input{std::move(name), kind, {1, kMinTokens}, {1, tokens}};
    };
    auto tokenRow = [&](std::string name, Kind kind, std::int64_t channels) {
        return Input{std::move(name), kind, {1, kMinTokens, channels}, {1, tokens, channels}};
    };
    auto pairInput = [&](std::string name, Kind kind) {
        return Input{
            std::move(name), kind, {1, kMinTokens, kMinTokens, pair}, {1, tokens, tokens, pair}};
    };
    auto atom = [&](std::string name, Kind kind, Shape tail = {}) {
        Shape min{1, kAtomBlock};
        Shape max{1, atoms};
        min.insert(min.end(), tail.begin(), tail.end());
        max.insert(max.end(), tail.begin(), tail.end());
        return Input{std::move(name), kind, std::move(min), std::move(max)};
    };
    const std::vector<Input> atom_features = {
        atom("ref_pos", Kind::kFloat, {3}),      atom("ref_charge", Kind::kFloat),
        atom("atom_mask", Kind::kFloat),         atom("ref_element", Kind::kInt),
        atom("ref_name_chars", Kind::kInt, {4}), atom("ref_space_uid", Kind::kInt),
    };
    Contract contract;
    if (plan == "esmc") {
        contract.inputs = {{"input_ids", Kind::kInt, {1, kMinTokens + 2}, {1, tokens + 2}}};
        contract.outputs = {{"lm_single", Kind::kCompute, {1, tokens + 2, pair}}};
    } else if (plan == "embed") {
        contract.inputs = {tokenRow("lm_single", Kind::kCompute, pair),
                           token("res_type"),
                           token("residue_index"),
                           token("token_index"),
                           token("asym_id"),
                           token("sym_id"),
                           token("entity_id")};
        contract.inputs.insert(contract.inputs.end(), atom_features.begin(), atom_features.end());
        contract.inputs.push_back(atom("atom_to_token", Kind::kInt));
        contract.outputs = {{"single_inputs", Kind::kCompute, {1, tokens, single}},
                            {"initial_pair", Kind::kCompute, pair_out},
                            {"relpos", Kind::kCompute, pair_out},
                            {"lm_pair", Kind::kCompute, pair_out}};
    } else if (plan == "msa") {
        contract.inputs = {pairInput("initial_pair", Kind::kCompute),
                           tokenRow("single_inputs", Kind::kCompute, single), token("res_type")};
        contract.outputs = {{"msa_pair", Kind::kCompute, pair_out}};
    } else if (plan == "loop") {
        contract.inputs = {pairInput("state", Kind::kCompute),
                           pairInput("base_pair", Kind::kCompute),
                           pairInput("lm_pair", Kind::kCompute), pairInput("lm_keep", Kind::kMask)};
        contract.outputs = {{"next_state", Kind::kCompute, pair_out}};
    } else if (plan == "decode") {
        contract.inputs = {pairInput("state", Kind::kCompute)};
        contract.outputs = {{"pair_trunk", Kind::kFloat, pair_out}};
    } else if (plan == "conditioning") {
        contract.inputs = {pairInput("pair_trunk", Kind::kFloat),
                           pairInput("relpos", Kind::kCompute),
                           tokenRow("single_inputs", Kind::kCompute, single)};
        contract.inputs.insert(contract.inputs.end(), atom_features.begin(), atom_features.end());
        contract.outputs = {
            {"token_bias", Kind::kCompute, {layers, heads, tokens, tokens}},
            {"cond_single", Kind::kCompute, {1, tokens, config.diffusion_hidden_size}},
            {"atom_embeds", Kind::kCompute, {1, atoms, width}},
            {"rope_cos", Kind::kCompute, {1, atoms, head}},
            {"rope_sin", Kind::kCompute, {1, atoms, head}}};
    } else if (plan == "denoise") {
        contract.inputs = {
            {"noisy_coords", Kind::kFloat, {1, kAtomBlock, 3}, {config.max_samples, atoms, 3}},
            {"noise_level", Kind::kFloat, {1, 1}, {config.max_samples, 1}},
            {"token_bias",
             Kind::kCompute,
             {layers, heads, kMinTokens, kMinTokens},
             {layers, heads, tokens, tokens}},
            tokenRow("cond_single", Kind::kCompute, config.diffusion_hidden_size),
            atom("atom_embeds", Kind::kCompute, {width}),
            atom("rope_cos", Kind::kCompute, {head}),
            atom("rope_sin", Kind::kCompute, {head}),
            atom("atom_mask", Kind::kFloat),
            atom("atom_to_token", Kind::kInt)};
        contract.outputs = {{"denoised", Kind::kFloat, {config.max_samples, atoms, 3}}};
    } else if (plan == "confidence") {
        contract.inputs = {
            tokenRow("single_inputs", Kind::kCompute, single),
            pairInput("pair_trunk", Kind::kFloat),
            pairInput("relpos", Kind::kCompute),
            atom("coords", Kind::kFloat, {3}),
            token("rep_atom"),
            atom("atom_to_token", Kind::kInt),
            atom("intra_index", Kind::kInt),
            {"tm_per_bin", Kind::kFloat, {config.pae_bins}, {config.pae_bins}, true}};
        contract.outputs = {{"plddt_atom", Kind::kFloat, {1, atoms}},
                            {"ptm_rows", Kind::kFloat, {1, tokens}}};
    } else {
        throw std::invalid_argument("ESMFold2 bundle has no plan named " + plan);
    }
    return contract;
}

DType dtypeOf(Kind kind, const BundleConfig& config) {
    switch (kind) {
    case Kind::kCompute:
        return config.bf16 ? DType::kBFloat16 : DType::kFloat32;
    case Kind::kInt:
        return DType::kInt32;
    default:
        return DType::kFloat32;
    }
}

const TensorFacts& factsFor(const std::string& plan, const PlanFacts& facts,
                            const std::string& name, bool input) {
    const auto found = facts.find(name);
    if (found == facts.end() || found->second.input != input)
        throw std::invalid_argument("ESMFold2 plan " + plan + " has no " +
                                    (input ? "input " : "output ") + name);
    return found->second;
}

} // namespace

void check_plan_contract(const std::string& plan, const PlanFacts& facts,
                         const BundleConfig& config) {
    const Contract contract = contractOf(plan, config);
    if (facts.size() != contract.inputs.size() + contract.outputs.size())
        throw std::invalid_argument(
            "ESMFold2 plan " + plan + " has " + std::to_string(facts.size()) +
            " I/O tensors, expected " +
            std::to_string(contract.inputs.size() + contract.outputs.size()));
    auto fail = [&](const std::string& name, const std::string& what) {
        throw std::invalid_argument("ESMFold2 plan " + plan + " tensor " + name + ": " + what);
    };
    for (const Input& expected : contract.inputs) {
        const TensorFacts& actual = factsFor(plan, facts, expected.name, true);
        if (expected.kind != Kind::kMask && actual.dtype != dtypeOf(expected.kind, config))
            fail(expected.name, "element type does not match runtime.json");
        if (expected.fixed) {
            if (actual.shape != expected.max ||
                (!actual.max_profile.empty() && actual.max_profile != expected.max))
                fail(expected.name,
                     "shape " + text(actual.shape) + ", expected " + text(expected.max));
        } else if (actual.min_profile != expected.min || actual.max_profile != expected.max) {
            fail(expected.name, "profile " + text(actual.min_profile) + " to " +
                                    text(actual.max_profile) + ", expected " + text(expected.min) +
                                    " to " + text(expected.max));
        }
    }
    for (const Output& expected : contract.outputs) {
        const TensorFacts& actual = factsFor(plan, facts, expected.name, false);
        if (actual.dtype != dtypeOf(expected.kind, config))
            fail(expected.name, "element type does not match runtime.json");
        if (actual.shape != expected.shape)
            fail(expected.name,
                 "shape " + text(actual.shape) + ", expected " + text(expected.shape));
    }
}

} // namespace trtmc::esmfold2
