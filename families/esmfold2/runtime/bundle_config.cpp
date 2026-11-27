/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/bundle_config.h"

#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::esmfold2 {
namespace {

using Json = nlohmann::json;

const Json& member(const Json& object, const char* key) {
    const auto found = object.find(key);
    if (found == object.end())
        throw std::invalid_argument(std::string("ESMFold2 runtime.json is missing ") + key);
    return *found;
}

std::int32_t integer(const Json& object, const char* key, std::int64_t minimum,
                     std::int64_t maximum) {
    const Json& value = member(object, key);
    if (!value.is_number_integer())
        throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                    " must be an integer");
    std::int64_t number = 0;
    if (value.is_number_unsigned()) {
        const auto unsigned_number = value.get<std::uint64_t>();
        if (unsigned_number > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                        " is out of range");
        number = static_cast<std::int64_t>(unsigned_number);
    } else {
        number = value.get<std::int64_t>();
    }
    if (number < minimum || number > maximum)
        throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                    " is out of range");
    return static_cast<std::int32_t>(number);
}

double real(const Json& object, const char* key, double minimum, double maximum) {
    const Json& value = member(object, key);
    if (!value.is_number())
        throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                    " must be a number");
    const double number = value.get<double>();
    if (!std::isfinite(number) || number < minimum || number > maximum)
        throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                    " is out of range");
    return number;
}

bool boolean(const Json& object, const char* key) {
    const Json& value = member(object, key);
    if (!value.is_boolean())
        throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                    " must be a boolean");
    return value.get<bool>();
}

void expect(std::int32_t actual, std::int32_t expected, const char* key) {
    if (actual != expected)
        throw std::invalid_argument(std::string("ESMFold2 runtime.json ") + key +
                                    " does not match the plans");
}

Json parseWithoutDuplicateKeys(std::string_view text) {
    std::vector<std::set<std::string>> scopes;
    bool duplicate = false;
    const Json::parser_callback_t callback = [&](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start)
            scopes.emplace_back();
        else if (event == Json::parse_event_t::object_end && !scopes.empty())
            scopes.pop_back();
        else if (event == Json::parse_event_t::key && !scopes.empty() &&
                 !scopes.back().insert(parsed.get<std::string>()).second)
            duplicate = true;
        return true;
    };
    Json root = Json::parse(text.begin(), text.end(), callback);
    if (duplicate)
        throw std::invalid_argument("ESMFold2 runtime.json repeats a key");
    return root;
}

} // namespace

BundleConfig parse_bundle_config(std::string_view text) {
    const Json root = parseWithoutDuplicateKeys(text);
    if (!root.is_object())
        throw std::invalid_argument("ESMFold2 runtime.json must hold an object");
    expect(integer(root, "format", 1, 1), 1, "format");
    BundleConfig config;
    const Json& precision = member(root, "precision");
    if (!precision.is_string() || (precision != "bf16" && precision != "fp32"))
        throw std::invalid_argument("ESMFold2 runtime.json precision must be bf16 or fp32");
    config.bf16 = precision == "bf16";
    constexpr std::int64_t kLarge = 1 << 20;
    config.max_tokens = integer(root, "max_tokens", kMinTokens, 4094);
    config.max_atoms = integer(root, "max_atoms", 32, kLarge);
    if (config.max_atoms % 32 != 0 || config.max_atoms < config.max_tokens * 4)
        throw std::invalid_argument(
            "ESMFold2 runtime.json max_atoms is inconsistent with max_tokens");
    config.max_samples = integer(root, "max_samples", 1, 1024);
    config.pair_size = integer(root, "pair_size", 1, 4096);
    config.single_inputs_size = integer(root, "single_inputs_size", 1, 8192);
    config.diffusion_hidden_size = integer(root, "diffusion_hidden_size", 1, 8192);
    config.diffusion_layers = integer(root, "diffusion_layers", 1, 256);
    config.diffusion_heads = integer(root, "diffusion_heads", 1, 256);
    config.atom_hidden_size = integer(root, "atom_hidden_size", 1, 4096);
    config.atom_head_dim = integer(root, "atom_head_dim", 2, 1024);
    config.max_atoms_per_token = integer(root, "max_atoms_per_token", 14, 64);
    config.pae_bins = integer(root, "pae_bins", 2, 4096);
    config.msa_encoder = boolean(root, "msa_encoder");
    config.lm_dropout = real(root, "lm_dropout", 0.0, 0.999);
    config.trunk_init_std = real(root, "trunk_init_std", 1.0e-6, 10.0);
    config.esmc_bos_id = integer(root, "esmc_bos_id", 0, 1 << 16);
    config.esmc_eos_id = integer(root, "esmc_eos_id", 0, 1 << 16);
    config.num_diffusion_samples = integer(root, "num_diffusion_samples", 1, config.max_samples);
    config.confidence_eps = real(root, "confidence_eps", 1.0e-30, 1.0);
    const Json& sampler = member(root, "sampler");
    if (!sampler.is_object())
        throw std::invalid_argument("ESMFold2 runtime.json sampler must be an object");
    config.sampler.sigma_data = real(sampler, "sigma_data", 1.0e-6, 1.0e6);
    config.sampler.gamma_0 = real(sampler, "gamma_0", 0.0, 1.0e3);
    config.sampler.gamma_min = real(sampler, "gamma_min", 0.0, 1.0e6);
    config.sampler.noise_scale = real(sampler, "noise_scale", 0.0, 1.0e3);
    config.sampler.step_scale = real(sampler, "step_scale", 0.0, 1.0e3);
    config.sampler.sigma_max_ratio = real(sampler, "sigma_max_ratio", 1.0e-6, 1.0e6);
    config.sampler.sigma_min_ratio = real(sampler, "sigma_min_ratio", 1.0e-12, 1.0e6);
    config.sampler.exponent = real(sampler, "exponent", 1.0e-3, 1.0e3);
    config.sampler.sigma_cap = real(sampler, "sigma_cap", 1.0e-6, 1.0e9);
    config.num_sampling_steps = integer(sampler, "num_steps", 1, 10000);
    if (config.sampler.sigma_min_ratio >= config.sampler.sigma_max_ratio)
        throw std::invalid_argument("ESMFold2 runtime.json noise schedule bounds are inverted");
    return config;
}

} // namespace trtmc::esmfold2
