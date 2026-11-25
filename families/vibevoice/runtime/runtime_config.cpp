/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/vibevoice/runtime/runtime_config.h"

#include "families/vibevoice/runtime/scheduler.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace trtmc::vibevoice {
namespace {

constexpr std::int64_t kInt32Max = std::numeric_limits<std::int32_t>::max();

[[noreturn]] void invalid(const std::string& what) {
    throw std::runtime_error("vibevoice runtime.json " + what);
}

// nlohmann keeps the last of two equal keys; the bundle is refused instead.
nlohmann::json parse_unique_keys(const std::string& text) {
    std::vector<std::set<std::string>> open;
    const nlohmann::json::parser_callback_t callback =
        [&open](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
            if (event == nlohmann::json::parse_event_t::object_start) {
                open.emplace_back();
            } else if (event == nlohmann::json::parse_event_t::object_end) {
                open.pop_back();
            } else if (event == nlohmann::json::parse_event_t::key &&
                       !open.back().insert(parsed.get<std::string>()).second) {
                invalid("repeats the key " + parsed.get<std::string>());
            }
            return true;
        };
    return nlohmann::json::parse(text, callback);
}

// A JSON integer in [low, high], checked before any conversion.
std::int64_t integer(const nlohmann::json& value, const std::string& name, std::int64_t low,
                     std::int64_t high) {
    if (!value.is_number_integer())
        invalid(name + " must be a JSON integer");
    if (value.is_number_unsigned()) {
        const auto number = value.get<std::uint64_t>();
        if (number > static_cast<std::uint64_t>(high) || static_cast<std::int64_t>(number) < low)
            invalid(name + " lies outside [" + std::to_string(low) + ", " + std::to_string(high) +
                    "]");
        return static_cast<std::int64_t>(number);
    }
    const auto number = value.get<std::int64_t>();
    if (number < low || number > high)
        invalid(name + " lies outside [" + std::to_string(low) + ", " + std::to_string(high) + "]");
    return number;
}

std::int32_t positive_int32(const nlohmann::json& json, const char* key) {
    return static_cast<std::int32_t>(integer(json.at(key), key, 1, kInt32Max));
}

std::int32_t token_id(const nlohmann::json& value, const std::string& name, std::int32_t vocab) {
    return static_cast<std::int32_t>(integer(value, name, 0, std::int64_t{vocab} - 1));
}

// A finite JSON number that float32 holds without overflow.
double finite(const nlohmann::json& value, const std::string& name) {
    if (!value.is_number())
        invalid(name + " must be a number");
    const auto number = value.get<double>();
    if (!std::isfinite(number) || std::fabs(number) > std::numeric_limits<float>::max())
        invalid(name + " must be a finite float32");
    return number;
}

std::vector<float> float_list(const nlohmann::json& value, const std::string& name) {
    if (!value.is_array())
        invalid(name + " must be a list");
    std::vector<float> result;
    result.reserve(value.size());
    for (const auto& item : value)
        result.push_back(static_cast<float>(finite(item, name + " item")));
    return result;
}

std::string text(const nlohmann::json& json, const char* key) {
    const auto& value = json.at(key);
    if (!value.is_string())
        invalid(std::string(key) + " must be a string");
    return value.get<std::string>();
}

// The product of sizes the runtime allocates, refused when it overflows int64.
std::int64_t checked_product(std::initializer_list<std::int64_t> factors, const char* what) {
    std::int64_t product = 1;
    for (const auto factor : factors) {
        if (factor != 0 && product > std::numeric_limits<std::int64_t>::max() / factor)
            invalid(std::string(what) + " overflows the runtime's sizes");
        product *= factor;
    }
    return product;
}

RuntimeConfig parse(const std::string& json_text) {
    const auto json = parse_unique_keys(json_text);
    if (!json.is_object())
        invalid("is not a JSON object");
    RuntimeConfig config;
    const auto precision = text(json, "precision");
    if (precision != "bf16" && precision != "fp32")
        invalid("precision must be bf16 or fp32");
    config.bfloat16 = precision == "bf16";
    config.hidden_size = positive_int32(json, "hidden_size");
    config.num_layers = positive_int32(json, "num_layers");
    config.num_kv_heads = positive_int32(json, "num_kv_heads");
    config.head_dim = positive_int32(json, "head_dim");
    config.vocab_size = positive_int32(json, "vocab_size");
    config.capacity = positive_int32(json, "capacity");
    config.prefill_chunk = positive_int32(json, "prefill_chunk");
    if (config.head_dim % 2 != 0 || config.prefill_chunk > config.capacity)
        invalid("geometry is inconsistent (odd head_dim or prefill_chunk above capacity)");

    // RoPE: the runtime forms position * inv_freq in float32 for positions below capacity.
    config.rope_inv_freq = float_list(json.at("rope_inv_freq"), "rope_inv_freq");
    if (static_cast<std::int64_t>(config.rope_inv_freq.size()) * 2 != config.head_dim ||
        std::any_of(config.rope_inv_freq.begin(), config.rope_inv_freq.end(),
                    [](float value) { return !(value > 0.0F && value <= 1.0F); }))
        invalid("rope_inv_freq needs head_dim / 2 values in (0, 1]");

    const auto& control = json.at("control_token_ids");
    if (!control.is_array() || control.size() != 4)
        invalid("control_token_ids must list four token ids");
    for (const auto& id : control)
        config.control_token_ids.push_back(
            token_id(id, "control_token_ids item", config.vocab_size));
    config.eos_token_id = token_id(json.at("eos_token_id"), "eos_token_id", config.vocab_size);
    config.audio_bos_token_id =
        token_id(json.at("audio_bos_token_id"), "audio_bos_token_id", config.vocab_size);
    config.audio_eos_token_id =
        token_id(json.at("audio_eos_token_id"), "audio_eos_token_id", config.vocab_size);
    config.audio_token_id =
        token_id(json.at("audio_token_id"), "audio_token_id", config.vocab_size);
    std::vector<std::int32_t> expected = {config.eos_token_id, config.audio_bos_token_id,
                                          config.audio_eos_token_id, config.audio_token_id};
    std::sort(expected.begin(), expected.end());
    if (std::adjacent_find(expected.begin(), expected.end()) != expected.end() ||
        config.control_token_ids != expected)
        invalid("control tokens must be the four distinct token ids in ascending order");

    config.frame.system_prompt = text(json, "system_prompt");
    config.frame.voice_input_header = text(json, "voice_input_header");
    config.frame.text_input_header = text(json, "text_input_header");
    config.frame.speech_output_header = text(json, "speech_output_header");
    config.frame.audio_bos_token = text(json, "audio_bos_token");
    config.frame.audio_eos_token = text(json, "audio_eos_token");
    config.frame.audio_token = text(json, "audio_token");
    if (config.frame.audio_bos_token.empty() || config.frame.audio_eos_token.empty() ||
        config.frame.audio_token.empty())
        invalid("audio tokens must be nonempty");

    config.sample_rate = positive_int32(json, "sample_rate");
    config.hop_length = positive_int32(json, "hop_length");
    config.latent_size = positive_int32(json, "latent_size");
    // Stored as float32, as prepare_voice uses them.
    config.target_dbfs = static_cast<float>(finite(json.at("target_dbfs"), "target_dbfs"));
    const double normalize_eps = finite(json.at("normalize_eps"), "normalize_eps");
    if (!usable_gain(config.target_dbfs, normalize_eps))
        invalid("target_dbfs and normalize_eps give no finite nonzero loudness gain");
    config.normalize_eps = static_cast<float>(normalize_eps);
    config.max_voice_samples = positive_int32(json, "max_voice_samples");
    if (config.max_voice_samples % config.hop_length != 0)
        invalid("max_voice_samples must be whole frames of hop_length");
    config.stream_state_size = positive_int32(json, "stream_state_size");

    // Sigmas of the training schedule, increasing with the timestep; the solver's
    // log-SNR steps divide by their differences.
    config.train_sigmas = float_list(json.at("train_sigmas"), "train_sigmas");
    if (config.train_sigmas.empty() ||
        static_cast<std::int64_t>(config.train_sigmas.size()) > kInt32Max ||
        !(config.train_sigmas.front() > 0.0F) ||
        std::adjacent_find(config.train_sigmas.begin(), config.train_sigmas.end(),
                           [](float lower, float upper) { return !(lower < upper); }) !=
            config.train_sigmas.end())
        invalid("train_sigmas must be positive and strictly increasing");
    config.default_guidance_scale =
        finite(json.at("default_guidance_scale"), "default_guidance_scale");
    if (config.default_guidance_scale < 0.0)
        invalid("default_guidance_scale must not be negative");
    config.default_diffusion_steps =
        integer(json.at("default_diffusion_steps"), "default_diffusion_steps", 1,
                static_cast<std::int64_t>(config.train_sigmas.size()));
    try {
        check_schedule(config.train_sigmas,
                       static_cast<std::int32_t>(config.default_diffusion_steps));
    } catch (const std::invalid_argument& error) {
        invalid(std::string("default_diffusion_steps: ") + error.what());
    }

    // Everything the pipeline allocates from these sizes.
    const std::int64_t element = config.bfloat16 ? 2 : 4;
    checked_product(
        {config.num_layers, 2, config.capacity, config.num_kv_heads, config.head_dim, element},
        "the KV cache");
    checked_product({config.capacity, config.hidden_size, element}, "the prompt buffer");
    checked_product({config.prefill_chunk, config.head_dim, 4}, "the rotary tables");
    checked_product({config.stream_state_size, element}, "the stream state");
    checked_product({config.max_voice_samples / config.hop_length, config.latent_size, 4},
                    "the voice-prompt noise");
    return config;
}

} // namespace

bool usable_gain(double target_dbfs, double eps) {
    if (!std::isfinite(target_dbfs) || !(eps >= std::numeric_limits<float>::min()) ||
        !(eps <= std::numeric_limits<float>::max()))
        return false;
    const float eps_float = static_cast<float>(eps);
    const float amplitude = static_cast<float>(std::pow(10.0, target_dbfs / 20.0));
    const float silence_gain = amplitude / eps_float;
    const float loudest = std::numeric_limits<float>::max() + eps_float;
    return amplitude >= std::numeric_limits<float>::min() && std::isfinite(silence_gain) &&
           silence_gain >= std::numeric_limits<float>::min() && std::isfinite(loudest) &&
           amplitude / loudest > 0.0F;
}

RuntimeConfig parse_runtime_config(const std::string& json_text) {
    try {
        return parse(json_text);
    } catch (const nlohmann::json::exception& error) {
        // A missing field or a value of the wrong JSON type.
        throw std::runtime_error(std::string("vibevoice runtime.json is malformed: ") +
                                 error.what());
    }
}

} // namespace trtmc::vibevoice
