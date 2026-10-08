/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/laya/runtime/record.h"

#include "families/laya/runtime/unicode.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace trtmc::laya {
namespace {
std::string python_json(const Json& value) {
    if (value.is_array() || value.is_object()) {
        std::string out = value.is_array() ? "[" : "{";
        bool first = true;
        for (const auto& item : value.items()) {
            if (!first)
                out += ", ";
            first = false;
            if (value.is_object())
                out += Json(item.key()).dump() + ": ";
            out += python_json(item.value());
        }
        return out + (value.is_array() ? "]" : "}");
    }
    auto out = value.dump();
    if (value.is_number_float()) {
        const bool negative = !out.empty() && out[0] == '-';
        auto digits = negative ? out.substr(1) : out;
        const auto e = digits.find('e');
        int exponent = e == std::string::npos ? 0 : std::stoi(digits.substr(e + 1));
        digits = digits.substr(0, e);
        const auto point = digits.find('.');
        exponent += static_cast<int>(point == std::string::npos ? digits.size() : point) - 1;
        if (point != std::string::npos)
            digits.erase(point, 1);
        while (digits.size() > 1 && digits.front() == '0') {
            digits.erase(0, 1);
            --exponent;
        }
        while (digits.size() > 1 && digits.back() == '0')
            digits.pop_back();
        if (digits == "0")
            return negative ? "-0.0" : "0.0";
        if (exponent < -4 || exponent >= 16) {
            out = digits.substr(0, 1);
            if (digits.size() > 1)
                out += "." + digits.substr(1);
            auto power = std::to_string(std::abs(exponent));
            if (power.size() < 2)
                power = "0" + power;
            out += std::string(exponent < 0 ? "e-" : "e+") + power;
        } else if (exponent < 0) {
            out = "0." + std::string(-exponent - 1, '0') + digits;
        } else if (static_cast<std::size_t>(exponent + 1) >= digits.size()) {
            out = digits + std::string(exponent + 1 - digits.size(), '0') + ".0";
        } else {
            out = digits;
            out.insert(exponent + 1, ".");
        }
        return negative ? "-" + out : out;
    }
    // Python uses two exponent digits in its shortest float representation.
    const auto e = out.find('e');
    if (e != std::string::npos && e + 3 == out.size() && (out[e + 1] == '+' || out[e + 1] == '-'))
        out.insert(e + 2, "0");
    return out;
}

std::string render(const Json& value) {
    return value.is_string() ? value.get<std::string>() : python_json(value);
}

std::string without_mask(std::string text, const std::string& mask) {
    for (std::size_t at = 0; (at = text.find(mask, at)) != std::string::npos;) {
        text.replace(at, mask.size(), " ");
        ++at;
    }
    return text;
}

std::string trim(std::string text) {
    return trim_unicode(text);
}

double rounded(double value) {
    return std::nearbyint(value * 10000.0) / 10000.0;
}
} // namespace

std::vector<Question> encode_record(const ITokenizer& tok, const Json& record, const Json& config) {
    if (!record.is_object() || !record.contains("state") || !record.contains("questions") ||
        !record["questions"].is_object())
        throw std::invalid_argument("Laya requires state and a questions object");
    if (record["questions"].empty())
        return {};
    const auto max_len = config.at("max_len").get<std::size_t>();
    const auto head_max = config.at("head_max_len").get<int>();
    const auto mask_id = config.at("mask_token_id").get<std::int32_t>();
    const auto mask = tok.token_for_id(mask_id);
    if (mask.empty())
        throw std::invalid_argument("Laya tokenizer has no mask token");
    const auto state = tok.encode(without_mask(render(record.at("state")), mask));
    std::vector<Question> result;
    for (const auto& item : record["questions"].items()) {
        const auto& q = item.value();
        if (!q.is_object() || !q.contains("type") || !q["type"].is_string() ||
            !q.contains("instructions"))
            throw std::invalid_argument("each Laya question requires type and instructions");
        const auto type = q["type"].get<std::string>();
        if (type != "choice" && type != "score" && type != "noul")
            throw std::invalid_argument("Laya question type must be choice, score, or noul");
        if (q.contains("labels") && type != "noul")
            throw std::invalid_argument("labels are only valid for noul questions");
        Question encoded{item.key(), type == "choice" ? 0 : type == "score" ? 1 : 2, {}, {}, {}};
        const auto criteria = q.value("criteria", Json());
        std::vector<std::string> descriptions;
        if (type == "choice") {
            if (!(criteria.is_array() || criteria.is_object()) || criteria.empty())
                throw std::invalid_argument("choice criteria must be a nonempty object or array");
            for (const auto& option : criteria.items()) {
                const auto key =
                    criteria.is_array() ? option.value().get<std::string>() : option.key();
                if (std::find(encoded.options.begin(), encoded.options.end(), key) !=
                    encoded.options.end())
                    continue; // The original list-to-dict conversion retains the first position.
                encoded.options.push_back(key);
                const auto value = criteria.is_array() ? Json() : option.value();
                descriptions.push_back(value.is_null() || value == "" ? key
                                                                      : key + ": " + render(value));
            }
        } else if (type == "score") {
            if (!criteria.is_array() || criteria.empty())
                throw std::invalid_argument("score criteria must be a nonempty array");
            for (std::size_t i = 0; i < criteria.size(); ++i) {
                encoded.options.push_back(std::to_string(i));
                descriptions.push_back("level " + std::to_string(i) + ": " + render(criteria[i]));
            }
        } else {
            if (!criteria.is_null() && !criteria.is_object())
                throw std::invalid_argument("noul criteria must be an object");
            Json normalized = Json::object();
            if (criteria.is_object())
                for (const auto& entry : criteria.items()) {
                    auto key = entry.key();
                    std::transform(key.begin(), key.end(), key.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    if (key != "false" && key != "true")
                        throw std::invalid_argument("unknown noul criterion");
                    normalized[key] = entry.value();
                }
            Json labels = q.value("labels", Json());
            if (labels.is_null())
                labels = Json{{"false", "false"}, {"true", "true"}};
            if (!labels.is_object() || labels.size() != 2 || !labels.contains("false") ||
                !labels.contains("true") || !labels["false"].is_string() ||
                !labels["true"].is_string())
                throw std::invalid_argument("noul labels must contain false and true strings");
            const auto a = trim(labels["false"].get<std::string>()),
                       b = trim(labels["true"].get<std::string>());
            if (a.empty() || b.empty() || a == b)
                throw std::invalid_argument("noul labels must be distinct and nonempty");
            for (const auto& key : {"false", "true"}) {
                encoded.options.push_back(key);
                const auto value = normalized.value(key, Json());
                const auto fallback = std::string(key) == "false"
                                          ? "no, the statement does not hold"
                                          : "yes, the statement holds";
                descriptions.push_back((std::string(key) == "false" ? a : b) + ": " +
                                       (value.is_null() || value == "" ? fallback : render(value)));
            }
        }
        auto head =
            tok.encode(type + " question: " + without_mask(render(q["instructions"]), mask));
        std::vector<std::vector<std::int32_t>> options;
        int option_tokens = 0;
        for (const auto& description : descriptions) {
            auto text = tok.encode(" " + without_mask(description, mask));
            text.resize(std::min<std::size_t>(48, text.size()));
            text.insert(text.begin(), mask_id);
            option_tokens += text.size();
            options.push_back(std::move(text));
        }
        int budget = head_max - option_tokens;
        if (budget < 16) {
            const auto per = static_cast<std::size_t>(
                std::max(4, (head_max - 16) / static_cast<int>(options.size())));
            option_tokens = 0;
            for (auto& option : options) {
                option.resize(std::min(per, option.size()));
                option_tokens += option.size();
            }
            budget = head_max - option_tokens;
        }
        head.resize(std::min(head.size(), static_cast<std::size_t>(std::max(8, budget))));
        auto& ids = encoded.tokens;
        ids.push_back(config.at("cls_token_id").get<std::int32_t>());
        ids.insert(ids.end(), head.begin(), head.end());
        ids.push_back(config.at("sep_token_id").get<std::int32_t>());
        for (const auto& option : options) {
            encoded.markers.push_back(ids.size());
            ids.insert(ids.end(), option.begin(), option.end());
        }
        ids.push_back(config.at("sep_token_id").get<std::int32_t>());
        const auto room = max_len > ids.size() + 1 ? max_len - ids.size() - 1 : 0;
        const auto count = std::min(room, state.size());
        const auto begin = record["state"].is_array() ? state.size() - count : 0;
        ids.insert(ids.end(), state.begin() + begin, state.begin() + begin + count);
        ids.push_back(config.at("sep_token_id").get<std::int32_t>());
        if (ids.size() > max_len)
            ids.resize(max_len);
        if (encoded.markers.back() >= static_cast<std::int32_t>(max_len))
            throw std::invalid_argument("question options exceed the Laya token budget");
        result.push_back(std::move(encoded));
    }
    return result;
}

double temperature(const Json& config, int type, std::size_t options) {
    const auto name = type == 0 ? "choice" : type == 1 ? "score" : "noul";
    const auto bucket = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+";
    const auto key = std::string(name) + ":" + bucket;
    const auto values = config.value("temperature", Json::array({1.0, 1.0, 1.0}));
    double value = values.at(type).get<double>();
    if (config.contains("temperature_by_options") && config["temperature_by_options"].contains(key))
        value = config["temperature_by_options"][key].get<double>();
    return std::isfinite(value) ? std::clamp(value, 0.5, 5.0) : 1.0;
}

Json format_answer(const Json& q, const std::vector<std::string>& options,
                   const std::vector<float>& p, float act_probability) {
    if (p.empty() || p.size() != options.size())
        throw std::invalid_argument("invalid Laya probabilities");
    const auto maximum = *std::max_element(p.begin(), p.end());
    double entropy = 0;
    for (auto value : p)
        entropy -= value * std::log(std::clamp<double>(value, 1e-12, 1.0));
    const auto confidence = p.size() < 2 ? 1.0 : 1.0 - entropy / std::log(p.size());
    const auto type = q["type"].get<std::string>();
    Json result{{"type", type}};
    if (type == "noul") {
        result["noul"] = rounded(p.at(1));
        result["confidence"] = rounded(std::max<double>(p[1], 1.0 - p[1]));
    } else {
        Json probabilities = Json::object();
        for (std::size_t i = 0; i < p.size(); ++i)
            probabilities[options[i]] = rounded(p[i]);
        if (type == "choice")
            result["choice"] = options[std::max_element(p.begin(), p.end()) - p.begin()];
        else {
            double score = 0;
            Json legend = Json::object();
            for (std::size_t i = 0; i < p.size(); ++i) {
                score += static_cast<double>(i) * static_cast<double>(p[i]);
                legend[options[i]] = q["criteria"][i];
            }
            result["score"] = rounded(score);
            result["legend"] = legend;
        }
        result["probabilities"] = probabilities;
        result["confidence"] = rounded(std::clamp(confidence, 0.0, 1.0));
    }
    result["answer_confidence"] = rounded(maximum);
    result["action"] = {{"act_probability", rounded(act_probability)}};
    return result;
}
} // namespace trtmc::laya
