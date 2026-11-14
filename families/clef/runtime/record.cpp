/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/clef/runtime/record.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace trtmc::clef {
namespace {
std::string render(const Json& value) {
    // Upstream preserves strings and serializes other values with sorted keys,
    // compact separators, and literal UTF-8. Preserve question insertion order
    // independently of the canonical JSON inside tokenized descriptions.
    return value.is_string() ? value.get<std::string>() : nlohmann::json(value).dump();
}

double rounded(double value) {
    return std::nearbyint(value * 10000.0) / 10000.0;
}

std::vector<std::pair<std::string, Json>> options(const Json& question) {
    const std::string type = question.at("type").get<std::string>();
    if (type == "noul") {
        Json criteria = {{"true", "The proposition is true or the answer is yes."},
                         {"false", "The proposition is false or the answer is no."}};
        if (question.contains("criteria") && !question["criteria"].is_null()) {
            if (!question["criteria"].is_object())
                throw std::invalid_argument("noul criteria must be an object");
            criteria.update(question["criteria"]);
        }
        return {{"true", criteria["true"]}, {"false", criteria["false"]}};
    }
    const auto& criteria = question.at("criteria");
    if (criteria.empty())
        throw std::invalid_argument("criteria must not be empty");
    std::vector<std::pair<std::string, Json>> result;
    if (type == "choice" && criteria.is_object()) {
        for (const auto& item : criteria.items())
            result.emplace_back(item.key(), item.value());
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    } else if (type == "score" && criteria.is_array()) {
        for (std::size_t i = 0; i < criteria.size(); ++i)
            result.emplace_back(std::to_string(i), criteria[i]);
    } else {
        throw std::invalid_argument(
            "choice criteria must be an object; score criteria must be an array");
    }
    return result;
}
} // namespace

Record encode_record(const ITokenizer& tokenizer, const Json& document, std::int32_t max_length,
                     std::int32_t max_state_tokens, const std::vector<std::int32_t>& media_ids) {
    if (max_length < 1 || max_state_tokens < -1)
        throw std::invalid_argument("invalid input token limit");
    if (!document.is_object() || !document.contains("state") || !document.contains("questions") ||
        !document["questions"].is_object() || document["questions"].empty())
        throw std::invalid_argument("state and at least one question are required");
    auto append = [&](std::vector<std::int32_t>& ids, const std::string& text) {
        const auto tokens = tokenizer.encode(text);
        ids.insert(ids.end(), tokens.begin(), tokens.end());
    };
    Record result;
    std::vector<std::int32_t> schema;
    append(schema, "\n\nSCHEMA FIELDS:\n");
    for (const auto& item : document["questions"].items()) {
        const auto& q = item.value();
        const auto type = q.at("type").get<std::string>();
        if (type != "noul" && type != "choice" && type != "score")
            throw std::invalid_argument("type must be noul, choice, or score");
        append(schema, "\nFIELD " + std::to_string(result.questions.size() + 1) +
                           "\nID: " + item.key() + "\nTYPE: " + type + "\nINSTRUCTION: ");
        Question question{item.key(),
                          type == "noul"     ? 0
                          : type == "choice" ? 1
                                             : 2,
                          {static_cast<std::int32_t>(schema.size()), 0},
                          {},
                          {}};
        const auto instructions = q.value("instructions", Json());
        append(schema,
               instructions.is_null() || instructions == "" ? item.key() : render(instructions));
        question.span.second = schema.size();
        if (question.span.first == question.span.second)
            throw std::invalid_argument("question instruction must have at least one token");
        append(schema, "\nALLOWED OPTIONS:\n");
        for (const auto& [id, description] : options(q)) {
            append(schema, "OPTION " + std::to_string(question.option_ids.size() + 1) + ": ");
            const auto start = static_cast<std::int32_t>(schema.size());
            Json semantics = {{"option_id", id}};
            if (!description.is_null())
                semantics["description"] = description;
            append(schema, render(semantics));
            question.option_spans.emplace_back(start, schema.size());
            question.option_ids.push_back(id);
            append(schema, "\n");
        }
        append(schema, "END FIELD\n");
        result.questions.push_back(std::move(question));
    }
    append(result.input_ids,
           "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. "
           "Each answer must be exactly one of that field's allowed options."
           "<|im_end|>\n<|im_start|>user\nSTATE:\n");
    result.input_ids.insert(result.input_ids.end(), media_ids.begin(), media_ids.end());
    auto state = tokenizer.encode(render(document.at("state")));
    const auto suffix = tokenizer.encode(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    const auto fixed = result.input_ids.size() + schema.size() + suffix.size();
    if (fixed > static_cast<std::size_t>(max_length))
        throw std::invalid_argument("schema exceeds maximum input length before state");
    auto state_size = std::min(state.size(), static_cast<std::size_t>(max_length) - fixed);
    if (max_state_tokens >= 0)
        state_size = std::min(state_size, static_cast<std::size_t>(max_state_tokens));
    result.input_ids.insert(result.input_ids.end(), state.begin(), state.begin() + state_size);
    const auto offset = result.input_ids.size();
    for (auto& q : result.questions) {
        q.span.first += offset;
        q.span.second += offset;
        for (auto& span : q.option_spans) {
            span.first += offset;
            span.second += offset;
        }
    }
    result.input_ids.insert(result.input_ids.end(), schema.begin(), schema.end());
    result.input_ids.insert(result.input_ids.end(), suffix.begin(), suffix.end());
    return result;
}

Json systemone_answer(const Json& question, const std::vector<std::string>& option_ids,
                      const std::vector<float>& probabilities) {
    if (option_ids.empty() || option_ids.size() != probabilities.size())
        throw std::invalid_argument("invalid option probabilities");
    std::map<std::string, double> p;
    for (std::size_t i = 0; i < option_ids.size(); ++i) {
        if (!std::isfinite(probabilities[i]))
            throw std::runtime_error("non-finite decision probability");
        p.emplace(option_ids[i], probabilities[i]);
    }
    const auto type = question.at("type").get<std::string>();
    if (type == "noul")
        return {{"type", type}, {"noul", rounded(p.at("true"))}};
    Json output = {{"type", type}};
    Json values = Json::object();
    if (type == "choice") {
        std::string best;
        double confidence = -1;
        for (const auto& option : question.at("criteria").items()) {
            const auto probability = p.at(option.key());
            values[option.key()] = rounded(probability);
            if (probability > confidence) {
                best = option.key();
                confidence = probability;
            }
        }
        output["choice"] = best;
        output["confidence"] = rounded(confidence);
    } else {
        double score = 0, confidence = 0;
        Json legend = Json::object();
        for (std::size_t i = 0; i < option_ids.size(); ++i) {
            const auto id = std::to_string(i);
            score += i * p.at(id);
            confidence = std::max(confidence, p.at(id));
            values[id] = rounded(p.at(id));
            legend[id] = question.at("criteria").at(i);
        }
        output["score"] = rounded(score);
        output["confidence"] = rounded(confidence);
        output["legend"] = legend;
    }
    output["probabilities"] = values;
    return output;
}
} // namespace trtmc::clef
