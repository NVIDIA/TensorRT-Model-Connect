/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/laya/runtime/routing.h"

#include "families/laya/runtime/unicode.h"

#include <algorithm>
#include <cmath>
#include <codecvt>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>

namespace trtmc::laya {
namespace {
std::u32string decode(const std::string& text) {
    return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>().from_bytes(text);
}
std::string encode(const std::u32string& text) {
    return std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>().to_bytes(text);
}
void leaves(const Json& value, int depth, std::u32string& result, bool& first) {
    if (depth > 6 || result.size() >= 4000)
        return;
    if (value.is_string()) {
        if (!first)
            result += U' ';
        first = false;
        const auto text = decode(value.get<std::string>());
        result += text.substr(0, 4000 - result.size());
    } else if (value.is_object() || value.is_array()) {
        for (const auto& part : value)
            leaves(part, depth + 1, result, first);
    }
}
double rounded(double value) {
    return std::nearbyint(value * 10000) / 10000;
}
std::string trimmed(std::string value) {
    return trim_unicode(value);
}
std::string percentage(double value) {
    return std::to_string(static_cast<int>(std::nearbyint(100 * value)));
}
} // namespace

Routing::Routing(const Json& tables)
    : scripts_(tables.at("scripts")), aliases_(tables.at("aliases")),
      workflows_(tables.at("workflows")) {
    for (const auto& row : tables.at("properties"))
        properties_.push_back(row.get<std::array<std::uint32_t, 3>>());
    for (const auto& row : tables.at("lowercase")) {
        auto& out = lowercase_[row[0].get<char32_t>()];
        for (const auto& code : row[1])
            out += code.get<char32_t>();
    }
    for (const auto& item : tables.at("stopwords").items())
        stopwords_.push_back({item.key(), item.value().get<std::unordered_set<std::string>>()});
    shared_ = tables.at("shared_words").get<std::unordered_set<std::string>>();
    diacritics_ = tables.at("diacritics").get<std::unordered_set<char32_t>>();
}

int Routing::flags(char32_t code) const {
    auto it = std::upper_bound(properties_.begin(), properties_.end(), code,
                               [](char32_t c, const auto& range) { return c < range[0]; });
    if (it == properties_.begin())
        return 0;
    --it;
    return code <= (*it)[1] ? (*it)[2] : 0;
}

std::u32string Routing::lower(const std::u32string& text) const {
    std::u32string out;
    for (auto code : text) {
        auto it = lowercase_.find(code);
        out += it == lowercase_.end() ? std::u32string(1, code) : it->second;
    }
    return out;
}

std::string Routing::quoted(const std::string& value) const {
    const char quote =
        value.find('\'') != std::string::npos && value.find('"') == std::string::npos ? '"' : '\'';
    std::string out(1, quote);
    for (auto code : decode(value)) {
        if (code == '\\' || code == char32_t(quote)) {
            out += '\\';
            out += static_cast<char>(code);
        } else if (code == '\n' || code == '\r' || code == '\t') {
            out += code == '\n' ? "\\n" : code == '\r' ? "\\r" : "\\t";
        } else if ((code >= 0x20 && code < 0x7F) || (flags(code) & 32)) {
            out += encode(std::u32string(1, code));
        } else {
            std::ostringstream escaped;
            escaped << (code <= 0xFF     ? "\\x"
                        : code <= 0xFFFF ? "\\u"
                                         : "\\U")
                    << std::hex << std::setfill('0')
                    << std::setw(code <= 0xFF     ? 2
                                 : code <= 0xFFFF ? 4
                                                  : 8)
                    << static_cast<std::uint32_t>(code);
            out += escaped.str();
        }
    }
    return out + quote;
}

std::string Routing::script(char32_t code, bool profile) const {
    if (code < (profile ? 0x02B0 : 0x0250) || (code >= 0x1E00 && code <= 0x1EFF) ||
        (code >= 0xFF21 && code <= 0xFF3A) || (code >= 0xFF41 && code <= 0xFF5A))
        return "latin";
    for (const auto& row : scripts_)
        for (const auto& range : row[1])
            if (code >= range[0].get<char32_t>() && code <= range[1].get<char32_t>())
                return row[0].get<std::string>();
    return "other";
}

Json Routing::analyse(const Json& state) const {
    std::u32string text;
    bool first = true;
    leaves(state, 0, text, first);
    Json counts = Json::object();
    int latin = 0, letters = 0;
    for (auto code : text) {
        if (!(flags(code) & 1))
            continue;
        ++letters;
        auto kind = script(code, true);
        if (kind == "latin")
            ++latin;
        else
            counts[kind] = counts.value(kind, 0) + 1;
    }
    counts["latin"] = latin; // detect_script breaks ties in insertion order.
    std::string dominant = "unknown";
    int maximum = 0;
    for (const auto& item : counts.items())
        if (item.value().get<int>() > maximum) {
            dominant = item.key();
            maximum = item.value();
        }
    Json profile = Json::object();
    if (latin)
        profile["latin"] = double(latin) / letters;
    for (const auto& item : counts.items())
        if (item.key() != "latin" && item.value().get<int>())
            profile[item.key()] = double(item.value().get<int>()) / letters;
    const double fraction = letters ? rounded(1.0 - double(latin) / letters) : 0;
    bool non_latin_words = false;
    std::u32string run;
    std::string run_script;
    auto finish_run = [&]() {
        if (run.size() >= 2 && !(flags(run.front()) & 8))
            non_latin_words = true;
        run.clear();
    };
    for (auto code : text) {
        if (flags(code) & 16)
            continue;
        const auto kind = script(code, false);
        const bool named = kind != "latin" && kind != "other";
        if (named && kind == run_script)
            run += code;
        else {
            finish_run();
            run_script = named ? kind : "";
            if (named)
                run += code;
        }
    }
    finish_run();
    if (dominant == "latin" && non_latin_words &&
        (fraction >= 0.2 || (fraction >= 0.1 && std::nearbyint(fraction * letters) >= 10))) {
        double best = 0;
        for (const auto& item : profile.items())
            if (item.key() != "latin" && item.value().get<double>() > best) {
                dominant = item.key();
                best = item.value();
            }
    }
    Json result{{"script", dominant},
                {"script_profile", profile},
                {"language", nullptr},
                {"is_english", dominant == "unknown"},
                {"language_undecided", true},
                {"diacritic_rate", 0.0},
                {"non_latin_fraction", fraction}};
    if (dominant != "latin")
        return result;
    const auto lowered = lower(text);
    const auto diac =
        std::count_if(lowered.begin(), lowered.end(), [&](auto c) { return diacritics_.count(c); });
    const double rate = double(diac) / std::max<std::size_t>(1, lowered.size());
    const bool non_english = rate >= 0.02;
    auto prose = text;
    auto identifier_char = [&](char32_t c) { return (flags(c) & 2) || c == U'-'; };
    for (std::size_t i = 0; i < prose.size();) {
        auto end = i;
        while (end < prose.size() && identifier_char(prose[end]))
            ++end;
        bool matched = false;
        while (end + 1 < prose.size() && (prose[end] == U'.' || prose[end] == U'@') &&
               identifier_char(prose[end + 1])) {
            matched = true;
            ++end;
            while (end < prose.size() && identifier_char(prose[end]))
                ++end;
        }
        if (matched) {
            std::fill(prose.begin() + i, prose.begin() + end, U' ');
            i = end;
        } else
            ++i;
    }
    std::replace(prose.begin(), prose.end(), char32_t(0x130), U'i');
    prose = lower(prose);
    std::vector<std::string> words;
    std::u32string word;
    for (auto code : prose) {
        if ((flags(code) & 2) && !(flags(code) & 4) && code != U'_')
            word += code;
        else if (!word.empty()) {
            words.push_back(encode(word));
            word.clear();
        }
    }
    if (!word.empty())
        words.push_back(encode(word));
    int english = 0, best = 0;
    std::string language;
    if (words.size() >= 4) {
        for (const auto& [key, stop] : stopwords_) {
            int score = 0;
            bool evidence = false;
            for (const auto& w : words)
                if (stop.count(w)) {
                    ++score;
                    evidence |= !shared_.count(w);
                }
            if (key == "en")
                english = score;
            else if (evidence && score > best) {
                language = key;
                best = score;
            }
        }
        if (!(best >= std::max(2, english + 2) || (non_english && best >= std::max(2, english))))
            language = english && !non_english ? "en" : "";
    }
    result["language"] = language.empty() ? Json() : Json(language);
    result["language_undecided"] = language.empty();
    result["is_english"] = language == "en" || (language.empty() && !non_english);
    result["diacritic_rate"] = rounded(rate);
    return result;
}

std::string Routing::name(std::string value) const {
    value = encode(lower(decode(trimmed(value))));
    value = aliases_.value(value, value);
    if (value != "english" && value != "multilingual" && value != "typed-decisions")
        throw std::invalid_argument("unknown Laya model variant");
    return value;
}

Json Routing::route(const Json& document) const {
    const auto& questions = document.at("questions");
    Json workflow;
    std::set<std::string> ids;
    for (const auto& q : questions.items())
        ids.insert(q.key());
    for (const auto& item : workflows_.items())
        if (ids == item.value().get<std::set<std::string>>())
            workflow = item.key();
    auto decision = [&](const std::string& variant, const std::string& reason,
                        const Json& detection, const Json& wf) {
        return Json{
            {"model", variant},
            {"repo", "convaiinnovations/laya" + (variant == "english" ? "" : "/" + variant)},
            {"reason", reason},
            {"detection", detection},
            {"workflow", wf}};
    };
    for (const auto& field : {"model", "task"})
        if (document.contains(field) && !document[field].is_null()) {
            const auto value = document[field].get<std::string>();
            return decision(name(value), std::string("explicit ") + field + "=" + quoted(value),
                            nullptr, nullptr);
        }
    if (!workflow.is_null() && document.value("auto_task_detection", false))
        return decision("typed-decisions",
                        "question ids match the " + quoted(workflow.get<std::string>()) +
                            " typed-decisions workflow",
                        nullptr, workflow);
    for (const auto& field : {"lang", "lang_guess"})
        if (document.contains(field) && !document[field].is_null()) {
            auto code = encode(lower(decode(trimmed(document[field].get<std::string>()))));
            code = code.substr(0, code.find('.'));
            std::replace(code.begin(), code.end(), '_', '-');
            code = code.substr(0, code.find('-'));
            if (!code.empty()) {
                const bool en = code == "en" || code == "eng" || code == "english";
                const auto reason =
                    std::string(field) == "lang"
                        ? "explicit lang=" + quoted(document[field].get<std::string>())
                        : "lang_guess: the caller identified this as " +
                              std::string(en ? "English" : "non-English") + " text";
                return decision(en ? "english" : "multilingual", reason, nullptr, workflow);
            }
        }
    const auto detection = analyse(document.at("state"));
    const auto dominant = detection.at("script").get<std::string>();
    const auto fallback = name(document.value("default", "english"));
    std::string variant, reason;
    if (dominant == "unknown") {
        variant = fallback;
        reason = "no letters detected in state; using default (" + fallback + ")";
    } else if (dominant != "latin") {
        variant = "multilingual";
        reason = "non-Latin script (" + dominant + ", " +
                 percentage(detection["non_latin_fraction"]) +
                 "% of letters); the English checkpoint cannot read it";
    } else if (!detection.at("is_english").get<bool>()) {
        variant = "multilingual";
        reason = detection["language"].is_null()
                     ? "Latin script, language not identified but " +
                           percentage(detection["diacritic_rate"]) +
                           "% non-English letters; not safe for the English checkpoint"
                     : "Latin script but language looks like " +
                           quoted(detection["language"].get<std::string>()) + ", not English";
    } else if (detection.at("language_undecided").get<bool>()) {
        variant = fallback;
        reason =
            "Latin script, language not identified and no non-English letters; using default (" +
            fallback + ")";
    } else {
        variant = "english";
        reason = "English Latin text";
    }
    return decision(variant, reason, detection, workflow);
}
} // namespace trtmc::laya
