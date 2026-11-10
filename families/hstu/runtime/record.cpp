/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/record.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>

namespace trtmc::hstu {
namespace {

std::int64_t positive(const Json& json, const char* key) {
    const auto value = json.at(key).get<std::int64_t>();
    if (value < 1)
        throw std::invalid_argument(std::string("HSTU runtime.json ") + key + " must be positive");
    return value;
}

Feature feature(const Json& json) {
    Feature result{json.at("name").get<std::string>(), json.at("vocab_size").get<std::int64_t>()};
    if (result.name.empty() || result.vocab_size < 1 ||
        result.vocab_size > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("HSTU runtime.json feature needs a name and an int32 vocab");
    return result;
}

// One value inside the record's user, history or candidates object, as the
// record law reads it: an integer id, an array of ids, or anything else.
struct Member {
    enum class Kind { kInteger, kArray, kOther };
    Kind kind{Kind::kOther};
    std::int64_t integer{0};       // kInteger
    std::vector<std::int64_t> ids; // kArray: its integers before the first non-integer element
    std::size_t elements{0};       // kArray: every element
    bool non_integer{false};       // kArray: an element is no integer
};

// The record's user, history or candidates object.
struct Section {
    bool present{false};
    bool object{false};
    std::vector<std::pair<std::string, Member>> members;

    const Member* find(const std::string& name) const {
        for (const auto& [key, member] : members)
            if (key == name)
                return &member;
        return nullptr;
    }
};

// What the record law reads of a record, filled by one SAX pass: no
// document tree is built (a 4096-event record's tree costs about as much host
// time to build and destroy as the HSTU block's attention takes on the GPU).
struct RecordSummary {
    bool object{false};
    std::optional<std::string> unknown;  // the first field outside the record's four
    std::optional<std::string> repeated; // the first field repeated in its object
    bool model_present{false};
    bool model_string{false};
    std::string model;
    Section user, history, candidates;
};

class RecordReader {
  public:
    using number_integer_t = nlohmann::json::number_integer_t;
    using number_unsigned_t = nlohmann::json::number_unsigned_t;
    using number_float_t = nlohmann::json::number_float_t;
    using string_t = nlohmann::json::string_t;
    using binary_t = nlohmann::json::binary_t;

    explicit RecordReader(RecordSummary& summary) : summary_(summary) {}

    bool null() { return scalar(); }
    bool boolean(bool) { return scalar(); }
    bool binary(binary_t&) { return scalar(); }
    bool number_float(number_float_t, const string_t&) { return scalar(); }
    bool number_integer(number_integer_t value) { return integer(value); }
    bool number_unsigned(number_unsigned_t value) {
        return integer(
            value > static_cast<number_unsigned_t>(std::numeric_limits<std::int64_t>::max())
                ? std::numeric_limits<std::int64_t>::max()
                : static_cast<std::int64_t>(value));
    }
    bool string(string_t& value) {
        if (skip_ == 0 && depth_ == 1 && field_ == Field::kModel) {
            summary_.model_string = true;
            summary_.model = std::move(value);
            return true;
        }
        return scalar();
    }
    bool start_object(std::size_t) {
        if (skip_ > 0 || depth_ >= 2) {
            nested();
            return true;
        }
        if (depth_ == 0) {
            summary_.object = true;
        } else if (section_ != nullptr) { // depth 1: user, history or candidates
            section_->object = true;
        } else {
            nested();
            return true;
        }
        ++depth_;
        return true;
    }
    bool key(string_t& name) {
        if (skip_ > 0)
            return true;
        if (depth_ == 1)
            return top_key(name);
        Section& section = *section_;
        if (section.find(name) != nullptr && !summary_.repeated)
            summary_.repeated = name;
        section.members.emplace_back(name, Member{});
        member_ = &section.members.back().second;
        return true;
    }
    bool end_object() {
        if (skip_ > 0) {
            --skip_;
            return true;
        }
        --depth_;
        if (depth_ == 1) {
            section_ = nullptr;
            member_ = nullptr;
        }
        return true;
    }
    bool start_array(std::size_t) {
        if (skip_ == 0 && depth_ == 2 && member_ != nullptr) {
            member_->kind = Member::Kind::kArray;
            ++depth_;
            return true;
        }
        nested();
        return true;
    }
    bool end_array() {
        if (skip_ > 0) {
            --skip_;
            return true;
        }
        --depth_;
        member_ = nullptr;
        return true;
    }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception& error) {
        throw std::invalid_argument(std::string("HSTU record is not JSON: ") + error.what());
    }

  private:
    enum class Field { kNone, kSection, kModel, kOther };

    bool top_key(const string_t& name) {
        if (std::find(seen_.begin(), seen_.end(), name) != seen_.end() && !summary_.repeated)
            summary_.repeated = name;
        seen_.push_back(name);
        section_ = nullptr;
        field_ = Field::kOther;
        if (name == "user" || name == "history" || name == "candidates") {
            section_ = name == "user"      ? &summary_.user
                       : name == "history" ? &summary_.history
                                           : &summary_.candidates;
            section_->present = true;
            field_ = Field::kSection;
        } else if (name == "model") {
            summary_.model_present = true;
            field_ = Field::kModel;
        } else if (!summary_.unknown) {
            summary_.unknown = name;
        }
        return true;
    }
    bool scalar() {
        if (skip_ == 0 && depth_ == 3)
            element(false, 0);
        return true;
    }
    bool integer(std::int64_t value) {
        if (skip_ > 0)
            return true;
        if (depth_ == 2 && member_ != nullptr) {
            member_->kind = Member::Kind::kInteger;
            member_->integer = value;
        } else if (depth_ == 3) {
            element(true, value);
        }
        return true;
    }
    void element(bool integral, std::int64_t value) {
        Member& member = *member_;
        if (!integral)
            member.non_integer = true;
        else if (!member.non_integer)
            member.ids.push_back(value);
        ++member.elements;
    }
    // A container the record law does not read: counted as one element (in
    // an id array) or one non-id value, and skipped to its end.
    void nested() {
        if (skip_ == 0 && depth_ == 3)
            element(false, 0);
        ++skip_;
    }

    RecordSummary& summary_;
    std::size_t depth_{0};
    std::size_t skip_{0};
    Field field_{Field::kNone};
    Section* section_{nullptr};
    Member* member_{nullptr};
    std::vector<std::string> seen_;
};

std::int32_t table_id(std::int64_t id, const Feature& table, const char* where) {
    if (id < 0 || id >= table.vocab_size)
        throw std::invalid_argument(std::string("HSTU record ") + where + " " + table.name +
                                    " id " + std::to_string(id) + " is outside its table of " +
                                    std::to_string(table.vocab_size));
    return static_cast<std::int32_t>(id);
}

[[noreturn]] void not_integers(const Feature& table, const char* where) {
    throw std::invalid_argument(std::string("HSTU record ") + where + " " + table.name +
                                " ids must be integers");
}

std::int32_t member_id(const Member* member, const Feature& table, const char* where) {
    if (member->kind != Member::Kind::kInteger)
        not_integers(table, where);
    return table_id(member->integer, table, where);
}

std::vector<std::int32_t> table_ids(const Section& section, const Feature& table,
                                    std::int64_t count, const char* where) {
    const Member* member = section.find(table.name);
    if (member == nullptr || member->kind != Member::Kind::kArray)
        throw std::invalid_argument(std::string("HSTU record ") + where + " must hold a " +
                                    table.name + " array");
    if (static_cast<std::int64_t>(member->elements) != count)
        throw std::invalid_argument(std::string("HSTU record ") + where + " " + table.name +
                                    " holds " + std::to_string(member->elements) +
                                    " ids, but the bundle's static geometry takes exactly " +
                                    std::to_string(count));
    std::vector<std::int32_t> ids;
    ids.reserve(member->elements);
    for (const auto id : member->ids) // the elements before the first non-integer, in order
        ids.push_back(table_id(id, table, where));
    if (ids.size() != member->elements)
        not_integers(table, where);
    return ids;
}

void require_section(const Section& section, const char* name) {
    if (!section.present)
        throw std::invalid_argument(std::string("HSTU record lacks ") + name);
}

} // namespace

Geometry parse_geometry(const Json& runtime) {
    Geometry geometry;
    geometry.precision = runtime.at("precision").get<std::string>();
    if (geometry.precision != "bf16" && geometry.precision != "fp32")
        throw std::invalid_argument("HSTU runtime.json precision must be bf16 or fp32");
    geometry.history_length = positive(runtime, "history_length");
    geometry.num_candidates = positive(runtime, "num_candidates");
    geometry.tokens = positive(runtime, "tokens");
    geometry.num_tasks = positive(runtime, "num_tasks");
    for (const auto& value : runtime.at("contextual_features"))
        geometry.contextual.push_back(feature(value));
    geometry.item = feature(runtime.at("item_feature"));
    geometry.action = feature(runtime.at("action_feature"));
    const auto contextual = static_cast<std::int64_t>(geometry.contextual.size());
    if (contextual < 1 ||
        geometry.tokens != contextual + 2 * geometry.history_length + geometry.num_candidates)
        throw std::invalid_argument("HSTU runtime.json tokens do not match its geometry");
    return geometry;
}

Request parse_request(const std::string& document, const Geometry& geometry) {
    RecordSummary record;
    RecordReader reader(record);
    if (!nlohmann::json::sax_parse(document, &reader))
        throw std::invalid_argument("HSTU record is not JSON");
    if (!record.object)
        throw std::invalid_argument("HSTU record must be a JSON object");
    if (record.unknown)
        throw std::invalid_argument("HSTU record has an unknown field: " + *record.unknown);
    if (record.repeated)
        throw std::invalid_argument("HSTU record repeats the field " + *record.repeated);
    Request request;
    if (record.model_present) {
        if (!record.model_string)
            throw std::invalid_argument("HSTU record model must be a string when supplied");
        request.model = record.model;
    }
    require_section(record.user, "user");
    if (!record.user.object || record.user.members.size() != geometry.contextual.size())
        throw std::invalid_argument("HSTU record user must hold exactly the contextual features");
    for (const auto& table : geometry.contextual) {
        const Member* member = record.user.find(table.name);
        if (member == nullptr)
            throw std::invalid_argument("HSTU record user lacks " + table.name);
        request.contextual_ids.push_back(member_id(member, table, "user"));
    }
    require_section(record.history, "history");
    if (!record.history.object || record.history.members.size() != 2)
        throw std::invalid_argument("HSTU record history must hold its item and action arrays");
    request.history_item_ids =
        table_ids(record.history, geometry.item, geometry.history_length, "history");
    request.history_action_ids =
        table_ids(record.history, geometry.action, geometry.history_length, "history");
    require_section(record.candidates, "candidates");
    if (!record.candidates.object || record.candidates.members.size() != 1)
        throw std::invalid_argument("HSTU record candidates must hold one item array");
    request.candidate_item_ids =
        table_ids(record.candidates, geometry.item, geometry.num_candidates, "candidates");
    auto sorted = request.candidate_item_ids;
    std::sort(sorted.begin(), sorted.end());
    if (const auto repeat = std::adjacent_find(sorted.begin(), sorted.end());
        repeat != sorted.end())
        throw std::invalid_argument("HSTU record candidates repeat " + geometry.item.name + " " +
                                    std::to_string(*repeat) + ": each candidate is one option");
    return request;
}

} // namespace trtmc::hstu
