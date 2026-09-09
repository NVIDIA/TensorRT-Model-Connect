/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/config.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using trtmc::Span;
using namespace trtmc::internal;

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

template <typename T>
bool rejects(const ConfigValue& value) {
    try {
        (void)config_value_as<T>(value);
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

void test_scalars_and_presence() {
    const ConfigValue integer{std::int64_t{9007199254740993LL}};
    check(config_kind(integer) == ConfigKind::I64, "integer kind");
    check(config_value_as<std::int64_t>(integer) == 9007199254740993LL,
          "int64 is not transported through double");
    check(rejects<double>(integer), "integer does not coerce to double");
    check(rejects<bool>(integer), "integer does not coerce to bool");

    const ConfigValue real{0.75};
    check(config_kind(real) == ConfigKind::F64 && config_value_as<double>(real) == 0.75,
          "double is preserved");
    check(rejects<std::int64_t>(real), "double does not truncate to integer");

    const ConfigField disabled{"emit_eos", ConfigKind::Bool, ConfigValue{false}, ""};
    const ConfigField zero{"top_k", ConfigKind::I64, ConfigValue{std::int64_t{0}}, ""};
    const ConfigField empty{"prefix", ConfigKind::String, ConfigValue{std::string_view{}}, ""};
    const ConfigField computed{"limit", ConfigKind::I64, std::nullopt, ""};
    check(disabled.default_value.has_value() &&
              config_kind(*disabled.default_value) == ConfigKind::Bool &&
              !config_value_as<bool>(*disabled.default_value),
          "explicit false is a fixed default");
    check(zero.default_value.has_value() && config_value_as<std::int64_t>(*zero.default_value) == 0,
          "explicit zero is a fixed default");
    check(empty.default_value.has_value() &&
              config_kind(*empty.default_value) == ConfigKind::String &&
              config_value_as<std::string_view>(*empty.default_value).empty(),
          "empty string is a fixed default");
    check(!computed.default_value.has_value(), "computed default has explicit absence");
}

void test_order_and_borrowing() {
    std::string key = "prefix";
    char text[] = {'a', '\0', 'b'};
    ConfigEntry entries[] = {
        {key, ConfigValue{std::string_view{text, sizeof(text)}}},
        {key, ConfigValue{std::string_view{"second"}}},
    };
    const ConfigView supplied{entries};
    check(supplied.size() == 2 && supplied[0].name == supplied[1].name,
          "transport preserves duplicate keys");
    check(config_value_as<std::string_view>(supplied[1].value) == "second",
          "transport preserves entry order");
    check(config_value_as<std::string_view>(supplied[0].value).size() == 3,
          "strings preserve embedded NUL and length");

    const ConfigValue copied_view = supplied[0].value;
    const std::string owned_copy(config_value_as<std::string_view>(copied_view));
    text[0] = 'x';
    key[0] = 'P';
    check(config_value_as<std::string_view>(copied_view)[0] == 'x',
          "copying a ConfigValue still borrows string storage");
    check(supplied[0].name == "Prefix", "entry names borrow their storage");
    check(owned_copy[0] == 'a', "family-owned copy is independent of caller mutation");
    check(ConfigView{}.empty(), "empty view represents no explicit overrides");
}

void test_lists_and_owned_snapshot() {
    std::vector<std::int64_t> owned_ids;
    std::vector<double> owned_schedule;
    std::vector<std::string> owned_stops;
    {
        std::int64_t ids[] = {1, 9007199254740993LL};
        double schedule[] = {1.0, 0.5, 0.0};
        std::string stop = "stop";
        std::string_view stops[] = {stop, std::string_view{}};
        const ConfigValue id_value{Span<const std::int64_t>{ids}};
        const ConfigValue schedule_value{Span<const double>{schedule}};
        const ConfigValue stop_value{Span<const std::string_view>{stops}};
        check(config_kind(id_value) == ConfigKind::I64List, "integer list kind");
        check(config_kind(schedule_value) == ConfigKind::F64List, "double list kind");
        check(config_kind(stop_value) == ConfigKind::StringList, "string list kind");
        check(rejects<Span<const double>>(id_value), "lists do not coerce element types");

        const auto id_view = config_value_as<Span<const std::int64_t>>(id_value);
        const auto schedule_view = config_value_as<Span<const double>>(schedule_value);
        const auto stop_view = config_value_as<Span<const std::string_view>>(stop_value);
        owned_ids.assign(id_view.begin(), id_view.end());
        owned_schedule.assign(schedule_view.begin(), schedule_view.end());
        for (const auto item : stop_view)
            owned_stops.emplace_back(item);

        ids[0] = 7;
        schedule[0] = 0.75;
        stop[0] = 'S';
        check(id_view[0] == 7 && schedule_view[0] == 0.75 && stop_view[0] == "Stop",
              "list payloads and string elements borrow caller storage");

        const ConfigValue empty_list{Span<const std::int64_t>{}};
        check(config_kind(empty_list) == ConfigKind::I64List &&
                  config_value_as<Span<const std::int64_t>>(empty_list).empty(),
              "empty list retains its element type");
    }
    check(owned_ids == std::vector<std::int64_t>({1, 9007199254740993LL}),
          "owned integer snapshot survives caller storage");
    check(owned_schedule == std::vector<double>({1.0, 0.5, 0.0}),
          "owned schedule survives caller storage");
    check(owned_stops == std::vector<std::string>({"stop", ""}),
          "owned string snapshot copies both list and string payloads");
}

} // namespace

int main() {
    test_scalars_and_presence();
    test_order_and_borrowing();
    test_lists_and_owned_snapshot();
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
