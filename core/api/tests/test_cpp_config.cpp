/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.hpp"

#include <iostream>

static_assert(std::is_same_v<decltype(trtmc::Model::load("model.bundle")), trtmc::Model>);
static_assert(
    std::is_same_v<
        decltype(std::declval<const trtmc::Model&>().supports<trtmc::TextContinuation>()), bool>);
static_assert(
    std::is_same_v<decltype(std::declval<const trtmc::Model&>().task<trtmc::TextContinuation>()),
                   trtmc::TextContinuation>);
static_assert(std::is_same_v<decltype(std::declval<const trtmc::TextContinuation&>().run(
                                 trtmc::TextContinuationRequest{"Hello"})),
                             trtmc::TextContinuationResult>);
static_assert(!std::is_copy_constructible_v<trtmc::TextContinuationResult>);
static_assert(std::is_nothrow_move_constructible_v<trtmc::TextContinuationResult>);

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::string string(trtmc_string_view value) {
    return value.size == 0 ? std::string{} : std::string(value.data, value.size);
}

void test_scalars_and_literal() {
    const trtmc::Config config{{"top_k", 40},       {"top_p", 0.9}, {"flag", false},
                               {"prefix", "Hello"}, {"zero", 0},    {"empty", ""}};
    auto wire = config.c_entries();
    check(wire.size() == 6 && wire.view().count == 6, "entry count preserved");
    check(wire.data()[0].value.kind == TRTMC_CONFIG_I64 && wire.data()[0].value.as.i64 == 40,
          "integer literal becomes int64");
    check(wire.data()[1].value.kind == TRTMC_CONFIG_F64 && wire.data()[1].value.as.f64 == 0.9,
          "floating-point value preserved");
    check(wire.data()[2].value.kind == TRTMC_CONFIG_BOOL && wire.data()[2].value.as.boolean == 0,
          "explicit false remains bool");
    check(wire.data()[3].value.kind == TRTMC_CONFIG_STRING &&
              string(wire.data()[3].value.as.string) == "Hello",
          "string literal never becomes bool");
    check(wire.data()[4].value.as.i64 == 0 && wire.data()[5].value.as.string.size == 0,
          "zero and empty string are not omitted");
    check(trtmc::Config{}.c_entries().size() == 0, "empty config has no overrides");

    bool overflow = false;
    try {
        (void)trtmc::ConfigValue(std::numeric_limits<std::uint64_t>::max());
    } catch (const std::out_of_range&) {
        overflow = true;
    }
    check(overflow, "oversized unsigned integer is not truncated");
    const trtmc::ConfigValue wide(std::int64_t{9007199254740993LL});
    check(wide.get<std::int64_t>() == 9007199254740993LL, "wide integer stays exact");
}

trtmc::Config make_owned_config() {
    std::string name = "temporary";
    std::string text = std::string("one\0two", 7);
    std::vector<std::string> stops{"stop", "", std::string("a\0b", 3)};
    trtmc::Config original{{name, std::string_view(text)},
                           {"stops", stops},
                           {"temporary", "duplicate"},
                           {"ids", std::vector<std::int64_t>{0, 9007199254740993LL}},
                           {"schedule", std::vector<double>{1.0, 0.5, 0.0}}};
    auto copied = original;
    original = trtmc::Config{};
    name.assign("changed");
    text.assign("changed");
    stops.clear();
    return copied;
}

void test_ownership_copies_and_moves() {
    auto config = make_owned_config();
    auto copied = config;
    config = trtmc::Config{};
    auto moved = std::move(copied);
    auto wire = moved.c_entries();
    auto moved_wire = std::move(wire);
    check(string(moved_wire.data()[0].name) == "temporary", "copied config owns keys");
    check(string(moved_wire.data()[0].value.as.string) == std::string("one\0two", 7),
          "copied config owns strings and embedded NUL");
    check(string(moved_wire.data()[2].name) == "temporary" &&
              string(moved_wire.data()[2].value.as.string) == "duplicate",
          "duplicates and ordering survive copies");
    const auto stops = moved_wire.data()[1].value.as.string_list;
    check(stops.size == 3 && string(stops.data[0]) == "stop" && stops.data[1].size == 0 &&
              string(stops.data[2]) == std::string("a\0b", 3),
          "string list descriptors remain valid after moving wire storage");
    const auto ids = moved_wire.data()[3].value.as.i64_list;
    const auto schedule = moved_wire.data()[4].value.as.f64_list;
    check(ids.size == 2 && ids.data[1] == 9007199254740993LL, "copied integer list stays exact");
    check(schedule.size == 3 && schedule.data[1] == 0.5, "copied floating-point list owns data");
}

void test_empty_lists_and_metadata_copy() {
    trtmc::Config config{{"ids", std::vector<std::int64_t>{}},
                         {"steps", std::vector<double>{}},
                         {"stops", std::vector<std::string>{}}};
    auto wire = config.c_entries();
    check(wire.data()[0].value.kind == TRTMC_CONFIG_I64_LIST &&
              wire.data()[0].value.as.i64_list.size == 0,
          "empty integer list retains type");
    check(wire.data()[1].value.kind == TRTMC_CONFIG_F64_LIST &&
              wire.data()[1].value.as.f64_list.size == 0,
          "empty floating-point list retains type");
    check(wire.data()[2].value.kind == TRTMC_CONFIG_STRING_LIST &&
              wire.data()[2].value.as.string_list.size == 0,
          "empty string list retains type");

    trtmc::ConfigValue owned("initial");
    {
        std::string text = "model default";
        trtmc_string_view strings[]{{text.data(), text.size()}};
        trtmc_config_value_v1 field{};
        field.kind = TRTMC_CONFIG_STRING_LIST;
        field.as.string_list = {strings, 1};
        owned = trtmc::detail::copy_value(field);
    }
    check(owned.get<std::vector<std::string>>() == std::vector<std::string>{"model default"},
          "metadata default copy does not retain model-owned pointers");
}

void test_large_copied_lists() {
    std::vector<std::int64_t> ids(4096, 9007199254740993LL);
    std::vector<double> schedule(4096, 0.125);
    std::vector<std::string> stops(4096, std::string(128, 'x'));
    trtmc::Config source{{"ids", ids}, {"schedule", schedule}, {"stops", stops}};
    auto copied = source;
    source = trtmc::Config{};
    ids.clear();
    schedule.clear();
    stops.clear();
    auto wire = copied.c_entries();
    auto replacement = trtmc::Config{}.c_entries();
    replacement = std::move(wire);
    const auto& values = replacement.data();
    check(values[0].value.as.i64_list.size == 4096 &&
              values[0].value.as.i64_list.data[4095] == 9007199254740993LL,
          "large integer list owns copied storage");
    check(values[1].value.as.f64_list.size == 4096 &&
              values[1].value.as.f64_list.data[4095] == 0.125,
          "large floating-point list owns copied storage");
    const auto list = values[2].value.as.string_list;
    check(list.size == 4096 && string(list.data[0]) == std::string(128, 'x') &&
              string(list.data[4095]) == std::string(128, 'x'),
          "large string-list descriptors survive wire move assignment");
    auto owned_default = trtmc::detail::copy_value(values[0].value);
    copied = trtmc::Config{};
    check(owned_default.get<std::vector<std::int64_t>>().back() == 9007199254740993LL,
          "integer-list metadata copy owns storage");
}

} // namespace

int main() {
    test_scalars_and_literal();
    test_ownership_copies_and_moves();
    test_empty_lists_and_metadata_copy();
    test_large_copied_lists();
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
