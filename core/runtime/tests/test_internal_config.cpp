/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/config.h"
#include "trtmc/internal/model.h"

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
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

template <class Exception, class Function>
bool throws_as(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

void test_table_driven_config() {
    const ConfigField fields[] = {
        {"count", ConfigKind::I64, ConfigValue{std::int64_t{4}}, ""},
        {"enabled", ConfigKind::Bool, ConfigValue{true}, ""},
        {"suffix", ConfigKind::String, ConfigValue{std::string_view{"!"}}, ""},
        {"budget", ConfigKind::I64, std::nullopt, ""},
    };
    validate_config(fields, {});
    check(config_get<std::int64_t>({}, fields, "count") == 4, "fixed default is resolved");
    check(!config_get<std::int64_t>({}, fields, "budget"), "dynamic default remains absent");
    check(!config_provided({}, "count"), "fixed default does not become explicit input");
    const ConfigEntry explicit_values[] = {
        {"count", std::int64_t{0}},
        {"enabled", false},
        {"suffix", std::string_view{}},
    };
    validate_config(fields, explicit_values);
    check(config_get<std::int64_t>(explicit_values, fields, "count") == 0,
          "explicit zero overrides the default");
    check(config_get<bool>(explicit_values, fields, "enabled") == false,
          "explicit false overrides the default");
    check(config_get<std::string_view>(explicit_values, fields, "suffix")->empty(),
          "explicit empty string overrides the default");
    check(config_provided(explicit_values, "count"), "explicit zero retains presence");
    const ConfigEntry unknown[] = {{"other", std::int64_t{1}}};
    const ConfigEntry duplicate[] = {{"count", std::int64_t{1}}, {"count", std::int64_t{2}}};
    const ConfigEntry wrong_type[] = {{"count", 1.0}};
    check(throws_as<ConfigError>([&] { validate_config(fields, unknown); }), "reject unknown key");
    check(throws_as<ConfigError>([&] { validate_config(fields, duplicate); }),
          "reject duplicate key");
    check(throws_as<ConfigError>([&] { validate_config(fields, wrong_type); }),
          "reject wrong kind");
    check(throws_as<std::logic_error>([&] { (void)config_get<double>({}, fields, "count"); }),
          "wrong accessor type is a programming error even without an override");
    check(throws_as<std::logic_error>([&] { (void)config_get<bool>({}, fields, "unknown"); }),
          "undeclared accessor key is not treated as missing");
    check(
        throws_as<ConfigError>([&] { (void)config_get<std::int64_t>(duplicate, fields, "count"); }),
        "accessor never chooses between duplicate overrides");
}

struct FirstTask {
    using TaskInterface = FirstTask;
    static constexpr std::string_view kTask = "first_test";
    virtual ~FirstTask() = default;
    virtual int first() const = 0;
};
struct SecondTask {
    using TaskInterface = SecondTask;
    static constexpr std::string_view kTask = "second_test";
    virtual ~SecondTask() = default;
    virtual int second() const = 0;
};
struct BoundModel final : FirstTask, SecondTask {
    explicit BoundModel(int id) : id(id) {}
    int first() const override { return id; }
    int second() const override { return id + 1; }
    int id;
};

struct LeadingBase {
    virtual ~LeadingBase() = default;
};
struct SingleTaskModel final : LeadingBase, FirstTask {
    int first() const override { return 42; }
};

template <class Interface, class Model, class = void>
struct CanBind : std::false_type {};
template <class Interface, class Model>
struct CanBind<Interface, Model, std::void_t<decltype(bind<Interface>(std::declval<Model&>()))>>
    : std::true_type {};
static_assert(CanBind<SecondTask, BoundModel>::value);
static_assert(!CanBind<SecondTask, FirstTask>::value,
              "a family cannot bind an interface it does not implement");
static_assert(CanBind<FirstTask, SingleTaskModel>::value);
static_assert(!CanBind<SingleTaskModel, SingleTaskModel>::value,
              "an inherited Task ID cannot turn a concrete model into its interface");
static_assert(!CanBind<BoundModel, BoundModel>::value,
              "a multiple-Task model cannot be bound as its own interface");

template <class Model, class = void>
struct CanInferBind : std::false_type {};
template <class Model>
struct CanInferBind<Model, std::void_t<decltype(bind(std::declval<Model&>()))>> : std::true_type {};
static_assert(CanInferBind<FirstTask>::value,
              "a reference already typed as its canonical interface is safe to bind");
static_assert(!CanInferBind<SingleTaskModel>::value,
              "omitting the interface must not erase an unadjusted model pointer");
static_assert(!CanInferBind<BoundModel>::value);

template <class Interface, class = void>
struct CanGetContractKey : std::false_type {};
template <class Interface>
struct CanGetContractKey<Interface, std::void_t<decltype(contract_key<Interface>())>>
    : std::true_type {};
static_assert(CanGetContractKey<FirstTask>::value);
static_assert(!CanGetContractKey<SingleTaskModel>::value);
static_assert(!CanGetContractKey<BoundModel>::value);

void test_interface_binding() {
    BoundModel first(10), second(20);
    const auto binding = bind<SecondTask>(first);
    const auto other = bind<SecondTask>(second);
    check(binding.key.id == SecondTask::kTask && binding.key.major == 1 && binding.key.minor == 0,
          "Task key comes from its contract");
    check(binding.implementation == static_cast<void*>(static_cast<SecondTask*>(&first)),
          "binding stores the adjusted base-interface address");
    check(static_cast<SecondTask*>(binding.implementation)->second() == 11,
          "restored interface dispatches to the correct implementation");
    check(static_cast<SecondTask*>(other.implementation)->second() == 21,
          "independent instances keep distinct bindings");
    check(binding.fields.empty(), "a Task can have no optional configuration");
    SingleTaskModel single;
    const auto single_binding = bind<FirstTask>(single);
    check(single_binding.implementation == static_cast<void*>(static_cast<FirstTask*>(&single)) &&
              static_cast<FirstTask*>(single_binding.implementation)->first() == 42,
          "a single-Task model with another polymorphic base still binds the adjusted interface");
}

} // namespace

int main() {
    test_scalars_and_presence();
    test_order_and_borrowing();
    test_lists_and_owned_snapshot();
    test_table_driven_config();
    test_interface_binding();
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
