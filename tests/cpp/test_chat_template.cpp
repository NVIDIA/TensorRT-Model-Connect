// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CHAT-TPL-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-CHAT-01
// Intent:         Chat template registry detection and application
// Preconditions:  None (pure string logic, no GPU or TRT required)
// Postconditions: Registered template IDs match markers and apply callbacks
//                 are used; unknown templates pass through unchanged.
// =============================================================================

#include "runtime/core/chat_template.h"

#include <cstdlib>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::string apply_unit_basic(const std::string& prompt, bool enable_thinking) {
    return std::string("basic:") + (enable_thinking ? "think:" : "plain:") + prompt;
}

static std::string apply_unit_specific(const std::string& prompt, bool enable_thinking) {
    return std::string("specific:") + (enable_thinking ? "think:" : "plain:") + prompt;
}

static void register_unit_templates() {
    trtmc::register_chat_template_format("unit_specific", {"UNIT_SPECIFIC"}, apply_unit_specific);
    trtmc::register_chat_template_format("unit_basic", {"UNIT_BASIC"}, apply_unit_basic);
}

static void test_detect_empty() {
    auto fmt = trtmc::detect_chat_template_format("");
    check(fmt.empty(), "empty -> no template");
}

static void test_detect_registered_template() {
    std::string tpl = "before UNIT_BASIC after";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == "unit_basic", "registered template detection");
}

static void test_specific_marker_wins_by_registration_order() {
    std::string tpl = "before UNIT_SPECIFIC and UNIT_BASIC after";
    auto fmt = trtmc::detect_chat_template_format(tpl);
    check(fmt == "unit_specific", "registered order detection");
}

static void test_detect_unknown() {
    auto fmt = trtmc::detect_chat_template_format("some random jinja template");
    check(fmt.empty(), "unknown -> no template");
}

static void test_apply_none() {
    auto result = trtmc::apply_chat_template("", "hello");
    check(result == "hello", "empty template passthrough");
}

static void test_apply_registered_template() {
    auto result = trtmc::apply_chat_template("unit_basic", "hello", false);
    check(result == "basic:plain:hello", "registered template application");
}

static void test_apply_unknown_template() {
    auto result = trtmc::apply_chat_template("__missing__", "hello");
    check(result == "hello", "unknown template passthrough");
}

int main() {
    register_unit_templates();

    test_detect_empty();
    test_detect_registered_template();
    test_specific_marker_wins_by_registration_order();
    test_detect_unknown();
    test_apply_none();
    test_apply_registered_template();
    test_apply_unknown_template();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All chat_template registry tests passed.\n";
    return 0;
}
