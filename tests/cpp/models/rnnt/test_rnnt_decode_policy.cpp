#include "runtime/models/rnnt/rnnt_config.h"

#include <iostream>

namespace {

int g_failures = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    }
}

void test_blank_advances_frame_without_emit() {
    const auto d = trtmc::make_rnnt_greedy_decision(4, 4, 0, 10);
    check(!d.emit_token, "blank should not emit");
    check(d.advance_frame, "blank should advance frame");
}

void test_nonblank_emits_without_advancing_under_limit() {
    const auto d = trtmc::make_rnnt_greedy_decision(2, 4, 0, 10);
    check(d.emit_token, "nonblank should emit");
    check(!d.advance_frame, "nonblank should stay on frame before limit");
}

void test_symbol_limit_advances_after_emit() {
    const auto d = trtmc::make_rnnt_greedy_decision(2, 4, 2, 3);
    check(d.emit_token, "limited nonblank still emits");
    check(d.advance_frame, "symbol limit should advance frame");
}

} // namespace

int main() {
    test_blank_advances_frame_without_emit();
    test_nonblank_emits_without_advancing_under_limit();
    test_symbol_limit_advances_after_emit();
    if (g_failures) {
        std::cerr << g_failures << " RNNT decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}
