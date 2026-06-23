// Unit tests for recurrent-owned output initializers.

#include "runtime/models/recurrent/recurrent_output_initializers.h"

#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_initialize_rwkv_outputs() {
    std::vector<float> logits;
    std::vector<std::vector<float>> attn;
    std::vector<std::vector<float>> ff;
    std::vector<std::vector<float>> num;
    std::vector<std::vector<float>> den;
    std::vector<std::vector<float>> maxv;

    trtmc::recurrent::initialize_rwkv_outputs(3, 11, 5, logits, attn, ff, num, den, maxv);

    check(logits.size() == 11, "rwkv outputs allocate logits");
    check(attn.size() == 3 && attn[0].size() == 5, "rwkv outputs allocate attn");
    check(ff.size() == 3 && ff[1].size() == 5, "rwkv outputs allocate ff");
    check(num.size() == 3 && num[2].size() == 5, "rwkv outputs allocate num");
    check(den.size() == 3 && den[0].size() == 5, "rwkv outputs allocate den");
    check(maxv.size() == 3 && maxv[1].size() == 5, "rwkv outputs allocate max");
}

void test_initialize_mamba_outputs() {
    std::vector<float> logits;
    std::vector<std::vector<float>> conv;
    std::vector<std::vector<float>> ssm;

    trtmc::recurrent::initialize_mamba_outputs(2, 13, 6, 7, logits, conv, ssm);

    check(logits.size() == 13, "mamba outputs allocate logits");
    check(conv.size() == 2 && conv[0].size() == 6, "mamba outputs allocate conv");
    check(ssm.size() == 2 && ssm[1].size() == 7, "mamba outputs allocate ssm");
}

} // namespace

int main() {
    test_initialize_rwkv_outputs();
    test_initialize_mamba_outputs();

    if (g_failures != 0) {
        std::cerr << g_failures << " recurrent output initializer test(s) failed\n";
        return 1;
    }
    return 0;
}
