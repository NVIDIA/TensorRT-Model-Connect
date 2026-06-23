// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-REC-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-REC-01
// Intent:         Recurrent step contracts: generic state validation
// Preconditions:  State specs with known layer counts and sizes
// Postconditions: Validation rejects mismatched layers and initializes generic layer outputs
// =============================================================================

#include "runtime/domains/recurrent/recurrent_step_contracts.h"

#include <cstdint>
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

void test_validate_state_layer_count_and_sizes() {
    std::vector<std::vector<float>> a(2, std::vector<float>(4, 1.0F));
    std::vector<std::vector<float>> b(2, std::vector<float>(4, 2.0F));
    std::vector<std::vector<float>> bad_layers(1, std::vector<float>(4, 0.0F));
    std::vector<std::vector<float>> bad_sizes = a;
    bad_sizes[1].resize(3);

    const auto ok_states = std::array<const std::vector<std::vector<float>>*, 2>{&a, &b};
    const auto bad_layer_states =
        std::array<const std::vector<std::vector<float>>*, 2>{&a, &bad_layers};

    check(trtmc::validate_state_layer_count(ok_states, 2), "layer count accepts matching inputs");
    check(!trtmc::validate_state_layer_count(bad_layer_states, 2), "layer count rejects mismatch");

    const auto ok_specs = std::array<trtmc::StateTensorView, 2>{trtmc::StateTensorView{&a, 4},
                                                                trtmc::StateTensorView{&b, 4}};
    const auto bad_specs = std::array<trtmc::StateTensorView, 2>{
        trtmc::StateTensorView{&a, 4}, trtmc::StateTensorView{&bad_sizes, 4}};

    check(trtmc::validate_state_tensor_sizes(ok_specs, 2), "tensor size accepts matching inputs");
    check(!trtmc::validate_state_tensor_sizes(bad_specs, 2), "tensor size rejects mismatch");
}

void test_initialize_layer_outputs_and_zero_layer_validation() {
    std::vector<std::vector<float>> outputs;
    trtmc::initialize_layer_outputs(3, 2, outputs);
    check(outputs.size() == 3, "layer outputs allocate requested layer count");
    check(outputs[0] == std::vector<float>({0.0F, 0.0F}), "layer outputs initialize with zeros");

    std::vector<std::vector<float>> empty_a;
    std::vector<std::vector<float>> empty_b;
    const auto states = std::array<const std::vector<std::vector<float>>*, 2>{&empty_a, &empty_b};
    check(trtmc::validate_state_layer_count(states, 0), "layer count accepts zero layers");

    const auto specs = std::array<trtmc::StateTensorView, 2>{trtmc::StateTensorView{&empty_a, 0},
                                                             trtmc::StateTensorView{&empty_b, 0}};
    check(trtmc::validate_state_tensor_sizes(specs, 0), "tensor size accepts zero layers");
}

} // namespace

int main() {
    test_validate_state_layer_count_and_sizes();
    test_initialize_layer_outputs_and_zero_layer_validation();

    if (g_failures != 0) {
        std::cerr << g_failures << " recurrent step contract test(s) failed\n";
        return 1;
    }
    return 0;
}
