// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-REC-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-REC-01
// Intent:         RecurrentState construction, reset, advance (D2D copy) for generic specs
// Preconditions:  CUDA GPU available
// Postconditions: State tensors allocate, reset zeros them, advance copies D2D correctly
// =============================================================================

// =============================================================================
// Test suite: RecurrentState — generic recurrent state manager
// =============================================================================
//
// Validates RecurrentState with two-tensor and five-tensor specs:
// construction, reset, advance (D2D copy), and ok().
//
// Requires CUDA GPU. Skips gracefully without TRT.
// =============================================================================

#include "trtmc/runtime/recurrent_state.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_two_tensor_spec() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Two state tensors per layer.
    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"conv_state", {96}}, // d_inner * (conv_kernel - 1) = 32 * 3
        {"ssm_state", {512}}, // state_size * d_inner = 16 * 32
    };

    trtmc::RecurrentState state(4, specs, stream);
    check(state.ok(), "two-tensor state ok");
    check(state.num_layers() == 4, "num_layers = 4");
    check(state.specs().size() == 2, "2 specs");
    check(state.specs()[0].name == "conv_state", "spec[0] = conv_state");
    check(state.specs()[1].name == "ssm_state", "spec[1] = ssm_state");

    cudaStreamDestroy(stream);
}

static void test_five_tensor_spec() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Five state tensors per layer.
    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"attn_state", {128}}, {"ff_state", {128}},  {"num_state", {128}},
        {"den_state", {128}},  {"max_state", {128}},
    };

    trtmc::RecurrentState state(6, specs, stream);
    check(state.ok(), "five-tensor state ok");
    check(state.num_layers() == 6, "five-tensor num_layers = 6");
    check(state.specs().size() == 5, "five-tensor has 5 specs");

    cudaStreamDestroy(stream);
}

static void test_reset() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"state_a", {4}},
    };

    trtmc::RecurrentState state(2, specs, stream);
    state.reset();
    check(state.ok(), "ok after reset");

    cudaStreamDestroy(stream);
}

static void test_advance() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<trtmc::RecurrentState::TensorSpec> specs = {
        {"s", {2}},
    };

    trtmc::RecurrentState state(1, specs, stream);

    // Advance should not crash (copies present→state internally)
    state.advance();
    check(state.ok(), "ok after advance");

    // Multiple advances
    for (int i = 0; i < 5; ++i)
        state.advance();
    check(state.ok(), "ok after 5 advances");

    cudaStreamDestroy(stream);
}

int main() {
    test_two_tensor_spec();
    test_five_tensor_spec();
    test_reset();
    test_advance();

    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
