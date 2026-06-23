// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DIFF-CPP-01
// Architecture:   ARCH-DIFF-001
// Unit Design:    UD-DIFF-01
// Intent:         Diffusion CPU math helpers: matmul, SiLU, GELU-tanh,
//                 sinusoidal embedding, timestep MLP
// Preconditions:  None (CPU-only, no GPU or TRT required)
// Postconditions: All math functions produce numerically correct results
// =============================================================================

// test_diffusion_math.cpp — Unit tests for
//   src/runtime/domains/diffusion/diffusion_math.h
//
// Purpose:
//   Validates the inline CPU math helpers shared across diffusion pipelines:
//   matrix multiply with optional bias, SiLU and
//   GELU-tanh activations, sinusoidal positional embedding, and the full
//   timestep MLP used to condition diffusion denoising steps.
//
// Dependencies:
//   - runtime/domains/diffusion/diffusion_math.h  (header-only, CPU)
//   No TRT, CUDA, or GPU required.

#include "runtime/domains/diffusion/diffusion_math.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static bool near(float a, float b, float tol = 1e-5F) {
    return std::abs(a - b) <= tol;
}

// ---------------------------------------------------------------------------
// cpu_matmul_bias
// ---------------------------------------------------------------------------

// Intention: 1x1 matmul with bias: [a] * [b] + c = a*b + c.
// Preconditions:  M=1, K=1, N=1
// Postconditions: out[0] == A[0]*B[0] + bias[0]
static bool test_matmul_1x1_with_bias() {
    const float A[1] = {3.0F};
    const float B[1] = {4.0F};
    const float bias[1] = {2.0F};
    float out[1] = {0.0F};

    trtmc::diffusion_math::cpu_matmul_bias(A, B, bias, out, 1, 1, 1);
    return near(out[0], 14.0F); // 3*4 + 2 = 14
}

// Intention: 2x2 matmul without bias (nullptr bias).
// Preconditions:  M=2, K=2, N=2, bias=nullptr
// Postconditions: out == A * B (standard matrix multiply)
static bool test_matmul_2x2_no_bias() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // A*B = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]]
    //     = [[19, 22], [43, 50]]
    const float A[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float B[4] = {5.0F, 6.0F, 7.0F, 8.0F};
    float out[4] = {0.0F};

    trtmc::diffusion_math::cpu_matmul_bias(A, B, nullptr, out, 2, 2, 2);

    return near(out[0], 19.0F) && near(out[1], 22.0F) && near(out[2], 43.0F) && near(out[3], 50.0F);
}

// ---------------------------------------------------------------------------
// cpu_silu_inplace
// ---------------------------------------------------------------------------

// Intention: SiLU(0) == 0 and SiLU(x) > 0 for large positive x.
// Preconditions:  data = {0, 2, -1}
// Postconditions: data[0] == 0; data[1] > data[2]
static bool test_silu_values() {
    float data[3] = {0.0F, 2.0F, -1.0F};
    trtmc::diffusion_math::cpu_silu_inplace(data, 3);

    // SiLU(0) = 0 / (1 + exp(0)) = 0
    if (!near(data[0], 0.0F))
        return false;
    // SiLU(2) = 2 * sigmoid(2) ≈ 1.761
    if (data[1] < 1.5F)
        return false;
    // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.269
    if (data[2] >= 0.0F)
        return false;
    return true;
}

// Intention: cpu_silu_inplace handles zero-length array without crash.
// Preconditions:  count = 0
// Postconditions: no side effects, no crash
static bool test_silu_empty() {
    trtmc::diffusion_math::cpu_silu_inplace(nullptr, 0);
    return true;
}

// ---------------------------------------------------------------------------
// cpu_gelu_tanh_inplace
// ---------------------------------------------------------------------------

// Intention: GELU-tanh(0) == 0 and output is positive for positive input.
// Preconditions:  data = {0, 1, -1}
// Postconditions: data[0] ≈ 0; data[1] > 0; data[2] < 0
static bool test_gelu_tanh_values() {
    float data[3] = {0.0F, 1.0F, -1.0F};
    trtmc::diffusion_math::cpu_gelu_tanh_inplace(data, 3);

    if (!near(data[0], 0.0F, 1e-4F))
        return false;
    if (data[1] <= 0.0F)
        return false; // GELU(1) ≈ 0.841
    if (data[2] >= 0.0F)
        return false; // GELU(-1) ≈ -0.159
    return true;
}

// ---------------------------------------------------------------------------
// fill_sinusoidal_embedding
// ---------------------------------------------------------------------------

// Intention: sinusoidal embedding has correct size and cos/sin structure.
// Preconditions:  value=0, freq_dim=4
// Postconditions: size=4, first half are cos (all 1.0 for value=0),
//                 second half are sin (all 0.0 for value=0)
static bool test_sinusoidal_embedding_zero_value() {
    std::vector<float> emb;
    trtmc::diffusion_math::fill_sinusoidal_embedding(0.0F, 4, emb);

    if (emb.size() != 4)
        return false;
    // cos(0 * freq) = 1.0 for any freq
    if (!near(emb[0], 1.0F))
        return false;
    if (!near(emb[1], 1.0F))
        return false;
    // sin(0 * freq) = 0.0 for any freq
    if (!near(emb[2], 0.0F))
        return false;
    if (!near(emb[3], 0.0F))
        return false;
    return true;
}

// Intention: sinusoidal embedding values lie in [-1, 1] for arbitrary input.
// Preconditions:  value=3.14, freq_dim=8
// Postconditions: all values in [-1.001, 1.001]
static bool test_sinusoidal_embedding_range() {
    std::vector<float> emb;
    trtmc::diffusion_math::fill_sinusoidal_embedding(3.14F, 8, emb);

    if (emb.size() != 8)
        return false;
    for (float v : emb) {
        if (v < -1.001F || v > 1.001F)
            return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// compute_timestep_mlp
// ---------------------------------------------------------------------------

// Intention: compute_timestep_mlp produces output of size=dim.
// Preconditions:  Identity-like weights (small dim=2, freq_dim=4)
// Postconditions: output.size() == 2
static bool test_timestep_mlp_output_size() {
    const int32_t freq_dim = 4;
    const int32_t dim = 2;

    // Layer 0 weights: [freq_dim x dim] = [4 x 2]
    const std::vector<float> w0(static_cast<std::size_t>(freq_dim * dim), 0.1F);
    const std::vector<float> b0(static_cast<std::size_t>(dim), 0.0F);

    // Layer 2 weights: [dim x dim] = [2 x 2]
    const std::vector<float> w2(static_cast<std::size_t>(dim * dim), 0.1F);
    const std::vector<float> b2(static_cast<std::size_t>(dim), 0.0F);

    std::vector<float> output;
    trtmc::diffusion_math::compute_timestep_mlp(1.0F, freq_dim, dim, w0, b0, w2, b2, output);

    return output.size() == static_cast<std::size_t>(dim);
}

// Intention: compute_timestep_mlp with zero weights produces near-zero output.
// Preconditions:  All weights and biases are 0
// Postconditions: output values are all ~0
static bool test_timestep_mlp_zero_weights() {
    const int32_t freq_dim = 4;
    const int32_t dim = 2;

    const std::vector<float> zeros_4x2(static_cast<std::size_t>(freq_dim * dim), 0.0F);
    const std::vector<float> zeros_2(static_cast<std::size_t>(dim), 0.0F);
    const std::vector<float> zeros_2x2(static_cast<std::size_t>(dim * dim), 0.0F);

    std::vector<float> output;
    trtmc::diffusion_math::compute_timestep_mlp(5.0F, freq_dim, dim, zeros_4x2, zeros_2, zeros_2x2,
                                                zeros_2, output);

    if (output.size() != static_cast<std::size_t>(dim))
        return false;
    for (float v : output) {
        if (std::abs(v) > 1e-5F)
            return false;
    }
    return true;
}

int main() {
    bool all_passed = true;
    std::cout << "test_diffusion_math:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << '\n';
        all_passed &= ok;
    };

    run("matmul_1x1_with_bias", test_matmul_1x1_with_bias);
    run("matmul_2x2_no_bias", test_matmul_2x2_no_bias);
    run("silu_values", test_silu_values);
    run("silu_empty", test_silu_empty);
    run("gelu_tanh_values", test_gelu_tanh_values);
    run("sinusoidal_embedding_zero_value", test_sinusoidal_embedding_zero_value);
    run("sinusoidal_embedding_range", test_sinusoidal_embedding_range);
    run("timestep_mlp_output_size", test_timestep_mlp_output_size);
    run("timestep_mlp_zero_weights", test_timestep_mlp_zero_weights);

    if (all_passed) {
        std::cout << "test_diffusion_math passed" << std::endl;
        return 0;
    }
    std::cerr << "test_diffusion_math FAILED" << std::endl;
    return 1;
}
