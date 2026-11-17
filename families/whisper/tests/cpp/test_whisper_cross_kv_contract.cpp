/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The host side of Whisper's cross-attention contract, without a GPU: which
// decoder input rank a bundle carries; the encoder_padded_rows values an
// encoder that projects the decoder's K/V is fed, which must mark the rows the
// raw path's plan zeroes and no other row; and the check and the binding the
// pipeline runs between such an encoder and its decoder, over stand-in modules.

#include "families/whisper/runtime/whisper_cross_kv_plan.h"
#include "families/whisper/runtime/whisper_host_plan.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// The few members of ITrtModule the contract helpers call.
struct Module {
    struct Tensor {
        std::vector<int64_t> shape;
        int dtype{0};
        void* device{nullptr};
    };

    std::map<std::string, Tensor> inputs;
    std::map<std::string, Tensor> outputs;
    std::vector<std::pair<std::string, void*>> bound;

    bool has_input(const std::string& name) const { return inputs.count(name) != 0; }
    bool has_output(const std::string& name) const { return outputs.count(name) != 0; }
    const Tensor* find(const std::string& name) const {
        const auto input = inputs.find(name);
        if (input != inputs.end())
            return &input->second;
        const auto output = outputs.find(name);
        return output != outputs.end() ? &output->second : nullptr;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const {
        const Tensor* tensor = find(name);
        return tensor != nullptr ? tensor->shape : std::vector<int64_t>{};
    }
    int tensor_dtype(const std::string& name) const {
        const Tensor* tensor = find(name);
        return tensor != nullptr ? tensor->dtype : -1;
    }
    void* device_ptr(const std::string& name) const {
        const Tensor* tensor = find(name);
        return tensor != nullptr ? tensor->device : nullptr;
    }
    void bind_external(const std::string& name, void* device) { bound.emplace_back(name, device); }
};

constexpr int kHalf = 1;
constexpr int kFloat = 0;
constexpr int32_t kLayers = 3;
const std::vector<int64_t> kHeads = {1, 6, 1500, 64};

// One buffer per tensor, so that an address identifies a layer's K or V.
char g_buffers[2 * kLayers];

Module projecting_encoder() {
    Module encoder;
    encoder.inputs[trtmc::kWhisperEncoderPaddedRows] = {{1500, 1}, kFloat, nullptr};
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        encoder.outputs["cross_k_" + std::to_string(layer)] = {kHeads, kHalf,
                                                               &g_buffers[2 * layer]};
        encoder.outputs["cross_v_" + std::to_string(layer)] = {kHeads, kHalf,
                                                               &g_buffers[2 * layer + 1]};
    }
    return encoder;
}

Module reading_decoder() {
    Module decoder;
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        decoder.inputs["cross_k_" + std::to_string(layer)] = {kHeads, kHalf, nullptr};
        decoder.inputs["cross_v_" + std::to_string(layer)] = {kHeads, kHalf, nullptr};
    }
    return decoder;
}

std::string mismatch(const Module& encoder, const Module& decoder) {
    return trtmc::whisper_projected_cross_kv_mismatch(encoder, decoder, kLayers,
                                                      trtmc::kWhisperEncoderPaddedRows);
}

std::size_t count_padded(const std::vector<float>& rows) {
    std::size_t padded = 0;
    for (const float row : rows) {
        if (row == 1.0F) {
            ++padded;
        }
    }
    return padded;
}

void test_cross_input_rank() {
    check(trtmc::whisper_cross_input_rank(1) == 4,
          "a single-GPU decoder reads projected K/V heads (rank 4)");
    check(trtmc::whisper_cross_input_rank(2) == 2,
          "a tensor-parallel decoder reads the raw encoder states (rank 2)");
    check(trtmc::whisper_cross_input_rank(4) == 2,
          "every tensor-parallel size reads the raw encoder states");
}

void test_padded_rows_values() {
    check(trtmc::build_whisper_encoder_padded_rows(0, 3).empty(), "no positions, no rows");
    check(trtmc::build_whisper_encoder_padded_rows(-1, 3).empty(), "a negative length has no rows");

    const std::vector<float> short_request = trtmc::build_whisper_encoder_padded_rows(4, 1);
    check(short_request == std::vector<float>({0.0F, 1.0F, 1.0F, 1.0F}),
          "rows past the valid length are 1, the others 0");
    const std::vector<float> last_row = trtmc::build_whisper_encoder_padded_rows(4, 3);
    check(last_row == std::vector<float>({0.0F, 0.0F, 0.0F, 1.0F}), "one padded row is marked");

    for (const int32_t full : {0, -5, 4, 9}) {
        check(trtmc::build_whisper_encoder_padded_rows(4, full) == std::vector<float>(4, 0.0F),
              "audio that fills the window (or an unknown length) marks no row");
    }
}

// The projected path must zero exactly the rows the raw path's plan zeroes.
void test_padded_rows_match_the_raw_plan() {
    const int32_t hidden = 384;
    for (const int32_t enc_seq : {1, 2, 7, 1500}) {
        for (const int32_t actual : {-2, 0, 1, 2, 6, 7, 8, 1166, 1499, 1500, 1501}) {
            const std::vector<float> rows =
                trtmc::build_whisper_encoder_padded_rows(enc_seq, actual);
            const trtmc::WhisperCrossKvPlan plan =
                trtmc::make_whisper_cross_kv_plan(enc_seq, hidden, actual);
            check(rows.size() == static_cast<std::size_t>(enc_seq), "one value per position");

            const std::size_t padded = count_padded(rows);
            const std::size_t valid = rows.size() - padded;
            const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(float);
            check(valid * row_bytes == plan.valid_bytes,
                  "the unmarked rows are the rows the raw plan keeps");
            check(padded * row_bytes == plan.pad_bytes,
                  "the marked rows are the rows the raw plan zeroes");
            check((padded > 0) == plan.zero_pad_encoder_output,
                  "a row is marked only when the raw plan zeroes");
            for (std::size_t row = 0; row < rows.size(); ++row) {
                check(rows[row] == (row < valid ? 0.0F : 1.0F),
                      "valid rows come first, every value is 0 or 1");
            }
        }
    }
}

void test_matching_engines_have_no_mismatch() {
    check(mismatch(projecting_encoder(), reading_decoder()).empty(),
          "an encoder and a decoder that agree on every layer's K/V are accepted");
}

void test_each_disagreement_is_named() {
    Module encoder = projecting_encoder();
    encoder.inputs.clear();
    check(mismatch(encoder, reading_decoder()) == "encoder has no encoder_padded_rows input",
          "an encoder without the padded-rows input is refused");

    encoder = projecting_encoder();
    encoder.outputs.erase("cross_v_2");
    check(mismatch(encoder, reading_decoder()) ==
              "encoder and decoder disagree on projected cross_v_2",
          "a missing encoder output is named");

    Module decoder = reading_decoder();
    decoder.inputs.erase("cross_k_1");
    check(mismatch(projecting_encoder(), decoder) ==
              "encoder and decoder disagree on projected cross_k_1",
          "a missing decoder input is named");

    decoder = reading_decoder();
    decoder.inputs["cross_k_0"].shape = {1500, 384};
    check(mismatch(projecting_encoder(), decoder) ==
              "encoder and decoder disagree on projected cross_k_0",
          "a shape the decoder does not read is named");

    decoder = reading_decoder();
    decoder.inputs["cross_v_1"].dtype = kFloat;
    check(mismatch(projecting_encoder(), decoder) ==
              "encoder and decoder disagree on projected cross_v_1",
          "a dtype the decoder does not read is named");

    // A tensor of the right shape and dtype on the wrong side is no match.
    encoder = projecting_encoder();
    encoder.inputs["cross_k_0"] = encoder.outputs["cross_k_0"];
    encoder.outputs.erase("cross_k_0");
    check(mismatch(encoder, reading_decoder()) ==
              "encoder and decoder disagree on projected cross_k_0",
          "an encoder input is not an encoder output");

    decoder = reading_decoder();
    decoder.outputs["cross_v_0"] = decoder.inputs["cross_v_0"];
    decoder.inputs.erase("cross_v_0");
    check(mismatch(projecting_encoder(), decoder) ==
              "encoder and decoder disagree on projected cross_v_0",
          "a decoder output is not a decoder input");

    // The last layer is checked too.
    encoder = projecting_encoder();
    encoder.outputs["cross_k_2"].dtype = kFloat;
    check(mismatch(encoder, reading_decoder()) ==
              "encoder and decoder disagree on projected cross_k_2",
          "every layer is compared");
}

void test_binding_gives_each_input_its_own_encoder_output() {
    const Module encoder = projecting_encoder();
    Module decoder = reading_decoder();
    trtmc::bind_whisper_projected_cross_kv(encoder, decoder, kLayers);

    check(decoder.bound.size() == static_cast<std::size_t>(2 * kLayers),
          "one binding per layer K and V");
    std::map<std::string, void*> bound(decoder.bound.begin(), decoder.bound.end());
    check(bound.size() == decoder.bound.size(), "no input is bound twice");
    for (int32_t layer = 0; layer < kLayers; ++layer) {
        const std::string key = "cross_k_" + std::to_string(layer);
        const std::string value = "cross_v_" + std::to_string(layer);
        check(bound[key] == &g_buffers[2 * layer], "a layer's K input reads that layer's K output");
        check(bound[value] == &g_buffers[2 * layer + 1],
              "a layer's V input reads that layer's V output");
    }
}

} // namespace

int main() {
    test_cross_input_rank();
    test_padded_rows_values();
    test_padded_rows_match_the_raw_plan();
    test_matching_engines_have_no_mismatch();
    test_each_disagreement_is_named();
    test_binding_gives_each_input_its_own_encoder_output();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("whisper cross-attention contract checks passed\n");
    return 0;
}
