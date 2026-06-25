// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-16
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech temporal embed plan: dual stream summation, vocab clamping, logit padding
// Preconditions:  Temporal embed config with valid stream parameters
// Postconditions: Dual streams summed correctly, tokens clamped to vocab, logits padded/truncated
// =============================================================================

#include "runtime/models/personaplex/speech_temporal_embed_plan.h"

#include <cmath>
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

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

trtmc::SpeechConfig make_test_config() {
    trtmc::SpeechConfig cfg;
    cfg.temporal_text_vocab = 4;
    cfg.audio_vocab_size = 3;
    cfg.temporal_text_embedding = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
    };
    cfg.audio_embeddings = {
        10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 20.0F, 21.0F, 22.0F, 23.0F, 24.0F, 25.0F,
        30.0F, 31.0F, 32.0F, 33.0F, 34.0F, 35.0F, 40.0F, 41.0F, 42.0F, 43.0F, 44.0F, 45.0F,
    };
    return cfg;
}

void test_dual_stream_embed_sums_text_moshi_and_user_streams() {
    auto cfg = make_test_config();
    const int32_t moshi_tokens[] = {1, 2};
    const int32_t user_tokens[] = {0, 2};
    float out[2] = {0.0F, 0.0F};
    trtmc::compute_dual_stream_summed_embed(cfg, 2, 2, moshi_tokens, user_tokens, 1, out);

    check_close(out[0], 3.0F + 12.0F + 24.0F + 30.0F + 44.0F, 1e-6F, "dual stream embed value 0");
    check_close(out[1], 4.0F + 13.0F + 25.0F + 31.0F + 45.0F, 1e-6F, "dual stream embed value 1");
}

void test_temporal_embed_clamps_tokens_to_vocab() {
    auto cfg = make_test_config();
    float out[2] = {0.0F, 0.0F};
    trtmc::add_temporal_text_embedding(cfg, 2, 100, out);
    trtmc::add_temporal_audio_embedding(cfg, 2, 0, 99, out);

    check_close(out[0], 7.0F + 14.0F, 1e-6F, "clamped embed value 0");
    check_close(out[1], 8.0F + 15.0F, 1e-6F, "clamped embed value 1");
}

void test_hidden_helpers_pad_or_truncate_logits() {
    std::vector<float> all_hidden;
    trtmc::append_hidden_from_logits(all_hidden, {1.0F, 2.0F}, 4);
    check(all_hidden.size() == 4, "append hidden extends output");
    check_close(all_hidden[0], 1.0F, 1e-6F, "append hidden first value");
    check_close(all_hidden[1], 2.0F, 1e-6F, "append hidden second value");
    check_close(all_hidden[2], 0.0F, 1e-6F, "append hidden pads third value");
    check_close(all_hidden[3], 0.0F, 1e-6F, "append hidden pads fourth value");

    std::vector<float> frame_hidden(3, -1.0F);
    trtmc::fill_hidden_from_logits(frame_hidden, {9.0F}, 3);
    check_close(frame_hidden[0], 9.0F, 1e-6F, "fill hidden first value");
    check_close(frame_hidden[1], 0.0F, 1e-6F, "fill hidden pads");
    check_close(frame_hidden[2], 0.0F, 1e-6F, "fill hidden pads tail");
}

} // namespace

int main() {
    test_dual_stream_embed_sums_text_moshi_and_user_streams();
    test_temporal_embed_clamps_tokens_to_vocab();
    test_hidden_helpers_pad_or_truncate_logits();

    if (g_failures != 0) {
        std::cerr << g_failures << " speech temporal embed plan test(s) failed\n";
        return 1;
    }
    return 0;
}
