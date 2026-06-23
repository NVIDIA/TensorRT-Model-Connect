// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-11
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech depth plan: projection view dimensions, input embedding, prev token
// resolution Preconditions:  Depth plan config with valid projection and embedding parameters
// Postconditions: Dimensions derived correctly, embeddings use text or audio seed, forced audio
// preferred
// =============================================================================

#include "runtime/models/speech/speech_depth_plan.h"

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

trtmc::SpeechConfig make_config() {
    trtmc::SpeechConfig cfg;
    cfg.depth_hidden_size = 2;
    cfg.temporal_hidden_size = 3;
    cfg.depth_text_vocab = 4;
    cfg.audio_vocab_size = 8;
    cfg.num_depformer_emb = 2;
    cfg.depth_text_embedding = {
        0.0F, 0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
    };
    cfg.depth_audio_embeddings = {
        10.0F, 11.0F, 12.0F, 13.0F, 14.0F, 15.0F, 16.0F, 17.0F, 18.0F, 19.0F, 20.0F,
        21.0F, 22.0F, 23.0F, 24.0F, 25.0F, 30.0F, 31.0F, 32.0F, 33.0F, 34.0F, 35.0F,
        36.0F, 37.0F, 38.0F, 39.0F, 40.0F, 41.0F, 42.0F, 43.0F, 44.0F, 45.0F,
    };
    cfg.depth_projection = {
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F, 0.5F, 0.0F,
    };
    return cfg;
}

void test_projection_view_derives_dimensions_and_flag() {
    const trtmc::SpeechConfig cfg = make_config();
    const float temporal_hidden[] = {2.0F, 4.0F, 6.0F};
    const auto view = trtmc::make_depth_projection_view(cfg, temporal_hidden);

    check(view.has_projection,
          "speech depth projection view enables projection when weights exist");
    check(view.depth_hidden == 2, "speech depth projection view tracks depth hidden size");
    check(view.temporal_hidden_dim == 3,
          "speech depth projection view tracks temporal hidden size");
    check(view.proj_size_per_cb == 6,
          "speech depth projection view computes per-codebook projection span");
}

void test_build_depth_input_embedding_uses_text_or_audio_seed_and_projection() {
    const trtmc::SpeechConfig cfg = make_config();
    const float temporal_hidden[] = {2.0F, 4.0F, 6.0F};
    const auto view = trtmc::make_depth_projection_view(cfg, temporal_hidden);

    std::vector<float> embed(2, 0.0F);
    trtmc::build_depth_input_embedding(cfg, view, 0, 2, 1, 2, embed);
    check_close(embed[0], 5.0F, 1e-6F, "speech depth text seed plus projection dim0");
    check_close(embed[1], 8.0F, 1e-6F, "speech depth text seed plus projection dim1");

    trtmc::build_depth_input_embedding(cfg, view, 1, 2, 3, 2, embed);
    check_close(embed[0], 17.0F, 1e-6F, "speech depth audio seed plus projection dim0");
    check_close(embed[1], 19.0F, 1e-6F, "speech depth audio seed plus projection dim1");
}

void test_resolve_depth_prev_token_prefers_forced_audio_when_present() {
    trtmc::SpeechConfig cfg = make_config();
    cfg.audio_vocab_size = 5;
    const int32_t forced_tokens[] = {9, 3};
    const uint8_t forced_provided[] = {0, 1};

    check(trtmc::resolve_depth_prev_token(0, 2, cfg, forced_tokens, forced_provided) == 2,
          "speech depth prev token keeps sampled token when forced token missing");
    check(trtmc::resolve_depth_prev_token(1, 2, cfg, forced_tokens, forced_provided) == 3,
          "speech depth prev token uses forced token when provided");
    check(trtmc::resolve_depth_prev_token(1, 2, cfg, nullptr, nullptr) == 2,
          "speech depth prev token falls back when no forced arrays exist");
}

} // namespace

int main() {
    test_projection_view_derives_dimensions_and_flag();
    test_build_depth_input_embedding_uses_text_or_audio_seed_and_projection();
    test_resolve_depth_prev_token_prefers_forced_audio_when_present();

    if (g_failures != 0) {
        std::cerr << g_failures << " speech depth plan test(s) failed\n";
        return 1;
    }
    return 0;
}
