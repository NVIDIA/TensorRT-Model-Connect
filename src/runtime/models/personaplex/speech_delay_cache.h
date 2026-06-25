#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct DelayCacheState {
    int32_t total_k{0};
    int32_t cache_size{0};
    int32_t max_delay{0};
    std::vector<int32_t> delays;
    std::vector<int32_t> cache;
    std::vector<uint8_t> provided;
};

inline std::vector<int32_t> make_default_speech_delays(int32_t num_codebooks) {
    const int32_t stream_cb = std::max(num_codebooks / 2, 0);
    std::vector<int32_t> delays(static_cast<std::size_t>(num_codebooks + 1), 1);
    if (delays.empty()) {
        return delays;
    }

    delays[0] = 0;
    if (num_codebooks > 0) {
        delays[1] = 0;
    }
    const int32_t first_user_stream = 1 + stream_cb;
    if (first_user_stream >= 0 && first_user_stream < static_cast<int32_t>(delays.size())) {
        delays[static_cast<std::size_t>(first_user_stream)] = 0;
    }
    return delays;
}

inline std::size_t delay_cache_index(const DelayCacheState& state, int32_t k, int32_t pos) {
    return static_cast<std::size_t>(k) * state.cache_size + (pos % state.cache_size);
}

inline DelayCacheState make_delay_cache_state(const std::vector<int32_t>& configured_delays,
                                              int32_t num_codebooks) {
    DelayCacheState state;
    state.total_k = num_codebooks + 1;
    state.delays = configured_delays.size() >= static_cast<std::size_t>(state.total_k)
                       ? std::vector<int32_t>(configured_delays.begin(),
                                              configured_delays.begin() + state.total_k)
                       : make_default_speech_delays(num_codebooks);

    state.max_delay = 0;
    for (int32_t delay : state.delays) {
        state.max_delay = std::max(state.max_delay, delay);
    }

    state.cache_size = state.max_delay + 3;
    constexpr int32_t kUngenerated = -2;
    state.cache.assign(static_cast<std::size_t>(state.total_k) * state.cache_size, kUngenerated);
    state.provided.assign(static_cast<std::size_t>(state.total_k) * state.cache_size, 0);
    return state;
}

inline void write_user_tokens_to_delay_cache(DelayCacheState& delay_state,
                                             const std::vector<int32_t>& codec_tokens,
                                             int32_t offset, int32_t stream_cb, int32_t num_frames,
                                             int32_t encode_codebooks, int32_t audio_bos) {
    for (int32_t cb = 0; cb < stream_cb; ++cb) {
        const int32_t k = stream_cb + 1 + cb;
        int32_t user_tok = audio_bos;
        if (offset < num_frames) {
            const auto tok_idx = static_cast<std::size_t>(offset) * encode_codebooks + cb;
            if (tok_idx < codec_tokens.size()) {
                user_tok = codec_tokens[tok_idx];
            }
        }
        const auto widx = delay_cache_index(
            delay_state, k, offset + delay_state.delays[static_cast<std::size_t>(k)]);
        delay_state.cache[widx] = user_tok;
        delay_state.provided[widx] = 1;
    }
}

inline void fill_initial_delay_tokens(DelayCacheState& delay_state, int32_t offset,
                                      int32_t text_bos, int32_t audio_bos) {
    for (int32_t k = 0; k < delay_state.total_k; ++k) {
        if (offset > delay_state.delays[static_cast<std::size_t>(k)]) {
            continue;
        }
        const int32_t init_tok = (k == 0) ? text_bos : audio_bos;
        const auto idx = delay_cache_index(delay_state, k, offset);
        delay_state.cache[idx] = init_tok;
        delay_state.provided[idx] = 1;
    }
}

inline void seed_delay_offset_zero(DelayCacheState& delay_state, int32_t text_bos,
                                   int32_t audio_bos) {
    for (int32_t k = 0; k < delay_state.total_k; ++k) {
        delay_state.cache[delay_cache_index(delay_state, k, 0)] = (k == 0) ? text_bos : audio_bos;
    }
}

inline void read_model_inputs_from_delay_cache(const DelayCacheState& delay_state,
                                               int32_t model_input_pos, int32_t stream_cb,
                                               int32_t& text_input,
                                               std::vector<int32_t>& moshi_input,
                                               std::vector<int32_t>& user_input) {
    text_input = delay_state.cache[delay_cache_index(delay_state, 0, model_input_pos)];
    for (int32_t cb = 0; cb < stream_cb; ++cb) {
        moshi_input[static_cast<std::size_t>(cb)] =
            delay_state.cache[delay_cache_index(delay_state, 1 + cb, model_input_pos)];
        user_input[static_cast<std::size_t>(cb)] =
            delay_state.cache[delay_cache_index(delay_state, stream_cb + 1 + cb, model_input_pos)];
    }
}

inline void build_target_audio_arrays(const DelayCacheState& delay_state, int32_t target_pos,
                                      int32_t num_cb, int32_t audio_bos,
                                      std::vector<int32_t>& target_audio_tokens,
                                      std::vector<uint8_t>& target_audio_provided) {
    std::fill(target_audio_tokens.begin(), target_audio_tokens.end(), audio_bos);
    std::fill(target_audio_provided.begin(), target_audio_provided.end(), 0);
    for (int32_t cb = 0; cb < num_cb; ++cb) {
        const auto idx = delay_cache_index(delay_state, 1 + cb, target_pos);
        target_audio_tokens[static_cast<std::size_t>(cb)] = delay_state.cache[idx];
        target_audio_provided[static_cast<std::size_t>(cb)] =
            static_cast<uint8_t>(delay_state.provided[idx] != 0);
    }
}

inline void clear_provided_flags_at_pos(DelayCacheState& delay_state, int32_t pos) {
    for (int32_t k = 0; k < delay_state.total_k; ++k) {
        delay_state.provided[delay_cache_index(delay_state, k, pos)] = 0;
    }
}

inline void write_generated_tokens_to_delay_cache(DelayCacheState& delay_state, int32_t target_pos,
                                                  int32_t sampled_text_token, bool text_provided,
                                                  const std::vector<int32_t>& frame_codes,
                                                  int32_t num_cb) {
    const auto text_target_idx = delay_cache_index(delay_state, 0, target_pos);
    if (!text_provided) {
        delay_state.cache[text_target_idx] = sampled_text_token;
    }

    for (int32_t cb = 0; cb < std::min(static_cast<int32_t>(frame_codes.size()), num_cb); ++cb) {
        const auto idx = delay_cache_index(delay_state, 1 + cb, target_pos);
        if (delay_state.provided[idx] == 0) {
            delay_state.cache[idx] = frame_codes[static_cast<std::size_t>(cb)];
        }
    }
}

inline bool collect_output_codes_from_delay_cache(const DelayCacheState& delay_state,
                                                  int32_t offset, int32_t max_delay,
                                                  int32_t mimi_cb,
                                                  std::vector<int32_t>& output_codes) {
    if (offset <= max_delay) {
        return false;
    }
    const int32_t out_pos = offset - max_delay;
    for (int32_t cb = 0; cb < mimi_cb; ++cb) {
        const int32_t k = 1 + cb;
        const int32_t gather_pos = out_pos + delay_state.delays[static_cast<std::size_t>(k)];
        const int32_t tok = delay_state.cache[delay_cache_index(delay_state, k, gather_pos)];
        output_codes.push_back(tok);
    }
    return true;
}

} // namespace trtmc
