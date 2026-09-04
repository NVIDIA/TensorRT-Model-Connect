/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {
namespace nemotron_voicechat {

constexpr int32_t kCodecSampleRate = 22050;
constexpr int32_t kCodecFrameSamples = 1764;
constexpr int32_t kCodecQuantizers = 31;
constexpr int32_t kCodecCodebookSize = 1024;
constexpr int32_t kCodecFftSize = 16;
constexpr int32_t kCodecHopLength = 4;
constexpr int32_t kCodecSpectralBins = kCodecFftSize / 2 + 1;
constexpr int32_t kCodecSpectralChannels = 2 * kCodecSpectralBins;
constexpr int32_t kCodecSpectralFramesPerFrame = kCodecFrameSamples / kCodecHopLength;
constexpr int32_t kCodecConvCacheWidth = 6;
constexpr int32_t kCodecSpectralCacheFrames = 4;
constexpr int32_t kCodecFlushFrames = 2;
constexpr int32_t kCodecConvBlocks = 9;

static_assert(kCodecSpectralFramesPerFrame == 441, "VoiceChat spectral frame ratio changed");
static_assert(kCodecSpectralFramesPerFrame * kCodecHopLength == kCodecFrameSamples,
              "VoiceChat codec frame must reconstruct to exactly 1764 samples");

// Channels for the nine causal ConvNeXt blocks, in TensorRT binding order.
constexpr std::array<int32_t, kCodecConvBlocks> kCodecConvCacheChannels = {
    1536, 1536, 1536, 768, 768, 768, 384, 384, 384,
};

struct CodecCacheBinding {
    const char* input_name;
    const char* output_name;
    int32_t channels;
};

const std::array<CodecCacheBinding, kCodecConvBlocks>& codec_cache_bindings();

// Host-side double-buffered storage for the explicit TensorRT causal-cache
// bindings.  A runtime may bind current_data() as inputs and next_data() as
// outputs, then call commit() after a successful enqueue.  Keeping this state
// per conversation prevents duplex sessions from leaking audio history.
class CodecCausalCache {
  public:
    CodecCausalCache();

    void reset();
    void commit();

    const float* current_data(int32_t block) const;
    float* next_data(int32_t block);
    std::size_t element_count(int32_t block) const;

  private:
    static std::size_t checked_block(int32_t block);

    std::array<std::vector<float>, kCodecConvBlocks> current_;
    std::array<std::vector<float>, kCodecConvBlocks> next_;
};

// Pure C++ reconstruction of NeMo's Latent2Wav spectral post-processing and
// cached spec_to_wav implementation.  Input is the channel-major TensorRT
// `spectral_params` output: [18, codec_frames * 441].
class CodecReconstruction {
  public:
    using Spectrum = std::array<std::complex<float>, kCodecSpectralBins>;

    CodecReconstruction();

    // Reconstruct one or more complete 80 ms codec frames.  The return size is
    // always codec_frames * 1764 samples.  Causal overlap state is retained.
    std::vector<float> push(const float* spectral_params, int32_t codec_frames);
    std::vector<float> push(const std::vector<float>& spectral_params, int32_t codec_frames);

    // Emit the eight-sample right tail used by NeMo's flush=True decode and
    // reset the spectral state.  Returns empty when no frames were pushed.
    std::vector<float> flush();

    // Drop all overlap history without emitting a tail.
    void reset();

    bool active() const noexcept { return active_; }

  private:
    static Spectrum decode_spectrum(const float* spectral_params, int32_t spectral_frames,
                                    int32_t frame);
    static std::vector<float> inverse_stft(const std::vector<Spectrum>& spectra);
    static std::vector<float> trim_streaming_padding(std::vector<float> waveform);

    std::array<Spectrum, kCodecSpectralCacheFrames> spectral_cache_{};
    bool active_{false};
};

} // namespace nemotron_voicechat
} // namespace trtmc
