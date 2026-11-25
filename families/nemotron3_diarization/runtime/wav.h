/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

namespace trtmc::nemotron3_diarization {

// One WAV file as interleaved float samples, for the family's command line.
// PCM16 decodes as value / 32768 and float32 as stored; every other encoding
// is refused. The model checks the sample rate and the samples themselves.
struct WavAudio {
    std::vector<float> samples;
    std::uint32_t sample_rate{0};
    std::uint32_t channels{0};
};

// The bytes of a RIFF/WAVE file: exactly one fmt and one data chunk, every
// chunk and its pad byte inside the RIFF payload, whole sample frames, at
// least one frame.
WavAudio parse_wav(const std::vector<char>& bytes);
// Reads the RIFF header of a stream of `size` bytes, then no more than the
// payload it declares: nothing after the RIFF chunk belongs to the audio.
WavAudio read_wav(std::istream& input, std::uint64_t size);
WavAudio read_wav(const std::string& path);

} // namespace trtmc::nemotron3_diarization
