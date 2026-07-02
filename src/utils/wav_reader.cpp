/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace {

struct ParsedWav
{
    uint16_t fmt_tag{0};
    uint16_t channels{0};
    uint32_t sample_rate{0};
    uint16_t bits_per_sample{0};
    const char* raw_ptr{nullptr};
    std::size_t raw_size{0};
    bool have_fmt{false};
    bool have_data{false};
};

std::vector<char> read_wav_bytes(const std::string& path)
{
    std::ifstream infile(path, std::ios::binary);
    if (!infile)
    {
        throw std::runtime_error("Failed to open WAV file: " + path);
    }

    return std::vector<char>(
        (std::istreambuf_iterator<char>(infile)),
        std::istreambuf_iterator<char>());
}

void validate_wav_container(const std::vector<char>& wav_bytes, const std::string& path)
{
    if (wav_bytes.size() < 44)
    {
        throw std::runtime_error("WAV file too small: " + path);
    }

    if (std::memcmp(wav_bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(wav_bytes.data() + 8, "WAVE", 4) != 0)
    {
        throw std::runtime_error("Invalid WAV container: " + path);
    }
}

uint32_t read_chunk_size(const char* chunk_header)
{
    uint32_t chunk_size = 0;
    std::memcpy(&chunk_size, chunk_header + 4, sizeof(uint32_t));
    return chunk_size;
}

bool chunk_fits_buffer(std::size_t pos, uint32_t chunk_size, std::size_t total_size)
{
    return pos + 8 + static_cast<std::size_t>(chunk_size) <= total_size;
}

void parse_fmt_chunk(const char* chunk_data, uint32_t chunk_size, ParsedWav& parsed)
{
    if (chunk_size < 16)
    {
        throw std::runtime_error("WAV fmt chunk too small");
    }

    std::memcpy(&parsed.fmt_tag, chunk_data + 0, sizeof(uint16_t));
    std::memcpy(&parsed.channels, chunk_data + 2, sizeof(uint16_t));
    std::memcpy(&parsed.sample_rate, chunk_data + 4, sizeof(uint32_t));
    std::memcpy(&parsed.bits_per_sample, chunk_data + 14, sizeof(uint16_t));
    parsed.have_fmt = true;
}

void parse_data_chunk(const char* chunk_data, uint32_t chunk_size, ParsedWav& parsed)
{
    parsed.raw_ptr = chunk_data;
    parsed.raw_size = static_cast<std::size_t>(chunk_size);
    parsed.have_data = true;
}

void parse_chunks(const std::vector<char>& wav_bytes, ParsedWav& parsed)
{
    std::size_t pos = 12;
    while (pos + 8 <= wav_bytes.size())
    {
        const char* chunk = wav_bytes.data() + pos;
        const char* chunk_data = chunk + 8;
        const uint32_t chunk_size = read_chunk_size(chunk);
        if (!chunk_fits_buffer(pos, chunk_size, wav_bytes.size()))
        {
            break;
        }

        if (std::memcmp(chunk, "fmt ", 4) == 0)
        {
            parse_fmt_chunk(chunk_data, chunk_size, parsed);
        }
        else if (std::memcmp(chunk, "data", 4) == 0)
        {
            parse_data_chunk(chunk_data, chunk_size, parsed);
        }

        pos += 8 + static_cast<std::size_t>(chunk_size);
        if ((chunk_size & 1U) != 0)
        {
            ++pos;  // RIFF chunks are word-aligned
        }
    }
}

void validate_required_chunks(const ParsedWav& parsed, const std::string& path)
{
    if (!parsed.have_fmt || !parsed.have_data || parsed.raw_ptr == nullptr || parsed.raw_size == 0)
    {
        throw std::runtime_error("WAV missing fmt/data chunk: " + path);
    }
}

std::vector<float> decode_float32_samples(const ParsedWav& parsed)
{
    const auto ns = static_cast<int32_t>(parsed.raw_size / sizeof(float));
    std::vector<float> samples(ns);
    std::memcpy(samples.data(), parsed.raw_ptr, static_cast<std::size_t>(ns) * sizeof(float));
    return samples;
}

std::vector<float> decode_pcm16_samples(const ParsedWav& parsed)
{
    const auto ns = static_cast<int32_t>(parsed.raw_size / sizeof(int16_t));
    std::vector<float> samples(ns);
    for (int32_t i = 0; i < ns; ++i)
    {
        int16_t pcm = 0;
        std::memcpy(&pcm, parsed.raw_ptr + static_cast<std::size_t>(i) * sizeof(int16_t), sizeof(int16_t));
        samples[i] = static_cast<float>(pcm) / 32768.0F;
    }
    return samples;
}

std::vector<float> decode_samples(const ParsedWav& parsed)
{
    if (parsed.fmt_tag == 3 && parsed.bits_per_sample == 32)
    {
        return decode_float32_samples(parsed);
    }

    if (parsed.fmt_tag == 1 && parsed.bits_per_sample == 16)
    {
        return decode_pcm16_samples(parsed);
    }

    throw std::runtime_error(
        "Unsupported WAV format: tag=" + std::to_string(parsed.fmt_tag) +
        " bits=" + std::to_string(parsed.bits_per_sample));
}

std::vector<float> stereo_to_mono(const std::vector<float>& samples)
{
    const auto mono_len = static_cast<int32_t>(samples.size()) / 2;
    std::vector<float> mono(mono_len);
    for (int32_t i = 0; i < mono_len; ++i)
    {
        mono[i] = (samples[2 * i] + samples[2 * i + 1]) * 0.5F;
    }
    return mono;
}

std::vector<float> first_channel_to_mono(const std::vector<float>& samples, uint16_t channels)
{
    const auto mono_len = static_cast<int32_t>(samples.size()) / channels;
    std::vector<float> mono(mono_len);
    for (int32_t i = 0; i < mono_len; ++i)
    {
        mono[i] = samples[static_cast<std::size_t>(i) * channels];
    }
    return mono;
}

std::vector<float> convert_to_mono(std::vector<float> samples, uint16_t channels)
{
    if (channels == 2)
    {
        return stereo_to_mono(samples);
    }

    if (channels > 2)
    {
        return first_channel_to_mono(samples, channels);
    }

    return samples;
}

int32_t compute_output_length(int32_t n_samples, int32_t source_rate, int32_t target_rate)
{
    return static_cast<int32_t>(
        static_cast<int64_t>(n_samples) * target_rate / source_rate);
}

double scaled_sinc(double distance, double cutoff)
{
    constexpr double kPi = 3.14159265358979323846;
    if (std::abs(distance) < 1e-12)
    {
        return cutoff;
    }
    return cutoff * std::sin(kPi * distance * cutoff) / (kPi * distance * cutoff);
}

double hann_window(double distance, int32_t half_taps)
{
    constexpr double kPi = 3.14159265358979323846;
    const double win_pos = (distance + static_cast<double>(half_taps)) /
        (2.0 * static_cast<double>(half_taps));
    return 0.5 * (1.0 - std::cos(2.0 * kPi * win_pos));
}

float resample_at_position(const float* samples, int32_t n_samples, double src_pos, double cutoff, int32_t half_taps)
{
    const auto center = static_cast<int32_t>(std::floor(src_pos));
    const int32_t lo = std::max(0, center - half_taps + 1);
    const int32_t hi = std::min(n_samples - 1, center + half_taps);

    double acc = 0.0;
    double weight_sum = 0.0;
    for (int32_t j = lo; j <= hi; ++j)
    {
        const double distance = static_cast<double>(j) - src_pos;
        const double weight = scaled_sinc(distance, cutoff) * hann_window(distance, half_taps);
        acc += static_cast<double>(samples[j]) * weight;
        weight_sum += weight;
    }

    if (weight_sum > 1e-12)
    {
        return static_cast<float>(acc / weight_sum);
    }
    return 0.0F;
}

} // namespace

WavData read_wav(const std::string& path)
{
    const std::vector<char> wav_bytes = read_wav_bytes(path);
    validate_wav_container(wav_bytes, path);

    ParsedWav parsed;
    parse_chunks(wav_bytes, parsed);
    validate_required_chunks(parsed, path);

    std::vector<float> samples = decode_samples(parsed);
    samples = convert_to_mono(std::move(samples), parsed.channels);

    WavData result;
    result.samples = std::move(samples);
    result.sample_rate = static_cast<int32_t>(parsed.sample_rate);
    return result;
}

std::vector<float> resample_linear(
    const float* samples, int32_t n_samples,
    int32_t source_rate, int32_t target_rate)
{
    if (source_rate == target_rate || n_samples <= 0)
        return std::vector<float>(samples, samples + n_samples);

    const int32_t out_len = compute_output_length(n_samples, source_rate, target_rate);
    std::vector<float> resampled(out_len);

    const int32_t half_taps = 16;
    const double cutoff = std::min(1.0, static_cast<double>(target_rate) /
                                        static_cast<double>(source_rate));

    for (int32_t i = 0; i < out_len; ++i)
    {
        const double src_pos = static_cast<double>(i) *
            static_cast<double>(source_rate) /
            static_cast<double>(target_rate);
        resampled[i] = resample_at_position(samples, n_samples, src_pos, cutoff, half_taps);
    }
    return resampled;
}

} // namespace trtmc
