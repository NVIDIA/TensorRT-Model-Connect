/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The family's command-line adapter without a model: its WAV reader, and the
// order in which it refuses a call (command, bundle identity, input, load).
#include "families/nemotron3_diarization/runtime/wav.h"
#include "trtmc/internal/cli.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
using Bytes = std::vector<char>;
using trtmc::nemotron3_diarization::parse_wav;
using trtmc::nemotron3_diarization::read_wav;
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void put(Bytes& out, std::uint32_t value, unsigned width) {
    for (unsigned i = 0; i < width; ++i)
        out.push_back(static_cast<char>((value >> (8U * i)) & 0xFFU));
}

Bytes chunk(const char* id, const Bytes& body) {
    Bytes out(id, id + 4);
    put(out, static_cast<std::uint32_t>(body.size()), 4);
    out.insert(out.end(), body.begin(), body.end());
    if (body.size() % 2 != 0)
        out.push_back(0);
    return out;
}

Bytes format(std::uint32_t encoding, std::uint32_t channels, std::uint32_t rate, std::uint32_t bits,
             std::uint32_t align) {
    Bytes body;
    put(body, encoding, 2);
    put(body, channels, 2);
    put(body, rate, 4);
    put(body, rate * align, 4);
    put(body, align, 2);
    put(body, bits, 2);
    return chunk("fmt ", body);
}

Bytes riff(const std::vector<Bytes>& chunks) {
    Bytes payload{'W', 'A', 'V', 'E'};
    for (const auto& part : chunks)
        payload.insert(payload.end(), part.begin(), part.end());
    Bytes out{'R', 'I', 'F', 'F'};
    put(out, static_cast<std::uint32_t>(payload.size()), 4);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// Rewrites the RIFF size so that the payload is every byte after the header.
void cover(Bytes& bytes) {
    Bytes size;
    put(size, static_cast<std::uint32_t>(bytes.size() - 8), 4);
    std::copy(size.begin(), size.end(), bytes.begin() + 4);
}

Bytes pcm16(const std::vector<std::int32_t>& codes) {
    Bytes body;
    for (const auto code : codes)
        put(body, static_cast<std::uint32_t>(code) & 0xFFFFU, 2);
    return body;
}

void refused(const Bytes& bytes, const std::string& reason) {
    try {
        parse_wav(bytes);
        check(false, "accepted a WAV that should fail with: " + reason);
    } catch (const std::invalid_argument& error) {
        check(std::string(error.what()).find(reason) != std::string::npos,
              "expected \"" + reason + "\", got \"" + error.what() + "\"");
    }
}

void test_wav_decoding() {
    const auto mono = parse_wav(
        riff({format(1, 1, 16000, 16, 2), chunk("data", pcm16({0, 16384, -32768, 32767}))}));
    check(mono.sample_rate == 16000 && mono.channels == 1 &&
              mono.samples == std::vector<float>{0.0F, 0.5F, -1.0F, 32767.0F / 32768.0F},
          "PCM16 decodes as value / 32768");

    const std::vector<float> values{0.25F, -0.5F, 1.5F, -0.0F};
    Bytes body;
    for (const float value : values) {
        std::uint32_t word = 0;
        std::memcpy(&word, &value, sizeof(word));
        put(body, word, 4);
    }
    // An odd-sized chunk ahead of fmt, with its pad byte, is skipped.
    const auto stereo = parse_wav(riff(
        {chunk("LIST", Bytes{'a', 'b', 'c'}), format(3, 2, 8000, 32, 8), chunk("data", body)}));
    check(stereo.sample_rate == 8000 && stereo.channels == 2 && stereo.samples == values &&
              std::signbit(stereo.samples[3]),
          "float32 keeps every sample bit for bit, interleaved");
}

void test_wav_refusals() {
    const auto fmt16 = format(1, 1, 16000, 16, 2);
    const auto data = chunk("data", pcm16({1, 2}));
    refused(Bytes{'R', 'I', 'F', 'F'}, "not a RIFF/WAVE file");
    auto riffx = riff({fmt16, data});
    riffx[3] = 'X';
    refused(riffx, "not a RIFF/WAVE file");
    auto small = riff({fmt16, data});
    small[4] = 3;
    small[5] = small[6] = small[7] = 0;
    refused(small, "does not cover its WAVE tag");
    auto truncated = riff({fmt16, data});
    truncated.pop_back();
    refused(truncated, "payload is truncated");
    auto stub = riff({fmt16, data});
    stub.insert(stub.end(), {'J', 'U', 'N', 'K'});
    cover(stub);
    refused(stub, "chunk header is truncated");
    auto unpadded = riff({fmt16, data});
    for (const char byte : chunk("LIST", Bytes{'a', 'b', 'c'}))
        unpadded.push_back(byte);
    unpadded.pop_back();
    cover(unpadded);
    refused(unpadded, "exceeds the RIFF payload");
    auto overlong = riff({fmt16, data});
    overlong[overlong.size() - 8] = 9;
    refused(overlong, "exceeds the RIFF payload");
    refused(riff({data}), "no fmt chunk");
    refused(riff({fmt16}), "no data chunk");
    refused(riff({fmt16, fmt16, data}), "more than one fmt chunk");
    refused(riff({fmt16, data, data}), "more than one data chunk");
    refused(riff({chunk("fmt ", Bytes(14, 0)), data}), "shorter than 16 bytes");
    refused(riff({format(1, 1, 16000, 24, 3), chunk("data", Bytes(6, 0))}),
            "only PCM16 and float32");
    refused(riff({format(3, 1, 16000, 64, 8), chunk("data", Bytes(8, 0))}),
            "only PCM16 and float32");
    refused(riff({format(0xFFFE, 1, 16000, 16, 2), data}), "only PCM16 and float32");
    refused(riff({format(1, 0, 16000, 16, 0), data}), "must be positive");
    refused(riff({format(1, 1, 0, 16, 2), data}), "must be positive");
    refused(riff({format(1, 2, 16000, 16, 2), data}), "block alignment");
    refused(riff({fmt16, chunk("data", Bytes{})}), "no sample frames");
    refused(riff({format(1, 2, 16000, 16, 4), chunk("data", pcm16({1, 2, 3}))}),
            "inside a sample frame");
}

// Serves bytes from memory and records how many the reader took.
class Tape : public std::streambuf {
  public:
    explicit Tape(Bytes bytes) : bytes_(std::move(bytes)) {
        setg(bytes_.data(), bytes_.data(), bytes_.data() + bytes_.size());
    }
    std::size_t consumed() const { return static_cast<std::size_t>(gptr() - eback()); }

  private:
    Bytes bytes_;
};

void test_wav_reads_only_its_riff_chunk() {
    auto bytes = riff({format(1, 1, 16000, 16, 2), chunk("data", pcm16({8192}))});
    const auto length = bytes.size();
    bytes.resize(length + 4096, 'x');
    Tape tape(bytes);
    std::istream input(&tape);
    const auto audio = read_wav(input, bytes.size());
    check(audio.samples == std::vector<float>{0.25F} && tape.consumed() == length,
          "the bytes after the RIFF chunk are never read");
}

void test_wav_files(const fs::path& root) {
    const auto path = root / "speech.wav";
    const auto bytes = riff({format(1, 1, 16000, 16, 2), chunk("data", pcm16({-16384}))});
    std::ofstream(path, std::ios::binary)
        .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    check(read_wav(path.string()).samples == std::vector<float>{-0.5F}, "a WAV file is read");
    for (const auto& missing : {root / "missing.wav", root}) {
        try {
            read_wav(missing.string());
            check(false, "read a WAV input that is not a file: " + missing.string());
        } catch (const std::invalid_argument& error) {
            check(std::string(error.what()).find("not a readable file") != std::string::npos,
                  "a missing file or a directory is refused before any read");
        }
    }
}

void write_bundle(const fs::path& path, const std::string& family) {
    const auto header =
        nlohmann::json{{"format", 1},
                       {"family", family},
                       {"task", "audio_to_timed_labels"},
                       {"backend", "trt"},
                       {"sections", {{"engine.plan", {{"offset", 0}, {"length", 4}}}}}}
            .dump();
    std::ofstream output(path, std::ios::binary);
    output.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write("PLAN", 4);
}

struct Capture {
    std::string output, error;
};
void capture_output(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->output.append(data, size);
}
void capture_error(void* context, const char* data, std::size_t size) {
    static_cast<Capture*>(context)->error.append(data, size);
}

struct Call {
    int status;
    Capture captured;
};
Call invoke(const char* handler, const std::string& values, const std::string& runtime_root) {
    Capture captured;
    const auto status = trtmc_family_cli_v1(handler, values.c_str(), runtime_root.c_str(),
                                            &captured, capture_output, capture_error);
    return {status, captured};
}

bool refuses(const Call& call, const std::string& reason) {
    return call.status == 1 && call.captured.output.empty() &&
           call.captured.error.rfind("Error: ", 0) == 0 &&
           call.captured.error.find(reason) != std::string::npos;
}

void test_handler(const fs::path& root) {
    const auto ours = root / "diarization.bundle";
    const auto other = root / "other.bundle";
    write_bundle(ours, "nemotron3_diarization");
    write_bundle(other, "another_family");
    const auto wav = root / "speech.wav";
    const auto missing = root / "missing.wav";
    const auto runtime = root / "runtime";
    fs::create_directories(runtime);
    const auto values = [&](const fs::path& bundle, const fs::path& audio) {
        return nlohmann::json{{"bundle", bundle.string()}, {"audio", audio.string()}}.dump();
    };

    check(refuses(invoke("detect", values(ours, wav), runtime.string()), "unknown Nemotron 3"),
          "an undeclared command is refused");
    check(refuses(invoke(nullptr, values(ours, wav), runtime.string()), "incomplete call"),
          "a call without a handler name is refused");
    check(refuses(invoke("diarize", "{", runtime.string()), "parse"),
          "malformed values are reported, not thrown across the C boundary");
    check(refuses(invoke("diarize", R"({"audio": "x.wav"})", runtime.string()), "bundle"),
          "a call without a bundle is refused");
    // Identity comes first: the missing audio of another family's bundle is never read.
    check(refuses(invoke("diarize", values(other, missing), runtime.string()),
                  "requires a nemotron3_diarization bundle"),
          "another family's bundle is refused before its input is read");
    // The input comes before the model: a bad WAV never loads an engine.
    check(
        refuses(invoke("diarize", values(ours, missing), runtime.string()), "not a readable file"),
        "the WAV input is checked before the model loads");
    const auto loaded = invoke("diarize", values(ours, wav), runtime.string());
    check(loaded.status == 1 && loaded.captured.output.empty() &&
              loaded.captured.error.rfind("Error: ", 0) == 0,
          "a model that cannot load is reported as an error");
}

} // namespace

int main() {
    const auto root =
        fs::temp_directory_path() / ("trtmc-nemotron3-diarization-cli-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        test_wav_decoding();
        test_wav_refusals();
        test_wav_reads_only_its_riff_chunk();
        test_wav_files(root);
        test_handler(root);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }
    fs::remove_all(root);
    return failures == 0 ? 0 : 1;
}
