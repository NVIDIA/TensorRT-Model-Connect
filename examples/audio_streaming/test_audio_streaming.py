# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-free CLI seam for the public persistent audio-streaming example."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import textwrap

import pytest


REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples/audio_streaming"


@pytest.fixture(scope="module")
def example_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("a C++ compiler is not installed")
    output = tmp_path_factory.mktemp("audio-streaming")
    fake_loader = output / "fake_loader.cpp"
    fake_loader.write_text(
        textwrap.dedent(
            r"""
            #include "trtmc/runtime/family_loader.h"
            #include "trtmc/task.h"
            #include "trtmc/trtmc.h"

            #include <cstdint>
            #include <memory>
            #include <stdexcept>
            #include <string>
            #include <vector>

            // Keep the existing loader-only unit seam independent of an SDK
            // installation. This stub supplies metadata for the OLD branch;
            // real SDK execution is tested separately against the loaded DSO.
            struct trtmc_bundle {};
            extern "C" trtmc_status TRTMC_CALL trtmc_get_api(
                uint32_t major, uint32_t minor, const trtmc_core_api_v1** out) {
                static const auto api = [] {
                    trtmc_core_api_v1 value{};
                    value.header = {1, 0, sizeof(value)};
                    value.bundle_open = [](trtmc_string_view, trtmc_bundle** result, trtmc_error** error) {
                        *error = nullptr;
                        *result = new trtmc_bundle;
                        return trtmc_status{TRTMC_OK};
                    };
                    value.bundle_release = [](trtmc_bundle* bundle) { delete bundle; };
                    value.bundle_info = [](const trtmc_bundle*, trtmc_bundle_info_v1* info, trtmc_error** error) {
                        *error = nullptr;
                        *info = {};
                        info->format = 1;
                        info->family = {"unit_fixture", 12};
                        info->task = {"audio_generation", 16};
                        info->backend = {"fake", 4};
                        return trtmc_status{TRTMC_OK};
                    };
                    value.error_release = [](trtmc_error*) {};
                    value.model_load = [](trtmc_string_view, const trtmc_load_options_v1*,
                                           trtmc_model** result, trtmc_error** error) {
                        *result = nullptr;
                        *error = nullptr;
                        return trtmc_status{TRTMC_UNSUPPORTED};
                    };
                    value.model_release = [](trtmc_model*) {};
                    return value;
                }();
                *out = nullptr;
                if (major != 1 || minor != 0)
                    return TRTMC_VERSION_MISMATCH;
                *out = &api;
                return TRTMC_OK;
            }

            namespace {

            class PlainAudio final : public trtmc::IAudioGeneration {
              public:
                trtmc::AudioResult generate_audio(
                    const std::string&, const trtmc::AudioGenerationConfig&) override {
                    return {};
                }
            };

            class StreamingAudio final : public trtmc::IAudioGeneration,
                                         public trtmc::IStreamingAudioGeneration {
              public:
                trtmc::AudioResult generate_audio(
                    const std::string&, const trtmc::AudioGenerationConfig&) override {
                    return {};
                }

                std::int32_t generate_audio_streaming(
                    const std::string& prompt, const trtmc::AudioGenerationConfig& config,
                    trtmc::AudioChunkCallback callback, std::int32_t chunk_frames) override {
                    if (config.max_new_tokens != 7 || chunk_frames != 2)
                        throw std::runtime_error("request options were not forwarded");
                    std::vector<std::vector<float>> chunks;
                    if (prompt == "first prompt")
                        chunks = {{0.25F, -0.5F}, {0.75F}};
                    else if (prompt == "second prompt")
                        chunks = {{-1.0F, 0.5F}};
                    else
                        throw std::runtime_error("unexpected prompt");
                    std::int32_t total = 0;
                    for (const auto& chunk : chunks) {
                        callback(chunk.data(), static_cast<std::int32_t>(chunk.size()), 22050);
                        total += static_cast<std::int32_t>(chunk.size());
                    }
                    return total;
                }
            };

            } // namespace

            namespace trtmc {

            std::unique_ptr<ITask> load_task(const std::string& bundle,
                                             const std::string& runtime_root,
                                             std::uint64_t kv_cache_size_bytes,
                                             const std::string& runtime_cache_path,
                                             bool cuda_graphs) {
                static int loads = 0;
                if (++loads != 1)
                    throw std::runtime_error("bundle was loaded more than once");
                if (kv_cache_size_bytes != 0 || !runtime_cache_path.empty() || cuda_graphs)
                    throw std::runtime_error("unexpected runtime options");
                if (runtime_root != "runtime")
                    throw std::runtime_error("unexpected runtime root");
                if (bundle == "model.bundle")
                    return std::make_unique<StreamingAudio>();
                if (bundle == "plain.bundle")
                    return std::make_unique<PlainAudio>();
                throw std::runtime_error("unexpected bundle");
            }

            } // namespace trtmc
            """
        ),
        encoding="utf-8",
    )
    binary = output / "trtmc_audio_streaming"
    completed = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(REPO / "core/runtime/include"),
            "-I",
            str(REPO / "core/api/include"),
            "-I",
            str(REPO / "apps"),
            str(EXAMPLE / "main.cpp"),
            str(fake_loader),
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    return binary


def test_streams_two_prompts_after_one_load(example_binary: Path) -> None:
    completed = subprocess.run(
        [
            str(example_binary),
            "model.bundle",
            "--runtime-root",
            "runtime",
            "--chunk-frames",
            "2",
            "--max-new-tokens",
            "7",
        ],
        input=b"  \nfirst prompt\n\nsecond prompt\n",
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert struct.unpack("=7f", completed.stdout) == pytest.approx(
        (0.25, -0.5, 0.75, 0.0, -1.0, 0.5, 0.0)
    )
    status = completed.stderr.decode()
    assert "ready; reading one prompt per line" in status
    assert "utterance 1 done; samples=3; sample_rate=22050; reported_samples=3" in status
    assert "utterance 2 done; samples=2; sample_rate=22050; reported_samples=2" in status
    assert "EOF; utterances=2" in status


@pytest.mark.parametrize(
    "arguments",
    (
        (),
        ("model.bundle",),
        ("model.bundle", "--runtime-root", "runtime", "extra.bundle"),
        ("model.bundle", "--runtime-root", "runtime", "--unknown"),
        ("model.bundle", "--runtime-root", "runtime", "--chunk-frames", "0"),
        ("model.bundle", "--runtime-root", "runtime", "--chunk-frames", "2x"),
        (
            "model.bundle",
            "--runtime-root",
            "runtime",
            "--runtime-root",
            "runtime",
        ),
    ),
)
def test_rejects_invalid_arguments_before_loading(
    example_binary: Path, arguments: tuple[str, ...]
) -> None:
    completed = subprocess.run(
        [str(example_binary), *arguments],
        input=b"first prompt\n",
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert completed.stdout == b""
    assert b"Usage:" in completed.stderr
    assert b"loading bundle" not in completed.stderr


def test_rejects_a_non_streaming_audio_task(example_binary: Path) -> None:
    completed = subprocess.run(
        [str(example_binary), "plain.bundle", "--runtime-root", "runtime"],
        input=b"first prompt\n",
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 1
    assert completed.stdout == b""
    assert b"does not implement streaming audio generation" in completed.stderr


def test_help_does_not_write_to_the_pcm_channel(example_binary: Path) -> None:
    completed = subprocess.run(
        [str(example_binary), "--help"],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stdout == b""
    assert b"Usage:" in completed.stderr


def test_cmake_links_public_sdk_and_existing_runtime_targets() -> None:
    cmake = (EXAMPLE / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (EXAMPLE / "main.cpp").read_text(encoding="utf-8")
    assert "find_package(trtmc CONFIG REQUIRED COMPONENTS runtime)" in cmake
    assert "trtmc::c" in cmake
    assert "trtmc::trtmc_runtime" in cmake
    assert "families/" not in cmake + source
