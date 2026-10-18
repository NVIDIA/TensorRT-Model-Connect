/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/personaplex/runtime/resampler.h"

#include <cmath>
#include <cstddef>
#include <dlfcn.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

// Opaque types from the public libsoxr C API. No libsoxr header or link-time
// dependency is needed by this family or by the other family DSOs.
extern "C" {
struct soxr;
struct soxr_io_spec;
struct soxr_quality_spec;
struct soxr_runtime_spec;
}

namespace trtmc {
namespace {

class SoxrLibrary {
  public:
    using Create = soxr* (*)(double, double, unsigned, const char**, const soxr_io_spec*,
                             const soxr_quality_spec*, const soxr_runtime_spec*);
    using Process = const char* (*)(soxr*, const void*, std::size_t, std::size_t*, void*,
                                    std::size_t, std::size_t*);
    using Delete = void (*)(soxr*);

    SoxrLibrary() {
        handle_.reset(dlopen("libsoxr.so.0", RTLD_NOW | RTLD_LOCAL));
        if (!handle_)
            throw std::runtime_error(
                std::string("PersonaPlex resampling requires libsoxr.so.0 (install libsoxr0): ") +
                dlerror());
        create = load<Create>("soxr_create");
        process = load<Process>("soxr_process");
        destroy = load<Delete>("soxr_delete");
    }

    Create create{};
    Process process{};
    Delete destroy{};

  private:
    template <typename T>
    T load(const char* symbol) const {
        dlerror();
        void* address = dlsym(handle_.get(), symbol);
        const char* error = dlerror();
        if (error || !address)
            throw std::runtime_error(std::string("PersonaPlex cannot resolve libsoxr symbol ") +
                                     symbol + (error ? std::string(": ") + error : ""));
        return reinterpret_cast<T>(address);
    }

    struct CloseLibrary {
        void operator()(void* handle) const noexcept { dlclose(handle); }
    };
    std::unique_ptr<void, CloseLibrary> handle_;
};

void check_soxr(const char* error) {
    if (error)
        throw std::runtime_error(std::string("PersonaPlex resampling failed: ") + error);
}

} // namespace

std::vector<float> resample_audio(const float* samples, std::int32_t num_samples,
                                  std::int32_t source_rate, std::int32_t target_rate) {
    if (!samples || num_samples < 0 || source_rate <= 0 || target_rate <= 0)
        throw std::invalid_argument("resampler requires samples and positive rates");
    if (source_rate == target_rate)
        return {samples, samples + num_samples};
    if (num_samples == 0)
        return {};

    // Match librosa's floating-point length rule and retain its zero padding
    // after the samples emitted by soxr, including the extra rounding sample.
    const double ratio = static_cast<double>(target_rate) / source_rate;
    const double count = std::ceil(static_cast<double>(num_samples) * ratio);
    if (count > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("resampled audio exceeds the supported sample count");
    std::vector<float> output(static_cast<std::size_t>(count), 0.0F);

    static const SoxrLibrary library;
    const char* error = nullptr;
    // soxr_create defaults to HQ, interleaved float32 I/O and one thread.
    // soxr_oneshot has a different default quality (LQ), so do not use it here.
    std::unique_ptr<soxr, SoxrLibrary::Delete> state(
        library.create(source_rate, target_rate, 1, &error, nullptr, nullptr, nullptr),
        library.destroy);
    check_soxr(error);
    if (!state)
        throw std::runtime_error("PersonaPlex could not create the libsoxr resampler");

    std::size_t consumed = 0, produced = 0;
    check_soxr(library.process(state.get(), samples, static_cast<std::size_t>(num_samples),
                               &consumed, output.data(), output.size(), &produced));
    if (consumed != static_cast<std::size_t>(num_samples))
        throw std::runtime_error("PersonaPlex libsoxr did not consume the full input");
    while (produced < output.size()) {
        std::size_t tail = 0;
        check_soxr(library.process(state.get(), nullptr, 0, nullptr, output.data() + produced,
                                   output.size() - produced, &tail));
        produced += tail;
        if (tail == 0)
            break;
    }
    return output;
}

} // namespace trtmc
