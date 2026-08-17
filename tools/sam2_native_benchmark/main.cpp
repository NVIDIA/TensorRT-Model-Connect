/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "runtime/models/sam2/sam2_jpeg_decoder.h"
#include "runtime/models/sam2/sam2_native_bundle_loader.h"
#include "runtime/models/sam2/sam2_native_video_processor.h"
#include "sam2_benchmark_accuracy.h"
#include "sam2_benchmark_protocol.h"
#include "sam2_checked_plan_module.h"
#include "tools/sam2_native_builder/sam2_engine_builder.h"
#include "utils/sha256.h"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace benchmark = trtmc::sam2::benchmark;
namespace native = trtmc::sam2::native;
using Json = nlohmann::json;

constexpr std::size_t kRgbElements = static_cast<std::size_t>(trtmc::sam2::kOriginalImageHeight) *
                                     trtmc::sam2::kOriginalImageWidth * 3U;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("SAM2 native benchmark: " + message);
}

[[noreturn]] void usageError(const std::string& message) {
    throw std::invalid_argument(
        message + "\nusage: sam2_native_benchmark --checkpoint PATH --config PATH --golden-dir DIR "
                  "--jpeg-dir DIR [--accuracy-only | --baseline-receipt PATH "
                  "--baseline-capture-script PATH] "
                  "--source-root DIR --bundle-output NEW.bundle "
                  "[--q3-receipt-output NEW.json] --receipt-output NEW.json --gpu-device N "
                  "--created-at YYYY-MM-DDTHH:MM:SSZ [--workspace-bytes N]");
}

void requireCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess)
        fail(operation + " failed: " + cudaGetErrorString(status));
}

std::string_view requireValue(int argc, char** argv, int& index, std::string_view option) {
    if (index + 1 >= argc)
        usageError(std::string(option) + " requires a value");
    ++index;
    const std::string_view result(argv[index]);
    if (result.empty())
        usageError(std::string(option) + " requires a nonempty value");
    return result;
}

std::uint64_t parseUnsigned(std::string_view value, std::string_view option) {
    if (value.empty() || (value.size() > 1U && value.front() == '0'))
        usageError(std::string(option) + " requires canonical unsigned decimal");
    std::uint64_t result = 0U;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        usageError(std::string(option) + " requires canonical unsigned decimal");
    return result;
}

struct Options {
    std::filesystem::path checkpoint;
    std::filesystem::path config;
    std::filesystem::path golden_dir;
    std::filesystem::path jpeg_dir;
    std::filesystem::path baseline_receipt;
    std::filesystem::path baseline_capture_script;
    std::filesystem::path source_root;
    std::filesystem::path bundle_output;
    std::filesystem::path q3_receipt_output;
    std::filesystem::path receipt_output;
    std::uint64_t workspace_bytes{native::kDefaultSam2WorkspaceBytes};
    std::int32_t gpu_device{-1};
    std::string created_at_utc;
    bool accuracy_only{false};
};

Options parseArguments(int argc, char** argv) {
    Options result;
    std::array<bool, 12> seen{};
    bool workspace_seen = false;
    bool accuracy_only_seen = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--help") {
            std::cout << "usage: sam2_native_benchmark --checkpoint PATH --config PATH "
                         "--golden-dir DIR --jpeg-dir DIR [--accuracy-only | "
                         "--baseline-receipt PATH --baseline-capture-script PATH] "
                         "--source-root DIR "
                         "--bundle-output NEW.bundle "
                         "[--q3-receipt-output NEW.json] "
                         "--receipt-output NEW.json --gpu-device N "
                         "--created-at YYYY-MM-DDTHH:MM:SSZ [--workspace-bytes N]\n";
            std::exit(0);
        }
        auto unique = [&](std::size_t slot) {
            if (seen[slot])
                usageError(std::string(option) + " may be specified only once");
            seen[slot] = true;
        };
        if (option == "--checkpoint") {
            unique(0);
            result.checkpoint = requireValue(argc, argv, index, option);
        } else if (option == "--config") {
            unique(1);
            result.config = requireValue(argc, argv, index, option);
        } else if (option == "--golden-dir") {
            unique(2);
            result.golden_dir = requireValue(argc, argv, index, option);
        } else if (option == "--jpeg-dir") {
            unique(3);
            result.jpeg_dir = requireValue(argc, argv, index, option);
        } else if (option == "--baseline-receipt") {
            unique(4);
            result.baseline_receipt = requireValue(argc, argv, index, option);
        } else if (option == "--baseline-capture-script") {
            unique(9);
            result.baseline_capture_script = requireValue(argc, argv, index, option);
        } else if (option == "--source-root") {
            unique(10);
            result.source_root = requireValue(argc, argv, index, option);
        } else if (option == "--bundle-output") {
            unique(5);
            result.bundle_output = requireValue(argc, argv, index, option);
        } else if (option == "--q3-receipt-output") {
            unique(11);
            result.q3_receipt_output = requireValue(argc, argv, index, option);
        } else if (option == "--receipt-output") {
            unique(6);
            result.receipt_output = requireValue(argc, argv, index, option);
        } else if (option == "--gpu-device") {
            unique(7);
            const auto value = parseUnsigned(requireValue(argc, argv, index, option), option);
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
                usageError("--gpu-device is outside int32 range");
            result.gpu_device = static_cast<std::int32_t>(value);
        } else if (option == "--created-at") {
            unique(8);
            result.created_at_utc = requireValue(argc, argv, index, option);
        } else if (option == "--workspace-bytes") {
            if (workspace_seen)
                usageError("--workspace-bytes may be specified only once");
            result.workspace_bytes = parseUnsigned(requireValue(argc, argv, index, option), option);
            workspace_seen = true;
        } else if (option == "--accuracy-only") {
            if (accuracy_only_seen)
                usageError("--accuracy-only may be specified only once");
            result.accuracy_only = true;
            accuracy_only_seen = true;
        } else {
            usageError("unsupported option: " + std::string(option));
        }
    }
    constexpr std::array<std::size_t, 9> always_required = {0U, 1U, 2U, 3U, 5U, 6U, 7U, 8U, 10U};
    if (!std::all_of(always_required.begin(), always_required.end(),
                     [&](std::size_t index) { return seen[index]; })) {
        usageError("all required arguments must be specified exactly once");
    }
    if (result.accuracy_only) {
        if (seen[4] || seen[9] || seen[11])
            usageError("baseline and pre-W3 Q3 inputs are not accepted with --accuracy-only");
    } else if (!seen[4] || !seen[9] || !seen[11]) {
        usageError("regular benchmark mode requires baseline inputs and --q3-receipt-output");
    }
    if (result.bundle_output == result.receipt_output ||
        (!result.accuracy_only && (result.bundle_output == result.q3_receipt_output ||
                                   result.receipt_output == result.q3_receipt_output))) {
        usageError("bundle, Q3 receipt, and regular receipt outputs must be different paths");
    }
    return result;
}

void requireAbsent(const std::filesystem::path& path, std::string_view label) {
    struct stat status{};
    if (::lstat(path.c_str(), &status) == 0)
        fail(std::string(label) + " already exists; refusing to overwrite it");
    if (errno != ENOENT)
        fail("unable to inspect " + std::string(label) + ": " + std::strerror(errno));
}

struct FileIdentity {
    dev_t device{};
    ino_t inode{};
    off_t size{};
    timespec modified{};
    timespec changed{};

    bool operator==(const FileIdentity& other) const noexcept {
        return device == other.device && inode == other.inode && size == other.size &&
               modified.tv_sec == other.modified.tv_sec &&
               modified.tv_nsec == other.modified.tv_nsec &&
               changed.tv_sec == other.changed.tv_sec && changed.tv_nsec == other.changed.tv_nsec;
    }
};

struct FileHash {
    std::string sha256;
    FileIdentity identity;
};

struct FileSnapshot {
    std::vector<std::uint8_t> bytes;
    FileHash file;
};

FileIdentity identity(const struct stat& status) {
    return {status.st_dev, status.st_ino, status.st_size, status.st_mtim, status.st_ctim};
}

FileHash hashRegularFile(const std::filesystem::path& path, std::uint64_t maximum_bytes) {
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0)
        fail("unable to open regular file for hashing: " + path.string() + ": " +
             std::strerror(errno));
    struct Closer {
        int descriptor;
        ~Closer() { (void)::close(descriptor); }
    } closer{descriptor};
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size <= 0)
        fail("hash input is not a nonempty regular file: " + path.string());
    if (static_cast<std::uint64_t>(before.st_size) > maximum_bytes)
        fail("hash input exceeds its safety limit: " + path.string());

    trtmc::internal::Sha256 hash;
    std::array<std::uint8_t, 1024U * 1024U> buffer{};
    off_t offset = 0;
    while (offset != before.st_size) {
        const auto requested = static_cast<std::size_t>(
            std::min<off_t>(static_cast<off_t>(buffer.size()), before.st_size - offset));
        ssize_t count = -1;
        do {
            count = ::pread(descriptor, buffer.data(), requested, offset);
        } while (count < 0 && errno == EINTR);
        if (count <= 0)
            fail("short read while hashing: " + path.string());
        hash.update(buffer.data(), static_cast<std::size_t>(count));
        offset += count;
    }
    struct stat after{};
    if (::fstat(descriptor, &after) != 0 || !(identity(before) == identity(after)))
        fail("file changed while it was hashed: " + path.string());
    return {hash.hex_digest(), identity(after)};
}

FileSnapshot snapshotRegularFile(const std::filesystem::path& path, std::uint64_t maximum_bytes) {
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0)
        fail("unable to open regular file snapshot: " + path.string() + ": " +
             std::strerror(errno));
    struct Closer {
        int descriptor;
        ~Closer() { (void)::close(descriptor); }
    } closer{descriptor};
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size <= 0)
        fail("snapshot input is not a nonempty regular file: " + path.string());
    if (static_cast<std::uint64_t>(before.st_size) > maximum_bytes ||
        static_cast<std::uint64_t>(before.st_size) >
            static_cast<std::uint64_t>(std::vector<std::uint8_t>().max_size())) {
        fail("snapshot input exceeds its safety limit: " + path.string());
    }
    FileSnapshot result;
    result.bytes.resize(static_cast<std::size_t>(before.st_size));
    trtmc::internal::Sha256 hash;
    std::size_t offset = 0U;
    while (offset != result.bytes.size()) {
        ssize_t count = -1;
        do {
            count = ::pread(descriptor, result.bytes.data() + offset, result.bytes.size() - offset,
                            static_cast<off_t>(offset));
        } while (count < 0 && errno == EINTR);
        if (count <= 0)
            fail("short read while snapshotting: " + path.string());
        hash.update(result.bytes.data() + offset, static_cast<std::size_t>(count));
        offset += static_cast<std::size_t>(count);
    }
    struct stat after{};
    if (::fstat(descriptor, &after) != 0 || !(identity(before) == identity(after)))
        fail("file changed while it was snapshotted: " + path.string());
    result.file = {hash.hex_digest(), identity(after)};
    return result;
}

std::string hashBytes(const void* data, std::size_t size) {
    trtmc::internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

bool isCanonicalSha256(std::string_view value) {
    return value.size() == 64U && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

void verifyPublishedBundle(const FileHash& actual,
                           const native::BundlePublicationFacts& published) {
    if (!published.filesystem_identity_available || !isCanonicalSha256(published.sha256) ||
        published.size_bytes == 0U || actual.identity.size < 0 ||
        actual.sha256 != published.sha256 ||
        static_cast<std::uint64_t>(actual.identity.size) != published.size_bytes ||
        static_cast<std::uint64_t>(actual.identity.device) != published.device ||
        static_cast<std::uint64_t>(actual.identity.inode) != published.inode) {
        fail("published bundle path does not match the builder's exact completed descriptor");
    }
}

struct SourceClosure {
    std::string manifest_sha256;
    std::string closure_sha256;
};

SourceClosure hashSourceClosure(const std::filesystem::path& source_root) {
    const auto manifest_path = source_root / "tools/sam2_native_benchmark/source_closure.txt";
    const auto manifest = snapshotRegularFile(manifest_path, 1024U * 1024U);
    const std::string text(manifest.bytes.begin(), manifest.bytes.end());
    std::istringstream lines(text);
    std::set<std::string> labels;
    std::string previous_label;
    std::string canonical;
    std::string label;
    while (std::getline(lines, label)) {
        if (label.empty() || label.front() == '/' || label.find('\0') != std::string::npos ||
            label.find("..") != std::string::npos ||
            std::filesystem::path(label).lexically_normal().generic_string() != label ||
            !labels.insert(label).second || (!previous_label.empty() && label <= previous_label)) {
            fail("benchmark source-closure manifest contains an unsafe, duplicate, or unsorted "
                 "path");
        }
        previous_label = label;
        const auto hashed = hashRegularFile(source_root / label, UINT64_C(2) << 30U);
        canonical +=
            label + "\t" + std::to_string(hashed.identity.size) + "\t" + hashed.sha256 + "\n";
    }
    if (labels.empty() || !lines.eof())
        fail("benchmark source-closure manifest is empty or unreadable");
    return {manifest.file.sha256, hashBytes(canonical.data(), canonical.size())};
}

FileHash hashSelfExecutable() {
    // /proc/self/exe is the kernel-owned link to the inode actually executing.
    // It must be followed, then the exact open descriptor is hashed and checked
    // for identity stability.
    const int descriptor = ::open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
        fail("unable to open executing image: " + std::string(std::strerror(errno)));
    struct Closer {
        int descriptor;
        ~Closer() { (void)::close(descriptor); }
    } closer{descriptor};
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<std::uint64_t>(before.st_size) > (UINT64_C(4) << 30U)) {
        fail("executing image is not a bounded nonempty regular file");
    }
    trtmc::internal::Sha256 hash;
    std::array<std::uint8_t, 1024U * 1024U> buffer{};
    off_t offset = 0;
    while (offset != before.st_size) {
        const auto requested = static_cast<std::size_t>(
            std::min<off_t>(static_cast<off_t>(buffer.size()), before.st_size - offset));
        ssize_t count = -1;
        do {
            count = ::pread(descriptor, buffer.data(), requested, offset);
        } while (count < 0 && errno == EINTR);
        if (count <= 0)
            fail("short read while hashing the executing image");
        hash.update(buffer.data(), static_cast<std::size_t>(count));
        offset += count;
    }
    struct stat after{};
    if (::fstat(descriptor, &after) != 0 || !(identity(before) == identity(after)))
        fail("executing image changed while it was hashed");
    return {hash.hex_digest(), identity(after)};
}

std::string currentUtcTimestamp() {
    const auto now = std::time(nullptr);
    if (now == static_cast<std::time_t>(-1))
        fail("unable to read UTC timestamp");
    std::tm value{};
    if (::gmtime_r(&now, &value) == nullptr)
        fail("unable to convert UTC timestamp");
    char result[21]{};
    if (std::snprintf(result, sizeof(result), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                      value.tm_year + 1900, value.tm_mon + 1, value.tm_mday, value.tm_hour,
                      value.tm_min, value.tm_sec) != 20) {
        fail("unable to format UTC timestamp");
    }
    return result;
}

std::string hostname() {
    std::array<char, 256> value{};
    if (::gethostname(value.data(), value.size()) != 0)
        fail("unable to query hostname: " + std::string(std::strerror(errno)));
    if (std::find(value.begin(), value.end(), '\0') == value.end())
        fail("hostname exceeds the benchmark receipt limit");
    return value.data();
}

std::string gpuUuid(const cudaUUID_t& uuid) {
    std::ostringstream output;
    output << "GPU-" << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < sizeof(uuid.bytes); ++index) {
        if (index == 4U || index == 6U || index == 8U || index == 10U)
            output << '-';
        output << std::setw(2)
               << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[index]));
    }
    return output.str();
}

std::string pciBusId(std::int32_t device) {
    std::array<char, 32> value{};
    requireCuda(cudaDeviceGetPCIBusId(value.data(), static_cast<int>(value.size()), device),
                "CUDA PCI bus ID query");
    return value.data();
}

void verifyBaselineReceipt(const FileSnapshot& snapshot) {
    if (snapshot.file.sha256 != benchmark::kBaselineReceiptSha256)
        fail("delivered W3/N100 baseline receipt hash mismatch");
    const auto value = Json::parse(snapshot.bytes.begin(), snapshot.bytes.end());
    if (value.at("warmup_iters").get<std::int32_t>() != 3 ||
        value.at("iters").get<std::int32_t>() != 100 ||
        value.at("num_images").get<std::int32_t>() != 5 || !value.at("rows").is_array() ||
        value.at("rows").size() != 100U)
        fail("delivered baseline W3/N100 workload contract drifted");
    const auto& stages = value.at("stage_summary");
    const auto& prefill = stages.at("preprocess_prefill_ms");
    const auto& tracker = stages.at("infer_tracker_ms");
    const auto& total = stages.at("total_loop_ms");
    const auto close = [](double left, double right) { return std::abs(left - right) <= 1e-12; };
    if (!close(prefill.at("mean_ms").get<double>(), 66.48435074836016) ||
        !close(tracker.at("mean_ms").get<double>(), 190.18253710120916) ||
        !close(total.at("mean_ms").get<double>(), 257.1344714984298) ||
        !close(total.at("median_ms").get<double>(), 253.67085821926594) ||
        !close(total.at("p90_ms").get<double>(), 265.56191593408585))
        fail("delivered baseline total summary drifted");
}

std::array<std::filesystem::path, 5> framePaths(const std::filesystem::path& directory) {
    std::array<std::filesystem::path, 5> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        char name[16]{};
        if (std::snprintf(name, sizeof(name), "%06zu.jpg", index) != 10)
            fail("unable to format exact frame filename");
        result[index] = directory / name;
    }
    return result;
}

class CudaStream final {
  public:
    explicit CudaStream(std::int32_t device) : device_(device) {
        requireCuda(cudaSetDevice(device_), "CUDA device selection");
        requireCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                    "nonblocking CUDA stream creation");
    }

    ~CudaStream() {
        if (stream_ == nullptr)
            return;
        std::int32_t previous = -1;
        if (cudaGetDevice(&previous) == cudaSuccess && previous != device_)
            (void)cudaSetDevice(device_);
        (void)cudaStreamDestroy(stream_);
        if (previous >= 0 && previous != device_)
            (void)cudaSetDevice(previous);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const {
        requireCuda(cudaStreamSynchronize(stream_), "CUDA stream synchronization");
    }

  private:
    cudaStream_t stream_{nullptr};
    std::int32_t device_{-1};
};

class StableFrames final {
  public:
    StableFrames() {
        for (std::size_t index = 0; index < storage_.size(); ++index) {
            storage_[index].resize(kRgbElements);
            views_[index] = {static_cast<std::int32_t>(index),
                             trtmc::sam2::kOriginalImageHeight,
                             trtmc::sam2::kOriginalImageWidth,
                             nullptr,
                             0U,
                             trtmc::Sam2VideoPixelFormat::kUint8Rgb,
                             storage_[index].data(),
                             storage_[index].size()};
        }
    }

    void decodeAll(const std::array<std::vector<std::uint8_t>, 5>& encoded) {
        std::array<std::future<trtmc::sam2::DecodedSam2Jpeg>, 5> decoded_tasks;
        for (std::size_t frame = 0; frame < encoded.size(); ++frame) {
            decoded_tasks[frame] = std::async(std::launch::async, [&encoded, frame] {
                // Keep the retained immutable lvalue copy and JPEG decode in
                // the timed envelope, now on one bounded worker per frame.
                return trtmc::sam2::decodeSam2JpegBytes(encoded[frame]);
            });
        }
        for (std::size_t frame = 0; frame < encoded.size(); ++frame) {
            // Observe errors and publish bytes in chronological frame order.
            // Destruction joins any outstanding future during unwinding.
            const auto decoded = decoded_tasks[frame].get();
            if (decoded.height != trtmc::sam2::kOriginalImageHeight ||
                decoded.width != trtmc::sam2::kOriginalImageWidth ||
                decoded.rgb_hwc.size() != storage_[frame].size())
                fail("decoded JPEG geometry drifted");
            std::copy(decoded.rgb_hwc.begin(), decoded.rgb_hwc.end(), storage_[frame].begin());
        }
    }

    const trtmc::Sam2VideoFrames& views() const noexcept { return views_; }

  private:
    std::array<std::vector<std::uint8_t>, 5> storage_;
    trtmc::Sam2VideoFrames views_{};
};

struct ReplayOutput {
    trtmc::Sam2VideoPromptResult prompt;
    trtmc::Sam2VideoFrameResults results;
};

ReplayOutput executeReplay(trtmc::Sam2VideoProcessor& processor, const CudaStream& stream,
                           StableFrames& frames,
                           const std::array<std::vector<std::uint8_t>, 5>& encoded) {
    processor.reset();
    stream.synchronize();
    frames.decodeAll(encoded);
    auto prompt = processor.run_bbox_prompt(frames.views());
    stream.synchronize();
    auto results = processor.propagate(prompt, frames.views());
    stream.synchronize();
    return {std::move(prompt), std::move(results)};
}

benchmark::TimingRow executeTimedRow(std::size_t index, trtmc::Sam2VideoProcessor& processor,
                                     const CudaStream& stream, StableFrames& frames,
                                     const std::array<std::vector<std::uint8_t>, 5>& encoded) {
    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();
    processor.reset();
    stream.synchronize();
    frames.decodeAll(encoded);
    auto prompt = processor.run_bbox_prompt(frames.views());
    stream.synchronize();
    const auto tracker_start = Clock::now();
    auto results = processor.propagate(prompt, frames.views());
    stream.synchronize();
    const auto total_end = Clock::now();
    (void)results;

    const auto to_nanoseconds = [](Clock::time_point value) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch())
            .count();
    };
    const auto total_start_ns = to_nanoseconds(total_start);
    const auto tracker_start_ns = to_nanoseconds(tracker_start);
    const auto total_end_ns = to_nanoseconds(total_end);
    const auto prefill = tracker_start_ns - total_start_ns;
    const auto tracker = total_end_ns - tracker_start_ns;
    const auto total = total_end_ns - total_start_ns;
    if (prefill <= 0 || tracker <= 0 || total <= 0 || prefill + tracker != total)
        fail("steady-clock timing row violated adjacent phase boundaries");
    return {index, static_cast<std::uint64_t>(prefill), static_cast<std::uint64_t>(tracker),
            static_cast<std::uint64_t>(total)};
}

std::string versionString(std::int32_t version) {
    if (version <= 0)
        fail("CUDA returned an invalid version");
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10) + "." +
           std::to_string(version % 10);
}

benchmark::RuntimeFacts inspectRuntime(std::int32_t device) {
    requireCuda(cudaSetDevice(device), "CUDA device selection");
    cudaDeviceProp properties{};
    requireCuda(cudaGetDeviceProperties(&properties, device), "CUDA device-properties query");
    std::int32_t runtime_version = 0;
    std::int32_t driver_version = 0;
    requireCuda(cudaRuntimeGetVersion(&runtime_version), "CUDA runtime-version query");
    requireCuda(cudaDriverGetVersion(&driver_version), "CUDA driver-version query");
    benchmark::RuntimeFacts result;
    result.gpu_device = device;
    result.gpu_name = properties.name;
    result.compute_major = properties.major;
    result.compute_minor = properties.minor;
    result.global_memory_bytes = properties.totalGlobalMem;
    result.tensorrt_version = std::to_string(getInferLibMajorVersion()) + "." +
                              std::to_string(getInferLibMinorVersion()) + "." +
                              std::to_string(getInferLibPatchVersion()) + "." +
                              std::to_string(getInferLibBuildVersion());
    result.tensorrt_abi =
        std::to_string(NV_TENSORRT_MAJOR) + "." + std::to_string(NV_TENSORRT_MINOR);
    result.cuda_runtime_version = versionString(runtime_version);
    result.cuda_driver_version = versionString(driver_version);
    result.hostname = hostname();
    result.gpu_uuid = gpuUuid(properties.uuid);
    result.pci_bus_id = pciBusId(device);
#if defined(__clang__)
    result.cxx_compiler_id = "Clang";
    result.cxx_compiler_version = __clang_version__;
#elif defined(__GNUC__)
    result.cxx_compiler_id = "GNU";
    result.cxx_compiler_version = __VERSION__;
#else
    result.cxx_compiler_id = "unknown";
    result.cxx_compiler_version = "unknown";
#endif
    result.cxx_language_standard = __cplusplus;
    return result;
}

benchmark::ImageAttentionFacts parseBuildReceipt(std::string_view bytes,
                                                 const benchmark::RuntimeFacts& runtime) {
    const auto receipt = Json::parse(bytes.begin(), bytes.end());
    if (receipt.at("schema_version").get<std::int32_t>() !=
            trtmc::sam2::kBuildReceiptSchemaVersion ||
        receipt.at("family").get<std::string>() != "sam2")
        fail("native build receipt identity drifted");
    const auto& assets = receipt.at("assets");
    if (assets.at("checkpoint_sha256").get<std::string>() != trtmc::sam2::kCheckpointSha256 ||
        assets.at("source_config_sha256").get<std::string>() != trtmc::sam2::kConfigSha256 ||
        assets.at("golden_manifest_sha256").get<std::string>() !=
            trtmc::sam2::kGoldenManifestSha256) {
        fail("native build receipt asset hashes drifted");
    }
    const auto& build = receipt.at("build");
    if (build.at("tensorrt_version").get<std::string>() != runtime.tensorrt_version ||
        build.at("tensorrt_abi").get<std::string>() != runtime.tensorrt_abi ||
        build.at("cuda_runtime_version").get<std::string>() != runtime.cuda_runtime_version ||
        build.at("cuda_driver_version").get<std::string>() != runtime.cuda_driver_version ||
        build.at("gpu").at("device").get<std::int32_t>() != runtime.gpu_device ||
        build.at("gpu").at("name").get<std::string>() != runtime.gpu_name ||
        build.at("gpu").at("compute_capability").get<std::string>() !=
            std::to_string(runtime.compute_major) + "." + std::to_string(runtime.compute_minor) ||
        build.at("gpu").at("global_memory_bytes").get<std::uint64_t>() !=
            runtime.global_memory_bytes ||
        build.at("network_mode").get<std::string>() != "strongly_typed" ||
        build.at("tf32_enabled").get<bool>() ||
        build.at("builder_optimization_level").get<std::int32_t>() !=
            trtmc::sam2::kBuilderOptimizationLevel ||
        build.at("plan_profiling_verbosity").get<std::string>() !=
            trtmc::sam2::kPlanProfilingVerbosity) {
        fail("native build receipt runtime does not match benchmark runtime");
    }
    const auto& value = receipt.at("image_attention");
    benchmark::ImageAttentionFacts result;
    result.implementation = value.at("implementation").get<std::string>();
    result.operator_name = value.at("operator").get<std::string>();
    result.api = value.at("api").get<std::string>();
    result.block_count = value.at("block_count").get<std::int32_t>();
    result.head_dimension = value.at("head_dimension").get<std::int32_t>();
    result.query_form = value.at("query_form").get<std::string>();
    result.key_value_form = value.at("key_value_form").get<std::string>();
    result.output_form = value.at("output_form").get<std::string>();
    result.normalization = value.at("normalization").get<std::string>();
    result.causal_mask = value.at("causal_mask").get<std::string>();
    result.decomposable = value.at("decomposable").get<bool>();
    result.fused_kernel_intent = value.at("fused_kernel_intent").get<bool>();
    result.metadata_prefix = value.at("metadata_prefix").get<std::string>();
    result.metadata_index_width = value.at("metadata_index_width").get<std::int32_t>();
    result.q_scale_formula = value.at("q_scale_formula").get<std::string>();
    result.k_scale_formula = value.at("k_scale_formula").get<std::string>();
    result.effective_score_scale = value.at("effective_score_scale").get<std::string>();
    result.scale_dtype = value.at("scale_dtype").get<std::string>();
    if (result.implementation != "tensorrt_iattention_v2" || result.operator_name != "IAttention" ||
        result.api != "addAttentionV2" || result.block_count != 16 || result.head_dimension != 96 ||
        result.query_form != "padded_bhnd" || result.key_value_form != "padded_bhnd" ||
        result.output_form != "padded_bhnd" || result.normalization != "softmax" ||
        result.causal_mask != "none" || result.decomposable || !result.fused_kernel_intent ||
        result.metadata_prefix != trtmc::sam2::kImageAttentionMetadataPrefix ||
        result.metadata_index_width != trtmc::sam2::kImageAttentionMetadataIndexWidth ||
        result.q_scale_formula != "1/sqrt(head_dimension)" || result.k_scale_formula != "none" ||
        result.effective_score_scale != "1/sqrt(head_dimension)" || result.scale_dtype != "bf16") {
        fail("native build receipt image attention contract drifted");
    }
    return result;
}

void verifyOutputAccuracyRepeat(const std::vector<benchmark::AccuracyReplay>& pre,
                                const benchmark::AccuracyReplay& post) {
    if (pre.size() != benchmark::kQualificationReplayCount)
        fail("prequalification replay count drifted");
    for (const auto& replay : pre) {
        if (!replay.passes)
            fail("prequalification semantic accuracy gate failed");
    }
    if (!post.passes)
        fail("postqualification semantic accuracy gate failed after W3/N100 reuse");
}

int run(const Options& options) {
    requireAbsent(options.bundle_output, "bundle output");
    requireAbsent(options.receipt_output, "receipt output");
    if (!options.accuracy_only)
        requireAbsent(options.q3_receipt_output, "Q3 receipt output");

    const auto paths = framePaths(options.jpeg_dir);
    std::array<std::vector<std::uint8_t>, 5> encoded_jpegs;
    benchmark::AssetFacts assets;
    for (std::size_t index = 0; index < paths.size(); ++index) {
        auto snapshot = snapshotRegularFile(paths[index], 64U * 1024U * 1024U);
        if (snapshot.file.sha256 != benchmark::kEncodedJpegSha256[index])
            fail("encoded JPEG hash mismatch: " + paths[index].string());
        assets.encoded_jpeg_sha256[index] = snapshot.file.sha256;
        encoded_jpegs[index] = std::move(snapshot.bytes);
        const auto decoded = trtmc::sam2::decodeSam2JpegBytes(encoded_jpegs[index]);
        const auto decoded_hash = hashBytes(decoded.rgb_hwc.data(), decoded.rgb_hwc.size());
        if (decoded_hash != benchmark::kDecodedJpegSha256[index])
            fail("decoded RGB hash mismatch: " + paths[index].string());
        assets.decoded_jpeg_sha256[index] = decoded_hash;
    }

    if (!options.accuracy_only) {
        const auto baseline = snapshotRegularFile(options.baseline_receipt, 16U * 1024U * 1024U);
        verifyBaselineReceipt(baseline);
        assets.baseline_receipt_sha256 = baseline.file.sha256;
        const auto baseline_capture =
            snapshotRegularFile(options.baseline_capture_script, 16U * 1024U * 1024U);
        if (baseline_capture.file.sha256 != benchmark::kBaselineCaptureScriptSha256)
            fail("reviewed baseline capture script hash mismatch");
        assets.baseline_capture_script_sha256 = baseline_capture.file.sha256;
    }
    const auto source_closure = hashSourceClosure(options.source_root);
    assets.benchmark_source_manifest_sha256 = source_closure.manifest_sha256;
    assets.benchmark_source_closure_sha256 = source_closure.closure_sha256;
    assets.benchmark_executable_sha256 = hashSelfExecutable().sha256;

    const auto golden = benchmark::loadGoldenEvidence(options.golden_dir);

    native::Sam2EngineBuildOptions build_options;
    build_options.checkpoint_path = options.checkpoint;
    build_options.source_config_path = options.config;
    build_options.output_path = options.bundle_output;
    build_options.workspace_bytes = options.workspace_bytes;
    build_options.gpu_device = options.gpu_device;
    build_options.created_at_utc = options.created_at_utc;
    const auto build_result = native::buildSam2NativeBundle(build_options);

    const auto bundle_before = hashRegularFile(options.bundle_output, UINT64_MAX);
    verifyPublishedBundle(bundle_before, build_result.bundle);
    if (build_result.build_receipt_json.empty() ||
        !isCanonicalSha256(build_result.build_receipt_sha256) ||
        hashBytes(build_result.build_receipt_json.data(), build_result.build_receipt_json.size()) !=
            build_result.build_receipt_sha256 ||
        !std::all_of(build_result.plan_sha256.begin(), build_result.plan_sha256.end(),
                     [](const auto& digest) { return isCanonicalSha256(digest); })) {
        fail("builder returned incomplete or inconsistent exact bundle evidence");
    }

    auto runtime = inspectRuntime(options.gpu_device);
    CudaStream stream(options.gpu_device);
    benchmark::CheckedPlanModuleFactory factory(stream.get());
    trtmc::sam2::NativeBundleRuntimeTarget target{
        runtime.tensorrt_version, runtime.tensorrt_abi, runtime.gpu_name,
        std::to_string(runtime.compute_major) + "." + std::to_string(runtime.compute_minor)};
    auto engines = trtmc::sam2::loadDiagnosticNativeVideoEngineSetFromBundleWithExpectedSha256(
        options.bundle_output.string(), build_result.bundle.sha256, target, factory.callback());
    auto processor = trtmc::sam2::makeNativeDeviceVideoProcessor(std::move(engines));

    const auto image_attention = parseBuildReceipt(build_result.build_receipt_json, runtime);
    const auto loaded_plans = factory.loadedPlanSha256();
    if (loaded_plans.size() != trtmc::sam2::kRequiredPlanSections.size())
        fail("checked loader did not deserialize exactly six plans");
    for (std::size_t index = 0; index < loaded_plans.size(); ++index) {
        if (loaded_plans[index].first != trtmc::sam2::kRequiredPlanSections[index] ||
            loaded_plans[index].second != build_result.plan_sha256[index]) {
            fail("plan bytes used for deserialization did not match the builder-returned ordered "
                 "plan evidence");
        }
        assets.native_plan_sha256[index] = loaded_plans[index].second;
    }
    runtime.engine_profiling_verbosity = std::string(trtmc::sam2::kPlanProfilingVerbosity);
    runtime.execution_context_nvtx_verbosity =
        std::string(trtmc::sam2::kBenchmarkExecutionContextNvtxVerbosity);
    const auto bundle_after = hashRegularFile(options.bundle_output, UINT64_MAX);
    if (!(bundle_before.identity == bundle_after.identity) ||
        bundle_before.sha256 != bundle_after.sha256)
        fail("native bundle changed across authenticated deserialization");
    assets.checkpoint_sha256 = std::string(trtmc::sam2::kCheckpointSha256);
    assets.source_config_sha256 = std::string(trtmc::sam2::kConfigSha256);
    assets.golden_manifest_sha256 = std::string(trtmc::sam2::kGoldenManifestSha256);
    assets.golden_masks_sha256 = std::string(trtmc::sam2::kGoldenMasksSha256);
    assets.native_bundle_sha256 = build_result.bundle.sha256;
    assets.native_build_receipt_sha256 = build_result.build_receipt_sha256;

    StableFrames frames;
    runtime.started_at_utc = currentUtcTimestamp();
    if (options.accuracy_only) {
        std::vector<benchmark::AccuracyReplay> replays;
        replays.reserve(benchmark::kQualificationReplayCount);
        for (std::size_t index = 0; index < benchmark::kQualificationReplayCount; ++index) {
            auto output = executeReplay(processor, stream, frames, encoded_jpegs);
            replays.push_back(
                benchmark::evaluateAccuracy(index, output.prompt, output.results, golden));
        }
        runtime.ended_at_utc = currentUtcTimestamp();

        const auto final_bundle = hashRegularFile(options.bundle_output, UINT64_MAX);
        if (!(bundle_after.identity == final_bundle.identity) ||
            bundle_after.sha256 != final_bundle.sha256) {
            fail("native bundle changed during the accuracy-only replay sequence");
        }

        benchmark::BenchmarkReceipt receipt;
        receipt.mode = benchmark::BenchmarkMode::kAccuracyOnly;
        receipt.assets = std::move(assets);
        receipt.runtime = runtime;
        receipt.image_attention = image_attention;
        receipt.accuracy_only_replays = std::move(replays);
        const std::string canonical = benchmark::makeCanonicalBenchmarkReceipt(receipt);
        benchmark::writeReceiptExclusive(options.receipt_output, canonical);

        std::cout << "Wrote accuracy-only SAM2 receipt: " << options.receipt_output << '\n'
                  << "receipt_sha256=" << hashBytes(canonical.data(), canonical.size()) << '\n'
                  << "Three reset-separated five-frame replays passed the semantic mask and "
                     "bbox gates. No timing or performance claim was performed.\n";
        return 0;
    }

    std::vector<benchmark::AccuracyReplay> prequalification;
    prequalification.reserve(benchmark::kQualificationReplayCount);
    for (std::size_t index = 0; index < benchmark::kQualificationReplayCount; ++index) {
        auto output = executeReplay(processor, stream, frames, encoded_jpegs);
        prequalification.push_back(
            benchmark::evaluateAccuracy(index, output.prompt, output.results, golden));
    }

    auto q3_assets = assets;
    q3_assets.baseline_receipt_sha256.clear();
    q3_assets.baseline_capture_script_sha256.clear();
    benchmark::BenchmarkReceipt q3_receipt;
    q3_receipt.mode = benchmark::BenchmarkMode::kAccuracyOnly;
    q3_receipt.assets = std::move(q3_assets);
    q3_receipt.runtime = runtime;
    q3_receipt.runtime.ended_at_utc = currentUtcTimestamp();
    q3_receipt.image_attention = image_attention;
    q3_receipt.accuracy_only_replays = prequalification;
    const std::string q3_canonical = benchmark::makeCanonicalBenchmarkReceipt(q3_receipt);
    benchmark::writeReceiptExclusive(options.q3_receipt_output, q3_canonical);
    const auto q3_snapshot = snapshotRegularFile(options.q3_receipt_output, 16U * 1024U * 1024U);
    if (q3_snapshot.bytes.size() != q3_canonical.size() ||
        !std::equal(q3_snapshot.bytes.begin(), q3_snapshot.bytes.end(), q3_canonical.begin())) {
        fail("exclusive pre-W3 Q3 receipt changed after publication");
    }
    assets.q3_receipt_sha256 = q3_snapshot.file.sha256;
    assets.q3_receipt_size_bytes = q3_snapshot.bytes.size();

    std::vector<benchmark::TimingRow> warmups;
    warmups.reserve(benchmark::kWarmupRowCount);
    for (std::size_t index = 0; index < benchmark::kWarmupRowCount; ++index) {
        warmups.push_back(executeTimedRow(index, processor, stream, frames, encoded_jpegs));
    }

    std::vector<benchmark::TimingRow> measurements;
    measurements.reserve(benchmark::kMeasurementRowCount);
    for (std::size_t index = 0; index < benchmark::kMeasurementRowCount; ++index) {
        measurements.push_back(executeTimedRow(index, processor, stream, frames, encoded_jpegs));
    }

    auto post_output = executeReplay(processor, stream, frames, encoded_jpegs);
    auto postqualification =
        benchmark::evaluateAccuracy(0U, post_output.prompt, post_output.results, golden);
    verifyOutputAccuracyRepeat(prequalification, postqualification);
    runtime.ended_at_utc = currentUtcTimestamp();

    const auto final_bundle = hashRegularFile(options.bundle_output, UINT64_MAX);
    if (!(bundle_after.identity == final_bundle.identity) ||
        bundle_after.sha256 != final_bundle.sha256)
        fail("native bundle changed during the formal benchmark sequence");

    benchmark::BenchmarkReceipt receipt;
    receipt.assets = std::move(assets);
    receipt.runtime = runtime;
    receipt.image_attention = image_attention;
    receipt.prequalification = std::move(prequalification);
    receipt.warmup_rows = std::move(warmups);
    receipt.measurement_rows = std::move(measurements);
    receipt.postqualification = std::move(postqualification);
    const std::string canonical = benchmark::makeCanonicalBenchmarkReceipt(receipt);
    benchmark::writeReceiptExclusive(options.receipt_output, canonical);

    const auto summary = benchmark::summarizeTimingRows(receipt.measurement_rows);
    std::cout << "Wrote diagnostic-only SAM2 benchmark receipt: " << options.receipt_output << '\n'
              << "receipt_sha256=" << hashBytes(canonical.data(), canonical.size()) << '\n'
              << "closest_envelope_total_mean_ms=" << summary.total.mean_milliseconds << '\n'
              << "No speedup or production runtime claim is made.\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parseArguments(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "sam2_native_benchmark: " << error.what() << '\n';
        return 1;
    }
}
