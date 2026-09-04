/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/pipeline.h"

#include "runtime/models/minimax_h3/conditioning.h"
#include "runtime/models/minimax_h3/fl2va_runtime.h"
#include "runtime/models/minimax_h3/ref2va_runtime.h"
#include "runtime/models/minimax_h3/torch_cuda_normal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int32_t kMinTextRows = 1;
constexpr int32_t kMaxTextRows = 2641;
constexpr int32_t kFiveSecondTextRows = 537;
constexpr int32_t kTextDim = 5120;
constexpr int32_t kAudioChannels = 32;
constexpr int32_t kLatentChannels = 24;
constexpr int32_t kPatchHeight = 2;
constexpr int32_t kPatchWidth = 2;
constexpr int32_t kPatchDim = 96;
constexpr int32_t kLayers = 50;
constexpr int32_t kHidden = 5376;
constexpr int32_t kTimestepSlots = 4;
constexpr int32_t kModalityCount = 3;
constexpr int32_t kAdalnRows = kTimestepSlots * kModalityCount;
constexpr int32_t kDefaultOutputFrames = kMiniMaxH3DefaultOutputFrames;
constexpr int32_t kMaxOutputFrames = 345;
constexpr int32_t kMaxVideoLatentFrames = 102;
constexpr int32_t kMaxAudioLatentFrames = 575;
constexpr int32_t kCanvasMultiple = kMiniMaxH3CanvasMultiple;
// Diffusers documents 960x544 as its smaller explicit performance canvas. It
// sits below the resolver envelope but is part of this finite TensorRT profile.
constexpr int32_t kExplicitCanvasHeight = kMiniMaxH3ExplicitCanvasHeight;
constexpr int32_t kExplicitCanvasWidth = kMiniMaxH3ExplicitCanvasWidth;
// The official resolver caps area before nearest-32 rounding. Its largest
// rounded canvas is 576x1856 (or the transpose), not 768x1344.
constexpr int32_t kMaxOutputPixels = 576 * 1856;
constexpr int32_t kMaxVideoSpatialRows = (576 / 32) * (1856 / 32);
constexpr int32_t kMaxTargetVideoRows = kMaxVideoLatentFrames * kMaxVideoSpatialRows;
constexpr int32_t kMaxConditionVideoRows = 2 * kMaxVideoSpatialRows;
constexpr int32_t kMaxVideoRows = kMaxTargetVideoRows + kMaxConditionVideoRows;
constexpr int32_t kMaxAudioRows = kMaxAudioLatentFrames * 2;
constexpr int32_t kMaxMediaRows = kMaxVideoRows + kMaxAudioRows;
constexpr int32_t kMaxSequenceRows = kMaxTextRows + kMaxMediaRows;
constexpr int32_t kDefaultOutputHeight = kMiniMaxH3DefaultOutputHeight;
constexpr int32_t kDefaultOutputWidth = kMiniMaxH3DefaultOutputWidth;
constexpr int32_t kAudioSampleRate = 32000;
constexpr int32_t kTileFrames = 28;
constexpr int32_t kTileSize = 256;
constexpr int32_t kTileMinOverlap = 64;
constexpr int32_t kTileAlignment = 16;
constexpr int32_t kTileLatentSize = 16;
constexpr int32_t kTileInputFrames = 7;
constexpr int32_t kMinTileBatch = 15;
constexpr int32_t kOptTileBatch = 28;
constexpr int32_t kMaxTileBatch = 33;
constexpr int32_t kMinVideoRows =
    37 * (kExplicitCanvasHeight / kCanvasMultiple) * (kExplicitCanvasWidth / kCanvasMultiple);
constexpr int32_t kMinPackedRows = kMinVideoRows + 414 + kMinTextRows;
constexpr int32_t kOptPackedRows = 37838;
constexpr int32_t kMaxPackedRows = 112367;
constexpr int32_t kFiveSecondVideoRows = 37296;
constexpr int32_t kFiveSecondAudioRows = 414;
constexpr int32_t kFiveSecondPackedRows =
    kFiveSecondTextRows + kFiveSecondAudioRows + kFiveSecondVideoRows;
static_assert(((kMaxOutputFrames - 5) / 17) * 5 + 2 == kMaxVideoLatentFrames);
static_assert(kMaxTargetVideoRows == 106488);
static_assert(kMaxVideoRows == 108576);
static_assert(kMaxSequenceRows == kMaxPackedRows);
static_assert(kMinVideoRows == 18870);
static_assert(kMinPackedRows == 19285);
static_assert(kFiveSecondPackedRows == 38247);

constexpr std::array<float, kLatentChannels> kLatentMean = {
    0.8580903411F,  -0.9606591463F, 1.0661640167F,  -0.5090325475F, -0.2727581859F, -1.3675414324F,
    -0.2553254962F, -0.2690755427F, -0.5376840830F, -0.0464097299F, 0.6657370329F,  0.1969012767F,
    -0.5460608006F, -0.4035342038F, -0.2368302494F, 0.2592845261F,  -0.3013394475F, 0.2113419920F,
    -1.1206848621F, 0.3581933379F,  -0.0422514379F, 0.2604829967F,  0.2286409289F,  0.7056031823F};
constexpr std::array<float, kLatentChannels> kLatentStd = {
    1.2223774195F, 1.2767263651F, 1.6831774712F, 1.7549455166F, 1.5636216402F, 2.1941435337F,
    0.9653137922F, 1.0569885969F, 0.8419489264F, 0.7729952931F, 1.8955937624F, 0.9468418360F,
    0.7996809483F, 0.4498890042F, 0.7197399735F, 0.6936293244F, 2.9610950947F, 2.7694199085F,
    3.0496184826F, 2.1088054180F, 3.2762262821F, 3.1627357006F, 2.2816812992F, 2.6127843857F};
constexpr std::array<float, kAudioChannels> kAudioLatentMean = {
    -0.0202116875F, 0.3876466480F,  -0.0439827980F, -0.2859151363F, 0.0817968622F,  -0.3578264117F,
    0.0406238101F,  -0.0155253448F, -0.2233624756F, 0.1821006835F,  0.2941778898F,  -0.0790116787F,
    -0.0568150729F, -0.3699028194F, -0.3161631525F, 0.5905951262F,  -0.0521395691F, 0.0136731602F,
    -0.0369164795F, 0.0973266065F,  -0.3394662440F, -0.3068567812F, -0.2450459898F, -0.0346985236F,
    0.0286803227F,  -0.2121777982F, -0.1678263098F, 0.3221288025F,  -0.1223055869F, 0.4356604815F,
    -0.0502599180F, 0.3979258239F};
constexpr std::array<float, kAudioChannels> kAudioLatentStd = {
    1.6895524263F, 2.7626373768F, 1.7945344448F, 1.6801681519F, 1.6390227079F, 2.7788298130F,
    1.7659090757F, 1.6199758053F, 2.6336526871F, 1.8539357185F, 2.5056498051F, 1.8110191822F,
    1.9579657316F, 1.6685497761F, 1.4922469854F, 3.2986702919F, 1.9491804838F, 1.8720003366F,
    1.8334080105F, 1.6488070488F, 1.6176958084F, 1.9131449461F, 1.5695245266F, 1.6943659782F,
    1.8318420649F, 1.5540637970F, 1.9344930649F, 1.5991982222F, 1.7180459499F, 1.6307219267F,
    1.8661226034F, 1.5613768101F};
constexpr std::array<float, 3> kPixelMean = {0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kPixelStd = {0.229F, 0.224F, 0.225F};

std::size_t video_latent_count(const MiniMaxH3Geometry& geometry) {
    return static_cast<std::size_t>(kLatentChannels) * geometry.video_latent_frames *
           geometry.latent_height * geometry.latent_width;
}

std::size_t audio_latent_count(const MiniMaxH3Geometry& geometry) {
    return static_cast<std::size_t>(geometry.audio_rows) * kAudioChannels;
}

minimax_h3::VaeLatentNormalization vae_latent_normalization() {
    minimax_h3::VaeLatentNormalization result{};
    std::copy(kLatentMean.begin(), kLatentMean.end(), result.mean);
    std::copy(kLatentStd.begin(), kLatentStd.end(), result.std);
    return result;
}

minimax_h3::VaePixelNormalization vae_pixel_normalization() {
    minimax_h3::VaePixelNormalization result{};
    std::copy(kPixelMean.begin(), kPixelMean.end(), result.mean);
    std::copy(kPixelStd.begin(), kPixelStd.end(), result.std);
    return result;
}

struct RawTensor {
    std::vector<std::byte> bytes;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
};

struct StepModulation {
    std::array<RawTensor, kLayers> blocks;
    RawTensor final;
};

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

const Tensor& require_output(const TensorMap& outputs, const std::string& name) {
    const auto it = outputs.find(name);
    if (it == outputs.end() || it->second.data == nullptr)
        throw std::runtime_error("MiniMax-H3 engine did not return " + name);
    return it->second;
}

RawTensor copy_raw(const Tensor& tensor, DType expected_dtype, std::size_t expected_numel,
                   const char* label) {
    if (tensor.dtype != expected_dtype || tensor.numel() != expected_numel)
        throw std::runtime_error(std::string("MiniMax-H3 invalid ") + label + " output");
    RawTensor result;
    result.shape = tensor.shape;
    result.dtype = tensor.dtype;
    result.bytes.resize(tensor.nbytes());
    std::memcpy(result.bytes.data(), tensor.data, tensor.nbytes());
    return result;
}

std::vector<float> copy_float(const Tensor& tensor, std::size_t expected_numel, const char* label) {
    if (tensor.dtype != DType::kFloat32 || tensor.numel() != expected_numel)
        throw std::runtime_error(std::string("MiniMax-H3 invalid ") + label + " output");
    const auto* begin = static_cast<const float*>(tensor.data);
    return std::vector<float>(begin, begin + expected_numel);
}

std::array<float, 256> timestep_features(float timestep) {
    std::array<float, 256> output{};
    for (int32_t index = 0; index < 128; ++index) {
        const double frequency = std::exp(-std::log(10000.0) * index / 128.0);
        const double phase = static_cast<double>(timestep) * frequency;
        output[index] = static_cast<float>(std::cos(phase));
        output[128 + index] = static_cast<float>(std::sin(phase));
    }
    return output;
}

std::vector<float> make_adaln_features(float video_timestep, float audio_timestep) {
    std::vector<float> result(kTimestepSlots * 256, 0.0F);
    const auto video = timestep_features(video_timestep);
    const auto audio = timestep_features(audio_timestep);
    // FL2VA keyframes stay at the released near-clean condition clock.  T2VA
    // never selects slot two, so filling it is bit-for-bit inert for existing
    // text-only rows and preserves the same plan/cache shape.
    const auto condition = timestep_features(std::max(video_timestep, 0.999F));
    std::copy(video.begin(), video.end(), result.begin());
    std::copy(audio.begin(), audio.end(), result.begin() + 256);
    std::copy(condition.begin(), condition.end(), result.begin() + 512);
    return result;
}

std::vector<float> patchify_video(const std::vector<float>& latent,
                                  const MiniMaxH3Geometry& geometry) {
    if (latent.size() != video_latent_count(geometry))
        throw std::invalid_argument("MiniMax-H3 video latent count is invalid");
    std::vector<float> rows(static_cast<std::size_t>(geometry.target_video_rows) * kPatchDim);
    std::size_t target = 0;
    for (int32_t frame = 0; frame < geometry.video_latent_frames; ++frame) {
        for (int32_t y = 0; y < geometry.latent_height; y += kPatchHeight) {
            for (int32_t x = 0; x < geometry.latent_width; x += kPatchWidth) {
                for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                    for (int32_t py = 0; py < kPatchHeight; ++py) {
                        for (int32_t px = 0; px < kPatchWidth; ++px) {
                            const auto source = ((((static_cast<std::size_t>(channel) *
                                                        geometry.video_latent_frames +
                                                    frame) *
                                                       geometry.latent_height +
                                                   y + py) *
                                                  geometry.latent_width) +
                                                 x + px);
                            rows[target++] = latent[source];
                        }
                    }
                }
            }
        }
    }
    return rows;
}

std::vector<float> unpatchify_video(const std::vector<float>& rows,
                                    const MiniMaxH3Geometry& geometry) {
    if (rows.size() != static_cast<std::size_t>(geometry.target_video_rows) * kPatchDim)
        throw std::invalid_argument("MiniMax-H3 video rows are invalid");
    std::vector<float> latent(video_latent_count(geometry));
    std::size_t source = 0;
    for (int32_t frame = 0; frame < geometry.video_latent_frames; ++frame) {
        for (int32_t y = 0; y < geometry.latent_height; y += kPatchHeight) {
            for (int32_t x = 0; x < geometry.latent_width; x += kPatchWidth) {
                for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
                    for (int32_t py = 0; py < kPatchHeight; ++py) {
                        for (int32_t px = 0; px < kPatchWidth; ++px) {
                            const auto target = ((((static_cast<std::size_t>(channel) *
                                                        geometry.video_latent_frames +
                                                    frame) *
                                                       geometry.latent_height +
                                                   y + py) *
                                                  geometry.latent_width) +
                                                 x + px);
                            latent[target] = rows[source++];
                        }
                    }
                }
            }
        }
    }
    return latent;
}

void fill_audio_position_ids(std::vector<float>& positions, const std::vector<double>& width_grid,
                             int32_t text_rows, int32_t audio_latent_frames) {
    for (int32_t channel = 0; channel < 2; ++channel) {
        for (int32_t index = 0; index < audio_latent_frames; ++index) {
            const int32_t row = text_rows + channel * audio_latent_frames + index;
            positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(text_rows + index);
            positions[static_cast<std::size_t>(row) * 3 + 2] =
                static_cast<float>(channel == 0 ? width_grid.front() : width_grid.back());
        }
    }
}

void validate_text_rows(int32_t text_rows) {
    if (text_rows < kMinTextRows || text_rows > kMaxTextRows)
        throw std::invalid_argument("MiniMax-H3 text rows must be between 1 and 2641");
}

void validate_prompt_token_count(std::size_t token_count, int32_t max_text_rows) {
    if (max_text_rows < kMinTextRows || max_text_rows > kMaxTextRows)
        throw std::invalid_argument("MiniMax-H3 prompt profile is invalid");
    if (token_count < static_cast<std::size_t>(kMinTextRows) ||
        token_count > static_cast<std::size_t>(max_text_rows)) {
        throw std::invalid_argument(
            "MiniMax-H3 native profile supports 1 to " + std::to_string(max_text_rows) +
            " prompt tokens without truncation; got " + std::to_string(token_count));
    }
}

double numpy_pairwise_sum(const std::vector<double>& values) {
    if (values.empty())
        return 0.0;
    if (values.size() < 8U)
        return std::accumulate(values.begin(), values.end(), -0.0);
    std::array<double, 8> lanes{};
    std::copy_n(values.begin(), 8, lanes.begin());
    std::size_t index = 8;
    for (; index + 7 < values.size(); index += 8) {
        for (std::size_t lane = 0; lane < lanes.size(); ++lane)
            lanes[lane] += values[index + lane];
    }
    double result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                    ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for (; index < values.size(); ++index)
        result += values[index];
    return result;
}

std::vector<float> make_position_ids(int32_t text_rows, const MiniMaxH3Geometry& geometry,
                                     const std::vector<int32_t>& keyframe_anchors = {}) {
    validate_text_rows(text_rows);
    if (static_cast<int32_t>(keyframe_anchors.size()) != geometry.condition_video_frames)
        throw std::invalid_argument("MiniMax-H3 keyframe anchors do not match condition frames");
    const int32_t sequence_rows = text_rows + geometry.audio_rows + geometry.video_rows;
    std::vector<float> positions(static_cast<std::size_t>(sequence_rows) * 3, 0.0F);
    for (int32_t index = 0; index < text_rows; ++index)
        positions[static_cast<std::size_t>(index) * 3] = static_cast<float>(index);

    const double sqrt_area =
        std::sqrt(static_cast<double>(geometry.latent_height * geometry.latent_width));
    const double height_ratio = geometry.latent_height / sqrt_area;
    const double width_ratio = geometry.latent_width / sqrt_area;
    std::vector<double> height_grid(
        static_cast<std::size_t>(geometry.latent_height / kPatchHeight));
    std::vector<double> width_grid(static_cast<std::size_t>(geometry.latent_width / kPatchWidth));
    const double height_left = (1.0 - height_ratio) / 2.0;
    const double width_left = (1.0 - width_ratio) / 2.0;
    const double height_step = ((height_left + height_ratio) - height_left) / height_grid.size();
    const double width_step = ((width_left + width_ratio) - width_left) / width_grid.size();
    for (std::size_t i = 0; i < height_grid.size(); ++i)
        height_grid[i] = (height_left + static_cast<double>(i) * height_step) * 32.0;
    for (std::size_t i = 0; i < width_grid.size(); ++i)
        width_grid[i] = (width_left + static_cast<double>(i) * width_step) * 32.0;

    fill_audio_position_ids(positions, width_grid, text_rows, geometry.audio_latent_frames);

    std::vector<double> frame_times(static_cast<std::size_t>(geometry.video_latent_frames));
    std::vector<double> frame_spans(static_cast<std::size_t>(geometry.video_latent_frames));
    double time = text_rows;
    for (int32_t frame = 0; frame < geometry.video_latent_frames; ++frame) {
        frame_times[static_cast<std::size_t>(frame)] = time;
        const int32_t multiple = frame % 5 == 0 ? 1 : 4;
        frame_spans[static_cast<std::size_t>(frame)] = (5.0 / 3.0) * multiple;
        time += frame_spans[static_cast<std::size_t>(frame)];
    }
    // NumPy's float64 sum in the released implementation includes the final
    // latent span and then subtracts 5/3 for the last-frame anchor.
    const double last_anchor_time =
        static_cast<double>(text_rows) + numpy_pairwise_sum(frame_spans) - 5.0 / 3.0;

    int32_t row = text_rows + geometry.audio_rows;
    for (int32_t condition = 0; condition < geometry.condition_video_frames; ++condition) {
        const int32_t anchor = keyframe_anchors[static_cast<std::size_t>(condition)];
        if (anchor != 0 && anchor != geometry.output_frames - 1)
            throw std::invalid_argument("MiniMax-H3 FL2VA keyframes must anchor an endpoint");
        const double condition_time =
            anchor == 0 ? static_cast<double>(text_rows) : last_anchor_time;
        for (double y : height_grid) {
            for (double x : width_grid) {
                positions[static_cast<std::size_t>(row) * 3] = static_cast<float>(condition_time);
                positions[static_cast<std::size_t>(row) * 3 + 1] = static_cast<float>(y);
                positions[static_cast<std::size_t>(row) * 3 + 2] = static_cast<float>(x);
                ++row;
            }
        }
    }
    for (int32_t frame = 0; frame < geometry.video_latent_frames; ++frame) {
        for (double y : height_grid) {
            for (double x : width_grid) {
                positions[static_cast<std::size_t>(row) * 3] =
                    static_cast<float>(frame_times[static_cast<std::size_t>(frame)]);
                positions[static_cast<std::size_t>(row) * 3 + 1] = static_cast<float>(y);
                positions[static_cast<std::size_t>(row) * 3 + 2] = static_cast<float>(x);
                ++row;
            }
        }
    }
    if (row != sequence_rows)
        throw std::logic_error("MiniMax-H3 position row construction failed");
    return positions;
}

MiniMaxH3DenoiserMetadata make_base_denoiser_metadata(int32_t text_rows,
                                                      const MiniMaxH3Geometry& geometry) {
    const int32_t sequence_rows = text_rows + geometry.audio_rows + geometry.video_rows;
    MiniMaxH3DenoiserMetadata result;
    result.positions = make_position_ids(text_rows, geometry);
    result.adaln_indices.resize(sequence_rows);
    result.timestep_indices.resize(sequence_rows);
    for (int32_t row = 0; row < sequence_rows; ++row) {
        int32_t tag = 0;
        int32_t timestep = 0;
        if (row < text_rows) {
            tag = 1;
        } else if (row < text_rows + geometry.audio_rows) {
            tag = 2;
            timestep = 1;
        }
        result.timestep_indices[row] = timestep;
        result.adaln_indices[row] = timestep * kModalityCount + tag;
    }
    return result;
}

std::vector<StepModulation> precompute_modulations(ITrtModule& module,
                                                   const MiniMaxH3Schedule& video_schedule,
                                                   const MiniMaxH3Schedule& audio_schedule) {
    std::vector<StepModulation> result(video_schedule.timesteps.size());
    for (std::size_t step = 0; step < result.size(); ++step) {
        auto features =
            make_adaln_features(video_schedule.timesteps[step], audio_schedule.timesteps[step]);
        TensorMap inputs;
        inputs.emplace("timestep_features",
                       Tensor{features.data(), {kTimestepSlots, 256}, DType::kFloat32});
        const auto outputs = module.forward(inputs);
        for (int32_t layer = 0; layer < kLayers; ++layer) {
            const std::string name = "block_modulation_" + std::to_string(layer);
            result[step].blocks[layer] =
                copy_raw(require_output(outputs, name), DType::kBFloat16,
                         static_cast<std::size_t>(kAdalnRows) * 6 * kHidden, name.c_str());
        }
        result[step].final =
            copy_raw(require_output(outputs, "final_modulation"), DType::kBFloat16,
                     static_cast<std::size_t>(kTimestepSlots) * 2 * kHidden, "final_modulation");
    }
    return result;
}

void append_block_modulation_inputs(TensorMap& inputs, StepModulation& modulation,
                                    int32_t first_layer, int32_t end_layer) {
    for (int32_t layer = first_layer; layer < end_layer; ++layer) {
        const std::string name = "block_modulation_" + std::to_string(layer);
        auto& value = modulation.blocks[layer];
        inputs.emplace(name, Tensor{value.bytes.data(), value.shape, value.dtype});
    }
}

void append_final_modulation_input(TensorMap& inputs, StepModulation& modulation) {
    inputs.emplace("final_modulation", Tensor{modulation.final.bytes.data(), modulation.final.shape,
                                              modulation.final.dtype});
}

ModuleExternalBinding external_binding(const char* name, DeviceTensor& tensor) {
    return ModuleExternalBinding{name, tensor.data(), tensor.nbytes()};
}

class StreamScopeSynchronizer {
  public:
    explicit StreamScopeSynchronizer(cudaStream_t stream) : stream_(stream) {}
    ~StreamScopeSynchronizer() {
        if (stream_ != nullptr)
            (void)cudaStreamSynchronize(stream_);
    }

  private:
    cudaStream_t stream_{nullptr};
};

void bind_external_dynamic_input_checked(ITrtModule& module, const char* name, void* pointer,
                                         DType dtype, std::initializer_list<int64_t> runtime_shape,
                                         std::initializer_list<int64_t> max_shape,
                                         int32_t profile_index = 0,
                                         int32_t profile_count = 1) {
    const std::vector<int64_t> actual(runtime_shape);
    const std::vector<int64_t> maximum(max_shape);
    if (pointer == nullptr || !module.has_input(name) || !module.input_is_dynamic(name) ||
        module.tensor_dtype(name) != dtype ||
        module.profile_idx() != profile_index ||
        module.optimization_profile_count() != profile_count ||
        module.input_profile_shape(name, profile_index, ProfileShapeSelector::kMax) != maximum)
        throw std::runtime_error(std::string("MiniMax-H3 dynamic split plan ABI mismatch for ") +
                                 name);
    module.bind_external(name, pointer, actual);
    if (module.device_ptr(name) != pointer || module.tensor_shape(name) != actual)
        throw std::runtime_error(std::string("MiniMax-H3 dynamic external binding failed for ") +
                                 name);
}

void bind_external_dynamic_output_checked(ITrtModule& module, const char* name, void* pointer,
                                          DType dtype, std::initializer_list<int64_t> max_shape) {
    const std::vector<int64_t> maximum(max_shape);
    if (pointer == nullptr || !module.has_output(name) || module.tensor_dtype(name) != dtype ||
        module.tensor_shape(name) != maximum)
        throw std::runtime_error(std::string("MiniMax-H3 dynamic split plan ABI mismatch for ") +
                                 name);
    module.bind_external(name, pointer);
    if (module.device_ptr(name) != pointer)
        throw std::runtime_error(std::string("MiniMax-H3 external binding failed for ") + name);
}

MiniMaxH3DenoiserMetadata make_fl2va_denoiser_metadata(const std::vector<int32_t>& text_token_tags,
                                                       const std::vector<int32_t>& keyframe_anchors,
                                                       const MiniMaxH3Geometry& geometry) {
    const int32_t text_rows = static_cast<int32_t>(text_token_tags.size());
    validate_text_rows(text_rows);
    if (geometry.condition_video_frames < 1 || geometry.condition_video_frames > 2 ||
        static_cast<int32_t>(keyframe_anchors.size()) != geometry.condition_video_frames)
        throw std::invalid_argument("MiniMax-H3 FL2VA conditioning geometry is inconsistent");
    const int32_t sequence_rows = text_rows + geometry.audio_rows + geometry.video_rows;
    MiniMaxH3DenoiserMetadata result;
    result.positions = make_position_ids(text_rows, geometry, keyframe_anchors);
    result.adaln_indices.resize(sequence_rows);
    result.timestep_indices.resize(sequence_rows);
    for (int32_t row = 0; row < sequence_rows; ++row) {
        int32_t tag = 0;
        int32_t timestep = 0;
        if (row < text_rows) {
            tag = text_token_tags[static_cast<std::size_t>(row)];
            if (tag != 0 && tag != 1)
                throw std::invalid_argument("MiniMax-H3 FL2VA text tags must be zero or one");
        } else if (row < text_rows + geometry.audio_rows) {
            tag = 2;
            timestep = 1;
        } else if (row < text_rows + geometry.audio_rows + geometry.condition_video_rows) {
            timestep = 2;
        }
        result.timestep_indices[static_cast<std::size_t>(row)] = timestep;
        result.adaln_indices[static_cast<std::size_t>(row)] = timestep * kModalityCount + tag;
    }
    return result;
}

int32_t ceil_div(int32_t numerator, int32_t denominator) {
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

struct TileAxisLayout {
    std::vector<int32_t> starts;
    std::vector<int32_t> overlaps;
};

TileAxisLayout make_tile_axis_layout(int32_t length) {
    if (length <= 0 || length % kTileAlignment != 0)
        throw std::invalid_argument("MiniMax-H3 VAE tile axis is not latent-aligned");
    if (length <= kTileSize)
        return {{0}, {}};

    int32_t tile_count = ceil_div(length, kTileSize);
    while (static_cast<int64_t>(kTileSize) * tile_count -
               static_cast<int64_t>(kTileMinOverlap) * (tile_count - 1) <
           length) {
        ++tile_count;
    }
    TileAxisLayout result;
    result.overlaps.assign(static_cast<std::size_t>(tile_count - 1), kTileMinOverlap);
    const int32_t remaining = kTileSize * tile_count - kTileMinOverlap * (tile_count - 1) - length;
    if (remaining < 0 || remaining % kTileAlignment != 0)
        throw std::logic_error("MiniMax-H3 VAE tile slack is not latent-aligned");
    const int32_t extra_steps = remaining / kTileAlignment;
    for (int32_t step = 0; step < extra_steps; ++step)
        result.overlaps[static_cast<std::size_t>(step % (tile_count - 1))] += kTileAlignment;

    result.starts.reserve(static_cast<std::size_t>(tile_count));
    result.starts.push_back(0);
    for (int32_t boundary = 0; boundary + 1 < tile_count; ++boundary) {
        result.starts.push_back(result.starts.back() + kTileSize -
                                result.overlaps[static_cast<std::size_t>(boundary)]);
    }
    if (result.starts.back() + kTileSize != length)
        throw std::logic_error("MiniMax-H3 VAE tile layout does not cover its canvas");
    return result;
}

std::vector<int64_t> vae_tile_shape(int32_t tile_count) {
    return {tile_count, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize};
}

std::vector<int64_t> vae_decoded_tile_shape(int32_t tile_count) {
    return {tile_count, 3, kTileFrames, kTileSize, kTileSize};
}

void validate_vae_plan_geometry(ITrtModule& module, const MiniMaxH3Geometry& geometry) {
    if (!module.has_input("latent_tiles") || !module.has_output("decoded_tiles") ||
        module.tensor_dtype("latent_tiles") != DType::kFloat32 ||
        module.tensor_dtype("decoded_tiles") != DType::kFloat32) {
        throw std::runtime_error("MiniMax-H3 VAE tile plan tensor ABI mismatch");
    }
    if (!module.input_is_dynamic("latent_tiles") || module.optimization_profile_count() != 1 ||
        module.input_profile_shape("latent_tiles", 0, ProfileShapeSelector::kMin) !=
            vae_tile_shape(kMinTileBatch) ||
        module.input_profile_shape("latent_tiles", 0, ProfileShapeSelector::kOpt) !=
            vae_tile_shape(kOptTileBatch) ||
        module.input_profile_shape("latent_tiles", 0, ProfileShapeSelector::kMax) !=
            vae_tile_shape(kMaxTileBatch) ||
        module.tensor_shape("decoded_tiles") != vae_decoded_tile_shape(kMaxTileBatch)) {
        throw std::runtime_error(
            "MiniMax-H3 dynamic VAE tile plan must use the native [15,28,33] batch profile");
    }
    if (geometry.vae_tile_count < kMinTileBatch || geometry.vae_tile_count > kMaxTileBatch)
        throw std::logic_error("MiniMax-H3 VAE runtime tile count exceeds its validated profile");
}

void denormalize_latents(std::vector<float>& latent, const MiniMaxH3Geometry& geometry) {
    const std::size_t per_channel = static_cast<std::size_t>(geometry.video_latent_frames) *
                                    geometry.latent_height * geometry.latent_width;
    for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
        float* values = latent.data() + static_cast<std::size_t>(channel) * per_channel;
        for (std::size_t index = 0; index < per_channel; ++index)
            values[index] = values[index] * kLatentStd[channel] + kLatentMean[channel];
    }
}

std::vector<float> extract_tiles(const std::vector<float>& latent, int32_t clip,
                                 const MiniMaxH3Geometry& geometry) {
    const auto layout =
        make_minimax_h3_vae_tile_layout(geometry.output_height, geometry.output_width);
    const std::size_t one_tile = static_cast<std::size_t>(kLatentChannels) * kTileInputFrames *
                                 kTileLatentSize * kTileLatentSize;
    std::vector<float> result(static_cast<std::size_t>(geometry.vae_tile_count) * one_tile);
    for (int32_t tile = 0; tile < geometry.vae_tile_count; ++tile) {
        const int32_t tile_y = tile / geometry.vae_tile_columns;
        const int32_t tile_x = tile % geometry.vae_tile_columns;
        const int32_t latent_y_start =
            layout.y_starts[static_cast<std::size_t>(tile_y)] / kTileAlignment;
        const int32_t latent_x_start =
            layout.x_starts[static_cast<std::size_t>(tile_x)] / kTileAlignment;
        for (int32_t channel = 0; channel < kLatentChannels; ++channel) {
            for (int32_t frame = 0; frame < kTileInputFrames; ++frame) {
                for (int32_t y = 0; y < kTileLatentSize; ++y) {
                    const auto source =
                        ((((static_cast<std::size_t>(channel) * geometry.video_latent_frames +
                            clip * 5 + frame) *
                               geometry.latent_height +
                           latent_y_start + y) *
                          geometry.latent_width) +
                         latent_x_start);
                    const auto target =
                        ((((static_cast<std::size_t>(tile) * kLatentChannels + channel) *
                               kTileInputFrames +
                           frame) *
                              kTileLatentSize +
                          y) *
                         kTileLatentSize);
                    std::copy_n(latent.begin() + static_cast<std::ptrdiff_t>(source),
                                kTileLatentSize,
                                result.begin() + static_cast<std::ptrdiff_t>(target));
                }
            }
        }
    }
    return result;
}

void stitch_one_spatial_tile(const float* tiles, std::vector<float>& clip, int32_t tile_y,
                             int32_t tile_x, int32_t kept_height, int32_t kept_width,
                             const MiniMaxH3VaeTileLayout& layout,
                             const MiniMaxH3Geometry& geometry) {
    const int32_t tile = tile_y * geometry.vae_tile_columns + tile_x;
    const auto tile_value = [&](int32_t source_tile, int32_t channel, int32_t frame, int32_t y,
                                int32_t x) {
        return tiles[((((static_cast<std::size_t>(source_tile) * 3 + channel) * kTileFrames +
                        frame) *
                           kTileSize +
                       y) *
                      kTileSize) +
                     x];
    };
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (int32_t frame = 0; frame < kTileFrames; ++frame) {
            for (int32_t y = 0; y < kept_height; ++y) {
                for (int32_t x = 0; x < kept_width; ++x) {
                    float value = tile_value(tile, channel, frame, y, x);
                    if (tile_y > 0 && y < layout.y_overlaps[static_cast<std::size_t>(tile_y - 1)]) {
                        const int32_t overlap =
                            layout.y_overlaps[static_cast<std::size_t>(tile_y - 1)];
                        const float weight_b = static_cast<float>(y) / overlap;
                        const float upper = tile_value(tile - geometry.vae_tile_columns, channel,
                                                       frame, kTileSize - overlap + y, x);
                        value = upper * (1.0F - weight_b) + value * weight_b;
                    }
                    if (tile_x > 0 && x < layout.x_overlaps[static_cast<std::size_t>(tile_x - 1)]) {
                        const int32_t overlap =
                            layout.x_overlaps[static_cast<std::size_t>(tile_x - 1)];
                        const float weight_b = static_cast<float>(x) / overlap;
                        const float left =
                            tile_value(tile - 1, channel, frame, y, kTileSize - overlap + x);
                        value = left * (1.0F - weight_b) + value * weight_b;
                    }
                    const auto target =
                        ((((static_cast<std::size_t>(channel) * kTileFrames + frame) *
                               geometry.output_height +
                           layout.y_starts[static_cast<std::size_t>(tile_y)] + y) *
                          geometry.output_width) +
                         layout.x_starts[static_cast<std::size_t>(tile_x)] + x);
                    clip[target] = value;
                }
            }
        }
    }
}

void stitch_spatial_tiles(const Tensor& tiles, std::vector<float>& clip,
                          const MiniMaxH3Geometry& geometry) {
    const auto layout =
        make_minimax_h3_vae_tile_layout(geometry.output_height, geometry.output_width);
    const std::size_t one_tile = static_cast<std::size_t>(3) * kTileFrames * kTileSize * kTileSize;
    if (tiles.dtype != DType::kFloat32 || tiles.data == nullptr ||
        tiles.numel() != static_cast<std::size_t>(geometry.vae_tile_count) * one_tile)
        throw std::runtime_error("MiniMax-H3 decoded VAE tile count is invalid");
    const auto* values = static_cast<const float*>(tiles.data);
    clip.resize(static_cast<std::size_t>(3) * kTileFrames * geometry.output_height *
                geometry.output_width);
    for (int32_t tile_y = 0; tile_y < geometry.vae_tile_rows; ++tile_y) {
        const int32_t kept_height =
            tile_y + 1 < geometry.vae_tile_rows ? kTileSize - layout.y_overlaps[tile_y] : kTileSize;
        for (int32_t tile_x = 0; tile_x < geometry.vae_tile_columns; ++tile_x) {
            const int32_t kept_width = tile_x + 1 < geometry.vae_tile_columns
                                           ? kTileSize - layout.x_overlaps[tile_x]
                                           : kTileSize;
            stitch_one_spatial_tile(values, clip, tile_y, tile_x, kept_height, kept_width, layout,
                                    geometry);
        }
    }
}

void write_temporal_chunk(std::vector<float>& video, std::size_t old_frames,
                          const std::vector<float>& clip,
                          const std::vector<float>& previous_overlap,
                          const MiniMaxH3Geometry& geometry) {
    constexpr int32_t chunk_frames = 17;
    constexpr int32_t pre_padding = 3;
    constexpr int32_t overlap_frames = 5;
    const std::size_t plane =
        static_cast<std::size_t>(geometry.output_height) * geometry.output_width;
    if (video.size() != static_cast<std::size_t>(3) * geometry.output_frames * plane ||
        old_frames + chunk_frames > static_cast<std::size_t>(geometry.output_frames))
        throw std::invalid_argument("MiniMax-H3 temporal output buffer is invalid");
    for (int32_t channel = 0; channel < 3; ++channel) {
        for (int32_t frame = 0; frame < chunk_frames; ++frame) {
            const auto source =
                (static_cast<std::size_t>(channel) * kTileFrames + pre_padding + frame) * plane;
            const auto target = (static_cast<std::size_t>(channel) * geometry.output_frames +
                                 old_frames + static_cast<std::size_t>(frame)) *
                                plane;
            if (!previous_overlap.empty() && frame < overlap_frames) {
                const float weight_b = static_cast<float>(frame) / overlap_frames;
                const auto prior =
                    (static_cast<std::size_t>(channel) * overlap_frames + frame) * plane;
                for (std::size_t pixel = 0; pixel < plane; ++pixel)
                    video[target + pixel] = previous_overlap[prior + pixel] * (1.0F - weight_b) +
                                            clip[source + pixel] * weight_b;
            } else {
                std::copy_n(clip.begin() + static_cast<std::ptrdiff_t>(source), plane,
                            video.begin() + static_cast<std::ptrdiff_t>(target));
            }
        }
    }
}

void update_trailing_overlap(const std::vector<float>& clip, std::vector<float>& result,
                             const MiniMaxH3Geometry& geometry) {
    constexpr int32_t overlap_frames = 5;
    constexpr int32_t start = 23;
    const std::size_t plane =
        static_cast<std::size_t>(geometry.output_height) * geometry.output_width;
    result.resize(static_cast<std::size_t>(3) * overlap_frames * plane);
    for (int32_t channel = 0; channel < 3; ++channel) {
        const auto source = (static_cast<std::size_t>(channel) * kTileFrames + start) * plane;
        const auto target = static_cast<std::size_t>(channel) * overlap_frames * plane;
        std::copy_n(clip.begin() + static_cast<std::ptrdiff_t>(source), overlap_frames * plane,
                    result.begin() + static_cast<std::ptrdiff_t>(target));
    }
}

void write_final_overlap(std::vector<float>& video, std::size_t old_frames,
                         const std::vector<float>& overlap, const MiniMaxH3Geometry& geometry) {
    constexpr int32_t overlap_frames = 5;
    const std::size_t plane =
        static_cast<std::size_t>(geometry.output_height) * geometry.output_width;
    if (video.size() != static_cast<std::size_t>(3) * geometry.output_frames * plane ||
        old_frames + overlap_frames != static_cast<std::size_t>(geometry.output_frames) ||
        overlap.size() != static_cast<std::size_t>(3) * overlap_frames * plane)
        throw std::invalid_argument("MiniMax-H3 final temporal overlap is invalid");
    for (int32_t channel = 0; channel < 3; ++channel) {
        std::copy_n(overlap.begin() + static_cast<std::ptrdiff_t>(channel * overlap_frames * plane),
                    overlap_frames * plane,
                    video.begin() + static_cast<std::ptrdiff_t>(
                                        (channel * geometry.output_frames + old_frames) * plane));
    }
}

void postprocess_video(std::vector<float>& video, const MiniMaxH3Geometry& geometry) {
    const std::size_t per_channel = static_cast<std::size_t>(geometry.output_frames) *
                                    geometry.output_height * geometry.output_width;
    for (int32_t channel = 0; channel < 3; ++channel) {
        float* values = video.data() + static_cast<std::size_t>(channel) * per_channel;
        for (std::size_t index = 0; index < per_channel; ++index)
            values[index] =
                std::clamp(values[index] * kPixelStd[channel] + kPixelMean[channel], 0.0F, 1.0F);
    }
}

std::vector<float> to_frame_major_rgb(const std::vector<float>& video,
                                      const MiniMaxH3Geometry& geometry) {
    const std::size_t plane =
        static_cast<std::size_t>(geometry.output_height) * geometry.output_width;
    const std::size_t per_channel = static_cast<std::size_t>(geometry.output_frames) * plane;
    std::vector<float> pixels(static_cast<std::size_t>(geometry.output_frames) * plane * 3);
    for (int32_t frame = 0; frame < geometry.output_frames; ++frame) {
        for (std::size_t pixel = 0; pixel < plane; ++pixel) {
            const auto target = (static_cast<std::size_t>(frame) * plane + pixel) * 3;
            for (int32_t channel = 0; channel < 3; ++channel) {
                const auto source = static_cast<std::size_t>(channel) * per_channel +
                                    static_cast<std::size_t>(frame) * plane + pixel;
                pixels[target + channel] = video[source];
            }
        }
    }
    return pixels;
}

MiniMaxH3Geometry resolve_generate_geometry(const GenerateConfig& cfg) {
    if ((cfg.height > 0) != (cfg.width > 0))
        throw std::invalid_argument("MiniMax-H3 height and width must be supplied together");
    const int32_t output_height = cfg.height > 0 ? cfg.height : kDefaultOutputHeight;
    const int32_t output_width = cfg.width > 0 ? cfg.width : kDefaultOutputWidth;
    const int32_t requested_frames =
        cfg.video_num_frames > 0 ? cfg.video_num_frames : kDefaultOutputFrames;
    const int32_t output_frames = align_minimax_h3_num_frames(requested_frames);
    return make_minimax_h3_geometry(output_frames, output_height, output_width);
}

struct DenoiserStats {
    int32_t full_steps{0};
    int32_t skipped_steps{0};
};

bool device_tensors_ready(std::initializer_list<const DeviceTensor*> tensors) {
    return std::all_of(tensors.begin(), tensors.end(), [](const DeviceTensor* tensor) {
        return tensor != nullptr && tensor->ok();
    });
}

} // namespace

int32_t select_minimax_h3_denoiser_profile(int32_t optimization_profile_count,
                                           int32_t text_rows,
                                           const MiniMaxH3Geometry& geometry) {
    if (optimization_profile_count == 1)
        return 0;
    if (optimization_profile_count != 2)
        throw std::invalid_argument(
            "MiniMax-H3 denoiser requires one or two optimization profiles");
    const bool qualified_five_second_request =
        text_rows == kFiveSecondTextRows && geometry.output_frames == kDefaultOutputFrames &&
        geometry.output_height == kDefaultOutputHeight &&
        geometry.output_width == kDefaultOutputWidth && geometry.condition_video_rows == 0 &&
        geometry.video_rows == kFiveSecondVideoRows &&
        geometry.audio_rows == kFiveSecondAudioRows;
    return qualified_five_second_request ? 0 : 1;
}

MiniMaxH3VaeTileLayout make_minimax_h3_vae_tile_layout(int32_t output_height,
                                                       int32_t output_width) {
    if (!is_minimax_h3_native_canvas(output_height, output_width))
        throw std::invalid_argument(
            "MiniMax-H3 VAE tiling supports the public 768p resolver canvases plus the explicit "
            "544x960/960x544 native profile");
    auto y = make_tile_axis_layout(output_height);
    auto x = make_tile_axis_layout(output_width);
    MiniMaxH3VaeTileLayout result;
    result.y_starts = std::move(y.starts);
    result.x_starts = std::move(x.starts);
    result.y_overlaps = std::move(y.overlaps);
    result.x_overlaps = std::move(x.overlaps);
    return result;
}

MiniMaxH3Geometry make_minimax_h3_geometry(int32_t output_frames, int32_t output_height,
                                           int32_t output_width) {
    if (output_frames % 17 != 5)
        throw std::invalid_argument("MiniMax-H3 output frames must have the form 17*n+5");
    if (output_frames < 5 * 24 || output_frames > 15 * 24)
        throw std::invalid_argument(
            "MiniMax-H3 released local profile supports output durations from 5 to 15 seconds");
    if (!is_minimax_h3_native_canvas(output_height, output_width))
        throw std::invalid_argument(
            "MiniMax-H3 output canvas must come from the public 768p resolver or be the explicit "
            "544x960/960x544 native profile; other multiple-of-32 canvases are not in the "
            "finite TensorRT profile");

    MiniMaxH3Geometry result;
    result.output_frames = output_frames;
    result.output_height = output_height;
    result.output_width = output_width;
    result.video_latent_frames = ((output_frames - 5) / 17) * 5 + 2;
    result.latent_height = output_height / 16;
    result.latent_width = output_width / 16;
    result.audio_latent_frames =
        static_cast<int32_t>(std::lround(static_cast<double>(output_frames) * 40.0 / 24.0));
    result.audio_rows = result.audio_latent_frames * 2;
    const int64_t video_rows = static_cast<int64_t>(result.video_latent_frames) *
                               (result.latent_height / 2) * (result.latent_width / 2);
    if (video_rows > kMaxTargetVideoRows)
        throw std::overflow_error("MiniMax-H3 packed video rows exceed the finite native profile");
    result.target_video_rows = static_cast<int32_t>(video_rows);
    result.video_rows = result.target_video_rows;

    const auto tile_layout = make_minimax_h3_vae_tile_layout(output_height, output_width);
    result.vae_tile_rows = static_cast<int32_t>(tile_layout.y_starts.size());
    result.vae_tile_columns = static_cast<int32_t>(tile_layout.x_starts.size());
    result.vae_tile_count = result.vae_tile_rows * result.vae_tile_columns;
    if (result.vae_tile_count < kMinTileBatch || result.vae_tile_count > kMaxTileBatch)
        throw std::logic_error("MiniMax-H3 native canvas exceeded the VAE tile profile");

    return result;
}

MiniMaxH3Geometry make_minimax_h3_fl2va_geometry(const MiniMaxH3Geometry& target_geometry,
                                                 int32_t keyframe_count) {
    if (keyframe_count < 1 || keyframe_count > 2)
        throw std::invalid_argument("MiniMax-H3 FL2VA requires one or two keyframes");
    // Re-resolve the base geometry so callers cannot smuggle mutually
    // inconsistent public canvas/duration fields into the frozen plan ABI.
    MiniMaxH3Geometry result = make_minimax_h3_geometry(
        target_geometry.output_frames, target_geometry.output_height, target_geometry.output_width);
    const int32_t rows_per_frame =
        (result.latent_height / kPatchHeight) * (result.latent_width / kPatchWidth);
    result.condition_video_frames = keyframe_count;
    result.condition_video_rows = keyframe_count * rows_per_frame;
    result.video_rows = result.condition_video_rows + result.target_video_rows;
    if (result.condition_video_rows > kMaxConditionVideoRows || result.video_rows > kMaxVideoRows)
        throw std::overflow_error("MiniMax-H3 FL2VA video rows exceed the frozen plan profile");

    return result;
}

MiniMaxH3DenoiserMetadata make_minimax_h3_denoiser_metadata(int32_t text_rows,
                                                            const MiniMaxH3Geometry& geometry) {
    return make_base_denoiser_metadata(text_rows, geometry);
}

MiniMaxH3DenoiserMetadata
make_minimax_h3_fl2va_denoiser_metadata(const std::vector<int32_t>& text_token_tags,
                                        const std::vector<int32_t>& keyframe_anchors,
                                        const MiniMaxH3Geometry& geometry) {
    return make_fl2va_denoiser_metadata(text_token_tags, keyframe_anchors, geometry);
}

void validate_minimax_h3_prompt_token_count(std::size_t token_count, int32_t max_text_rows) {
    validate_prompt_token_count(token_count, max_text_rows);
}

std::vector<float> make_minimax_h3_position_ids(int32_t text_rows) {
    return make_position_ids(
        text_rows,
        make_minimax_h3_geometry(kDefaultOutputFrames, kDefaultOutputHeight, kDefaultOutputWidth));
}

std::vector<float> make_minimax_h3_position_ids(int32_t text_rows,
                                                const MiniMaxH3Geometry& geometry) {
    return make_position_ids(text_rows, geometry);
}

std::vector<float> unpack_and_denormalize_minimax_h3_audio(const std::vector<float>& audio_rows,
                                                           int32_t audio_latent_frames) {
    if (audio_latent_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 audio latent frame count must be positive");
    const std::size_t expected = static_cast<std::size_t>(2) * audio_latent_frames * kAudioChannels;
    if (audio_rows.size() != expected)
        throw std::invalid_argument("MiniMax-H3 audio row count does not match its latent frames");

    std::vector<float> channel_major(expected);
    for (int32_t channel = 0; channel < 2; ++channel) {
        for (int32_t frame = 0; frame < audio_latent_frames; ++frame) {
            for (int32_t latent_channel = 0; latent_channel < kAudioChannels; ++latent_channel) {
                const auto source =
                    (static_cast<std::size_t>(channel) * audio_latent_frames + frame) *
                        kAudioChannels +
                    latent_channel;
                const auto target =
                    (static_cast<std::size_t>(channel) * kAudioChannels + latent_channel) *
                        audio_latent_frames +
                    frame;
                channel_major[target] = audio_rows[source] * kAudioLatentStd[latent_channel] +
                                        kAudioLatentMean[latent_channel];
            }
        }
    }
    return channel_major;
}

std::vector<float>
duplicate_minimax_h3_audio_decoder_channel(const std::vector<float>& channel_major_latents,
                                           int32_t audio_latent_frames, int32_t stereo_channel) {
    if (audio_latent_frames <= 0)
        throw std::invalid_argument("MiniMax-H3 audio latent frame count must be positive");
    if (stereo_channel < 0 || stereo_channel >= 2)
        throw std::invalid_argument("MiniMax-H3 stereo channel must be zero or one");
    const std::size_t channel_values =
        static_cast<std::size_t>(kAudioChannels) * audio_latent_frames;
    if (channel_major_latents.size() != 2U * channel_values)
        throw std::invalid_argument("MiniMax-H3 channel-major audio latent size is invalid");

    const auto source_begin = channel_major_latents.begin() +
                              static_cast<std::ptrdiff_t>(stereo_channel * channel_values);
    const auto source_end = source_begin + static_cast<std::ptrdiff_t>(channel_values);
    std::vector<float> duplicated(2U * channel_values);
    std::copy(source_begin, source_end, duplicated.begin());
    std::copy(source_begin, source_end,
              duplicated.begin() + static_cast<std::ptrdiff_t>(channel_values));
    return duplicated;
}

struct MiniMaxH3Pipeline::ResidentState {
    std::string prompt;
    std::vector<float> text_embeddings;
    std::vector<int32_t> text_token_tags;
    int32_t text_rows{0};
    int32_t denoiser_profile_index{-1};
    MiniMaxH3Geometry denoiser_geometry{};
    std::vector<StepModulation> modulations;
    std::unique_ptr<DeviceTensor> head_hidden;
    std::unique_ptr<DeviceTensor> head_residual;
    std::unique_ptr<DeviceTensor> previous_head_residual;
    std::unique_ptr<DeviceTensor> tail_residual;
    std::unique_ptr<DeviceTensor> video_rows;
    std::unique_ptr<DeviceTensor> audio_rows;
    std::unique_ptr<DeviceTensor> video_velocity;
    std::unique_ptr<DeviceTensor> audio_velocity;
    std::unique_ptr<DeviceTensor> vae_latent_tiles;
    std::unique_ptr<DeviceTensor> vae_decoded_tiles;
    std::unique_ptr<DeviceTensor> vae_overlap;
    std::unique_ptr<DeviceTensor> frame_major_rgb;
    std::unique_ptr<ITrtModule> denoiser_head;
    std::unique_ptr<ITrtModule> denoiser_tail;
    std::unique_ptr<ITrtModule> denoiser_finish;
    std::unique_ptr<ITrtModule> vae;

    void load_text_embeddings(const std::string& requested_prompt, ITokenizer& tokenizer,
                              const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                              int32_t max_text_rows);
    std::vector<std::vector<float>>
    load_fl2va_conditioning(const std::string& requested_prompt,
                            const MiniMaxH3PreparedKeyframes& keyframes, ITokenizer& tokenizer,
                            const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    void load_modulations(const MiniMaxH3Schedule& video_schedule,
                          const MiniMaxH3Schedule& audio_schedule,
                          const MiniMaxH3ModuleLoader& loader, cudaStream_t stream);
    bool prepare_denoiser(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                          int32_t optimization_profile_count,
                          const MiniMaxH3Geometry& geometry);
    DenoiserStats run_denoiser(MiniMaxH3DenoiserMetadata& metadata,
                               const MiniMaxH3Schedule& video_schedule,
                               const MiniMaxH3Schedule& audio_schedule,
                               std::vector<float>& video_rows_host,
                               std::vector<float>& audio_rows_host, float cache_threshold,
                               cudaStream_t stream);
    bool prepare_vae(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                     const MiniMaxH3Geometry& geometry);
    std::vector<float> decode_vae(std::size_t expected_pixels,
                                  const MiniMaxH3Geometry& geometry, cudaStream_t stream);
    void prepare_ref2va_vae(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                            const MiniMaxH3Geometry& geometry);
    std::vector<float> decode_ref2va_vae(const std::vector<float>& latent,
                                         std::size_t expected_pixels,
                                         const MiniMaxH3Geometry& geometry);
    AudioResult decode_audio(const std::vector<float>& audio_rows_host,
                             const MiniMaxH3Geometry& geometry, const MiniMaxH3ModuleLoader& loader,
                             cudaStream_t stream);

    void release_denoiser_stage(bool preserve_video_rows);
    void release_vae_stage();
    bool denoiser_is_resident(int32_t profile_index) const;
    void load_first_block_cache_denoiser(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                                         const MiniMaxH3Geometry& geometry, int32_t profile_index,
                                         int32_t profile_count);
    void bind_first_block_cache_shapes(const MiniMaxH3Geometry& geometry);
    bool vae_is_resident() const;
    void load_first_block_cache_vae(const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
                                    const MiniMaxH3Geometry& geometry);
};

namespace {

void sync_and_reset(std::unique_ptr<ITrtModule>& module) {
    if (module)
        module->sync();
    module.reset();
}

} // namespace

void MiniMaxH3Pipeline::ResidentState::release_denoiser_stage(bool preserve_video_rows) {
    // Contexts must be destroyed before buffers prebound into them.
    sync_and_reset(denoiser_tail);
    sync_and_reset(denoiser_head);
    sync_and_reset(denoiser_finish);
    head_hidden.reset();
    head_residual.reset();
    previous_head_residual.reset();
    tail_residual.reset();
    if (!preserve_video_rows)
        video_rows.reset();
    audio_rows.reset();
    video_velocity.reset();
    audio_velocity.reset();
    denoiser_geometry = {};
    denoiser_profile_index = -1;
}

void MiniMaxH3Pipeline::ResidentState::release_vae_stage() {
    sync_and_reset(vae);
    vae_latent_tiles.reset();
    vae_decoded_tiles.reset();
    vae_overlap.reset();
    frame_major_rgb.reset();
}

void MiniMaxH3Pipeline::ResidentState::load_text_embeddings(const std::string& requested_prompt,
                                                            ITokenizer& tokenizer,
                                                            const MiniMaxH3ModuleLoader& loader,
                                                            cudaStream_t stream,
                                                            int32_t max_text_rows) {
    // The text encoder is the largest plan. Drop resident execution modules
    // before loading it so prompt changes retain the previous peak-memory
    // behavior on smaller devices.
    release_denoiser_stage(/*preserve_video_rows=*/false);
    release_vae_stage();
    prompt.clear();
    text_embeddings.clear();
    text_token_tags.clear();
    text_rows = 0;
    const auto ids = tokenizer.encode(requested_prompt);
    validate_prompt_token_count(ids.size(), max_text_rows);
    const int32_t requested_text_rows = static_cast<int32_t>(ids.size());
    std::vector<int32_t> position_ids(ids.size());
    for (int32_t index = 0; index < requested_text_rows; ++index)
        position_ids[static_cast<std::size_t>(index)] = index;
    auto module = loader("text_encoder_plan", stream, {}, 0);
    module->set_timing_label("text_encoder_plan");
    TensorMap inputs;
    inputs.emplace("input_ids",
                   Tensor{const_cast<int32_t*>(ids.data()), {requested_text_rows}, DType::kInt32});
    if (!module->has_input("mrope_position_ids") || module->input_info().size() != 9U ||
        module->has_input("position_ids")) {
        throw std::runtime_error("MiniMax-H3 unified text plan input ABI mismatch");
    }
    std::vector<int32_t> mrope_positions(static_cast<std::size_t>(3) * requested_text_rows);
    for (int32_t axis = 0; axis < 3; ++axis) {
        std::copy(position_ids.begin(), position_ids.end(),
                  mrope_positions.begin() +
                      static_cast<std::ptrdiff_t>(axis * requested_text_rows));
    }
    std::vector<float> vision_mask(static_cast<std::size_t>(requested_text_rows), 0.0F);
    int32_t vision_count = 0;
    int32_t dummy_vision_index = 0;
    std::vector<float> dummy_vision(static_cast<std::size_t>(kTextDim), 0.0F);
    inputs.emplace("mrope_position_ids",
                   Tensor{mrope_positions.data(), {3, requested_text_rows}, DType::kInt32});
    inputs.emplace("vision_mask",
                   Tensor{vision_mask.data(), {requested_text_rows, 1}, DType::kFloat32});
    inputs.emplace("vision_count", Tensor{&vision_count, {1}, DType::kInt32});
    inputs.emplace("vision_row_indices", Tensor{&dummy_vision_index, {1}, DType::kInt32});
    for (const char* name : {"vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"}) {
        inputs.emplace(name, Tensor{dummy_vision.data(), {1, kTextDim}, DType::kFloat32});
    }
    const auto outputs = module->forward(inputs);
    text_embeddings =
        copy_float(require_output(outputs, "encoder_hidden_states"),
                   static_cast<std::size_t>(requested_text_rows) * kTextDim, "text encoder");
    module->sync();
    text_rows = requested_text_rows;
    text_token_tags.assign(static_cast<std::size_t>(text_rows), 1);
    prompt = requested_prompt;
}

std::vector<std::vector<float>> MiniMaxH3Pipeline::ResidentState::load_fl2va_conditioning(
    const std::string& requested_prompt, const MiniMaxH3PreparedKeyframes& keyframes,
    ITokenizer& tokenizer, const MiniMaxH3ModuleLoader& loader, cudaStream_t stream) {
    if (keyframes.images.empty() || keyframes.images.size() > 2U ||
        keyframes.images.size() != keyframes.anchors.size())
        throw std::invalid_argument("MiniMax-H3 FL2VA prepared keyframes are inconsistent");
    release_denoiser_stage(/*preserve_video_rows=*/false);
    release_vae_stage();
    prompt.clear();
    text_embeddings.clear();
    text_token_tags.clear();
    text_rows = 0;

    auto conditioning = minimax_h3::run_fl2va_conditioning(
        requested_prompt, keyframes, tokenizer,
        [&](const std::string& section) { return loader(section, stream, {}, 0); });
    text_embeddings = std::move(conditioning.text_embeddings);
    text_token_tags = std::move(conditioning.text_token_tags);
    text_rows = static_cast<int32_t>(text_token_tags.size());
    // Media participates in the conditioning cache key; until an explicit
    // content hash is part of the public request ABI, FL2VA intentionally does
    // not reuse a prompt-only cache entry.
    prompt.clear();
    return std::move(conditioning.keyframe_latents);
}

void MiniMaxH3Pipeline::ResidentState::load_modulations(const MiniMaxH3Schedule& video_schedule,
                                                        const MiniMaxH3Schedule& audio_schedule,
                                                        const MiniMaxH3ModuleLoader& loader,
                                                        cudaStream_t stream) {
    auto module = loader("adaln_precompute_plan", stream, {}, 0);
    module->set_timing_label("adaln_precompute_plan");
    modulations = precompute_modulations(*module, video_schedule, audio_schedule);
    module->sync();
}

bool MiniMaxH3Pipeline::ResidentState::denoiser_is_resident(int32_t profile_index) const {
    return denoiser_profile_index == profile_index && denoiser_head != nullptr &&
           denoiser_tail != nullptr && denoiser_finish != nullptr &&
           device_tensors_ready({head_hidden.get(), head_residual.get(),
                                 previous_head_residual.get(), tail_residual.get(),
                                 video_rows.get(), audio_rows.get(), video_velocity.get(),
                                 audio_velocity.get()});
}

void MiniMaxH3Pipeline::ResidentState::load_first_block_cache_denoiser(
    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream, const MiniMaxH3Geometry& geometry,
    int32_t profile_index, int32_t profile_count) {
    if (text_rows < kMinTextRows || text_rows > kMaxTextRows)
        throw std::logic_error("MiniMax-H3 text embeddings are not prepared");
    const bool five_second_profile = profile_count == 2 && profile_index == 0;
    const int64_t profile_sequence_rows =
        five_second_profile ? kFiveSecondPackedRows : kMaxSequenceRows;
    const int64_t profile_video_rows =
        five_second_profile ? kFiveSecondVideoRows : kMaxVideoRows;
    const int64_t profile_audio_rows =
        five_second_profile ? kFiveSecondAudioRows : kMaxAudioRows;
    std::cerr << "[minimax-h3] denoiser optimization_profile=" << profile_index << '/'
              << profile_count << " packed_rows=" << profile_sequence_rows << '\n';

    DeviceTensor new_head_hidden({profile_sequence_rows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_head_residual({profile_sequence_rows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_previous_head_residual({profile_sequence_rows, kHidden}, DType::kBFloat16,
                                            stream);
    DeviceTensor new_tail_residual({profile_sequence_rows, kHidden}, DType::kBFloat16, stream);
    DeviceTensor new_video_rows({profile_video_rows, kPatchDim}, DType::kFloat32, stream);
    DeviceTensor new_audio_rows({profile_audio_rows, kAudioChannels}, DType::kFloat32, stream);
    DeviceTensor new_video_velocity({profile_video_rows, kPatchDim}, DType::kFloat32, stream);
    DeviceTensor new_audio_velocity({profile_audio_rows, kAudioChannels}, DType::kFloat32, stream);
    if (!device_tensors_ready({&new_head_hidden, &new_head_residual, &new_previous_head_residual,
                               &new_tail_residual, &new_video_rows, &new_audio_rows,
                               &new_video_velocity, &new_audio_velocity}))
        throw std::runtime_error("MiniMax-H3 failed to allocate FirstBlockCache buffers");

    auto resident_head_hidden = std::make_unique<DeviceTensor>(std::move(new_head_hidden));
    auto resident_head_residual = std::make_unique<DeviceTensor>(std::move(new_head_residual));
    auto resident_previous_head_residual =
        std::make_unique<DeviceTensor>(std::move(new_previous_head_residual));
    auto resident_tail_residual = std::make_unique<DeviceTensor>(std::move(new_tail_residual));
    auto resident_video_rows = std::make_unique<DeviceTensor>(std::move(new_video_rows));
    auto resident_audio_rows = std::make_unique<DeviceTensor>(std::move(new_audio_rows));
    auto resident_video_velocity = std::make_unique<DeviceTensor>(std::move(new_video_velocity));
    auto resident_audio_velocity = std::make_unique<DeviceTensor>(std::move(new_audio_velocity));

    // Prebind max-capacity buffers during deserialization so dynamic plans do
    // not first allocate a second set of max-profile buffers. Runtime shapes
    // are selected immediately after module creation below.
    const std::vector<ModuleExternalBinding> head_bindings = {
        external_binding("head_hidden", *resident_head_hidden),
        external_binding("head_residual", *resident_head_residual),
        external_binding("previous_head_residual", *resident_previous_head_residual),
        external_binding("video_hidden_states", *resident_video_rows),
        external_binding("audio_hidden_states", *resident_audio_rows),
    };
    const std::vector<ModuleExternalBinding> tail_bindings = {
        external_binding("head_hidden", *resident_head_hidden),
        external_binding("tail_residual", *resident_tail_residual),
    };
    const std::vector<ModuleExternalBinding> finish_bindings = {
        external_binding("head_hidden", *resident_head_hidden),
        external_binding("tail_residual", *resident_tail_residual),
        external_binding("video_hidden_states", *resident_video_rows),
        external_binding("audio_hidden_states", *resident_audio_rows),
        external_binding("video_velocity", *resident_video_velocity),
        external_binding("audio_velocity", *resident_audio_velocity),
    };
    auto head = loader("denoiser_head_plan", stream, head_bindings, profile_index);
    auto tail = loader("denoiser_tail_plan", stream, tail_bindings, profile_index);
    auto finish = loader("denoiser_finish_plan", stream, finish_bindings, profile_index);
    head->set_timing_label("denoiser_head_plan");
    tail->set_timing_label("denoiser_tail_plan");
    finish->set_timing_label("denoiser_finish_plan");

    bind_external_dynamic_output_checked(*head, "head_hidden", resident_head_hidden->data(),
                                         DType::kBFloat16, {profile_sequence_rows, kHidden});
    bind_external_dynamic_output_checked(*head, "head_residual", resident_head_residual->data(),
                                         DType::kBFloat16, {profile_sequence_rows, kHidden});
    bind_external_dynamic_output_checked(*tail, "tail_residual", resident_tail_residual->data(),
                                         DType::kBFloat16, {profile_sequence_rows, kHidden});
    bind_external_dynamic_output_checked(*finish, "video_velocity", resident_video_velocity->data(),
                                         DType::kFloat32, {profile_video_rows, kPatchDim});
    bind_external_dynamic_output_checked(*finish, "audio_velocity", resident_audio_velocity->data(),
                                         DType::kFloat32, {profile_audio_rows, kAudioChannels});

    denoiser_head = std::move(head);
    denoiser_tail = std::move(tail);
    denoiser_finish = std::move(finish);
    head_hidden = std::move(resident_head_hidden);
    head_residual = std::move(resident_head_residual);
    previous_head_residual = std::move(resident_previous_head_residual);
    tail_residual = std::move(resident_tail_residual);
    video_rows = std::move(resident_video_rows);
    audio_rows = std::move(resident_audio_rows);
    video_velocity = std::move(resident_video_velocity);
    audio_velocity = std::move(resident_audio_velocity);
    denoiser_profile_index = profile_index;
    bind_first_block_cache_shapes(geometry);
}

void MiniMaxH3Pipeline::ResidentState::bind_first_block_cache_shapes(
    const MiniMaxH3Geometry& geometry) {
    if (!denoiser_head || !denoiser_tail || !denoiser_finish)
        throw std::logic_error("MiniMax-H3 split denoiser is not loaded");
    const int64_t sequence_rows =
        static_cast<int64_t>(text_rows) + geometry.audio_rows + geometry.video_rows;
    const int32_t profile_count = denoiser_head->optimization_profile_count();
    const bool five_second_profile = profile_count == 2 && denoiser_profile_index == 0;
    const int64_t profile_sequence_rows =
        five_second_profile ? kFiveSecondPackedRows : kMaxSequenceRows;
    const int64_t profile_video_rows =
        five_second_profile ? kFiveSecondVideoRows : kMaxVideoRows;
    const int64_t profile_audio_rows =
        five_second_profile ? kFiveSecondAudioRows : kMaxAudioRows;
    bind_external_dynamic_input_checked(*denoiser_head, "previous_head_residual",
                                        previous_head_residual->data(), DType::kBFloat16,
                                        {sequence_rows, kHidden}, {profile_sequence_rows, kHidden},
                                        denoiser_profile_index, profile_count);
    bind_external_dynamic_input_checked(*denoiser_head, "video_hidden_states", video_rows->data(),
                                        DType::kFloat32, {geometry.video_rows, kPatchDim},
                                        {profile_video_rows, kPatchDim}, denoiser_profile_index,
                                        profile_count);
    bind_external_dynamic_input_checked(*denoiser_head, "audio_hidden_states", audio_rows->data(),
                                        DType::kFloat32, {geometry.audio_rows, kAudioChannels},
                                        {profile_audio_rows, kAudioChannels},
                                        denoiser_profile_index, profile_count);
    bind_external_dynamic_input_checked(*denoiser_tail, "head_hidden", head_hidden->data(),
                                        DType::kBFloat16, {sequence_rows, kHidden},
                                        {profile_sequence_rows, kHidden}, denoiser_profile_index,
                                        profile_count);
    bind_external_dynamic_input_checked(*denoiser_finish, "head_hidden", head_hidden->data(),
                                        DType::kBFloat16, {sequence_rows, kHidden},
                                        {profile_sequence_rows, kHidden}, denoiser_profile_index,
                                        profile_count);
    bind_external_dynamic_input_checked(*denoiser_finish, "tail_residual", tail_residual->data(),
                                        DType::kBFloat16, {sequence_rows, kHidden},
                                        {profile_sequence_rows, kHidden}, denoiser_profile_index,
                                        profile_count);
    bind_external_dynamic_input_checked(*denoiser_finish, "video_hidden_states", video_rows->data(),
                                        DType::kFloat32, {geometry.video_rows, kPatchDim},
                                        {profile_video_rows, kPatchDim}, denoiser_profile_index,
                                        profile_count);
    bind_external_dynamic_input_checked(*denoiser_finish, "audio_hidden_states", audio_rows->data(),
                                        DType::kFloat32, {geometry.audio_rows, kAudioChannels},
                                        {profile_audio_rows, kAudioChannels},
                                        denoiser_profile_index, profile_count);
    denoiser_geometry = geometry;
}

bool MiniMaxH3Pipeline::ResidentState::prepare_denoiser(const MiniMaxH3ModuleLoader& loader,
                                                        cudaStream_t stream,
                                                        int32_t optimization_profile_count,
                                                        const MiniMaxH3Geometry& geometry) {
    // The previous request synchronized its VAE before returning. Release it
    // before deserializing the denoiser so the two large stages never overlap.
    release_vae_stage();
    const int32_t profile_index =
        select_minimax_h3_denoiser_profile(optimization_profile_count, text_rows, geometry);
    const bool same_shape = denoiser_geometry.video_rows == geometry.video_rows &&
                            denoiser_geometry.audio_rows == geometry.audio_rows;
    const bool resident_hit = same_shape && denoiser_is_resident(profile_index);
    if (!resident_hit)
        release_denoiser_stage(/*preserve_video_rows=*/false);
    if (resident_hit) {
        bind_first_block_cache_shapes(geometry);
        return true;
    }
    load_first_block_cache_denoiser(loader, stream, geometry, profile_index,
                                    optimization_profile_count);
    return false;
}

DenoiserStats MiniMaxH3Pipeline::ResidentState::run_denoiser(
    MiniMaxH3DenoiserMetadata& metadata, const MiniMaxH3Schedule& video_schedule,
    const MiniMaxH3Schedule& audio_schedule, std::vector<float>& video_rows_host,
    std::vector<float>& audio_rows_host, float cache_threshold, cudaStream_t stream) {
    DenoiserStats stats;
    auto& head = *denoiser_head;
    auto& tail = *denoiser_tail;
    auto& finish = *denoiser_finish;
    if (video_rows_host.size() !=
            static_cast<std::size_t>(denoiser_geometry.video_rows) * kPatchDim ||
        audio_rows_host.size() !=
            static_cast<std::size_t>(denoiser_geometry.audio_rows) * kAudioChannels)
        throw std::invalid_argument("MiniMax-H3 denoiser latents do not match request geometry");
    const int64_t sequence_rows = static_cast<int64_t>(metadata.adaln_indices.size());
    const int64_t expected_sequence_rows = static_cast<int64_t>(text_rows) +
                                           denoiser_geometry.audio_rows +
                                           denoiser_geometry.video_rows;
    if (sequence_rows != expected_sequence_rows || sequence_rows < kMinPackedRows ||
        sequence_rows > kMaxPackedRows ||
        metadata.positions.size() != static_cast<std::size_t>(sequence_rows) * 3U ||
        metadata.timestep_indices.size() != static_cast<std::size_t>(sequence_rows)) {
        throw std::invalid_argument("MiniMax-H3 FirstBlockCache metadata does not match geometry");
    }
    head.reset_execution_context();
    tail.reset_execution_context();
    finish.reset_execution_context();
    const std::size_t sequence_bytes =
        static_cast<std::size_t>(sequence_rows) * kHidden * sizeof(uint16_t);
    const std::size_t video_bytes = video_rows_host.size() * sizeof(float);
    const std::size_t audio_bytes = audio_rows_host.size() * sizeof(float);
    if (cudaMemsetAsync(previous_head_residual->data(), 0, sequence_bytes, stream) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to reset FirstBlockCache state");
    if (cudaMemcpyAsync(video_rows->data(), video_rows_host.data(), video_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(audio_rows->data(), audio_rows_host.data(), audio_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to upload FirstBlockCache latents");

    for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
        auto& modulation = modulations[step];
        TensorMap head_inputs;
        head_inputs.emplace("encoder_hidden_states",
                            Tensor{text_embeddings.data(), {text_rows, kTextDim}, DType::kFloat32});
        head_inputs.emplace("position_ids",
                            Tensor{metadata.positions.data(), {sequence_rows, 3}, DType::kFloat32});
        head_inputs.emplace("adaln_indices",
                            Tensor{metadata.adaln_indices.data(), {sequence_rows}, DType::kInt32});
        append_block_modulation_inputs(head_inputs, modulation, 0, 1);
        const auto head_outputs = head.forward(head_inputs);
        const float metric =
            copy_float(require_output(head_outputs, "cache_metric"), 1, "cache metric")[0];
        const bool compute_tail = should_compute_minimax_h3_tail(
            step, video_schedule.timesteps.size(), metric, cache_threshold);

        if (compute_tail) {
            TensorMap tail_inputs;
            tail_inputs.emplace(
                "position_ids",
                Tensor{metadata.positions.data(), {sequence_rows, 3}, DType::kFloat32});
            tail_inputs.emplace(
                "adaln_indices",
                Tensor{metadata.adaln_indices.data(), {sequence_rows}, DType::kInt32});
            append_block_modulation_inputs(tail_inputs, modulation, 1, kLayers);
            tail.forward_async(tail_inputs);
            if (cudaMemcpyAsync(previous_head_residual->data(), head_residual->data(),
                                sequence_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess)
                throw std::runtime_error("MiniMax-H3 failed to update FirstBlockCache state");
            ++stats.full_steps;
        } else {
            ++stats.skipped_steps;
        }

        TensorMap finish_inputs;
        finish_inputs.emplace(
            "timestep_indices",
            Tensor{metadata.timestep_indices.data(), {sequence_rows}, DType::kInt32});
        append_final_modulation_input(finish_inputs, modulation);
        finish.forward_async(finish_inputs);
        const std::size_t condition_elements =
            static_cast<std::size_t>(denoiser_geometry.condition_video_rows) * kPatchDim;
        const std::size_t target_elements =
            static_cast<std::size_t>(denoiser_geometry.target_video_rows) * kPatchDim;
        minimax_h3::scheduler_step_cuda_async(
            static_cast<float*>(video_rows->data()) + condition_elements,
            static_cast<const float*>(video_velocity->data()) + condition_elements, target_elements,
            video_schedule.timesteps[step], video_schedule.sigmas[step],
            video_schedule.sigmas[step + 1], stream);
        minimax_h3::scheduler_step_cuda_async(
            static_cast<float*>(audio_rows->data()),
            static_cast<const float*>(audio_velocity->data()), audio_rows_host.size(),
            audio_schedule.timesteps[step], audio_schedule.sigmas[step],
            audio_schedule.sigmas[step + 1], stream);
        std::cerr << "[minimax-h3] denoiser " << (step + 1) << '/'
                  << video_schedule.timesteps.size() << " cache_metric=" << metric
                  << " compute_tail=" << static_cast<int>(compute_tail) << '\n';
    }
    finish.sync();
    if (cudaMemcpy(audio_rows_host.data(), audio_rows->data(), audio_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to download final audio latents");
    return stats;
}

bool MiniMaxH3Pipeline::ResidentState::vae_is_resident() const {
    return vae != nullptr && device_tensors_ready({vae_latent_tiles.get(), vae_decoded_tiles.get(),
                                                   vae_overlap.get(), frame_major_rgb.get()});
}

void MiniMaxH3Pipeline::ResidentState::load_first_block_cache_vae(
    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream, const MiniMaxH3Geometry& geometry) {
    DeviceTensor latent_tiles(
        {kMaxTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize},
        DType::kFloat32, stream);
    DeviceTensor decoded_tiles({kMaxTileBatch, 3, kTileFrames, kTileSize, kTileSize},
                               DType::kFloat32, stream);
    DeviceTensor overlap({3, 5, kMaxOutputPixels}, DType::kFloat32, stream);
    DeviceTensor output_pixels({kMaxOutputFrames, kMaxOutputPixels, 3}, DType::kFloat32, stream);
    if (!device_tensors_ready({&latent_tiles, &decoded_tiles, &overlap, &output_pixels}))
        throw std::runtime_error("MiniMax-H3 failed to allocate CUDA VAE buffers");

    auto resident_latent_tiles = std::make_unique<DeviceTensor>(std::move(latent_tiles));
    auto resident_decoded_tiles = std::make_unique<DeviceTensor>(std::move(decoded_tiles));
    auto resident_overlap = std::make_unique<DeviceTensor>(std::move(overlap));
    auto resident_frame_major_rgb = std::make_unique<DeviceTensor>(std::move(output_pixels));
    const std::vector<ModuleExternalBinding> vae_bindings = {
        external_binding("latent_tiles", *resident_latent_tiles),
        external_binding("decoded_tiles", *resident_decoded_tiles),
    };
    auto module = loader("vae_tile_decoder_plan", stream, vae_bindings, 0);
    module->set_timing_label("vae_tile_decoder_plan");
    validate_vae_plan_geometry(*module, geometry);
    bind_external_dynamic_input_checked(
        *module, "latent_tiles", resident_latent_tiles->data(), DType::kFloat32,
        {geometry.vae_tile_count, kLatentChannels, kTileInputFrames, kTileLatentSize,
         kTileLatentSize},
        {kMaxTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize});
    bind_external_dynamic_output_checked(*module, "decoded_tiles",
                                         resident_decoded_tiles->data(), DType::kFloat32,
                                         {kMaxTileBatch, 3, kTileFrames, kTileSize, kTileSize});

    vae = std::move(module);
    vae_latent_tiles = std::move(resident_latent_tiles);
    vae_decoded_tiles = std::move(resident_decoded_tiles);
    vae_overlap = std::move(resident_overlap);
    frame_major_rgb = std::move(resident_frame_major_rgb);
}

bool MiniMaxH3Pipeline::ResidentState::prepare_vae(const MiniMaxH3ModuleLoader& loader,
                                                   cudaStream_t stream,
                                                   const MiniMaxH3Geometry& geometry) {
    const bool resident_hit = vae_is_resident();
    // Keep final FirstBlockCache video rows for tiled VAE extraction, but
    // release every denoiser context and all other bound buffers first.
    release_denoiser_stage(/*preserve_video_rows=*/true);
    if (resident_hit) {
        validate_vae_plan_geometry(*vae, geometry);
        bind_external_dynamic_input_checked(
            *vae, "latent_tiles", vae_latent_tiles->data(), DType::kFloat32,
            {geometry.vae_tile_count, kLatentChannels, kTileInputFrames, kTileLatentSize,
             kTileLatentSize},
            {kMaxTileBatch, kLatentChannels, kTileInputFrames, kTileLatentSize, kTileLatentSize});
        return true;
    }
    load_first_block_cache_vae(loader, stream, geometry);
    return false;
}

std::vector<float> MiniMaxH3Pipeline::ResidentState::decode_vae(
    std::size_t expected_pixels, const MiniMaxH3Geometry& geometry, cudaStream_t stream) {
    auto& module = *vae;
    module.reset_execution_context();
    const auto latent_normalization = vae_latent_normalization();
    const auto pixel_normalization = vae_pixel_normalization();
    const int32_t clip_count = (geometry.output_frames - 5) / 17;
    const auto* target_video_rows =
        static_cast<const float*>(video_rows->data()) +
        static_cast<std::size_t>(geometry.condition_video_rows) * kPatchDim;
    TensorMap no_inputs;
    for (int32_t clip_index = 0; clip_index < clip_count; ++clip_index) {
        minimax_h3::extract_vae_tiles_cuda_async(
            target_video_rows, static_cast<float*>(vae_latent_tiles->data()), clip_index,
            clip_count, geometry.output_height, geometry.output_width, geometry.vae_tile_rows,
            geometry.vae_tile_columns, latent_normalization, stream);
        module.forward_async(no_inputs);
        minimax_h3::assemble_vae_clip_cuda_async(
            static_cast<const float*>(vae_decoded_tiles->data()),
            static_cast<float*>(vae_overlap->data()), static_cast<float*>(frame_major_rgb->data()),
            clip_index, clip_count, geometry.output_height, geometry.output_width,
            geometry.vae_tile_rows, geometry.vae_tile_columns, pixel_normalization, stream);
        std::cerr << "[minimax-h3] VAE clip " << (clip_index + 1) << '/' << clip_count << '\n';
    }
    module.sync();
    std::vector<float> pixels(expected_pixels);
    if (cudaMemcpy(pixels.data(), frame_major_rgb->data(), expected_pixels * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to download CUDA VAE output");
    release_vae_stage();
    return pixels;
}

void MiniMaxH3Pipeline::ResidentState::prepare_ref2va_vae(
    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream,
    const MiniMaxH3Geometry& geometry) {
    release_denoiser_stage(/*preserve_video_rows=*/false);
    release_vae_stage();
    vae = loader("vae_tile_decoder_plan", stream, {}, 0);
    vae->set_timing_label("vae_tile_decoder_plan");
    validate_vae_plan_geometry(*vae, geometry);
}

std::vector<float>
MiniMaxH3Pipeline::ResidentState::decode_ref2va_vae(const std::vector<float>& latent,
                                                    std::size_t expected_pixels,
                                                    const MiniMaxH3Geometry& geometry) {
    std::vector<float> video(expected_pixels);
    std::size_t decoded_frames = 0;
    std::vector<float> overlap;
    std::vector<float> clip;
    auto& module = *vae;
    module.reset_execution_context();
    const int32_t clip_count = (geometry.output_frames - 5) / 17;
    const std::size_t output_count =
        static_cast<std::size_t>(geometry.vae_tile_count) * 3 * kTileFrames * kTileSize * kTileSize;
    for (int32_t clip_index = 0; clip_index < clip_count; ++clip_index) {
        auto latent_tiles = extract_tiles(latent, clip_index, geometry);
        TensorMap inputs;
        inputs.emplace("latent_tiles", Tensor{latent_tiles.data(),
                                              {geometry.vae_tile_count, kLatentChannels,
                                               kTileInputFrames, kTileLatentSize, kTileLatentSize},
                                              DType::kFloat32});
        const auto outputs = module.forward(inputs);
        const Tensor decoded_tiles = require_output(outputs, "decoded_tiles");
        if (decoded_tiles.numel() != output_count)
            throw std::runtime_error("MiniMax-H3 invalid VAE decoded tiles output");
        stitch_spatial_tiles(decoded_tiles, clip, geometry);
        write_temporal_chunk(video, decoded_frames, clip, overlap, geometry);
        decoded_frames += 17;
        update_trailing_overlap(clip, overlap, geometry);
        std::cerr << "[minimax-h3] VAE clip " << (clip_index + 1) << '/' << clip_count << '\n';
    }
    module.sync();
    write_final_overlap(video, decoded_frames, overlap, geometry);
    decoded_frames += 5;
    if (video.size() != expected_pixels ||
        decoded_frames != static_cast<std::size_t>(geometry.output_frames))
        throw std::runtime_error("MiniMax-H3 VAE produced the wrong video geometry");
    postprocess_video(video, geometry);
    auto pixels = to_frame_major_rgb(video, geometry);
    release_vae_stage();
    return pixels;
}

AudioResult MiniMaxH3Pipeline::ResidentState::decode_audio(
    const std::vector<float>& audio_rows_host, const MiniMaxH3Geometry& geometry,
    const MiniMaxH3ModuleLoader& loader, cudaStream_t stream) {
    // AudioVAE is loaded only after the much larger denoiser and video VAE
    // have been released, keeping the native A/V path within the same staged
    // peak-memory envelope as the historical video-only path.
    release_denoiser_stage(/*preserve_video_rows=*/false);
    release_vae_stage();
    auto audio_latents =
        unpack_and_denormalize_minimax_h3_audio(audio_rows_host, geometry.audio_latent_frames);
    auto module = loader("audio_vae_decoder_plan", stream, {}, 0);
    module->set_timing_label("audio_vae_decoder_plan");
    const int32_t audio_output_samples = geometry.audio_latent_frames * 800;

    AudioResult result;
    result.channels = 2;
    result.sample_rate = kAudioSampleRate;
    result.samples.resize(static_cast<std::size_t>(2) * audio_output_samples);

    // The released AudioVAE applies one shared mono decoder independently to
    // the left and right channels. TensorRT-RTX 1.6.1 on SM121 can return an
    // unstable second batch item for this dynamic deconvolution graph while
    // batch item zero remains stable. Preserve the released model semantics by
    // decoding each real stereo latent through item zero. Duplicating the
    // selected channel satisfies the plan's fixed batch-two profile; the
    // duplicate output is deliberately ignored.
    for (int32_t channel = 0; channel < 2; ++channel) {
        auto decoder_input = duplicate_minimax_h3_audio_decoder_channel(
            audio_latents, geometry.audio_latent_frames, channel);
        TensorMap inputs;
        inputs.emplace("audio_latents", Tensor{decoder_input.data(),
                                               {2, kAudioChannels, geometry.audio_latent_frames},
                                               DType::kFloat32});
        const auto outputs = module->forward(inputs);
        auto duplicated_planar =
            copy_float(require_output(outputs, "decoded_audio"),
                       static_cast<std::size_t>(2) * audio_output_samples, "AudioVAE decoder");
        for (int32_t sample = 0; sample < audio_output_samples; ++sample) {
            result.samples[static_cast<std::size_t>(sample) * 2 + channel] =
                duplicated_planar[sample];
        }
    }
    module->sync();
    if (result.samples.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::overflow_error("MiniMax-H3 decoded audio exceeds the public result capacity");
    result.num_samples = static_cast<int32_t>(result.samples.size());
    return result;
}

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift) {
    if (grid_points < 2 || shift <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 schedule arguments are invalid");
    MiniMaxH3Schedule result;
    result.sigmas.reserve(grid_points);
    for (int32_t index = 0; index < grid_points; ++index) {
        const float base = static_cast<float>(1.0 - static_cast<double>(index) / (grid_points - 1));
        const float sigma = shift * base / (1.0F + (shift - 1.0F) * base);
        if (result.sigmas.empty() || sigma != result.sigmas.back())
            result.sigmas.push_back(sigma);
    }
    if (result.sigmas.size() < 2 || result.sigmas.back() != 0.0F)
        throw std::runtime_error("MiniMax-H3 sigma grid collapsed unexpectedly");
    result.timesteps.reserve(result.sigmas.size() - 1);
    for (std::size_t index = 0; index + 1 < result.sigmas.size(); ++index)
        result.timesteps.push_back(1.0F - result.sigmas[index]);
    return result;
}

bool should_compute_minimax_h3_tail(std::size_t step, std::size_t transformer_forwards,
                                    float metric, float cache_threshold) {
    // The final Euler update takes sigma to zero, so its velocity directly determines the
    // final latent and must come from a fresh tail evaluation rather than cached tail state.
    const bool boundary_step =
        step == 0 || (transformer_forwards != 0 && step == transformer_forwards - 1);
    return boundary_step || !std::isfinite(metric) || metric > cache_threshold;
}

void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next) {
    if (sample == nullptr || velocity == nullptr || !(sigma > 0.0F))
        throw std::invalid_argument("MiniMax-H3 scheduler received invalid inputs");
    const float sigma_from_timestep = 1.0F - timestep;
    const float ratio = sigma_next / sigma;
    for (std::size_t index = 0; index < count; ++index) {
        const float denoised = sample[index] + sigma_from_timestep * velocity[index];
        sample[index] = ratio * sample[index] + (1.0F - ratio) * denoised;
    }
}

MiniMaxH3Pipeline::MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader,
                                     std::unique_ptr<ITokenizer> tokenizer, std::string model_id,
                                     float cache_threshold,
                                     MiniMaxH3DenoiserConfig denoiser_config,
                                     MiniMaxH3Ref2VAConfig ref2va_config,
                                     std::function<void()> runtime_cache_finalize)
    : loader_(std::move(loader)), tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id)),
      resident_(std::make_unique<ResidentState>()), cache_threshold_(cache_threshold),
      denoiser_config_(denoiser_config),
      ref2va_config_(ref2va_config), runtime_cache_finalize_(std::move(runtime_cache_finalize)) {
    if (!loader_ || !tokenizer_)
        throw std::invalid_argument("MiniMax-H3 pipeline requires a loader and tokenizer");
    if (!std::isfinite(cache_threshold_) || cache_threshold_ <= 0.0F)
        throw std::invalid_argument("MiniMax-H3 cache threshold must be finite and positive");
    if (denoiser_config_.max_text_rows < kMinTextRows ||
        denoiser_config_.max_text_rows > kMaxTextRows) {
        throw std::invalid_argument("MiniMax-H3 denoiser text profile is invalid");
    }
    if (denoiser_config_.scheduler_grid_points < 2 ||
        denoiser_config_.transformer_forwards != denoiser_config_.scheduler_grid_points - 1 ||
        !std::isfinite(denoiser_config_.guidance_scale) ||
        denoiser_config_.guidance_scale != 1.0F) {
        throw std::invalid_argument("MiniMax-H3 denoiser schedule contract is invalid");
    }
    if (denoiser_config_.scheduler_grid_points != 50 ||
        denoiser_config_.transformer_forwards != 49) {
        throw std::invalid_argument("MiniMax-H3 dense execution requires its 50-point schedule");
    }
    if (ref2va_config_.enabled) {
        if (ref2va_config_.scheduler_grid_points != 50 ||
            ref2va_config_.transformer_forwards != 49 ||
            !std::isfinite(ref2va_config_.video_shift) || ref2va_config_.video_shift != 12.0F ||
            !std::isfinite(ref2va_config_.audio_shift) || ref2va_config_.audio_shift != 3.0F ||
            !std::isfinite(ref2va_config_.guidance_scale) ||
            ref2va_config_.guidance_scale != 1.0F || !ref2va_config_.guidance_distilled) {
            throw std::invalid_argument(
                "MiniMax-H3 Ref2VA requires its dedicated 50-point, shift-12/3 distilled "
                "schedule");
        }
        for (std::size_t index = 0; index < ref2va_config_.audio_latent_std.size(); ++index) {
            if (!std::isfinite(ref2va_config_.audio_latent_mean[index]) ||
                !std::isfinite(ref2va_config_.audio_latent_std[index]) ||
                ref2va_config_.audio_latent_std[index] <= 0.0F) {
                throw std::invalid_argument(
                    "MiniMax-H3 Ref2VA audio latent normalization is invalid");
            }
        }
    }
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess)
        throw std::runtime_error("MiniMax-H3 failed to create its CUDA stream");
}

MiniMaxH3Pipeline::~MiniMaxH3Pipeline() {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    if (stream_ != nullptr) {
        const cudaError_t status = cudaStreamSynchronize(stream_);
        if (status != cudaSuccess) {
            std::cerr << "[trtmc] Failed to synchronize MiniMax-H3 before cache cleanup: "
                      << cudaGetErrorString(status) << '\n';
        }
    }
    resident_.reset();
    if (runtime_cache_finalize_) {
        try {
            runtime_cache_finalize_();
            runtime_cache_finalize_ = {};
        } catch (const std::exception& error) {
            std::cerr << "[trtmc] Failed to persist RTX runtime cache: " << error.what() << '\n';
        } catch (...) {
            std::cerr << "[trtmc] Failed to persist RTX runtime cache: unknown error\n";
        }
    }
    if (stream_ != nullptr)
        cudaStreamDestroy(stream_);
}

VideoResult MiniMaxH3Pipeline::generate_video(const std::string& prompt,
                                              const GenerateConfig& cfg) {
    return generate_video_impl(prompt, cfg, /*include_audio=*/true);
}

VideoResult MiniMaxH3Pipeline::generate_video(const VideoGenerationRequest& request) {
    return generate_video_request_impl(request, /*include_audio=*/true);
}

std::optional<ReferenceMediaDecodePolicy> MiniMaxH3Pipeline::reference_media_decode_policy() const {
    return ReferenceMediaDecodePolicy{
        15,
        24,
        240,
        768,
        768ULL * 1344,
        32,
        0.25,
        4.0,
    };
}

VideoResult MiniMaxH3Pipeline::generate_video_impl(const std::string& prompt,
                                                   const GenerateConfig& cfg, bool include_audio) {
    VideoGenerationRequest request;
    request.prompt = prompt;
    request.config = cfg;
    request.mode = VideoGenerationMode::kTextToVideoAudio;
    return generate_video_request_impl(request, include_audio);
}

VideoResult MiniMaxH3Pipeline::generate_ref2va_request_impl(const VideoGenerationRequest& request,
                                                            bool include_audio) {
    if (request.mode != VideoGenerationMode::kReferenceToVideoAudio)
        throw std::invalid_argument("MiniMax-H3 Ref2VA dispatch received the wrong mode");
    if (request.config.num_steps > 0 && request.config.num_steps != 50)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA requires 50 sigma grid points and 49 transformer forwards");
    if (request.config.guidance_scale >= 0.0F && request.config.guidance_scale != 1.0F)
        throw std::invalid_argument(
            "MiniMax-H3 Ref2VA is guidance-distilled and requires guidance_scale=1");

    // Ref2VA target geometry is independent of every reference. With no
    // explicit --height/--width this resolves to the public 768x1344 canvas.
    const MiniMaxH3Geometry geometry = resolve_generate_geometry(request.config);
    auto prepared = minimax_h3::prepare_ref2va_request(request, geometry.output_frames);
    const int64_t seed = request.config.seed >= 0 ? request.config.seed : 0;
    const auto total_begin = Clock::now();

    resident_->release_denoiser_stage(/*preserve_video_rows=*/false);
    resident_->release_vae_stage();

    const auto text_begin = Clock::now();
    const auto blueprint =
        minimax_h3::make_ref2va_presentation_blueprint(request.prompt, prepared.references);
    const auto presentation = minimax_h3::materialize_ref2va_presentation(blueprint, *tokenizer_);
    minimax_h3::Ref2vaVisionFeatures vision_features;
    if (!blueprint.vision_invocations.empty()) {
        auto vision_module = loader_("vision_encoder_plan", stream_, {}, 0);
        vision_module->set_timing_label("ref2va_shared_vision_encoder_plan");
        vision_features = minimax_h3::run_ref2va_reference_vision_encoder(
            *vision_module, prepared.references, blueprint);
        vision_module->sync();
    }
    if (vision_features.rows != presentation.vision_rows)
        throw std::runtime_error("MiniMax-H3 Ref2VA Qwen vision/presentation row counts disagree");
    auto text_module = loader_("text_encoder_plan", stream_, {}, 0);
    text_module->set_timing_label("ref2va_shared_text_encoder_plan");
    auto text_embeddings =
        minimax_h3::run_ref2va_text_encoder(*text_module, presentation, vision_features);
    text_module->sync();
    text_module.reset();
    vision_features = {};
    const auto text_end = Clock::now();

    const auto condition_begin = Clock::now();
    std::vector<minimax_h3::Ref2vaEncodedCondition> conditions(prepared.references.size());
    if (prepared.summary.image_count > 0) {
        auto module = loader_("fl2va_keyframe_vae_encoder_plan", stream_, {}, 0);
        module->set_timing_label("ref2va_image_vae_encoder_plan");
        for (std::size_t index = 0; index < prepared.references.size(); ++index) {
            if (prepared.references[index].kind == VideoReferenceKind::kImage)
                conditions[index] = minimax_h3::run_ref2va_image_vae_encoder(
                    *module, prepared.references[index].image);
        }
        module->sync();
    }
    if (prepared.summary.video_count > 0) {
        auto module = loader_("ref2va_video_vae_encoder_plan", stream_, {}, 0);
        module->set_timing_label("ref2va_video_vae_encoder_plan");
        for (std::size_t index = 0; index < prepared.references.size(); ++index) {
            if (prepared.references[index].kind == VideoReferenceKind::kVideo)
                conditions[index] = minimax_h3::run_ref2va_video_vae_encoder(
                    *module, prepared.references[index].video);
        }
        module->sync();
    }
    if (prepared.summary.audio_bearing_count > 0) {
        auto module = loader_("ref2va_audio_vae_encoder_plan", stream_, {}, 0);
        module->set_timing_label("ref2va_audio_vae_encoder_plan");
        for (std::size_t index = 0; index < prepared.references.size(); ++index) {
            const auto& reference = prepared.references[index];
            const AudioResult* source = nullptr;
            if (reference.kind == VideoReferenceKind::kAudio)
                source = &reference.audio;
            else if (reference.kind == VideoReferenceKind::kVideo &&
                     !reference.video.soundtrack.samples.empty())
                source = &reference.video.soundtrack;
            if (source == nullptr)
                continue;
            auto encoded = minimax_h3::run_ref2va_audio_vae_encoder(
                *module, *source, ref2va_config_.audio_latent_mean,
                ref2va_config_.audio_latent_std);
            if (reference.kind == VideoReferenceKind::kAudio) {
                conditions[index] = std::move(encoded);
            } else {
                conditions[index].geometry.audio_latents = encoded.geometry.audio_latents;
                conditions[index].audio_hidden_states = std::move(encoded.audio_hidden_states);
            }
        }
        module->sync();
    }

    uint64_t generator_offset = 0;
    for (auto& condition : conditions) {
        if (condition.video_hidden_states.empty())
            continue;
        MiniMaxH3Geometry condition_geometry;
        condition_geometry.video_latent_frames = condition.geometry.latent_frames;
        condition_geometry.latent_height = condition.geometry.latent_height;
        condition_geometry.latent_width = condition.geometry.latent_width;
        condition_geometry.video_rows = condition.geometry.video_rows();
        auto latent = unpatchify_video(condition.video_hidden_states, condition_geometry);
        auto noise = minimax_h3::torch_cuda_normal(latent.size(), static_cast<uint64_t>(seed),
                                                   generator_offset);
        generator_offset += minimax_h3::torch_cuda_normal_consumed_offset(latent.size());
        for (std::size_t index = 0; index < latent.size(); ++index)
            latent[index] = 0.999F * latent[index] + 0.001F * noise[index];
        condition.video_hidden_states = patchify_video(latent, condition_geometry);
    }
    const auto condition_end = Clock::now();

    auto target_video_latent = minimax_h3::torch_cuda_normal(
        video_latent_count(geometry), static_cast<uint64_t>(seed), generator_offset);
    generator_offset += minimax_h3::torch_cuda_normal_consumed_offset(target_video_latent.size());
    auto target_video_rows = patchify_video(target_video_latent, geometry);
    target_video_latent.clear();
    target_video_latent.shrink_to_fit();
    auto target_audio_rows = minimax_h3::torch_cuda_normal(
        audio_latent_count(geometry), static_cast<uint64_t>(seed), generator_offset);

    minimax_h3::Ref2vaDenoiserInputs denoiser_inputs;
    std::vector<minimax_h3::Ref2vaEncodedReferenceGeometry> reference_geometries;
    reference_geometries.reserve(conditions.size());
    for (auto& condition : conditions) {
        reference_geometries.push_back(condition.geometry);
        denoiser_inputs.video_hidden_states.insert(denoiser_inputs.video_hidden_states.end(),
                                                   condition.video_hidden_states.begin(),
                                                   condition.video_hidden_states.end());
        denoiser_inputs.audio_hidden_states.insert(denoiser_inputs.audio_hidden_states.end(),
                                                   condition.audio_hidden_states.begin(),
                                                   condition.audio_hidden_states.end());
    }
    denoiser_inputs.video_hidden_states.insert(denoiser_inputs.video_hidden_states.end(),
                                               target_video_rows.begin(), target_video_rows.end());
    denoiser_inputs.audio_hidden_states.insert(denoiser_inputs.audio_hidden_states.end(),
                                               target_audio_rows.begin(), target_audio_rows.end());
    denoiser_inputs.encoder_hidden_states = std::move(text_embeddings);
    denoiser_inputs.layout = minimax_h3::make_ref2va_packed_layout(
        presentation.h3_token_tags, reference_geometries, geometry.video_latent_frames,
        geometry.latent_height, geometry.latent_width, geometry.audio_latent_frames);
    conditions.clear();
    conditions.shrink_to_fit();
    prepared.references.clear();
    prepared.references.shrink_to_fit();

    const auto video_schedule =
        make_minimax_h3_schedule(ref2va_config_.scheduler_grid_points, ref2va_config_.video_shift);
    const auto audio_schedule =
        make_minimax_h3_schedule(ref2va_config_.scheduler_grid_points, ref2va_config_.audio_shift);
    if (video_schedule.timesteps.size() != 49U || audio_schedule.timesteps.size() != 49U)
        throw std::logic_error("MiniMax-H3 Ref2VA scheduler did not produce 49 forwards");
    const auto adaln_begin = Clock::now();
    auto adaln_module = loader_("ref2va_adaln_precompute_plan", stream_, {}, 0);
    adaln_module->set_timing_label("ref2va_adaln_precompute_plan");
    std::vector<minimax_h3::Ref2vaModulations> modulations;
    modulations.reserve(video_schedule.timesteps.size());
    for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
        const auto row_timesteps = minimax_h3::make_ref2va_row_timesteps(
            denoiser_inputs.layout, video_schedule.timesteps[step], audio_schedule.timesteps[step]);
        const auto table = minimax_h3::pad_ref2va_timesteps(row_timesteps.unique_timesteps);
        modulations.push_back(minimax_h3::run_ref2va_adaln_precompute(*adaln_module, table));
    }
    adaln_module->sync();
    adaln_module.reset();
    const auto adaln_end = Clock::now();

    const auto denoiser_begin = Clock::now();
    auto denoiser = loader_("ref2va_denoiser_plan", stream_, {}, 0);
    denoiser->set_timing_label("ref2va_denoiser_plan");
    minimax_h3::validate_ref2va_plan(*denoiser, minimax_h3::Ref2vaPlanKind::kDenoiser);
    denoiser->reset_execution_context();
    const std::size_t condition_video_values =
        static_cast<std::size_t>(denoiser_inputs.layout.condition_video_rows) * kPatchDim;
    const std::size_t condition_audio_values =
        static_cast<std::size_t>(denoiser_inputs.layout.condition_audio_rows) * kAudioChannels;
    for (std::size_t step = 0; step < video_schedule.timesteps.size(); ++step) {
        auto row_timesteps = minimax_h3::make_ref2va_row_timesteps(
            denoiser_inputs.layout, video_schedule.timesteps[step], audio_schedule.timesteps[step]);
        denoiser_inputs.timestep_indices = std::move(row_timesteps.timestep_indices);
        denoiser_inputs.adaln_indices = std::move(row_timesteps.adaln_indices);
        auto velocity =
            minimax_h3::run_ref2va_denoiser(*denoiser, denoiser_inputs, modulations[step]);
        minimax_h3_scheduler_step(denoiser_inputs.video_hidden_states.data() +
                                      condition_video_values,
                                  velocity.video.data() + condition_video_values,
                                  target_video_rows.size(), video_schedule.timesteps[step],
                                  video_schedule.sigmas[step], video_schedule.sigmas[step + 1]);
        minimax_h3_scheduler_step(denoiser_inputs.audio_hidden_states.data() +
                                      condition_audio_values,
                                  velocity.audio.data() + condition_audio_values,
                                  target_audio_rows.size(), audio_schedule.timesteps[step],
                                  audio_schedule.sigmas[step], audio_schedule.sigmas[step + 1]);
        std::cerr << "[minimax-h3.ref2va] denoiser " << (step + 1) << '/'
                  << video_schedule.timesteps.size() << '\n';
    }
    denoiser->sync();
    denoiser.reset();
    modulations.clear();
    modulations.shrink_to_fit();
    const auto denoiser_end = Clock::now();

    target_video_rows.assign(denoiser_inputs.video_hidden_states.begin() +
                                 static_cast<std::ptrdiff_t>(condition_video_values),
                             denoiser_inputs.video_hidden_states.end());
    target_audio_rows.assign(denoiser_inputs.audio_hidden_states.begin() +
                                 static_cast<std::ptrdiff_t>(condition_audio_values),
                             denoiser_inputs.audio_hidden_states.end());
    denoiser_inputs = {};

    auto latent = unpatchify_video(target_video_rows, geometry);
    denormalize_latents(latent, geometry);
    target_video_rows.clear();
    target_video_rows.shrink_to_fit();
    const std::size_t expected_pixels = static_cast<std::size_t>(3) * geometry.output_frames *
                                        geometry.output_height * geometry.output_width;
    const auto vae_begin = Clock::now();
    resident_->prepare_ref2va_vae(loader_, stream_, geometry);
    auto pixels = resident_->decode_ref2va_vae(latent, expected_pixels, geometry);
    latent.clear();
    latent.shrink_to_fit();
    const auto vae_end = Clock::now();

    const auto audio_vae_begin = Clock::now();
    AudioResult audio;
    if (include_audio)
        audio = resident_->decode_audio(target_audio_rows, geometry, loader_, stream_);
    const auto audio_vae_end = Clock::now();
    target_audio_rows.clear();
    target_audio_rows.shrink_to_fit();

    const auto total_end = Clock::now();
    std::cerr << std::fixed << std::setprecision(3) << "[minimax-h3.perf] workflow=ref2va"
              << " text_vision_encoder_ms=" << milliseconds(text_begin, text_end)
              << " condition_encoder_ms=" << milliseconds(condition_begin, condition_end)
              << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
              << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
              << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
              << " audio_vae_decoder_ms=" << milliseconds(audio_vae_begin, audio_vae_end)
              << " total_ms=" << milliseconds(total_begin, total_end) << " transformer_forwards=49"
              << " output_frames=" << geometry.output_frames
              << " output_height=" << geometry.output_height
              << " output_width=" << geometry.output_width << '\n';
    VideoResult result;
    result.frames.height = geometry.output_height;
    result.frames.width = geometry.output_width;
    result.frames.channels = 3;
    result.frames.num_frames = geometry.output_frames;
    result.frames.pixels = std::move(pixels);
    result.audio = std::move(audio);
    result.fps = 24;
    return result;
}

VideoResult MiniMaxH3Pipeline::generate_video_request_impl(const VideoGenerationRequest& request,
                                                           bool include_audio) {
    std::lock_guard<std::mutex> lock(generation_mutex_);
    StreamScopeSynchronizer synchronize_on_exit(stream_);
    try {
        if (!request.config.initial_latents.empty())
            throw std::invalid_argument(
                "MiniMax-H3 native runtime does not accept initial_latents");
        if (!request.config.negative_prompt.empty())
            throw std::invalid_argument(
                "MiniMax-H3 is guidance-distilled and does not accept negative_prompt");
        if (request.mode == VideoGenerationMode::kReferenceToVideoAudio) {
            if (!ref2va_config_.enabled)
                throw std::runtime_error(
                    "MiniMax-H3 bundle does not contain the authenticated Ref2VA plans");
            return generate_ref2va_request_impl(request, include_audio);
        }
        const bool fl2va = request.mode == VideoGenerationMode::kFirstLastFrameToVideoAudio;
        if (request.mode == VideoGenerationMode::kTextToVideoAudio) {
            if (request.first_frame || request.last_frame || !request.references.empty())
                throw std::invalid_argument("MiniMax-H3 T2VA request cannot carry media");
        } else if (fl2va) {
            if ((!request.first_frame && !request.last_frame) || !request.references.empty())
                throw std::invalid_argument(
                    "MiniMax-H3 FL2VA needs an endpoint keyframe and cannot carry references");
        } else {
            throw std::invalid_argument("MiniMax-H3 video generation mode is invalid");
        }

        GenerateConfig resolved_config = request.config;
        if (fl2va && resolved_config.height == 0 && resolved_config.width == 0) {
            const VideoImageInput& geometry_anchor =
                request.first_frame ? *request.first_frame : *request.last_frame;
            const MiniMaxH3Canvas canvas =
                resolve_minimax_h3_canvas(geometry_anchor.width, geometry_anchor.height);
            resolved_config.height = canvas.height;
            resolved_config.width = canvas.width;
        }
        const GenerateConfig& cfg = resolved_config;
        const MiniMaxH3Geometry target_geometry = resolve_generate_geometry(cfg);
        MiniMaxH3PreparedKeyframes prepared_keyframes;
        MiniMaxH3Geometry geometry = target_geometry;
        if (fl2va) {
            prepared_keyframes = prepare_minimax_h3_keyframes(
                request.first_frame, request.last_frame, target_geometry.output_height,
                target_geometry.output_width, target_geometry.output_frames);
            geometry = make_minimax_h3_fl2va_geometry(
                target_geometry, static_cast<int32_t>(prepared_keyframes.images.size()));
        }
        const int32_t requested_steps = denoiser_config_.scheduler_grid_points;
        if (cfg.num_steps > 0 && cfg.num_steps != requested_steps) {
            throw std::invalid_argument("MiniMax-H3 request num_steps does not match the bundle");
        }
        if (cfg.guidance_scale >= 0.0F && cfg.guidance_scale != denoiser_config_.guidance_scale) {
            throw std::invalid_argument(
                "MiniMax-H3 request guidance_scale does not match the bundle");
        }
        const int64_t seed = cfg.seed >= 0 ? cfg.seed : 0;
        const auto total_begin = Clock::now();

        const bool text_cache_hit =
            !fl2va && resident_->prompt == request.prompt && !resident_->text_embeddings.empty();
        const auto text_begin = Clock::now();
        std::vector<std::vector<float>> keyframe_latents;
        if (fl2va) {
            keyframe_latents = resident_->load_fl2va_conditioning(
                request.prompt, prepared_keyframes, *tokenizer_, loader_, stream_);
        } else if (!text_cache_hit) {
            resident_->load_text_embeddings(request.prompt, *tokenizer_, loader_, stream_,
                                             denoiser_config_.max_text_rows);
        }
        const auto text_end = Clock::now();

        const auto video_schedule =
            make_minimax_h3_schedule(denoiser_config_.scheduler_grid_points, 12.0F);
        const auto audio_schedule =
            make_minimax_h3_schedule(denoiser_config_.scheduler_grid_points, 3.0F);
        const bool adaln_cache_hit = !resident_->modulations.empty();
        const auto adaln_begin = Clock::now();
        if (!adaln_cache_hit)
            resident_->load_modulations(video_schedule, audio_schedule, loader_, stream_);
        const auto adaln_end = Clock::now();

        uint64_t generator_offset = 0;
        std::vector<float> video_rows;
        if (fl2va) {
            video_rows.reserve(static_cast<std::size_t>(geometry.video_rows) * kPatchDim);
            const std::size_t expected_keyframe_latent_count =
                static_cast<std::size_t>(kLatentChannels) * geometry.latent_height *
                geometry.latent_width;
            for (auto& latent : keyframe_latents) {
                if (latent.size() != expected_keyframe_latent_count)
                    throw std::runtime_error(
                        "MiniMax-H3 keyframe encoder returned the wrong latent geometry");
                auto noise = minimax_h3::torch_cuda_normal(
                    latent.size(), static_cast<uint64_t>(seed), generator_offset);
                generator_offset += minimax_h3::torch_cuda_normal_consumed_offset(latent.size());
                for (std::size_t index = 0; index < latent.size(); ++index)
                    latent[index] = 0.999F * latent[index] + 0.001F * noise[index];
                auto rows = minimax_h3::patchify_fl2va_keyframe_latent(
                    latent, geometry.latent_height, geometry.latent_width);
                video_rows.insert(video_rows.end(), rows.begin(), rows.end());
            }
        }

        const std::size_t current_video_latent_count = video_latent_count(geometry);
        auto video_tensor = minimax_h3::torch_cuda_normal(
            current_video_latent_count, static_cast<uint64_t>(seed), generator_offset);
        generator_offset +=
            minimax_h3::torch_cuda_normal_consumed_offset(current_video_latent_count);
        auto audio_rows = minimax_h3::torch_cuda_normal(
            audio_latent_count(geometry), static_cast<uint64_t>(seed), generator_offset);
        auto target_video_rows = patchify_video(video_tensor, geometry);
        video_rows.insert(video_rows.end(), target_video_rows.begin(), target_video_rows.end());
        if (video_rows.size() != static_cast<std::size_t>(geometry.video_rows) * kPatchDim)
            throw std::logic_error("MiniMax-H3 packed FL2VA video row accounting failed");
        video_tensor.clear();
        video_tensor.shrink_to_fit();
        MiniMaxH3DenoiserMetadata metadata =
            fl2va ? make_minimax_h3_fl2va_denoiser_metadata(resident_->text_token_tags,
                                                            prepared_keyframes.anchors, geometry)
                  : make_minimax_h3_denoiser_metadata(resident_->text_rows, geometry);

        const auto denoiser_begin = Clock::now();
        const bool denoiser_resident_hit = resident_->prepare_denoiser(
            loader_, stream_, denoiser_config_.optimization_profile_count, geometry);
        const DenoiserStats denoiser_stats = resident_->run_denoiser(
            metadata, video_schedule, audio_schedule, video_rows, audio_rows, cache_threshold_,
            stream_);
        const auto denoiser_end = Clock::now();
        video_rows.clear();
        video_rows.shrink_to_fit();
        const std::size_t expected_pixels = static_cast<std::size_t>(3) * geometry.output_frames *
                                            geometry.output_height * geometry.output_width;
        const auto vae_begin = Clock::now();
        const bool vae_resident_hit = resident_->prepare_vae(loader_, stream_, geometry);
        auto pixels = resident_->decode_vae(expected_pixels, geometry, stream_);
        const auto vae_end = Clock::now();

        const auto audio_vae_begin = Clock::now();
        AudioResult audio;
        if (include_audio)
            audio = resident_->decode_audio(audio_rows, geometry, loader_, stream_);
        const auto audio_vae_end = Clock::now();
        audio_rows.clear();
        audio_rows.shrink_to_fit();

        const auto total_end = Clock::now();
        std::cerr << std::fixed << std::setprecision(3)
                  << "[minimax-h3.perf] text_encoder_ms=" << milliseconds(text_begin, text_end)
                  << " adaln_ms=" << milliseconds(adaln_begin, adaln_end)
                  << " denoiser_ms=" << milliseconds(denoiser_begin, denoiser_end)
                  << " vae_decoder_ms=" << milliseconds(vae_begin, vae_end)
                  << " audio_vae_decoder_ms=" << milliseconds(audio_vae_begin, audio_vae_end)
                  << " total_ms=" << milliseconds(total_begin, total_end)
                  << " text_cache_hit=" << static_cast<int>(text_cache_hit)
                  << " adaln_cache_hit=" << static_cast<int>(adaln_cache_hit)
                  << " denoiser_resident_hit=" << static_cast<int>(denoiser_resident_hit)
                  << " vae_resident_hit=" << static_cast<int>(vae_resident_hit)
                  << " first_block_cache=1"
                  << " workflow=" << (fl2va ? "fl2va" : "t2va")
                  << " condition_video_rows=" << geometry.condition_video_rows
                  << " output_frames=" << geometry.output_frames
                  << " output_height=" << geometry.output_height
                  << " output_width=" << geometry.output_width
                  << " vae_tile_count=" << geometry.vae_tile_count
                  << " attention_mode=dense"
                  << " transformer_forwards=" << denoiser_config_.transformer_forwards
                  << " cache_threshold=" << cache_threshold_
                  << " full_denoiser_steps=" << denoiser_stats.full_steps
                  << " skipped_denoiser_steps=" << denoiser_stats.skipped_steps << '\n';
        VideoResult result;
        result.frames.height = geometry.output_height;
        result.frames.width = geometry.output_width;
        result.frames.channels = 3;
        result.frames.num_frames = geometry.output_frames;
        result.frames.pixels = std::move(pixels);
        result.audio = std::move(audio);
        result.fps = 24;
        return result;
    } catch (...) {
        const auto error = std::current_exception();
        (void)cudaStreamSynchronize(stream_);
        try {
            resident_->release_denoiser_stage(/*preserve_video_rows=*/false);
            resident_->release_vae_stage();
        } catch (...) {
            // Preserve the request failure after best-effort staged cleanup.
        }
        std::rethrow_exception(error);
    }
}

} // namespace trtmc
