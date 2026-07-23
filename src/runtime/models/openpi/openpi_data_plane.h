/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::openpi {

inline constexpr std::size_t kModelActionDimension = 32;
inline constexpr int32_t kImageHeight = 224;
inline constexpr int32_t kImageWidth = 224;
inline constexpr std::array<std::string_view, 3> kCameraNames = {
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
};

struct QuantileStats {
    std::vector<float> q01;
    std::vector<float> q99;
};

// Matches OpenPI transforms.Normalize(use_quantiles=True). Quantile statistics
// are broadcast over every leading dimension and sliced to last_dimension.
std::vector<float> quantile_normalize(const std::vector<float>& values, std::size_t last_dimension,
                                      const QuantileStats& stats);

// Matches OpenPI transforms.Unnormalize(use_quantiles=True). When the stored
// statistics cover fewer dimensions than the model's 32-D action tensor, the
// uncalibrated tail is preserved verbatim.
std::vector<float> quantile_unnormalize(const std::vector<float>& values,
                                        std::size_t last_dimension, const QuantileStats& stats);

// Matches np.digitize(x, np.linspace(-1, 1, 257)[:-1]) - 1 exactly at bin
// boundaries. Values outside the nominal range are intentionally not clipped:
// OpenPI maps values below -1 to -1 and values at or above the last boundary to
// 255. NaN follows NumPy and maps to 255.
int32_t discretize_state_value(float value);
std::vector<int32_t> discretize_state(const std::vector<float>& state);

// Matches prompt.strip().replace("_", " ").replace("\n", " ") from the
// upstream PaligemmaTokenizer. Invalid UTF-8 is rejected instead of being
// accepted differently by different C++ standard libraries.
std::string clean_prompt(std::string_view prompt);

// Formats the released pi0.5 discrete-state language input:
//   Task: <clean prompt>, State: <space-separated bin ids>;\nAction:
std::string format_pi05_prompt(std::string_view prompt, const std::vector<float>& state);

enum class CameraPixelType : uint8_t {
    kUint8,
    kFloat32,
};

// Non-owning HWC RGB view. Float pixels use OpenPI's normalized [-1, 1]
// convention. Even masked camera slots carry a correctly shaped black image,
// as they do in the upstream DROID adapter.
struct CameraFrameView {
    std::string_view name;
    int32_t height{0};
    int32_t width{0};
    int32_t channels{0};
    bool valid{false};
    CameraPixelType pixel_type{CameraPixelType::kUint8};
    const uint8_t* uint8_data{nullptr};
    const float* float_data{nullptr};
    std::size_t element_count{0};
};

using OrderedCameraSlots = std::array<CameraFrameView, 3>;

// Validates names, geometry, pointers, ranges, duplicates, and masked-camera
// black pixels, then returns the canonical OpenPI camera order.
OrderedCameraSlots validate_and_order_camera_slots(const std::vector<CameraFrameView>& cameras);

// The released pi0.5 DROID checkpoint uses two real views followed by one
// black, masked right-wrist slot.
void validate_pi05_two_camera_masks(const OrderedCameraSlots& cameras);

struct ResizeWithPadGeometry {
    int32_t source_height{0};
    int32_t source_width{0};
    int32_t target_height{0};
    int32_t target_width{0};
    int32_t resized_height{0};
    int32_t resized_width{0};
    int32_t pad_top{0};
    int32_t pad_bottom{0};
    int32_t pad_left{0};
    int32_t pad_right{0};
};

ResizeWithPadGeometry compute_resize_with_pad_geometry(int32_t source_height, int32_t source_width,
                                                       int32_t target_height, int32_t target_width);

// Dependency-free implementation of OpenPI's JAX resize_with_pad path:
// half-pixel centers, LINEAR triangle kernel, antialiasing while downsampling,
// round-to-nearest-even for uint8, and centered black padding.
std::vector<uint8_t> resize_with_pad_uint8(const uint8_t* pixels, int32_t source_height,
                                           int32_t source_width, int32_t channels,
                                           int32_t target_height, int32_t target_width);

// Float input and output use OpenPI's [-1, 1] convention. Padding is -1 and
// resized values are clipped to [-1, 1], matching the upstream JAX helper.
std::vector<float> resize_with_pad_float(const float* pixels, int32_t source_height,
                                         int32_t source_width, int32_t channels,
                                         int32_t target_height, int32_t target_width);

// Matches the exact float32 values produced by the pinned JAX/XLA lowering,
// including its reciprocal approximation rather than ideal real arithmetic.
std::vector<float> uint8_to_openpi_float(const std::vector<uint8_t>& pixels);

struct EulerSchedule {
    float dt{0.0F};
    std::vector<float> timesteps;
};

// Uses iterative float32 time updates and OpenPI's robust `time >= -dt / 2`
// stop condition rather than reconstructing timesteps in higher precision.
EulerSchedule make_euler_schedule(int32_t num_steps);

// Validates and copies caller-provided noise for reproducible parity runs.
std::vector<float> make_fixed_noise(const std::vector<float>& noise, int32_t batch_size,
                                    int32_t action_horizon, int32_t action_dimension);

// Performs x_t <- x_t + dt * v_t in float32 and rejects mismatched buffers.
void euler_step_in_place(std::vector<float>& sample, const std::vector<float>& velocity, float dt);

} // namespace trtmc::openpi
