#include "runtime/models/sana_wm/pipeline.h"

#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"
#include "trtmc/trtmc_io.hpp"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

constexpr float kDefaultPitchLimitDeg = 85.0F;
constexpr float kPi = 3.14159265358979323846F;
constexpr std::array<float, 4> kRefinerSigmas = {0.909375F, 0.725F, 0.421875F, 0.0F};
using half_bits_t = uint16_t;

std::string first_nonempty(std::string a, const std::string& b) {
    if (!a.empty())
        return a;
    return b;
}

std::string trim_ascii_whitespace(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    auto begin = std::find_if_not(value.begin(), value.end(), [&](char ch) {
        return is_space(static_cast<unsigned char>(ch));
    });
    auto end = std::find_if_not(value.rbegin(), value.rend(), [&](char ch) {
                   return is_space(static_cast<unsigned char>(ch));
               }).base();
    if (begin >= end)
        return {};
    return std::string(begin, end);
}

using Mat3 = std::array<std::array<float, 3>, 3>;
using Mat4 = std::array<float, 16>;
using Vec3 = std::array<float, 3>;

Mat3 identity3() {
    return {{{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}}};
}

Mat3 rot_x(float angle_rad) {
    const float c = std::cos(angle_rad);
    const float s = std::sin(angle_rad);
    return {{{1.0F, 0.0F, 0.0F}, {0.0F, c, -s}, {0.0F, s, c}}};
}

Mat3 rot_y(float angle_rad) {
    const float c = std::cos(angle_rad);
    const float s = std::sin(angle_rad);
    return {{{c, 0.0F, s}, {0.0F, 1.0F, 0.0F}, {-s, 0.0F, c}}};
}

Mat3 matmul3(const Mat3& a, const Mat3& b) {
    Mat3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float acc = 0.0F;
            for (int k = 0; k < 3; ++k)
                acc += a[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] *
                       b[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)];
            out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = acc;
        }
    }
    return out;
}

Vec3 column(const Mat3& m, int c) {
    return {m[0][static_cast<std::size_t>(c)], m[1][static_cast<std::size_t>(c)],
            m[2][static_cast<std::size_t>(c)]};
}

float m4_at(const Mat4& m, int row, int col) {
    return m[static_cast<std::size_t>(row * 4 + col)];
}

void m4_set(Mat4& m, int row, int col, float value) {
    m[static_cast<std::size_t>(row * 4 + col)] = value;
}

Mat4 identity4() {
    return {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
}

Mat4 matmul4(const Mat4& a, const Mat4& b) {
    Mat4 out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            float acc = 0.0F;
            for (int k = 0; k < 4; ++k)
                acc += m4_at(a, r, k) * m4_at(b, k, c);
            m4_set(out, r, c, acc);
        }
    }
    return out;
}

Mat4 inverse_rigid_pose(const Mat4& pose) {
    Mat4 out = identity4();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            m4_set(out, r, c, m4_at(pose, c, r));
    }
    for (int r = 0; r < 3; ++r) {
        float translated = 0.0F;
        for (int k = 0; k < 3; ++k)
            translated += m4_at(out, r, k) * m4_at(pose, k, 3);
        m4_set(out, r, 3, -translated);
    }
    return out;
}

Vec3 pose_origin(const Mat4& pose) {
    return {m4_at(pose, 0, 3), m4_at(pose, 1, 3), m4_at(pose, 2, 3)};
}

Vec3 rotate_direction(const Mat4& pose, const Vec3& direction) {
    return {
        m4_at(pose, 0, 0) * direction[0] + m4_at(pose, 0, 1) * direction[1] +
            m4_at(pose, 0, 2) * direction[2],
        m4_at(pose, 1, 0) * direction[0] + m4_at(pose, 1, 1) * direction[1] +
            m4_at(pose, 1, 2) * direction[2],
        m4_at(pose, 2, 0) * direction[0] + m4_at(pose, 2, 1) * direction[1] +
            m4_at(pose, 2, 2) * direction[2],
    };
}

Vec3 normalized(Vec3 v) {
    const float norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (norm <= 0.0F)
        return {0.0F, 0.0F, 0.0F};
    const float inv = 1.0F / norm;
    return {v[0] * inv, v[1] * inv, v[2] * inv};
}

Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

void normalize_horizontal(Vec3& v) {
    v[1] = 0.0F;
    const float norm = std::sqrt(v[0] * v[0] + v[2] * v[2]);
    if (norm > 0.0F) {
        // Mirrors upstream: divide by norm + 1e-6 after the positive-norm test.
        const float inv = 1.0F / (norm + 1.0e-6F);
        v[0] *= inv;
        v[2] *= inv;
    }
}

SanaWmPose make_pose(const Mat3& r, const Vec3& t) {
    SanaWmPose pose;
    pose.c2w = {r[0][0], r[0][1], r[0][2], t[0], r[1][0], r[1][1], r[1][2], t[1],
                r[2][0], r[2][1], r[2][2], t[2], 0.0F,    0.0F,    0.0F,    1.0F};
    return pose;
}

struct ParsedActionSegment {
    std::vector<char> keys;
    int32_t duration{0};
};

std::string remove_action_whitespace(const std::string& action) {
    std::string out;
    out.reserve(action.size());
    for (unsigned char ch : action) {
        if (!std::isspace(ch))
            out.push_back(static_cast<char>(ch));
    }
    return out;
}

bool is_action_key(char key) {
    switch (key) {
    case 'w':
    case 'a':
    case 's':
    case 'd':
    case 'i':
    case 'j':
    case 'k':
    case 'l':
        return true;
    default:
        return false;
    }
}

std::string lowercase_keys(const std::string& keys_part) {
    std::string keys_lower;
    keys_lower.reserve(keys_part.size());
    for (unsigned char ch : keys_part)
        keys_lower.push_back(static_cast<char>(std::tolower(ch)));
    return keys_lower;
}

int32_t parse_action_duration(const std::string& segment, const std::string& duration_text) {
    int32_t duration = 0;
    for (char ch : duration_text) {
        if (!std::isdigit(static_cast<unsigned char>(ch)))
            throw std::invalid_argument("Invalid SANA-WM action duration: " + segment);
        duration = duration * 10 + (ch - '0');
    }
    if (duration <= 0)
        throw std::invalid_argument("SANA-WM action duration must be positive: " + segment);
    return duration;
}

std::vector<char> parse_action_keys(const std::string& segment, const std::string& keys_part) {
    const std::string keys_lower = lowercase_keys(keys_part);
    if (keys_lower == "none")
        return {};

    std::set<char> unique;
    for (char key : keys_lower) {
        if (!is_action_key(key))
            throw std::invalid_argument("Unknown SANA-WM action key in segment: " + segment);
        unique.insert(key);
    }
    return {unique.begin(), unique.end()};
}

ParsedActionSegment parse_action_segment(const std::string& segment) {
    if (segment.empty())
        throw std::invalid_argument("Invalid empty SANA-WM action segment");

    const std::size_t dash = segment.rfind('-');
    if (dash == std::string::npos || dash == 0 || dash + 1 >= segment.size())
        throw std::invalid_argument("Invalid SANA-WM action segment: " + segment);

    const std::string keys_part = segment.substr(0, dash);
    const std::string duration_text = segment.substr(dash + 1);
    return {parse_action_keys(segment, keys_part), parse_action_duration(segment, duration_text)};
}

void append_segment_frames(std::vector<std::vector<char>>& per_frame,
                           const ParsedActionSegment& segment) {
    for (int32_t i = 0; i < segment.duration; ++i)
        per_frame.push_back(segment.keys);
}

std::vector<std::vector<char>> parse_action_string(const std::string& action) {
    const std::string cleaned = remove_action_whitespace(action);
    if (cleaned.empty())
        throw std::invalid_argument("SANA-WM action string is empty");

    std::vector<std::vector<char>> per_frame;
    std::size_t start = 0;
    while (start <= cleaned.size()) {
        const std::size_t end = cleaned.find(',', start);
        append_segment_frames(
            per_frame, parse_action_segment(cleaned.substr(
                           start, end == std::string::npos ? std::string::npos : end - start)));
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return per_frame;
}

bool has_key(const std::vector<char>& keys, char key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

int32_t key_direction(const std::vector<char>& keys, char positive, char negative) {
    const int32_t plus = has_key(keys, positive) ? 1 : 0;
    const int32_t minus = has_key(keys, negative) ? 1 : 0;
    return plus - minus;
}

float limited_pitch_delta(const std::vector<char>& keys, float rotate_rad, float current_pitch,
                          float pitch_limit_rad, float& next_pitch) {
    const float pitch_delta = static_cast<float>(key_direction(keys, 'i', 'k')) * rotate_rad;
    next_pitch = current_pitch;
    const float candidate = current_pitch + pitch_delta;
    if (candidate < -pitch_limit_rad || candidate > pitch_limit_rad)
        return 0.0F;
    next_pitch = candidate;
    return pitch_delta;
}

Vec3 camera_ground_motion(const std::vector<char>& keys, const Mat3& r, float translation_speed) {
    Vec3 forward = column(r, 2);
    Vec3 right = column(r, 0);
    normalize_horizontal(forward);
    normalize_horizontal(right);

    const float forward_step = static_cast<float>(key_direction(keys, 'w', 's'));
    const float right_step = static_cast<float>(key_direction(keys, 'd', 'a'));
    return {
        (forward[0] * forward_step + right[0] * right_step) * translation_speed,
        0.0F,
        (forward[2] * forward_step + right[2] * right_step) * translation_speed,
    };
}

int32_t python_round_to_int(double value) {
    const double floored = std::floor(value);
    const double frac = value - floored;
    if (frac < 0.5)
        return static_cast<int32_t>(floored);
    if (frac > 0.5)
        return static_cast<int32_t>(floored + 1.0);
    const auto floor_int = static_cast<long long>(floored);
    return static_cast<int32_t>((floor_int % 2LL == 0LL) ? floor_int : floor_int + 1LL);
}

void validate_camera_condition_inputs(const std::vector<SanaWmPose>& c2w,
                                      const std::vector<SanaWmIntrinsics>& intrinsics,
                                      int32_t target_height, int32_t target_width,
                                      int32_t vae_time_stride, int32_t vae_spatial_stride) {
    if (c2w.empty())
        throw std::invalid_argument("SANA-WM camera conditioning requires at least one pose");
    if (intrinsics.empty())
        throw std::invalid_argument("SANA-WM camera conditioning requires intrinsics");
    if (intrinsics.size() != 1 && intrinsics.size() != c2w.size())
        throw std::invalid_argument("SANA-WM intrinsics must have one row or match pose count");
    if (target_height <= 0 || target_width <= 0 || vae_time_stride <= 0 ||
        vae_spatial_stride <= 0) {
        throw std::invalid_argument("SANA-WM camera conditioning dimensions must be positive");
    }
}

SanaWmIntrinsics intrinsics_at(const std::vector<SanaWmIntrinsics>& intrinsics, std::size_t idx) {
    return intrinsics.size() == 1 ? intrinsics.front() : intrinsics[idx];
}

SanaWmIntrinsics scale_intrinsics_to_latent(const SanaWmIntrinsics& intrinsics, int32_t latent_h,
                                            int32_t latent_w, int32_t target_height,
                                            int32_t target_width) {
    if (intrinsics.fx <= 0.0F || intrinsics.fy <= 0.0F)
        throw std::invalid_argument("SANA-WM intrinsics fx/fy must be positive");
    return {
        intrinsics.fx * static_cast<float>(latent_w) / static_cast<float>(target_width),
        intrinsics.fy * static_cast<float>(latent_h) / static_cast<float>(target_height),
        intrinsics.cx * static_cast<float>(latent_w) / static_cast<float>(target_width),
        intrinsics.cy * static_cast<float>(latent_h) / static_cast<float>(target_height),
    };
}

std::vector<Mat4> relative_poses_from_first(const std::vector<SanaWmPose>& c2w) {
    std::vector<Mat4> poses;
    poses.reserve(c2w.size());
    const Mat4 first_inv = inverse_rigid_pose(c2w.front().c2w);
    poses.push_back(identity4());
    for (std::size_t i = 1; i < c2w.size(); ++i)
        poses.push_back(matmul4(first_inv, c2w[i].c2w));
    return poses;
}

std::vector<int32_t> camera_time_indices(int32_t num_frames, int32_t latent_frames,
                                         int32_t vae_time_stride) {
    std::vector<int32_t> indices;
    for (int32_t t = 0; t < num_frames; t += vae_time_stride) {
        if (static_cast<int32_t>(indices.size()) >= latent_frames)
            break;
        indices.push_back(t);
    }
    return indices;
}

Vec3 camera_ray_direction(const Mat4& pose, const SanaWmIntrinsics& intrinsics, int32_t y,
                          int32_t x) {
    const Vec3 camera_dir{
        (static_cast<float>(x) - intrinsics.cx) / intrinsics.fx,
        (static_cast<float>(y) - intrinsics.cy) / intrinsics.fy,
        1.0F,
    };
    return normalized(rotate_direction(pose, camera_dir));
}

std::array<float, 6> plucker_for_pixel(const Mat4& pose, const SanaWmIntrinsics& intrinsics,
                                       int32_t y, int32_t x) {
    const Vec3 direction = camera_ray_direction(pose, intrinsics, y, x);
    const Vec3 moment = cross3(pose_origin(pose), direction);
    return {direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]};
}

void pack_raymap_row(std::vector<float>& raymap, std::size_t row, const Mat4& pose,
                     const SanaWmIntrinsics& intrinsics) {
    constexpr std::size_t kWidth = 20;
    const std::size_t offset = row * kWidth;
    std::copy(pose.begin(), pose.end(), raymap.begin() + static_cast<std::ptrdiff_t>(offset));
    raymap[offset + 16] = intrinsics.fx;
    raymap[offset + 17] = intrinsics.fy;
    raymap[offset + 18] = intrinsics.cx;
    raymap[offset + 19] = intrinsics.cy;
}

std::size_t chunk_plucker_index(int32_t channel, int32_t chunk, int32_t y, int32_t x,
                                int32_t chunk_count, int32_t latent_h, int32_t latent_w) {
    return (((static_cast<std::size_t>(channel) * static_cast<std::size_t>(chunk_count) +
              static_cast<std::size_t>(chunk)) *
                 static_cast<std::size_t>(latent_h) +
             static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(latent_w) +
            static_cast<std::size_t>(x));
}

void pack_chunk_plucker(std::vector<float>& chunk_plucker, const std::vector<Mat4>& poses,
                        const std::vector<SanaWmIntrinsics>& intrinsics, int32_t chunk,
                        int32_t time_index, int32_t vae_time_stride, int32_t latent_h,
                        int32_t latent_w, int32_t chunk_count) {
    const int32_t start = std::max(0, time_index - (vae_time_stride - 1));
    const std::size_t max_pose_idx = poses.size() - 1;
    for (int32_t local_t = 0; local_t < vae_time_stride; ++local_t) {
        const auto pose_idx = std::min(static_cast<std::size_t>(start + local_t), max_pose_idx);
        for (int32_t y = 0; y < latent_h; ++y) {
            for (int32_t x = 0; x < latent_w; ++x) {
                const auto plucker = plucker_for_pixel(poses[pose_idx], intrinsics[pose_idx], y, x);
                for (int32_t c = 0; c < 6; ++c) {
                    const int32_t channel = local_t * 6 + c;
                    chunk_plucker[chunk_plucker_index(channel, chunk, y, x, chunk_count, latent_h,
                                                      latent_w)] =
                        plucker[static_cast<std::size_t>(c)];
                }
            }
        }
    }
}

std::size_t raymat_index(int32_t frame, int32_t y, int32_t x, int32_t row, int32_t col,
                         int32_t latent_h, int32_t latent_w) {
    return (((((static_cast<std::size_t>(frame) * static_cast<std::size_t>(latent_h) +
                static_cast<std::size_t>(y)) *
                   static_cast<std::size_t>(latent_w) +
               static_cast<std::size_t>(x)) *
                  4U +
              static_cast<std::size_t>(row)) *
                 4U) +
            static_cast<std::size_t>(col));
}

void pack_ucpe_raymat(std::vector<float>& raymats, std::vector<float>& raymats_inv,
                      int32_t frame, int32_t y, int32_t x, const Mat4& pose,
                      const SanaWmIntrinsics& intrinsics, int32_t latent_h,
                      int32_t latent_w) {
    const Vec3 z_ray = camera_ray_direction(pose, intrinsics, y, x);
    const Vec3 cam_y = {m4_at(pose, 0, 1), m4_at(pose, 1, 1), m4_at(pose, 2, 1)};
    const Vec3 x_ray = normalized(cross3(cam_y, z_ray));
    const Vec3 y_ray = normalized(cross3(z_ray, x_ray));
    const Vec3 origin = pose_origin(pose);
    const std::array<Vec3, 3> rows{x_ray, y_ray, z_ray};
    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            raymats[raymat_index(frame, y, x, row, col, latent_h, latent_w)] =
                rows[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
        }
        const float translated =
            -(rows[static_cast<std::size_t>(row)][0] * origin[0] +
              rows[static_cast<std::size_t>(row)][1] * origin[1] +
              rows[static_cast<std::size_t>(row)][2] * origin[2]);
        raymats[raymat_index(frame, y, x, row, 3, latent_h, latent_w)] = translated;
    }
    raymats[raymat_index(frame, y, x, 3, 0, latent_h, latent_w)] = 0.0F;
    raymats[raymat_index(frame, y, x, 3, 1, latent_h, latent_w)] = 0.0F;
    raymats[raymat_index(frame, y, x, 3, 2, latent_h, latent_w)] = 0.0F;
    raymats[raymat_index(frame, y, x, 3, 3, latent_h, latent_w)] = 1.0F;

    for (int32_t row = 0; row < 3; ++row) {
        for (int32_t col = 0; col < 3; ++col) {
            raymats_inv[raymat_index(frame, y, x, row, col, latent_h, latent_w)] =
                rows[static_cast<std::size_t>(col)][static_cast<std::size_t>(row)];
        }
        raymats_inv[raymat_index(frame, y, x, row, 3, latent_h, latent_w)] =
            origin[static_cast<std::size_t>(row)];
    }
    raymats_inv[raymat_index(frame, y, x, 3, 0, latent_h, latent_w)] = 0.0F;
    raymats_inv[raymat_index(frame, y, x, 3, 1, latent_h, latent_w)] = 0.0F;
    raymats_inv[raymat_index(frame, y, x, 3, 2, latent_h, latent_w)] = 0.0F;
    raymats_inv[raymat_index(frame, y, x, 3, 3, latent_h, latent_w)] = 1.0F;
}

std::size_t stage1_latent_index(int32_t channel, int32_t frame, int32_t y, int32_t x,
                                int32_t frames, int32_t height, int32_t width) {
    return (((static_cast<std::size_t>(channel) * static_cast<std::size_t>(frames) +
              static_cast<std::size_t>(frame)) *
                 static_cast<std::size_t>(height) +
             static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x));
}

void validate_stage1_latent_dims(int32_t channels, int32_t frames, int32_t height, int32_t width) {
    if (channels <= 0 || frames <= 0 || height <= 0 || width <= 0)
        throw std::invalid_argument("SANA-WM Stage-1 latent dimensions must be positive");
}

std::size_t stage1_latent_count(int32_t channels, int32_t frames, int32_t height, int32_t width) {
    validate_stage1_latent_dims(channels, frames, height, width);
    return static_cast<std::size_t>(channels) * static_cast<std::size_t>(frames) *
           static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
}

uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t& high) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    high = static_cast<uint32_t>(product >> 32U);
    return static_cast<uint32_t>(product);
}

std::array<uint32_t, 4> philox_single_round(std::array<uint32_t, 4> counter,
                                            std::array<uint32_t, 2> key) {
    constexpr uint32_t kPhiloxSA = 0xD2511F53U;
    constexpr uint32_t kPhiloxSB = 0xCD9E8D57U;
    uint32_t high0 = 0U;
    uint32_t high1 = 0U;
    const uint32_t low0 = mulhilo32(kPhiloxSA, counter[0], high0);
    const uint32_t low1 = mulhilo32(kPhiloxSB, counter[2], high1);
    return {high1 ^ counter[1] ^ key[0], low1, high0 ^ counter[3] ^ key[1], low0};
}

std::array<uint32_t, 4> philox4x32_10(uint64_t seed, uint64_t subsequence, uint64_t offset) {
    constexpr uint32_t kPhilox10A = 0x9E3779B9U;
    constexpr uint32_t kPhilox10B = 0xBB67AE85U;
    std::array<uint32_t, 4> counter{
        static_cast<uint32_t>(offset),
        static_cast<uint32_t>(offset >> 32U),
        static_cast<uint32_t>(subsequence),
        static_cast<uint32_t>(subsequence >> 32U),
    };
    std::array<uint32_t, 2> key{
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32U),
    };
    for (int round = 0; round < 9; ++round) {
        counter = philox_single_round(counter, key);
        key[0] += kPhilox10A;
        key[1] += kPhilox10B;
    }
    return philox_single_round(counter, key);
}

float round_stage1_noise_to_bfloat16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    bits &= 0xFFFF0000U;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::array<float, 4> torch_cuda_bfloat16_normal4(uint64_t seed, uint64_t subsequence,
                                                 uint64_t offset) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    constexpr double kUint32ToUnit = 2.3283064365386962890625e-10;
    const auto random = philox4x32_10(seed, subsequence, offset);
    const double u0 = (static_cast<double>(random[0]) + 1.0) * kUint32ToUnit;
    const double u1 = (static_cast<double>(random[1]) + 1.0) * kUint32ToUnit;
    const double u2 = (static_cast<double>(random[2]) + 1.0) * kUint32ToUnit;
    const double u3 = (static_cast<double>(random[3]) + 1.0) * kUint32ToUnit;
    const double radius0 = std::sqrt(-2.0 * std::log(u0));
    const double theta0 = kTwoPi * u1;
    const double radius1 = std::sqrt(-2.0 * std::log(u2));
    const double theta1 = kTwoPi * u3;
    return {
        round_stage1_noise_to_bfloat16(static_cast<float>(radius0 * std::sin(theta0))),
        round_stage1_noise_to_bfloat16(static_cast<float>(radius0 * std::cos(theta0))),
        round_stage1_noise_to_bfloat16(static_cast<float>(radius1 * std::sin(theta1))),
        round_stage1_noise_to_bfloat16(static_cast<float>(radius1 * std::cos(theta1))),
    };
}

uint64_t torch_cuda_distribution_thread_count(std::size_t count) {
    constexpr uint32_t kBlockSize = 256U;
    constexpr uint32_t kFallbackGridBlocks = 1216U;
    if (count == 0)
        return 0;

    const auto grid_for_count =
        static_cast<uint32_t>((count + static_cast<std::size_t>(kBlockSize) - 1U) /
                              static_cast<std::size_t>(kBlockSize));
    uint32_t grid_limit = kFallbackGridBlocks;
    int device = 0;
    cudaDeviceProp props{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&props, device) == cudaSuccess &&
        props.maxThreadsPerMultiProcessor >= static_cast<int>(kBlockSize) &&
        props.multiProcessorCount > 0) {
        const auto blocks_per_sm =
            static_cast<uint32_t>(props.maxThreadsPerMultiProcessor) / kBlockSize;
        grid_limit = static_cast<uint32_t>(props.multiProcessorCount) * blocks_per_sm;
    }
    const uint32_t grid = std::max(1U, std::min(grid_limit, grid_for_count));
    return static_cast<uint64_t>(kBlockSize) * static_cast<uint64_t>(grid);
}

float torch_cuda_bfloat16_normal(uint64_t seed, uint64_t element_index, uint64_t thread_count) {
    constexpr uint64_t kUnrollFactor = 4U;
    const uint64_t values_per_grid_stride = thread_count * kUnrollFactor;
    const uint64_t offset = element_index / values_per_grid_stride;
    const uint64_t within = element_index % values_per_grid_stride;
    const auto lane = static_cast<std::size_t>(within / thread_count);
    const uint64_t subsequence = within % thread_count;
    const auto normal4 = torch_cuda_bfloat16_normal4(seed, subsequence, offset);
    return normal4[lane];
}

std::vector<float> sample_stage1_noise(std::size_t count, uint64_t seed) {
    std::vector<float> values(count);
    const uint64_t thread_count = torch_cuda_distribution_thread_count(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = torch_cuda_bfloat16_normal(seed, static_cast<uint64_t>(i), thread_count);
    return values;
}

void overwrite_first_latent_frame(std::vector<float>& latents,
                                  const std::vector<float>& first_frame, int32_t channels,
                                  int32_t frames, int32_t height, int32_t width) {
    const auto expected_first = static_cast<std::size_t>(channels) *
                                static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    if (first_frame.size() != expected_first) {
        throw std::invalid_argument("SANA-WM first-frame latent size does not match [C,H,W]");
    }

    for (int32_t c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const auto src = (static_cast<std::size_t>(c) * static_cast<std::size_t>(height) +
                                  static_cast<std::size_t>(y)) *
                                     static_cast<std::size_t>(width) +
                                 static_cast<std::size_t>(x);
                latents[stage1_latent_index(c, 0, y, x, frames, height, width)] = first_frame[src];
            }
        }
    }
}

std::size_t chw_index(int32_t channel, int32_t y, int32_t x, int32_t height, int32_t width) {
    return (static_cast<std::size_t>(channel) * static_cast<std::size_t>(height) +
            static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x);
}

float lanczos3(float x) {
    const float ax = std::fabs(x);
    if (ax < 1.0e-7F)
        return 1.0F;
    if (ax >= 3.0F)
        return 0.0F;
    const float pix = kPi * ax;
    return (std::sin(pix) / pix) * (std::sin(pix / 3.0F) / (pix / 3.0F));
}

struct ResizeContribution {
    std::vector<std::pair<int32_t, float>> weights;
};

std::vector<ResizeContribution> make_pillow_lanczos_contributions(int32_t src_size,
                                                                  int32_t dst_size) {
    if (src_size <= 0 || dst_size <= 0)
        throw std::invalid_argument("SANA-WM resize contribution dimensions must be positive");

    const double scale = static_cast<double>(src_size) / static_cast<double>(dst_size);
    const double filter_scale = std::max(scale, 1.0);
    const double support = 3.0 * filter_scale;
    const double inv_filter_scale = 1.0 / filter_scale;
    std::vector<ResizeContribution> out(static_cast<std::size_t>(dst_size));
    for (int32_t dst = 0; dst < dst_size; ++dst) {
        const double center = (static_cast<double>(dst) + 0.5) * scale;
        int32_t xmin = static_cast<int32_t>(center - support + 0.5);
        xmin = std::max(xmin, 0);
        int32_t xmax = static_cast<int32_t>(center + support + 0.5);
        xmax = std::min(xmax, src_size);
        const int32_t count = std::max(0, xmax - xmin);
        double sum = 0.0;
        auto& weights = out[static_cast<std::size_t>(dst)].weights;
        weights.reserve(static_cast<std::size_t>(count));
        for (int32_t x = 0; x < count; ++x) {
            const float weight = lanczos3(
                static_cast<float>((static_cast<double>(x + xmin) - center + 0.5) *
                                   inv_filter_scale));
            weights.emplace_back(xmin + x, weight);
            sum += static_cast<double>(weight);
        }
        if (std::fabs(sum) <= 1.0e-12) {
            const int32_t nearest =
                std::max(0, std::min(src_size - 1, static_cast<int32_t>(std::round(center - 0.5))));
            weights = {{nearest, 1.0F}};
            continue;
        }
        for (auto& [_, weight] : weights)
            weight = static_cast<float>(static_cast<double>(weight) / sum);
    }
    return out;
}

uint8_t float_to_uint8_pixel(float value) {
    const float clamped = std::max(0.0F, std::min(1.0F, value));
    const long rounded = std::lround(static_cast<double>(clamped) * 255.0);
    return static_cast<uint8_t>(std::max<long>(0, std::min<long>(255, rounded)));
}

uint8_t round_to_uint8_pixel(double value) {
    const long rounded = std::lround(value);
    return static_cast<uint8_t>(std::max<long>(0, std::min<long>(255, rounded)));
}

bool resize_lanczos3_hwc(const std::vector<float>& src_hwc, int32_t src_width,
                         int32_t src_height, int32_t dst_width, int32_t dst_height,
                         std::vector<float>& dst_hwc) {
    const auto expected =
        static_cast<std::size_t>(src_width) * static_cast<std::size_t>(src_height) * 3U;
    if (src_hwc.size() != expected)
        return false;
    if (src_width == dst_width && src_height == dst_height) {
        dst_hwc = src_hwc;
        return true;
    }

    const auto x_weights = make_pillow_lanczos_contributions(src_width, dst_width);
    const auto y_weights = make_pillow_lanczos_contributions(src_height, dst_height);
    std::vector<uint8_t> src_u8(src_hwc.size());
    for (std::size_t i = 0; i < src_hwc.size(); ++i)
        src_u8[i] = float_to_uint8_pixel(src_hwc[i]);

    std::vector<uint8_t> horizontal(static_cast<std::size_t>(src_height) *
                                    static_cast<std::size_t>(dst_width) * 3U);
    for (int32_t y = 0; y < src_height; ++y) {
        for (int32_t x = 0; x < dst_width; ++x) {
            const auto& contrib = x_weights[static_cast<std::size_t>(x)].weights;
            for (int32_t c = 0; c < 3; ++c) {
                double acc = 0.0;
                for (const auto& [src_x, weight] : contrib) {
                    const auto src_idx =
                        (static_cast<std::size_t>(y) * static_cast<std::size_t>(src_width) +
                         static_cast<std::size_t>(src_x)) *
                            3U +
                        static_cast<std::size_t>(c);
                    acc += static_cast<double>(src_u8[src_idx]) * static_cast<double>(weight);
                }
                const auto dst_idx =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) +
                     static_cast<std::size_t>(x)) *
                        3U +
                    static_cast<std::size_t>(c);
                horizontal[dst_idx] = round_to_uint8_pixel(acc);
            }
        }
    }

    dst_hwc.assign(static_cast<std::size_t>(dst_width) * static_cast<std::size_t>(dst_height) * 3U,
                   0.0F);
    for (int32_t y = 0; y < dst_height; ++y) {
        const auto& contrib = y_weights[static_cast<std::size_t>(y)].weights;
        for (int32_t x = 0; x < dst_width; ++x) {
            for (int32_t c = 0; c < 3; ++c) {
                double acc = 0.0;
                for (const auto& [src_y, weight] : contrib) {
                    const auto src_idx =
                        (static_cast<std::size_t>(src_y) * static_cast<std::size_t>(dst_width) +
                         static_cast<std::size_t>(x)) *
                            3U +
                        static_cast<std::size_t>(c);
                    acc += static_cast<double>(horizontal[src_idx]) * static_cast<double>(weight);
                }
                const auto dst_idx =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_width) +
                     static_cast<std::size_t>(x)) *
                        3U +
                    static_cast<std::size_t>(c);
                dst_hwc[dst_idx] = static_cast<float>(round_to_uint8_pixel(acc)) / 255.0F;
            }
        }
    }
    return true;
}

half_bits_t fp32_to_fp16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFFU;
    if (exp <= 0)
        return static_cast<half_bits_t>(sign);
    if (exp >= 31)
        return static_cast<half_bits_t>(sign | 0x7C00U);
    return static_cast<half_bits_t>(sign | (static_cast<uint32_t>(exp) << 10U) | (mant >> 13U));
}

float fp16_to_fp32(half_bits_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16U;
    const uint32_t exp = (h >> 10U) & 0x1FU;
    const uint32_t mant = h & 0x3FFU;
    uint32_t bits = sign;
    if (exp == 31U) {
        bits |= 0x7F800000U | (mant << 13U);
    } else if (exp != 0U) {
        const auto fp32_exp = static_cast<uint32_t>(static_cast<int32_t>(exp) - 15 + 127);
        bits |= fp32_exp << 23U;
        bits |= mant << 13U;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

half_bits_t fp32_to_bf16(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<half_bits_t>(bits >> 16U);
}

float bf16_to_fp32(half_bits_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16U;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<half_bits_t> convert_float_to_16(const std::vector<float>& src, DType dtype) {
    std::vector<half_bits_t> dst(src.size());
    for (std::size_t i = 0; i < src.size(); ++i)
        dst[i] = dtype == DType::kBFloat16 ? fp32_to_bf16(src[i]) : fp32_to_fp16(src[i]);
    return dst;
}

Tensor make_model_tensor(const std::vector<float>& values, std::vector<half_bits_t>& scratch16,
                         DType dtype, std::vector<int64_t> shape) {
    if (dtype == DType::kFloat32)
        return Tensor{const_cast<float*>(values.data()), std::move(shape), DType::kFloat32};
    if (dtype != DType::kFloat16 && dtype != DType::kBFloat16) {
        throw std::runtime_error("SANA-WM float tensor input has unsupported dtype");
    }
    scratch16 = convert_float_to_16(values, dtype);
    return Tensor{scratch16.data(), std::move(shape), dtype};
}

void log_float_stats(const char* label, const std::vector<float>& values);

std::vector<float> tensor_to_float_vector(const Tensor& tensor, std::size_t count,
                                          const std::string& label) {
    if (tensor.data == nullptr)
        throw std::runtime_error("SANA-WM " + label + " output tensor is null");
    if (tensor.numel() < count) {
        throw std::runtime_error("SANA-WM " + label + " output tensor has " +
                                 std::to_string(tensor.numel()) + " values, expected at least " +
                                 std::to_string(count));
    }

    std::vector<float> out(count, 0.0F);
    if (tensor.dtype == DType::kFloat32) {
        const auto* src = static_cast<const float*>(tensor.data);
        std::copy_n(src, count, out.data());
        log_float_stats(label.c_str(), out);
        return out;
    }
    if (tensor.dtype == DType::kFloat16) {
        const auto* src = static_cast<const half_bits_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = fp16_to_fp32(src[i]);
        log_float_stats(label.c_str(), out);
        return out;
    }
    if (tensor.dtype == DType::kBFloat16) {
        const auto* src = static_cast<const half_bits_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = bf16_to_fp32(src[i]);
        log_float_stats(label.c_str(), out);
        return out;
    }
    if (tensor.dtype == DType::kInt32) {
        const auto* src = static_cast<const int32_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = static_cast<float>(src[i]);
        log_float_stats(label.c_str(), out);
        return out;
    }
    if (tensor.dtype == DType::kInt8) {
        const auto* src = static_cast<const int8_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = static_cast<float>(src[i]);
        log_float_stats(label.c_str(), out);
        return out;
    }
    throw std::runtime_error("SANA-WM " + label + " output tensor has unsupported dtype");
}

bool debug_stats_enabled() {
    const char* value = std::getenv("TRTMC_SANA_WM_DEBUG_STATS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

std::string debug_dump_dir() {
    const char* value = std::getenv("TRTMC_SANA_WM_DEBUG_DUMP_DIR");
    if (value == nullptr || value[0] == '\0' || std::string(value) == "0")
        return {};
    return value;
}

bool debug_step_dumps_enabled() {
    const char* value = std::getenv("TRTMC_SANA_WM_DEBUG_STEP_DUMPS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

bool debug_vae_tensor_dumps_disabled() {
    const char* value = std::getenv("TRTMC_SANA_WM_DEBUG_NO_VAE_DUMPS");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

std::string stage1_step_debug_label(const char* base, int32_t step_index) {
    if (step_index < 0 || !debug_step_dumps_enabled())
        return std::string(base);
    return std::string(base) + " step" + std::to_string(step_index);
}

std::string debug_dump_filename(const char* label) {
    std::string name(label == nullptr ? "tensor" : label);
    for (char& ch : name) {
        const bool keep = std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-';
        if (!keep)
            ch = '_';
    }
    return name + ".f32";
}

void maybe_dump_float_tensor(const char* label, const std::vector<float>& values) {
    const std::string dir = debug_dump_dir();
    if (dir.empty() || values.empty())
        return;
    std::ofstream out(dir + "/" + debug_dump_filename(label), std::ios::binary);
    if (!out)
        return;
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::string debug_input_dir() {
    const char* value = std::getenv("TRTMC_SANA_WM_DEBUG_INPUT_DIR");
    if (value == nullptr || value[0] == '\0' || std::string(value) == "0")
        return {};
    return value;
}

void maybe_load_float_tensor_override(const char* label, std::vector<float>& values) {
    const std::string dir = debug_input_dir();
    if (dir.empty() || values.empty())
        return;
    std::ifstream in(dir + "/" + debug_dump_filename(label), std::ios::binary);
    if (!in)
        return;
    std::vector<float> loaded(values.size());
    in.read(reinterpret_cast<char*>(loaded.data()),
            static_cast<std::streamsize>(loaded.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(loaded.size() * sizeof(float))) {
        std::cerr << "[sana_wm.debug] Ignoring tensor override with wrong size for " << label
                  << std::endl;
        return;
    }
    values = std::move(loaded);
    std::cerr << "[sana_wm.debug] Loaded tensor override for " << label << std::endl;
}

void log_float_stats(const char* label, const std::vector<float>& values) {
    if (!debug_stats_enabled() || values.empty())
        return;
    float min_value = values.front();
    float max_value = values.front();
    long double sum = 0.0L;
    std::size_t finite_count = 0;
    std::size_t negative_count = 0;
    std::size_t above_one_count = 0;
    std::size_t nonfinite_count = 0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            ++nonfinite_count;
            continue;
        }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        sum += static_cast<long double>(value);
        ++finite_count;
        if (value < 0.0F)
            ++negative_count;
        if (value > 1.0F)
            ++above_one_count;
    }
    const long double mean = finite_count ? sum / static_cast<long double>(finite_count) : 0.0L;
    std::cerr << "[sana_wm.stats] " << label << " count=" << values.size() << " finite="
              << finite_count << " nonfinite=" << nonfinite_count << " min=" << min_value
              << " max=" << max_value << " mean=" << static_cast<double>(mean)
              << " lt0=" << negative_count << " gt1=" << above_one_count << std::endl;
    const std::string label_string(label);
    const bool is_vae_tensor = label_string.rfind("VAE encoder", 0) == 0 ||
                               label_string.rfind("VAE decoder", 0) == 0 ||
                               label_string.rfind("refiner VAE decoder", 0) == 0;
    if (label_string.rfind("Stage-1 denoiser", 0) == 0 ||
        label_string.rfind("Stage-1 final latent", 0) == 0 ||
        (is_vae_tensor && !debug_vae_tensor_dumps_disabled())) {
        maybe_dump_float_tensor(label, values);
    }
}

void log_float_stats_slice(const char* label, const std::vector<float>& values,
                           std::size_t offset, std::size_t count) {
    if (!debug_stats_enabled())
        return;
    if (offset > values.size() || count > values.size() - offset)
        throw std::runtime_error("SANA-WM debug stats slice is out of bounds");
    std::vector<float> slice(values.begin() + static_cast<std::ptrdiff_t>(offset),
                             values.begin() + static_cast<std::ptrdiff_t>(offset + count));
    log_float_stats(label, slice);
}

void log_int_token_summary(const char* label, const std::vector<int32_t>& values) {
    if (!debug_stats_enabled())
        return;
    std::size_t nonzero = 0;
    for (int32_t value : values) {
        if (value != 0)
            ++nonzero;
    }
    std::cerr << "[sana_wm.tokens] " << label << " count=" << values.size()
              << " nonzero=" << nonzero << " head=[";
    const std::size_t head_count = std::min<std::size_t>(values.size(), 20);
    for (std::size_t i = 0; i < head_count; ++i) {
        if (i != 0)
            std::cerr << ',';
        std::cerr << values[i];
    }
    std::cerr << "] tail=[";
    const std::size_t tail_count = std::min<std::size_t>(values.size(), 20);
    const std::size_t tail_start = values.size() - tail_count;
    for (std::size_t i = tail_start; i < values.size(); ++i) {
        if (i != tail_start)
            std::cerr << ',';
        std::cerr << values[i];
    }
    std::cerr << "]" << std::endl;
}

DType input_dtype_or(const ITrtModule& module, const std::string& name, DType fallback) {
    for (const auto& info : module.input_info()) {
        if (info.name == name)
            return info.dtype;
    }
    return fallback;
}

std::string pick_input_name(const ITrtModule& module, std::initializer_list<const char*> names,
                            const std::string& label) {
    for (const char* name : names) {
        if (module.has_input(name))
            return name;
    }
    const auto inputs = module.input_info();
    if (inputs.size() == 1U)
        return inputs.front().name;
    throw std::runtime_error("SANA-WM " + label + " input tensor not found");
}

std::string pick_optional_input_name(const ITrtModule& module,
                                     std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (module.has_input(name))
            return name;
    }
    return {};
}

TensorMap::const_iterator find_output_tensor(const TensorMap& outputs,
                                             std::initializer_list<const char*> names) {
    for (const char* name : names) {
        auto it = outputs.find(name);
        if (it != outputs.end())
            return it;
    }
    if (outputs.size() == 1U)
        return outputs.begin();
    return outputs.end();
}

bool has_positive_shape(const std::vector<int64_t>& shape) {
    if (shape.empty())
        return false;
    for (int64_t dim : shape) {
        if (dim <= 0)
            return false;
    }
    return true;
}

std::size_t checked_shape_numel(const std::vector<int64_t>& shape, const std::string& label) {
    if (!has_positive_shape(shape))
        throw std::runtime_error("SANA-WM native " + label + " tensor shape is unresolved");
    std::size_t count = 1;
    for (int64_t dim : shape) {
        if (static_cast<std::uint64_t>(dim) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / count)) {
            throw std::runtime_error("SANA-WM native " + label + " tensor shape is too large");
        }
        count *= static_cast<std::size_t>(dim);
    }
    return count;
}

std::vector<int64_t> input_shape_or_profile_max(const ITrtModule& module, const std::string& name,
                                                const std::string& label) {
    auto shape = module.tensor_shape(name);
    if (has_positive_shape(shape))
        return shape;
    for (const auto& info : module.input_info()) {
        if (info.name == name && has_positive_shape(info.shape))
            return info.shape;
    }
    const int32_t profile_count = module.optimization_profile_count();
    for (int32_t profile_idx = 0; profile_idx < profile_count; ++profile_idx) {
        shape = module.input_profile_shape(name, profile_idx, ProfileShapeSelector::kMax);
        if (has_positive_shape(shape))
            return shape;
    }
    throw std::runtime_error("SANA-WM native " + label + " input tensor shape not found");
}

int32_t trailing_nonnegative_suffix(const std::string& name, const std::string& prefix) {
    if (name.rfind(prefix, 0) != 0)
        return -1;
    if (name.size() == prefix.size())
        return -1;
    int32_t value = 0;
    for (std::size_t i = prefix.size(); i < name.size(); ++i) {
        const char ch = name[i];
        if (ch < '0' || ch > '9')
            return -1;
        value = value * 10 + (ch - '0');
    }
    return value;
}

std::vector<std::string> sorted_layer_input_names(const ITrtModule& module,
                                                  const std::string& prefix) {
    std::vector<std::pair<int32_t, std::string>> indexed;
    for (const auto& info : module.input_info()) {
        const int32_t idx = trailing_nonnegative_suffix(info.name, prefix);
        if (idx >= 0)
            indexed.emplace_back(idx, info.name);
    }
    std::sort(indexed.begin(), indexed.end());

    std::vector<std::string> names;
    names.reserve(indexed.size());
    for (const auto& [_, name] : indexed)
        names.push_back(name);
    return names;
}

struct DecoderCacheTensor {
    std::string input_name;
    std::string output_name;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};
    std::vector<float> values32;
    std::vector<half_bits_t> values16;

    void allocate(const ITrtModule& module, const std::string& label) {
        shape = input_shape_or_profile_max(module, input_name, label);
        const auto count = checked_shape_numel(shape, label);
        dtype = input_dtype_or(module, input_name, DType::kFloat32);
        if (dtype == DType::kFloat32) {
            values32.assign(count, 0.0F);
            values16.clear();
            return;
        }
        if (dtype == DType::kFloat16 || dtype == DType::kBFloat16) {
            values16.assign(count, 0);
            values32.clear();
            return;
        }
        throw std::runtime_error("SANA-WM native " + label + " cache dtype is unsupported");
    }

    void* data() {
        return dtype == DType::kFloat32 ? static_cast<void*>(values32.data())
                                        : static_cast<void*>(values16.data());
    }

    std::size_t numel() const {
        return dtype == DType::kFloat32 ? values32.size() : values16.size();
    }

    std::size_t row_count() const {
        if (shape.empty())
            return 0;
        return static_cast<std::size_t>(shape.front());
    }

    std::size_t row_width() const {
        const std::size_t rows = row_count();
        return rows == 0 ? 0 : numel() / rows;
    }
};

struct DecoderCacheLayer {
    DecoderCacheTensor k;
    DecoderCacheTensor v;
};

std::string present_name_from_cache_name(const std::string& cache_name, const char* cache_prefix,
                                         const char* present_prefix) {
    const int32_t idx = trailing_nonnegative_suffix(cache_name, cache_prefix);
    if (idx < 0)
        return {};
    return std::string(present_prefix) + std::to_string(idx);
}

std::vector<DecoderCacheLayer> collect_decoder_cache_layers(const ITrtModule& module) {
    const auto k_names = sorted_layer_input_names(module, "cache_k_");
    const auto v_names = sorted_layer_input_names(module, "cache_v_");
    if (k_names.size() != v_names.size()) {
        throw std::runtime_error("SANA-WM native decoder text encoder has mismatched KV cache "
                                 "input tensors");
    }

    std::vector<DecoderCacheLayer> layers;
    layers.reserve(k_names.size());
    for (std::size_t i = 0; i < k_names.size(); ++i) {
        DecoderCacheLayer layer;
        layer.k.input_name = k_names[i];
        layer.v.input_name = v_names[i];
        layer.k.output_name =
            present_name_from_cache_name(layer.k.input_name, "cache_k_", "present_k_");
        layer.v.output_name =
            present_name_from_cache_name(layer.v.input_name, "cache_v_", "present_v_");
        layer.k.allocate(module, "text cache K");
        layer.v.allocate(module, "text cache V");
        layers.push_back(std::move(layer));
    }
    return layers;
}

Tensor make_cache_tensor(DecoderCacheTensor& cache) {
    return Tensor{cache.data(), cache.shape, cache.dtype};
}

std::vector<float> decoder_attention_mask(int32_t token_index, std::size_t width) {
    std::vector<float> mask(width, -10000.0F);
    if (mask.empty())
        return mask;
    const std::size_t visible_cache = std::min<std::size_t>(
        static_cast<std::size_t>(std::max(token_index, 0)), width > 0 ? width - 1U : 0U);
    for (std::size_t i = 0; i < visible_cache; ++i)
        mask[i] = 0.0F;
    mask.back() = 0.0F;
    return mask;
}

std::vector<float> decoder_attention_mask(int32_t token_index, std::size_t width,
                                          const std::vector<int32_t>& key_mask) {
    std::vector<float> mask(width, -10000.0F);
    if (mask.empty())
        return mask;
    if (key_mask.empty())
        return decoder_attention_mask(token_index, width);

    const std::size_t current_index = static_cast<std::size_t>(std::max(token_index, 0));
    const std::size_t cache_width = width - 1U;
    const std::size_t visible_cache = std::min<std::size_t>(current_index, cache_width);
    for (std::size_t i = 0; i < visible_cache; ++i) {
        if (i < key_mask.size() && key_mask[i] != 0)
            mask[i] = 0.0F;
    }
    if (current_index < key_mask.size() && key_mask[current_index] != 0)
        mask.back() = 0.0F;
    return mask;
}

void write_cache_values(DecoderCacheTensor& cache, const std::vector<float>& values,
                        std::size_t offset) {
    if (offset + values.size() > cache.numel()) {
        throw std::runtime_error("SANA-WM native text cache update exceeds cache tensor size");
    }
    if (cache.dtype == DType::kFloat32) {
        std::copy(values.begin(), values.end(), cache.values32.begin() + offset);
        return;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        cache.values16[offset + i] =
            cache.dtype == DType::kBFloat16 ? fp32_to_bf16(values[i]) : fp32_to_fp16(values[i]);
    }
}

void update_decoder_cache_tensor(DecoderCacheTensor& cache, const TensorMap& outputs,
                                 int32_t token_index, const std::string& label) {
    if (cache.output_name.empty())
        return;
    const auto it = outputs.find(cache.output_name);
    if (it == outputs.end())
        return;

    const auto cache_count = cache.numel();
    const auto row_width = cache.row_width();
    if (row_width == 0)
        throw std::runtime_error("SANA-WM native " + label + " cache row width is zero");
    const auto rows = cache.row_count();
    const auto present_count = it->second.numel();
    if (present_count >= cache_count) {
        auto values = tensor_to_float_vector(it->second, cache_count, label + " cache");
        write_cache_values(cache, values, 0);
        return;
    }
    if (present_count < row_width) {
        throw std::runtime_error("SANA-WM native " + label +
                                 " present cache tensor is smaller than one cache row");
    }
    const auto row =
        std::min<std::size_t>(static_cast<std::size_t>(std::max(token_index, 0)), rows - 1U);
    auto values = tensor_to_float_vector(it->second, row_width, label + " cache");
    write_cache_values(cache, values, row * row_width);
}

void update_decoder_caches(std::vector<DecoderCacheLayer>& layers, const TensorMap& outputs,
                           int32_t token_index) {
    for (auto& layer : layers) {
        update_decoder_cache_tensor(layer.k, outputs, token_index, "text K");
        update_decoder_cache_tensor(layer.v, outputs, token_index, "text V");
    }
}

struct SanaWmRequest {
    std::string image_path;
    std::string action;
    float translation_speed;
    float rotation_speed_deg;
    int32_t num_frames;
};

SanaWmRequest resolve_request(const SanaWmRuntimeConfig& config, const GenerateConfig& cfg) {
    return {
        first_nonempty(cfg.image_path, config.default_image),
        first_nonempty(cfg.camera_action, config.action),
        cfg.translation_speed > 0.0F ? cfg.translation_speed : config.translation_speed,
        cfg.rotation_speed_deg > 0.0F ? cfg.rotation_speed_deg : config.rotation_speed_deg,
        cfg.num_frames > 0 ? cfg.num_frames : config.num_frames,
    };
}

struct SanaWmNativeInputs {
    SanaWmVaeInputImage first_frame;
    SanaWmCameraConditions camera;
};

struct SanaWmTextConditioning {
    std::vector<float> cond;
    std::vector<float> neg;
    std::vector<int32_t> cond_mask;
    std::vector<int32_t> neg_mask;
};

struct SanaWmTextEncoding {
    std::vector<float> values;
    std::vector<int64_t> shape;
};

struct SanaWmRefinerText {
    std::vector<float> values;
    std::vector<int64_t> shape;
    std::vector<float> attention_mask;
    std::vector<int64_t> attention_mask_shape;
};

std::vector<float> repeat_batch(const std::vector<float>& values, int32_t batch) {
    if (batch <= 1)
        return values;
    std::vector<float> out(values.size() * static_cast<std::size_t>(batch));
    for (int32_t b = 0; b < batch; ++b) {
        std::copy(values.begin(), values.end(),
                  out.begin() + static_cast<std::size_t>(b) * values.size());
    }
    return out;
}

template <typename T>
std::vector<T> concat_two(const std::vector<T>& first, const std::vector<T>& second,
                          const std::string& label) {
    if (first.size() != second.size()) {
        throw std::runtime_error("SANA-WM " + label + " CFG tensors have mismatched sizes");
    }
    std::vector<T> out;
    out.reserve(first.size() + second.size());
    out.insert(out.end(), first.begin(), first.end());
    out.insert(out.end(), second.begin(), second.end());
    return out;
}

std::vector<float> stage1_text_input(const SanaWmTextConditioning& text, bool do_cfg) {
    return do_cfg ? concat_two(text.neg, text.cond, "text conditioning") : text.cond;
}

std::vector<int32_t> stage1_mask_input(const SanaWmTextConditioning& text, bool do_cfg) {
    return do_cfg ? concat_two(text.neg_mask, text.cond_mask, "text mask") : text.cond_mask;
}

std::vector<float> stage1_frame_timestep(int32_t batch, int32_t frames, float timestep) {
    std::vector<float> out(static_cast<std::size_t>(batch) * static_cast<std::size_t>(frames),
                           timestep);
    for (int32_t b = 0; b < batch; ++b)
        out[static_cast<std::size_t>(b) * static_cast<std::size_t>(frames)] = 0.0F;
    return out;
}

void keep_stage1_anchor_frame(std::vector<float>& next, const std::vector<float>& current,
                              int32_t channels, int32_t frames, int32_t height, int32_t width) {
    for (int32_t c = 0; c < channels; ++c) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const auto idx = stage1_latent_index(c, 0, y, x, frames, height, width);
                next[idx] = current[idx];
            }
        }
    }
}

bool stage1_ltx_updates_non_anchor_tokens(float timestep) {
    // PR379 compares this scalar against a BF16 condition mask. Values close enough to 1.0
    // round back to 1.0 and do not denoise non-anchor tokens for that step.
    const float mask_lhs = round_stage1_noise_to_bfloat16(timestep / 1000.0F - 1.0e-6F);
    return mask_lhs < 1.0F;
}

void round_stage1_latents_to_bfloat16(std::vector<float>& values) {
    for (float& value : values)
        value = round_stage1_noise_to_bfloat16(value);
}

std::vector<float> stage1_velocity_from_model_output(const std::vector<float>& model_output,
                                                     std::size_t latent_count, float cfg_scale) {
    std::vector<float> velocity(latent_count, 0.0F);
    if (cfg_scale <= 1.0F) {
        if (model_output.size() != latent_count)
            throw std::runtime_error("SANA-WM Stage-1 denoiser output size mismatch");
        for (std::size_t i = 0; i < latent_count; ++i)
            velocity[i] = model_output[i];
        return velocity;
    }
    if (model_output.size() != latent_count * 2U)
        throw std::runtime_error("SANA-WM Stage-1 CFG denoiser output size mismatch");
    for (std::size_t i = 0; i < latent_count; ++i) {
        const float uncond = model_output[i];
        const float cond = model_output[latent_count + i];
        velocity[i] = uncond + cfg_scale * (cond - uncond);
    }
    return velocity;
}

std::size_t refiner_token_count(int32_t frames, int32_t height, int32_t width) {
    return static_cast<std::size_t>(frames) * static_cast<std::size_t>(height) *
           static_cast<std::size_t>(width);
}

std::size_t refiner_patched_index(int32_t token, int32_t channel, int32_t channels) {
    return static_cast<std::size_t>(token) * static_cast<std::size_t>(channels) +
           static_cast<std::size_t>(channel);
}

std::vector<float> patchify_refiner_latents(const std::vector<float>& cthw, int32_t channels,
                                            int32_t frame_offset, int32_t frames,
                                            int32_t total_frames, int32_t height, int32_t width) {
    std::vector<float> out(refiner_token_count(frames, height, width) *
                           static_cast<std::size_t>(channels));
    int32_t token = 0;
    for (int32_t f = 0; f < frames; ++f) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x, ++token) {
                for (int32_t c = 0; c < channels; ++c) {
                    out[refiner_patched_index(token, c, channels)] = cthw[stage1_latent_index(
                        c, frame_offset + f, y, x, total_frames, height, width)];
                }
            }
        }
    }
    return out;
}

std::vector<float> unpatchify_refiner_current(const std::vector<float>& tokens, int32_t channels,
                                              int32_t frames, int32_t height, int32_t width) {
    std::vector<float> out(stage1_latent_count(channels, frames, height, width));
    int32_t token = 0;
    for (int32_t f = 0; f < frames; ++f) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x, ++token) {
                for (int32_t c = 0; c < channels; ++c) {
                    out[stage1_latent_index(c, f, y, x, frames, height, width)] =
                        tokens[refiner_patched_index(token, c, channels)];
                }
            }
        }
    }
    return out;
}

std::vector<float> refiner_positions(int32_t sink_frames, int32_t current_frames, int32_t height,
                                     int32_t width, int32_t fps) {
    const int32_t total_frames = sink_frames + current_frames;
    const auto tokens = refiner_token_count(total_frames, height, width);
    std::vector<float> out(3U * tokens * 2U, 0.0F);
    int32_t token = 0;
    const float inv_fps = 1.0F / static_cast<float>(std::max(fps, 1));
    for (int32_t f = 0; f < total_frames; ++f) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x, ++token) {
                const auto base = static_cast<std::size_t>(token) * 2U;
                out[base] = std::max(0.0F, static_cast<float>(f * 8 + 1 - 8)) * inv_fps;
                out[base + 1U] = std::max(0.0F, static_cast<float>((f + 1) * 8 + 1 - 8)) * inv_fps;
                const auto y_base = (tokens + static_cast<std::size_t>(token)) * 2U;
                out[y_base] = static_cast<float>(y * 32);
                out[y_base + 1U] = static_cast<float>((y + 1) * 32);
                const auto x_base = (2U * tokens + static_cast<std::size_t>(token)) * 2U;
                out[x_base] = static_cast<float>(x * 32);
                out[x_base + 1U] = static_cast<float>((x + 1) * 32);
            }
        }
    }
    return out;
}

std::vector<float> concatenate_float_vectors(const std::vector<float>& a,
                                             const std::vector<float>& b) {
    std::vector<float> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

std::vector<SanaWmIntrinsics> crop_intrinsics(const std::vector<SanaWmIntrinsics>& intrinsics,
                                              const SanaWmResizeCropPlan& plan) {
    std::vector<SanaWmIntrinsics> out;
    out.reserve(intrinsics.size());
    for (const auto& value : intrinsics)
        out.push_back(sana_wm_transform_intrinsics_for_crop(value, plan));
    return out;
}

std::vector<SanaWmPose> resolve_native_poses(const SanaWmRequest& request,
                                             const GenerateConfig& cfg) {
    if (!cfg.camera_poses.empty())
        return sana_wm_row_major_c2w_to_poses(cfg.camera_poses);
    return sana_wm_action_to_c2w(request.action, request.translation_speed,
                                 request.rotation_speed_deg);
}

const std::vector<float>& resolve_native_intrinsics_values(const SanaWmRuntimeConfig& config,
                                                           const GenerateConfig& cfg) {
    if (!cfg.camera_intrinsics.empty())
        return cfg.camera_intrinsics;
    return config.default_intrinsics;
}

SanaWmNativeInputs prepare_native_inputs(const SanaWmRuntimeConfig& config,
                                         const SanaWmRequest& request, const GenerateConfig& cfg) {
    const auto image = io::read_image(request.image_path);
    if (image.empty())
        throw std::runtime_error("SANA-WM native runtime failed to load image: " +
                                 request.image_path);

    auto poses = resolve_native_poses(request, cfg);
    if (request.num_frames > 0 && static_cast<int32_t>(poses.size()) != request.num_frames) {
        throw std::runtime_error("SANA-WM native camera pose count does not match num_frames");
    }
    const auto& intrinsics_values = resolve_native_intrinsics_values(config, cfg);
    if (intrinsics_values.empty()) {
        throw std::runtime_error(
            "SANA-WM native runtime requires camera_intrinsics or sana_wm_default_intrinsics; "
            "Pi3X intrinsics estimation is not implemented in C++");
    }

    auto first_frame = sana_wm_prepare_vae_input_image(image.pixels, image.width, image.height,
                                                       config.height, config.width);
    if (!first_frame.ok)
        throw std::runtime_error("SANA-WM native runtime failed to preprocess first frame");

    auto intrinsics = crop_intrinsics(
        sana_wm_expand_intrinsics(intrinsics_values, static_cast<int32_t>(poses.size())),
        first_frame.plan);
    auto camera =
        sana_wm_prepare_camera_conditions(poses, intrinsics, config.height, config.width,
                                          config.vae_time_stride, config.vae_spatial_stride);
    return {std::move(first_frame), std::move(camera)};
}

std::vector<float> run_native_vae_encoder(ITrtModule& vae_encoder,
                                          const SanaWmVaeInputImage& first_frame,
                                          const SanaWmCameraConditions& camera,
                                          int32_t expected_channels) {
    if (!vae_encoder.ok())
        throw std::runtime_error("SANA-WM native VAE encoder is not ready");

    const auto input_name =
        pick_input_name(vae_encoder, {"sample", "pixel_values", "images", "x"}, "VAE encoder");
    const DType input_dtype = input_dtype_or(vae_encoder, input_name, DType::kBFloat16);
    std::vector<float> vae_input = first_frame.pixels_chw;
    maybe_load_float_tensor_override("VAE encoder input", vae_input);
    log_float_stats("VAE encoder input", vae_input);

    std::vector<half_bits_t> input16;
    TensorMap inputs;
    inputs[input_name] = make_model_tensor(vae_input, input16, input_dtype,
                                           {1, 3, 1, static_cast<int64_t>(first_frame.height),
                                            static_cast<int64_t>(first_frame.width)});

    auto outputs = vae_encoder.forward(inputs);
    const auto it =
        find_output_tensor(outputs, {"latent", "latents", "output0", "sample", "encoder_output"});
    if (it == outputs.end())
        throw std::runtime_error("SANA-WM native VAE encoder output tensor not found");

    const auto count = static_cast<std::size_t>(expected_channels) *
                       static_cast<std::size_t>(camera.latent_height) *
                       static_cast<std::size_t>(camera.latent_width);
    return tensor_to_float_vector(it->second, count, "VAE encoder");
}

std::vector<int32_t> tokenize_fixed(const ITokenizer& tokenizer, const std::string& text,
                                    int32_t length) {
    auto ids = tokenizer.encode(text);
    if (static_cast<int32_t>(ids.size()) > length)
        ids.resize(static_cast<std::size_t>(length));
    ids.resize(static_cast<std::size_t>(length), 0);
    return ids;
}

std::vector<int32_t> tokenize_fixed_left_padded(const ITokenizer& tokenizer,
                                                const std::string& text, int32_t length) {
    auto ids = tokenizer.encode(text);
    if (static_cast<int32_t>(ids.size()) > length)
        ids.resize(static_cast<std::size_t>(length));
    std::vector<int32_t> out(static_cast<std::size_t>(length), 0);
    const auto offset = out.size() - ids.size();
    std::copy(ids.begin(), ids.end(), out.begin() + static_cast<std::ptrdiff_t>(offset));
    return out;
}

std::vector<int32_t> attention_mask_from_tokens(const std::vector<int32_t>& ids) {
    std::vector<int32_t> mask(ids.size(), 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != 0)
            mask[i] = 1;
    }
    return mask;
}

SanaWmTextEncoding run_native_full_sequence_text_encoder(ITrtModule& text_encoder,
                                                         const std::vector<int32_t>& input_ids,
                                                         const std::vector<int32_t>& attention_mask,
                                                         int32_t text_dim, const char* label) {
    const int64_t seq_len = static_cast<int64_t>(input_ids.size());
    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{const_cast<int32_t*>(input_ids.data()), {1, seq_len}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{const_cast<int32_t*>(attention_mask.data()), {1, seq_len}, DType::kInt32};

    auto outputs = text_encoder.forward(inputs);
    const auto it =
        find_output_tensor(outputs, {"last_hidden_state", "hidden_states", "text_embeddings",
                                     "video_encoding", "encoder_hidden_states", "output0"});
    if (it == outputs.end())
        throw std::runtime_error(std::string("SANA-WM native ") + label +
                                 " text encoder output tensor not found");
    const auto count = text_dim > 0 ? static_cast<std::size_t>(input_ids.size()) *
                                          static_cast<std::size_t>(text_dim)
                                    : it->second.numel();
    auto shape = it->second.shape;
    if (shape.empty() && text_dim > 0)
        shape = {1, seq_len, text_dim};
    return {tensor_to_float_vector(it->second, count, label), std::move(shape)};
}

SanaWmTextEncoding run_native_decoder_text_encoder(ITrtModule& text_encoder,
                                                   const std::vector<int32_t>& input_ids,
                                                   const std::vector<int32_t>& attention_mask,
                                                   int32_t text_dim, const char* label) {
    const auto attention_shape =
        input_shape_or_profile_max(text_encoder, "attention_mask", "text attention mask");
    const auto attention_count = checked_shape_numel(attention_shape, "text attention mask");
    auto caches = collect_decoder_cache_layers(text_encoder);

    int32_t output_dim = text_dim;
    std::vector<float> encoded;
    if (output_dim > 0)
        encoded.reserve(input_ids.size() * static_cast<std::size_t>(output_dim));

    for (std::size_t pos = 0; pos < input_ids.size(); ++pos) {
        int32_t token_id = input_ids[pos];
        int32_t position_id = static_cast<int32_t>(pos);
        auto attention = decoder_attention_mask(position_id, attention_count, attention_mask);
        std::vector<half_bits_t> attention16;

        TensorMap inputs;
        inputs["token_id"] = Tensor{const_cast<int32_t*>(&token_id), {1}, DType::kInt32};
        inputs["position_id"] = Tensor{const_cast<int32_t*>(&position_id), {1}, DType::kInt32};
        inputs["attention_mask"] = make_model_tensor(
            attention, attention16, input_dtype_or(text_encoder, "attention_mask", DType::kFloat32),
            attention_shape);

        for (auto& layer : caches) {
            inputs[layer.k.input_name] = make_cache_tensor(layer.k);
            inputs[layer.v.input_name] = make_cache_tensor(layer.v);
        }

        auto outputs = text_encoder.forward(inputs);
        const auto it =
            find_output_tensor(outputs, {"hidden_state", "last_hidden_state", "hidden_states",
                                         "text_embeddings", "output0"});
        if (it == outputs.end())
            throw std::runtime_error(std::string("SANA-WM native ") + label +
                                     " decoder text encoder output tensor not found");
        const auto token_dim =
            output_dim > 0 ? static_cast<std::size_t>(output_dim) : it->second.numel();
        if (token_dim == 0 ||
            token_dim > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(std::string("SANA-WM native ") + label +
                                     " decoder text encoder output size is invalid");
        }
        if (output_dim <= 0) {
            output_dim = static_cast<int32_t>(token_dim);
            encoded.reserve(input_ids.size() * token_dim);
        }
        auto token_encoded = tensor_to_float_vector(it->second, token_dim, label);
        encoded.insert(encoded.end(), token_encoded.begin(), token_encoded.end());
        update_decoder_caches(caches, outputs, position_id);
    }

    return {std::move(encoded), {1, static_cast<int64_t>(input_ids.size()), output_dim}};
}

std::vector<std::string> sorted_refiner_debug_output_names(const TensorMap& outputs) {
    std::vector<std::string> names;
    if (outputs.find("debug_embed") != outputs.end())
        names.push_back("debug_embed");

    std::vector<std::pair<int32_t, std::string>> indexed;
    for (const auto& [name, _] : outputs) {
        const int32_t idx = trailing_nonnegative_suffix(name, "debug_hidden_");
        if (idx >= 0)
            indexed.emplace_back(idx, name);
    }
    std::sort(indexed.begin(), indexed.end());
    for (const auto& [_, name] : indexed)
        names.push_back(name);
    return names;
}

SanaWmTextEncoding
run_native_refiner_decoder_hidden_stack(ITrtModule& text_encoder,
                                        const std::vector<int32_t>& input_ids,
                                        const std::vector<int32_t>& attention_mask) {
    const auto attention_shape =
        input_shape_or_profile_max(text_encoder, "attention_mask", "refiner text attention mask");
    const auto attention_count =
        checked_shape_numel(attention_shape, "refiner text attention mask");
    auto caches = collect_decoder_cache_layers(text_encoder);

    std::vector<std::vector<float>> encoded_rows(input_ids.size());
    std::vector<std::string> hidden_output_names;
    int32_t hidden_dim = 0;
    int32_t packed_dim = 0;

    for (std::size_t pos = 0; pos < input_ids.size(); ++pos) {
        if (pos >= attention_mask.size() || attention_mask[pos] == 0)
            continue;
        int32_t token_id = input_ids[pos];
        int32_t position_id = static_cast<int32_t>(pos);
        auto attention = decoder_attention_mask(position_id, attention_count, attention_mask);
        std::vector<half_bits_t> attention16;

        TensorMap inputs;
        inputs["token_id"] = Tensor{const_cast<int32_t*>(&token_id), {1}, DType::kInt32};
        inputs["position_id"] = Tensor{const_cast<int32_t*>(&position_id), {1}, DType::kInt32};
        inputs["attention_mask"] = make_model_tensor(
            attention, attention16, input_dtype_or(text_encoder, "attention_mask", DType::kFloat32),
            attention_shape);

        for (auto& layer : caches) {
            inputs[layer.k.input_name] = make_cache_tensor(layer.k);
            inputs[layer.v.input_name] = make_cache_tensor(layer.v);
        }

        auto outputs = text_encoder.forward(inputs);
        if (hidden_output_names.empty()) {
            hidden_output_names = sorted_refiner_debug_output_names(outputs);
            if (hidden_output_names.empty()) {
                throw std::runtime_error("SANA-WM native refiner text connector requires "
                                         "debug_embed/debug_hidden_* outputs from the Gemma "
                                         "text encoder plan");
            }
        }

        std::vector<std::vector<float>> per_layer;
        per_layer.reserve(hidden_output_names.size());
        for (const auto& name : hidden_output_names) {
            const auto it = outputs.find(name);
            if (it == outputs.end()) {
                throw std::runtime_error("SANA-WM native refiner text encoder missing output " +
                                         name);
            }
            auto values = tensor_to_float_vector(it->second, it->second.numel(),
                                                 "refiner text hidden " + name);
            if (values.empty() ||
                values.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
                throw std::runtime_error(
                    "SANA-WM native refiner text hidden-state output size is invalid");
            }
            if (hidden_dim == 0) {
                hidden_dim = static_cast<int32_t>(values.size());
                packed_dim = hidden_dim * static_cast<int32_t>(hidden_output_names.size());
            } else if (values.size() != static_cast<std::size_t>(hidden_dim)) {
                throw std::runtime_error("SANA-WM native refiner text hidden-state outputs have "
                                         "mismatched hidden dimensions");
            }
            per_layer.push_back(std::move(values));
        }

        auto& row = encoded_rows[pos];
        row.reserve(static_cast<std::size_t>(packed_dim));
        for (int32_t h = 0; h < hidden_dim; ++h) {
            for (const auto& values : per_layer)
                row.push_back(values[static_cast<std::size_t>(h)]);
        }
        update_decoder_caches(caches, outputs, position_id);
    }

    if (packed_dim <= 0)
        throw std::runtime_error("SANA-WM native refiner text prompt contains no valid tokens");

    std::vector<float> encoded;
    encoded.reserve(input_ids.size() * static_cast<std::size_t>(packed_dim));
    const std::vector<float> pad_row(static_cast<std::size_t>(packed_dim), 0.0F);
    for (const auto& row : encoded_rows) {
        const auto& values = row.empty() ? pad_row : row;
        encoded.insert(encoded.end(), values.begin(), values.end());
    }

    return {std::move(encoded), {1, static_cast<int64_t>(input_ids.size()), packed_dim}};
}

SanaWmTextEncoding
run_native_refiner_full_sequence_hidden_stack(ITrtModule& text_encoder,
                                              const std::vector<int32_t>& input_ids,
                                              const std::vector<int32_t>& attention_mask) {
    const int64_t seq_len = static_cast<int64_t>(input_ids.size());
    TensorMap inputs;
    inputs["input_ids"] = Tensor{const_cast<int32_t*>(input_ids.data()), {1, seq_len},
                                 DType::kInt32};
    inputs["attention_mask"] =
        Tensor{const_cast<int32_t*>(attention_mask.data()), {1, seq_len}, DType::kInt32};

    auto outputs = text_encoder.forward(inputs);
    const auto it =
        find_output_tensor(outputs, {"text_hidden_states", "all_hidden_states",
                                     "stacked_hidden_states", "packed_hidden_states",
                                     "hidden_states", "output0"});
    if (it == outputs.end()) {
        throw std::runtime_error("SANA-WM native refiner text connector requires a packed "
                                 "all-hidden-state output from the text encoder plan");
    }
    auto shape = it->second.shape;
    auto values = tensor_to_float_vector(it->second, it->second.numel(), "refiner text hidden");
    if (shape.empty())
        shape = {1, seq_len, static_cast<int64_t>(values.size()) / std::max<int64_t>(seq_len, 1)};
    return {std::move(values), std::move(shape)};
}

SanaWmTextEncoding run_native_text_encoder(ITrtModule& text_encoder,
                                           const std::vector<int32_t>& input_ids,
                                           const std::vector<int32_t>& attention_mask,
                                           int32_t text_dim, const char* label) {
    if (!text_encoder.ok())
        throw std::runtime_error(std::string("SANA-WM native ") + label +
                                 " text encoder is not ready");
    if (text_encoder.has_input("input_ids")) {
        return run_native_full_sequence_text_encoder(text_encoder, input_ids, attention_mask,
                                                     text_dim, label);
    }
    if (text_encoder.has_input("token_id") && text_encoder.has_input("position_id") &&
        text_encoder.has_input("attention_mask")) {
        return run_native_decoder_text_encoder(text_encoder, input_ids, attention_mask, text_dim,
                                               label);
    }
    throw std::runtime_error(std::string("SANA-WM native ") + label +
                             " text encoder input tensors not found");
}

std::vector<float> select_stage1_text_window(const std::vector<float>& encoded, int32_t encoded_len,
                                             int32_t max_length, int32_t text_dim) {
    std::vector<float> out(
        static_cast<std::size_t>(max_length) * static_cast<std::size_t>(text_dim), 0.0F);
    auto copy_token = [&](int32_t src_token, int32_t dst_token) {
        const auto src = static_cast<std::size_t>(src_token) * static_cast<std::size_t>(text_dim);
        const auto dst = static_cast<std::size_t>(dst_token) * static_cast<std::size_t>(text_dim);
        std::copy_n(encoded.data() + src, static_cast<std::size_t>(text_dim), out.data() + dst);
    };
    copy_token(0, 0);
    const int32_t tail_start = std::max(1, encoded_len - max_length + 1);
    for (int32_t i = 1; i < max_length; ++i)
        copy_token(tail_start + i - 1, i);
    return out;
}

std::vector<int32_t> select_stage1_mask_window(const std::vector<int32_t>& mask,
                                               int32_t max_length) {
    std::vector<int32_t> out(static_cast<std::size_t>(max_length), 0);
    out[0] = mask.empty() ? 0 : mask.front();
    const int32_t encoded_len = static_cast<int32_t>(mask.size());
    const int32_t tail_start = std::max(1, encoded_len - max_length + 1);
    for (int32_t i = 1; i < max_length; ++i)
        out[static_cast<std::size_t>(i)] = mask[static_cast<std::size_t>(tail_start + i - 1)];
    return out;
}

SanaWmTextConditioning run_native_text_conditioning(ITrtModule& text_encoder,
                                                    const ITokenizer& tokenizer,
                                                    const SanaWmRuntimeConfig& config,
                                                    const std::string& prompt,
                                                    const std::string& negative_prompt) {
    const auto conditioning_prompt = sana_wm_make_conditioning_prompt(prompt, config.chi_prompt);
    const int32_t chi_tokens =
        config.chi_prompt.empty()
            ? 0
            : static_cast<int32_t>(tokenizer.encode(config.chi_prompt).size());
    const int32_t cond_len = config.chi_prompt.empty()
                                 ? config.text_encoder_max_length
                                 : chi_tokens + config.text_encoder_max_length - 2;

    auto cond_ids = tokenize_fixed(tokenizer, conditioning_prompt, cond_len);
    auto cond_mask_full = attention_mask_from_tokens(cond_ids);
    log_int_token_summary("stage1 cond ids", cond_ids);
    log_int_token_summary("stage1 cond mask full", cond_mask_full);
    auto cond_full = run_native_text_encoder(text_encoder, cond_ids, cond_mask_full,
                                             config.text_encoder_dim, "cond")
                         .values;

    auto neg_ids = tokenize_fixed(tokenizer, negative_prompt, config.text_encoder_max_length);
    auto neg_mask = attention_mask_from_tokens(neg_ids);
    log_int_token_summary("stage1 negative ids", neg_ids);
    log_int_token_summary("stage1 negative mask", neg_mask);
    auto neg = run_native_text_encoder(text_encoder, neg_ids, neg_mask, config.text_encoder_dim,
                                       "negative")
                   .values;

    auto cond = select_stage1_text_window(cond_full, cond_len, config.text_encoder_max_length,
                                          config.text_encoder_dim);
    auto cond_mask = select_stage1_mask_window(cond_mask_full, config.text_encoder_max_length);
    log_int_token_summary("stage1 cond mask selected", cond_mask);

    return {std::move(cond), std::move(neg), std::move(cond_mask), std::move(neg_mask)};
}

Tensor make_mask_tensor(const std::vector<int32_t>& values, std::vector<float>& scratch,
                        std::vector<half_bits_t>& scratch16, DType dtype,
                        std::vector<int64_t> shape) {
    if (dtype == DType::kInt32)
        return Tensor{const_cast<int32_t*>(values.data()), std::move(shape), DType::kInt32};
    scratch.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
        scratch[i] = static_cast<float>(values[i]);
    return make_model_tensor(scratch, scratch16, dtype, std::move(shape));
}

std::vector<float>
run_native_stage1_denoiser(ITrtModule& denoiser, const SanaWmStage1Latents& latents,
                           const SanaWmTextConditioning& text, const SanaWmCameraConditions& camera,
                           const SanaWmRuntimeConfig& config, float timestep, float cfg_scale,
                           int32_t step_index) {
    if (!denoiser.ok())
        throw std::runtime_error("SANA-WM native Stage-1 denoiser is not ready");

    const bool do_cfg = cfg_scale > 1.0F;
    const int32_t batch = do_cfg ? 2 : 1;
    auto latent_input = repeat_batch(latents.values, batch);
    auto text_input = stage1_text_input(text, do_cfg);
    auto mask_input = stage1_mask_input(text, do_cfg);
    auto raymap_input = repeat_batch(camera.raymap, batch);
    auto raymats_input = repeat_batch(camera.raymats, batch);
    auto raymats_inv_input = repeat_batch(camera.raymats_inv, batch);
    auto plucker_input = repeat_batch(camera.chunk_plucker, batch);
    auto timestep_input = stage1_frame_timestep(batch, latents.frames, timestep);
    maybe_load_float_tensor_override("Stage-1 denoiser input x", latent_input);
    maybe_load_float_tensor_override("Stage-1 denoiser input timestep", timestep_input);
    maybe_load_float_tensor_override("Stage-1 denoiser input y", text_input);
    maybe_load_float_tensor_override("Stage-1 denoiser input camera_conditions", raymap_input);
    maybe_load_float_tensor_override("Stage-1 denoiser input raymats", raymats_input);
    maybe_load_float_tensor_override("Stage-1 denoiser input raymats_inv", raymats_inv_input);
    maybe_load_float_tensor_override("Stage-1 denoiser input chunk_plucker", plucker_input);
    const auto x_label = stage1_step_debug_label("Stage-1 denoiser input x", step_index);
    const auto timestep_label =
        stage1_step_debug_label("Stage-1 denoiser input timestep", step_index);
    const auto y_label = stage1_step_debug_label("Stage-1 denoiser input y", step_index);
    const auto camera_label =
        stage1_step_debug_label("Stage-1 denoiser input camera_conditions", step_index);
    const auto raymats_label =
        stage1_step_debug_label("Stage-1 denoiser input raymats", step_index);
    const auto raymats_inv_label =
        stage1_step_debug_label("Stage-1 denoiser input raymats_inv", step_index);
    const auto plucker_label =
        stage1_step_debug_label("Stage-1 denoiser input chunk_plucker", step_index);
    log_float_stats(x_label.c_str(), latent_input);
    log_float_stats(timestep_label.c_str(), timestep_input);
    log_float_stats(y_label.c_str(), text_input);
    log_float_stats(camera_label.c_str(), raymap_input);
    log_float_stats(raymats_label.c_str(), raymats_input);
    log_float_stats(raymats_inv_label.c_str(), raymats_inv_input);
    log_float_stats(plucker_label.c_str(), plucker_input);

    const auto latent_name = pick_input_name(
        denoiser, {"x", "sample", "latents", "hidden_states", "noisy_image_or_video"},
        "Stage-1 denoiser latent");
    const auto timestep_name =
        pick_input_name(denoiser, {"timestep", "timesteps", "t"}, "Stage-1 denoiser timestep");
    const auto text_name =
        pick_input_name(denoiser, {"y", "encoder_hidden_states", "condition", "prompt_embeds"},
                        "Stage-1 denoiser text");
    const auto mask_name = pick_input_name(
        denoiser, {"mask", "encoder_attention_mask", "attention_mask"}, "Stage-1 denoiser mask");
    const auto camera_name = pick_input_name(denoiser, {"camera_conditions", "raymap", "camera"},
                                             "Stage-1 denoiser camera");
    const auto plucker_name = pick_input_name(denoiser, {"chunk_plucker", "plucker", "plucker_emb"},
                                              "Stage-1 denoiser chunk plucker");

    std::vector<half_bits_t> latent16;
    std::vector<half_bits_t> text16;
    std::vector<half_bits_t> timestep16;
    std::vector<half_bits_t> mask16;
    std::vector<half_bits_t> raymap16;
    std::vector<half_bits_t> raymats16;
    std::vector<half_bits_t> raymats_inv16;
    std::vector<half_bits_t> plucker16;
    std::vector<float> mask_float;
    TensorMap inputs;
    inputs[latent_name] = make_model_tensor(
        latent_input, latent16, input_dtype_or(denoiser, latent_name, DType::kBFloat16),
        {batch, latents.channels, latents.frames, latents.height, latents.width});
    inputs[timestep_name] = make_model_tensor(
        timestep_input, timestep16, input_dtype_or(denoiser, timestep_name, DType::kFloat32),
        {batch, 1, latents.frames});
    inputs[text_name] =
        make_model_tensor(text_input, text16, input_dtype_or(denoiser, text_name, DType::kBFloat16),
                          {batch, 1, config.text_encoder_max_length, config.text_encoder_dim});
    inputs[mask_name] = make_mask_tensor(mask_input, mask_float, mask16,
                                         input_dtype_or(denoiser, mask_name, DType::kInt32),
                                         {batch, config.text_encoder_max_length});
    inputs[camera_name] = make_model_tensor(raymap_input, raymap16,
                                            input_dtype_or(denoiser, camera_name, DType::kBFloat16),
                                            {batch, camera.latent_frames, camera.raymap_width});
    if (denoiser.has_input("raymats") || denoiser.has_input("ucpe_raymats")) {
        const auto raymats_name = denoiser.has_input("raymats") ? "raymats" : "ucpe_raymats";
        const auto token_count = static_cast<int64_t>(camera.latent_frames) *
                                 static_cast<int64_t>(camera.latent_height) *
                                 static_cast<int64_t>(camera.latent_width);
        inputs[raymats_name] = make_model_tensor(
            raymats_input, raymats16, input_dtype_or(denoiser, raymats_name, DType::kBFloat16),
            {batch, token_count, 4, 4});
    }
    if (denoiser.has_input("raymats_inv") || denoiser.has_input("ucpe_raymats_inv")) {
        const auto raymats_inv_name =
            denoiser.has_input("raymats_inv") ? "raymats_inv" : "ucpe_raymats_inv";
        const auto token_count = static_cast<int64_t>(camera.latent_frames) *
                                 static_cast<int64_t>(camera.latent_height) *
                                 static_cast<int64_t>(camera.latent_width);
        inputs[raymats_inv_name] =
            make_model_tensor(raymats_inv_input, raymats_inv16,
                              input_dtype_or(denoiser, raymats_inv_name, DType::kBFloat16),
                              {batch, token_count, 4, 4});
    }
    inputs[plucker_name] = make_model_tensor(
        plucker_input, plucker16, input_dtype_or(denoiser, plucker_name, DType::kBFloat16),
        {batch, camera.chunk_plucker_channels, camera.latent_frames, camera.latent_height,
         camera.latent_width});

    auto outputs = denoiser.forward(inputs);
    const auto it =
        find_output_tensor(outputs, {"output0", "sample", "noise_pred", "velocity", "x"});
    if (it == outputs.end())
        throw std::runtime_error("SANA-WM native Stage-1 denoiser output tensor not found");
    const auto count = static_cast<std::size_t>(batch) * latents.values.size();
    const auto denoiser_label = stage1_step_debug_label("Stage-1 denoiser", step_index);
    auto values = tensor_to_float_vector(it->second, count, denoiser_label);
    if (do_cfg) {
        log_float_stats_slice("Stage-1 denoiser uncond half", values, 0, latents.values.size());
        log_float_stats_slice("Stage-1 denoiser cond half", values, latents.values.size(),
                              latents.values.size());
    }
    return values;
}

SanaWmStage1Latents run_native_stage1_solver(ITrtModule& denoiser, SanaWmStage1Latents latents,
                                             const SanaWmTextConditioning& text,
                                             const SanaWmCameraConditions& camera,
                                             const SanaWmRuntimeConfig& config, int32_t num_steps,
                                             float cfg_scale) {
    if (num_steps <= 0)
        throw std::runtime_error("SANA-WM Stage-1 solver requires num_steps > 0");
    diffusion::FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = 1000;
    scheduler.shift = config.flow_shift;
    scheduler.set_timesteps(num_steps);
    std::vector<float> next(latents.values.size(), 0.0F);

    for (int32_t step = 0; step < num_steps; ++step) {
        const float timestep = scheduler.timesteps[static_cast<std::size_t>(step)];
        auto model_output = run_native_stage1_denoiser(denoiser, latents, text, camera, config,
                                                       timestep, cfg_scale, step);
        if (!stage1_ltx_updates_non_anchor_tokens(timestep))
            continue;
        auto velocity =
            stage1_velocity_from_model_output(model_output, latents.values.size(), cfg_scale);
        scheduler.step(velocity.data(), latents.values.data(), next.data(), latents.values.size(),
                       step);
        keep_stage1_anchor_frame(next, latents.values, latents.channels, latents.frames,
                                 latents.height, latents.width);
        round_stage1_latents_to_bfloat16(next);
        latents.values.swap(next);
    }
    log_float_stats("Stage-1 final latent", latents.values);
    return latents;
}

SanaWmStage1Latents run_native_stage1_path(SanaWmNativeModules& modules,
                                           const std::shared_ptr<ITokenizer>& tokenizer,
                                           const SanaWmRuntimeConfig& config,
                                           const SanaWmRequest& request, const GenerateConfig& cfg,
                                           const std::string& prompt) {
    auto native_inputs = prepare_native_inputs(config, request, cfg);
    if (!modules.vae_encoder) {
        throw std::runtime_error("SANA-WM native TensorRT execution requires a VAE encoder module");
    }
    auto first_latent = run_native_vae_encoder(*modules.vae_encoder, native_inputs.first_frame,
                                               native_inputs.camera, config.vae_latent_dim);
    if (!modules.text_encoder || !tokenizer) {
        throw std::runtime_error(
            "SANA-WM native TensorRT execution requires text encoder and tokenizer");
    }
    auto text = run_native_text_conditioning(*modules.text_encoder, *tokenizer, config, prompt,
                                             cfg.negative_prompt);
    const auto seed =
        cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed) : static_cast<uint64_t>(config.seed);
    auto latents = sana_wm_prepare_stage1_latents(
        first_latent, cfg.initial_latents, config.vae_latent_dim,
        native_inputs.camera.latent_frames, native_inputs.camera.latent_height,
        native_inputs.camera.latent_width, seed);
    if (!modules.stage1_denoiser) {
        throw std::runtime_error(
            "SANA-WM native TensorRT execution requires a Stage-1 denoiser module");
    }
    const int32_t num_steps = cfg.num_steps > 0 ? cfg.num_steps : config.num_steps;
    const float cfg_scale = cfg.cfg_scale >= 0.0F ? cfg.cfg_scale
                                                  : (cfg.guidance_scale >= 0.0F ? cfg.guidance_scale
                                                                                : config.cfg_scale);
    return run_native_stage1_solver(*modules.stage1_denoiser, std::move(latents), text,
                                    native_inputs.camera, config, num_steps, cfg_scale);
}

SanaWmRefinerText run_native_refiner_text_connector(ITrtModule& connector,
                                                    const SanaWmTextEncoding& hidden_states,
                                                    const std::vector<int32_t>& attention_mask) {
    if (!connector.ok())
        throw std::runtime_error("SANA-WM native refiner text connector is not ready");

    const auto hidden_name = pick_input_name(connector,
                                            {"text_hidden_states", "hidden_states",
                                             "prompt_embeds", "input_hidden_states", "input0"},
                                            "refiner text connector hidden states");
    const auto mask_name =
        pick_input_name(connector, {"attention_mask", "text_attention_mask",
                                    "prompt_attention_mask", "input_attention_mask"},
                        "refiner text connector attention mask");

    std::vector<half_bits_t> hidden16;
    std::vector<float> mask_float;
    std::vector<half_bits_t> mask16;
    TensorMap inputs;
    inputs[hidden_name] =
        make_model_tensor(hidden_states.values, hidden16,
                          input_dtype_or(connector, hidden_name, DType::kBFloat16),
                          hidden_states.shape);
    inputs[mask_name] = make_mask_tensor(
        attention_mask, mask_float, mask16, input_dtype_or(connector, mask_name, DType::kInt32),
        {1, static_cast<int64_t>(attention_mask.size())});

    auto outputs = connector.forward(inputs);
    const auto context_it =
        find_output_tensor(outputs, {"v_context", "video_text_embedding",
                                     "video_text_embeddings", "connector_prompt_embeds",
                                     "prompt_embeds", "encoder_hidden_states", "output0"});
    if (context_it == outputs.end())
        throw std::runtime_error("SANA-WM native refiner text connector output tensor not found");

    auto values = tensor_to_float_vector(context_it->second, context_it->second.numel(),
                                         "refiner text connector");
    auto shape = context_it->second.shape;
    if (shape.empty())
        shape = {1, static_cast<int64_t>(values.size())};

    std::vector<float> fallback_mask(attention_mask.begin(), attention_mask.end());
    auto mask_values = std::move(fallback_mask);
    std::vector<int64_t> mask_shape{1, static_cast<int64_t>(attention_mask.size())};
    const auto mask_it =
        find_output_tensor(outputs, {"v_attention_mask", "video_attention_mask",
                                     "connector_attention_mask", "encoder_attention_mask",
                                     "text_attention_mask", "prompt_attention_mask"});
    if (mask_it != outputs.end() && mask_it != context_it) {
        mask_values = tensor_to_float_vector(mask_it->second, mask_it->second.numel(),
                                             "refiner connector attention mask");
        mask_shape = mask_it->second.shape;
        if (mask_shape.empty())
            mask_shape = {1, static_cast<int64_t>(mask_values.size())};
    }

    return {std::move(values), std::move(shape), std::move(mask_values), std::move(mask_shape)};
}

SanaWmRefinerText run_native_refiner_text_encoder(ITrtModule& text_encoder,
                                                  ITrtModule* text_connector,
                                                  const ITokenizer& tokenizer, int32_t max_length,
                                                  const std::string& prompt) {
    if (!text_encoder.ok())
        throw std::runtime_error("SANA-WM native refiner text encoder is not ready");
    auto input_ids = tokenize_fixed_left_padded(tokenizer, prompt, max_length);
    auto attention_mask = attention_mask_from_tokens(input_ids);
    std::vector<float> fallback_mask(attention_mask.begin(), attention_mask.end());

    if (text_connector != nullptr) {
        SanaWmTextEncoding hidden_states;
        if (text_encoder.has_input("input_ids")) {
            hidden_states = run_native_refiner_full_sequence_hidden_stack(text_encoder, input_ids,
                                                                          attention_mask);
        } else if (text_encoder.has_input("token_id") && text_encoder.has_input("position_id") &&
                   text_encoder.has_input("attention_mask")) {
            hidden_states =
                run_native_refiner_decoder_hidden_stack(text_encoder, input_ids, attention_mask);
        } else {
            throw std::runtime_error("SANA-WM native refiner text encoder input tensors not found");
        }
        if (hidden_states.shape.empty())
            hidden_states.shape = {1, static_cast<int64_t>(hidden_states.values.size())};
        return run_native_refiner_text_connector(*text_connector, hidden_states, attention_mask);
    }

    if (text_encoder.has_input("input_ids")) {
        const int64_t seq_len = static_cast<int64_t>(input_ids.size());
        TensorMap inputs;
        inputs["input_ids"] =
            Tensor{const_cast<int32_t*>(input_ids.data()), {1, seq_len}, DType::kInt32};
        inputs["attention_mask"] =
            Tensor{const_cast<int32_t*>(attention_mask.data()), {1, seq_len}, DType::kInt32};

        auto outputs = text_encoder.forward(inputs);
        const auto context_it = find_output_tensor(
            outputs, {"v_context", "video_text_embedding", "video_text_embeddings",
                      "connector_prompt_embeds", "prompt_embeds", "last_hidden_state",
                      "hidden_states", "text_embeddings", "encoder_hidden_states", "output0"});
        if (context_it == outputs.end()) {
            throw std::runtime_error(
                "SANA-WM native refiner text encoder output tensor not found");
        }

        auto shape = context_it->second.shape;
        auto values = tensor_to_float_vector(context_it->second, context_it->second.numel(),
                                             "refiner");
        if (shape.empty())
            shape = {1, static_cast<int64_t>(values.size())};

        auto mask_values = std::move(fallback_mask);
        std::vector<int64_t> mask_shape{1, seq_len};
        const auto mask_it =
            find_output_tensor(outputs, {"v_attention_mask", "video_attention_mask",
                                         "connector_attention_mask", "encoder_attention_mask",
                                         "text_attention_mask", "prompt_attention_mask"});
        if (mask_it != outputs.end() && mask_it != context_it) {
            mask_values = tensor_to_float_vector(mask_it->second, mask_it->second.numel(),
                                                 "refiner attention mask");
            mask_shape = mask_it->second.shape;
            if (mask_shape.empty())
                mask_shape = {1, static_cast<int64_t>(mask_values.size())};
        }

        return {std::move(values), std::move(shape), std::move(mask_values),
                std::move(mask_shape)};
    }

    auto encoded = run_native_text_encoder(text_encoder, input_ids, attention_mask, 0, "refiner");
    if (encoded.shape.empty())
        encoded.shape = {1, static_cast<int64_t>(encoded.values.size())};
    return {std::move(encoded.values), std::move(encoded.shape), std::move(fallback_mask),
            {1, static_cast<int64_t>(input_ids.size())}};
}

std::vector<float> run_native_refiner_denoiser(ITrtModule& denoiser,
                                               const std::vector<float>& combined_latent,
                                               const std::vector<float>& clean_latent,
                                               const std::vector<float>& denoise_mask,
                                               const std::vector<float>& positions,
                                               const SanaWmRefinerText& text, float sigma,
                                               int32_t total_tokens, int32_t channels) {
    if (!denoiser.ok())
        throw std::runtime_error("SANA-WM native refiner denoiser is not ready");

    const auto latent_name = pick_input_name(denoiser, {"latent", "hidden_states", "video_latent"},
                                             "refiner denoiser latent");
    const auto clean_name = pick_input_name(denoiser, {"clean_latent", "clean_video_latent"},
                                            "refiner denoiser clean latent");
    const auto mask_name =
        pick_input_name(denoiser, {"denoise_mask", "mask"}, "refiner denoiser mask");
    const auto positions_name =
        pick_input_name(denoiser, {"positions", "video_positions"}, "refiner denoiser positions");
    const auto text_name =
        pick_input_name(denoiser, {"v_context", "encoder_hidden_states", "text_embeddings"},
                        "refiner denoiser text");
    const auto text_mask_name = pick_optional_input_name(
        denoiser, {"v_attention_mask", "encoder_attention_mask", "attention_mask",
                   "text_attention_mask", "prompt_attention_mask"});
    const auto sigma_name =
        pick_input_name(denoiser, {"sigma", "timestep"}, "refiner denoiser sigma");

    std::vector<half_bits_t> latent16;
    std::vector<half_bits_t> clean16;
    std::vector<half_bits_t> positions16;
    std::vector<half_bits_t> text16;
    std::vector<half_bits_t> text_mask16;
    std::vector<half_bits_t> sigma16;
    std::vector<float> sigma_vec{sigma};
    TensorMap inputs;
    inputs[latent_name] = make_model_tensor(combined_latent, latent16,
                                            input_dtype_or(denoiser, latent_name, DType::kBFloat16),
                                            {1, total_tokens, channels});
    inputs[clean_name] = make_model_tensor(clean_latent, clean16,
                                           input_dtype_or(denoiser, clean_name, DType::kBFloat16),
                                           {1, total_tokens, channels});
    inputs[mask_name] =
        Tensor{const_cast<float*>(denoise_mask.data()), {1, total_tokens, 1}, DType::kFloat32};
    inputs[positions_name] = make_model_tensor(
        positions, positions16, input_dtype_or(denoiser, positions_name, DType::kBFloat16),
        {1, 3, total_tokens, 2});
    inputs[text_name] = make_model_tensor(
        text.values, text16, input_dtype_or(denoiser, text_name, DType::kBFloat16), text.shape);
    if (!text_mask_name.empty()) {
        inputs[text_mask_name] =
            make_model_tensor(text.attention_mask, text_mask16,
                              input_dtype_or(denoiser, text_mask_name, DType::kFloat32),
                              text.attention_mask_shape);
    }
    inputs[sigma_name] = make_model_tensor(
        sigma_vec, sigma16, input_dtype_or(denoiser, sigma_name, DType::kFloat32), {1});

    auto outputs = denoiser.forward(inputs);
    const auto it = find_output_tensor(outputs, {"output0", "denoised", "sample", "pred", "x0"});
    if (it == outputs.end())
        throw std::runtime_error("SANA-WM native refiner denoiser output tensor not found");
    return tensor_to_float_vector(it->second, it->second.numel(), "refiner denoiser");
}

std::vector<float> refiner_current_prediction(const std::vector<float>& output,
                                              std::size_t context_values,
                                              std::size_t current_values) {
    if (output.size() == current_values)
        return output;
    if (output.size() == context_values + current_values) {
        return std::vector<float>(output.begin() + static_cast<std::ptrdiff_t>(context_values),
                                  output.end());
    }
    if (output.size() > current_values)
        return std::vector<float>(output.begin(),
                                  output.begin() + static_cast<std::ptrdiff_t>(current_values));
    throw std::runtime_error(
        "SANA-WM native refiner denoiser output is smaller than current tokens");
}

std::vector<float> refiner_euler_step(const std::vector<float>& sample,
                                      const std::vector<float>& denoised, float sigma,
                                      float sigma_next) {
    if (sample.size() != denoised.size())
        throw std::runtime_error("SANA-WM native refiner sample/prediction size mismatch");
    std::vector<float> out(sample.size(), 0.0F);
    const float dt = sigma_next - sigma;
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const float velocity = (sample[i] - denoised[i]) / std::max(sigma, 1.0e-6F);
        out[i] = round_stage1_noise_to_bfloat16(sample[i] + velocity * dt);
    }
    return out;
}

SanaWmStage1Latents run_native_refiner(ITrtModule& text_encoder, ITrtModule* text_connector,
                                       ITrtModule& denoiser, const ITokenizer& tokenizer,
                                       const SanaWmStage1Latents& stage1,
                                       const SanaWmRuntimeConfig& config, const GenerateConfig& cfg,
                                       const std::string& prompt) {
    constexpr int32_t kSinkFrames = 1;
    if (stage1.frames <= kSinkFrames)
        throw std::runtime_error("SANA-WM native refiner requires more than one latent frame");
    const int32_t current_frames = stage1.frames - kSinkFrames;
    const auto text = run_native_refiner_text_encoder(
        text_encoder, text_connector, tokenizer, config.refiner_text_encoder_max_length, prompt);
    auto sink = patchify_refiner_latents(stage1.values, stage1.channels, 0, kSinkFrames,
                                         stage1.frames, stage1.height, stage1.width);
    auto current_clean =
        patchify_refiner_latents(stage1.values, stage1.channels, kSinkFrames, current_frames,
                                 stage1.frames, stage1.height, stage1.width);
    auto noise_cthw = sample_stage1_noise(stage1_latent_count(stage1.channels, current_frames,
                                                              stage1.height, stage1.width),
                                          cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed)
                                                        : static_cast<uint64_t>(config.seed));
    auto noise = patchify_refiner_latents(noise_cthw, stage1.channels, 0, current_frames,
                                          current_frames, stage1.height, stage1.width);
    std::vector<float> current(current_clean.size(), 0.0F);
    for (std::size_t i = 0; i < current.size(); ++i)
        current[i] = round_stage1_noise_to_bfloat16(
            (1.0F - kRefinerSigmas[0]) * current_clean[i] + kRefinerSigmas[0] * noise[i]);
    log_float_stats("refiner input stage1 latent", stage1.values);
    log_float_stats("refiner current clean", current_clean);
    log_float_stats("refiner initial noise", noise);
    log_float_stats("refiner noisy current step0", current);

    const auto positions =
        refiner_positions(kSinkFrames, current_frames, stage1.height, stage1.width, config.fps);
    const auto context_values = sink.size();
    const auto current_values = current.size();
    const int32_t total_tokens = static_cast<int32_t>((context_values + current_values) /
                                                      static_cast<std::size_t>(stage1.channels));
    std::vector<float> clean =
        concatenate_float_vectors(sink, std::vector<float>(current_values, 0.0F));
    std::vector<float> mask(static_cast<std::size_t>(total_tokens), 0.0F);
    std::fill(mask.begin() + static_cast<std::ptrdiff_t>(context_values / stage1.channels),
              mask.end(), 1.0F);

    for (std::size_t i = 0; i + 1U < kRefinerSigmas.size(); ++i) {
        auto combined = concatenate_float_vectors(sink, current);
        auto output = run_native_refiner_denoiser(denoiser, combined, clean, mask, positions, text,
                                                  kRefinerSigmas[i], total_tokens, stage1.channels);
        auto pred = refiner_current_prediction(output, context_values, current_values);
        log_float_stats((std::string("refiner denoised step") + std::to_string(i)).c_str(), pred);
        current = refiner_euler_step(current, pred, kRefinerSigmas[i], kRefinerSigmas[i + 1U]);
        log_float_stats((std::string("refiner noisy current step") + std::to_string(i + 1U)).c_str(),
                        current);
    }

    auto current_cthw = unpatchify_refiner_current(current, stage1.channels, current_frames,
                                                   stage1.height, stage1.width);
    SanaWmStage1Latents refined;
    refined.values = stage1.values;
    refined.channels = stage1.channels;
    refined.frames = stage1.frames;
    refined.height = stage1.height;
    refined.width = stage1.width;
    for (int32_t c = 0; c < stage1.channels; ++c)
        for (int32_t f = 0; f < current_frames; ++f)
            for (int32_t y = 0; y < stage1.height; ++y)
                for (int32_t x = 0; x < stage1.width; ++x)
                    refined.values[stage1_latent_index(c, f + kSinkFrames, y, x, stage1.frames,
                                                       stage1.height, stage1.width)] =
                        current_cthw[stage1_latent_index(c, f, y, x, current_frames, stage1.height,
                                                         stage1.width)];
    return refined;
}

bool has_any_refiner_module(const SanaWmNativeModules& modules) {
    return modules.refiner_text_encoder || modules.refiner_text_connector ||
           modules.refiner_denoiser || modules.refiner_vae_decoder ||
           !modules.refiner_vae_decoder_tiles.empty();
}

bool has_stage1_core_modules(const SanaWmNativeModules& modules) {
    return modules.text_encoder && modules.stage1_denoiser && modules.vae_encoder;
}

void validate_native_module_set(const SanaWmNativeModules& modules) {
    if (!modules.has_any())
        return;
    if (!has_stage1_core_modules(modules)) {
        throw std::runtime_error("SANA-WM native TensorRT execution requires a complete stage1 "
                                 "module set: text encoder, denoiser, and VAE encoder");
    }
    if (has_any_refiner_module(modules)) {
        if (!modules.has_refiner()) {
            throw std::runtime_error("SANA-WM native TensorRT execution requires a complete "
                                     "refiner module set: refiner text encoder, optional refiner "
                                     "text connector, refiner denoiser, and refiner VAE decoder");
        }
        return;
    }
    if (!modules.vae_decoder && modules.vae_decoder_tiles.empty()) {
        throw std::runtime_error("SANA-WM native TensorRT execution requires a VAE decoder module "
                                 "when no native refiner is bundled");
    }
}

struct SanaWmDecodedCthw {
    std::vector<float> values;
    int32_t frames{0};
    int32_t height{0};
    int32_t width{0};
};

std::size_t vae_video_index(int32_t channel, int32_t frame, int32_t y, int32_t x,
                            int32_t frames, int32_t height, int32_t width) {
    return (((static_cast<std::size_t>(channel) * static_cast<std::size_t>(frames) +
              static_cast<std::size_t>(frame)) *
                 static_cast<std::size_t>(height) +
             static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x));
}

int32_t decoded_frames_for_latent_frames(int32_t latent_frames, int32_t stride) {
    if (latent_frames <= 0)
        return 0;
    return (latent_frames - 1) * stride + 1;
}

std::vector<float> extract_latent_tile(const SanaWmStage1Latents& latents, int32_t frame_start,
                                       int32_t tile_frames, int32_t y_start,
                                       int32_t tile_height, int32_t x_start,
                                       int32_t tile_width) {
    std::vector<float> out(static_cast<std::size_t>(latents.channels) *
                               static_cast<std::size_t>(tile_frames) *
                               static_cast<std::size_t>(tile_height) *
                               static_cast<std::size_t>(tile_width),
                           0.0F);
    for (int32_t c = 0; c < latents.channels; ++c)
        for (int32_t t = 0; t < tile_frames; ++t)
            for (int32_t y = 0; y < tile_height; ++y)
                for (int32_t x = 0; x < tile_width; ++x) {
                    out[stage1_latent_index(c, t, y, x, tile_frames, tile_height, tile_width)] =
                        latents.values[stage1_latent_index(
                            c, frame_start + t, y_start + y, x_start + x, latents.frames,
                            latents.height, latents.width)];
                }
    return out;
}

const SanaWmVaeDecoderTile* find_vae_decoder_tile(const std::vector<SanaWmVaeDecoderTile>& tiles,
                                                  int32_t frames, int32_t height,
                                                  int32_t width) {
    for (const auto& tile : tiles) {
        if (tile.latent_frames == frames && tile.latent_height == height &&
            tile.latent_width == width && tile.module) {
            return &tile;
        }
    }
    return nullptr;
}

SanaWmDecodedCthw run_native_vae_decoder_cthw(ITrtModule& vae_decoder,
                                              const std::vector<float>& latent_values,
                                              int32_t channels, int32_t latent_frames,
                                              int32_t latent_height, int32_t latent_width,
                                              int32_t vae_time_stride,
                                              int32_t vae_spatial_stride,
                                              const char* label) {
    if (!vae_decoder.ok())
        throw std::runtime_error(std::string("SANA-WM native ") + label + " is not ready");

    const auto input_name = pick_input_name(vae_decoder, {"latents", "sample", "z"}, label);
    std::vector<half_bits_t> latent16;
    TensorMap inputs;
    inputs[input_name] = make_model_tensor(
        latent_values, latent16, input_dtype_or(vae_decoder, input_name, DType::kBFloat16),
        {1, channels, latent_frames, latent_height, latent_width});

    auto outputs = vae_decoder.forward(inputs);
    const auto it = find_output_tensor(outputs, {"output0", "sample", "decoder_output", "video"});
    if (it == outputs.end())
        throw std::runtime_error(std::string("SANA-WM native ") + label +
                                 " output tensor not found");

    const int32_t out_frames = decoded_frames_for_latent_frames(latent_frames, vae_time_stride);
    const int32_t out_height = latent_height * vae_spatial_stride;
    const int32_t out_width = latent_width * vae_spatial_stride;
    const auto raw_count = static_cast<std::size_t>(3) * static_cast<std::size_t>(out_frames) *
                           static_cast<std::size_t>(out_height) *
                           static_cast<std::size_t>(out_width);
    return {tensor_to_float_vector(it->second, raw_count, label), out_frames, out_height,
            out_width};
}

void blend_vertical(SanaWmDecodedCthw& above, SanaWmDecodedCthw& current, int32_t blend_extent) {
    blend_extent = std::min({above.height, current.height, blend_extent});
    for (int32_t y = 0; y < blend_extent; ++y) {
        const float alpha = static_cast<float>(y) / static_cast<float>(blend_extent);
        for (int32_t c = 0; c < 3; ++c)
            for (int32_t t = 0; t < current.frames; ++t)
                for (int32_t x = 0; x < current.width; ++x) {
                    const auto src = vae_video_index(c, t, above.height - blend_extent + y, x,
                                                     above.frames, above.height, above.width);
                    const auto dst =
                        vae_video_index(c, t, y, x, current.frames, current.height, current.width);
                    current.values[dst] = above.values[src] * (1.0F - alpha) +
                                          current.values[dst] * alpha;
                }
    }
}

void blend_horizontal(SanaWmDecodedCthw& left, SanaWmDecodedCthw& current,
                      int32_t blend_extent) {
    blend_extent = std::min({left.width, current.width, blend_extent});
    for (int32_t x = 0; x < blend_extent; ++x) {
        const float alpha = static_cast<float>(x) / static_cast<float>(blend_extent);
        for (int32_t c = 0; c < 3; ++c)
            for (int32_t t = 0; t < current.frames; ++t)
                for (int32_t y = 0; y < current.height; ++y) {
                    const auto src = vae_video_index(c, t, y, left.width - blend_extent + x,
                                                     left.frames, left.height, left.width);
                    const auto dst =
                        vae_video_index(c, t, y, x, current.frames, current.height, current.width);
                    current.values[dst] = left.values[src] * (1.0F - alpha) +
                                          current.values[dst] * alpha;
                }
    }
}

void blend_temporal(SanaWmDecodedCthw& previous, SanaWmDecodedCthw& current,
                    int32_t blend_extent) {
    blend_extent = std::min({previous.frames, current.frames, blend_extent});
    for (int32_t t = 0; t < blend_extent; ++t) {
        const float alpha = static_cast<float>(t) / static_cast<float>(blend_extent);
        for (int32_t c = 0; c < 3; ++c)
            for (int32_t y = 0; y < current.height; ++y)
                for (int32_t x = 0; x < current.width; ++x) {
                    const auto src =
                        vae_video_index(c, previous.frames - blend_extent + t, y, x,
                                        previous.frames, previous.height, previous.width);
                    const auto dst =
                        vae_video_index(c, t, y, x, current.frames, current.height, current.width);
                    current.values[dst] = previous.values[src] * (1.0F - alpha) +
                                          current.values[dst] * alpha;
                }
    }
}

SanaWmDecodedCthw drop_last_frame(SanaWmDecodedCthw input) {
    if (input.frames <= 0)
        return input;
    const int32_t out_frames = input.frames - 1;
    std::vector<float> out(static_cast<std::size_t>(3) * static_cast<std::size_t>(out_frames) *
                               static_cast<std::size_t>(input.height) *
                               static_cast<std::size_t>(input.width),
                           0.0F);
    for (int32_t c = 0; c < 3; ++c)
        for (int32_t t = 0; t < out_frames; ++t)
            for (int32_t y = 0; y < input.height; ++y)
                for (int32_t x = 0; x < input.width; ++x)
                    out[vae_video_index(c, t, y, x, out_frames, input.height, input.width)] =
                        input.values[vae_video_index(c, t, y, x, input.frames, input.height,
                                                     input.width)];
    input.values = std::move(out);
    input.frames = out_frames;
    return input;
}

SanaWmDecodedCthw decode_native_vae_spatial_tiled(
    const std::vector<SanaWmVaeDecoderTile>& tiles, const SanaWmStage1Latents& latents,
    const SanaWmRuntimeConfig& config, int32_t frame_start, int32_t tile_frames,
    const char* label) {
    const int32_t tile_latent_min_height =
        std::max(1, config.vae_tile_sample_min_height / config.vae_spatial_stride);
    const int32_t tile_latent_min_width =
        std::max(1, config.vae_tile_sample_min_width / config.vae_spatial_stride);
    const int32_t tile_latent_stride_height =
        std::max(1, config.vae_tile_sample_stride_height / config.vae_spatial_stride);
    const int32_t tile_latent_stride_width =
        std::max(1, config.vae_tile_sample_stride_width / config.vae_spatial_stride);
    const int32_t blend_height =
        std::max(0, config.vae_tile_sample_min_height - config.vae_tile_sample_stride_height);
    const int32_t blend_width =
        std::max(0, config.vae_tile_sample_min_width - config.vae_tile_sample_stride_width);
    if (!config.vae_use_spatial_tiling ||
        (latents.height <= tile_latent_min_height && latents.width <= tile_latent_min_width)) {
        const auto* tile_module =
            find_vae_decoder_tile(tiles, tile_frames, latents.height, latents.width);
        if (!tile_module) {
            throw std::runtime_error("SANA-WM native tiled VAE missing tile decoder for [" +
                                     std::to_string(tile_frames) + "," +
                                     std::to_string(latents.height) + "," +
                                     std::to_string(latents.width) + "]");
        }
        auto latent_tile = extract_latent_tile(latents, frame_start, tile_frames, 0,
                                               latents.height, 0, latents.width);
        return run_native_vae_decoder_cthw(
            *tile_module->module, latent_tile, latents.channels, tile_frames, latents.height,
            latents.width, config.vae_time_stride, config.vae_spatial_stride, label);
    }
    const int32_t output_frames = decoded_frames_for_latent_frames(tile_frames, config.vae_time_stride);
    SanaWmDecodedCthw out;
    out.frames = output_frames;
    out.height = latents.height * config.vae_spatial_stride;
    out.width = latents.width * config.vae_spatial_stride;
    out.values.resize(static_cast<std::size_t>(3) * static_cast<std::size_t>(out.frames) *
                          static_cast<std::size_t>(out.height) *
                          static_cast<std::size_t>(out.width),
                      0.0F);

    std::vector<std::vector<SanaWmDecodedCthw>> rows;
    int32_t dst_y = 0;
    for (int32_t y0 = 0; y0 < latents.height; y0 += tile_latent_stride_height) {
        std::vector<SanaWmDecodedCthw> row;
        for (int32_t x0 = 0; x0 < latents.width; x0 += tile_latent_stride_width) {
            const int32_t tile_height = std::min(tile_latent_min_height, latents.height - y0);
            const int32_t tile_width = std::min(tile_latent_min_width, latents.width - x0);
            const auto* tile_module =
                find_vae_decoder_tile(tiles, tile_frames, tile_height, tile_width);
            if (!tile_module) {
                throw std::runtime_error("SANA-WM native tiled VAE missing tile decoder for [" +
                                         std::to_string(tile_frames) + "," +
                                         std::to_string(tile_height) + "," +
                                         std::to_string(tile_width) + "]");
            }
            auto latent_tile = extract_latent_tile(latents, frame_start, tile_frames, y0,
                                                   tile_height, x0, tile_width);
            row.push_back(run_native_vae_decoder_cthw(
                *tile_module->module, latent_tile, latents.channels, tile_frames, tile_height,
                tile_width, config.vae_time_stride, config.vae_spatial_stride, label));
        }

        int32_t dst_x = 0;
        int32_t row_crop_height = 0;
        const auto row_index = rows.size();
        for (std::size_t j = 0; j < row.size(); ++j) {
            auto& tile = row[j];
            if (row_index > 0)
                blend_vertical(rows[row_index - 1U][j], tile, blend_height);
            if (j > 0)
                blend_horizontal(row[j - 1U], tile, blend_width);
            const int32_t crop_height =
                std::min(config.vae_tile_sample_stride_height, tile.height);
            const int32_t crop_width = std::min(config.vae_tile_sample_stride_width, tile.width);
            row_crop_height = std::max(row_crop_height, crop_height);
            for (int32_t c = 0; c < 3; ++c)
                for (int32_t t = 0; t < tile.frames; ++t)
                    for (int32_t y = 0; y < crop_height && dst_y + y < out.height; ++y)
                        for (int32_t x = 0; x < crop_width && dst_x + x < out.width; ++x)
                            out.values[vae_video_index(c, t, dst_y + y, dst_x + x, out.frames,
                                                       out.height, out.width)] =
                                tile.values[vae_video_index(c, t, y, x, tile.frames, tile.height,
                                                            tile.width)];
            dst_x += crop_width;
        }
        dst_y += row_crop_height;
        rows.push_back(std::move(row));
    }
    return out;
}

SanaWmDecodedCthw decode_native_vae_tiled(const std::vector<SanaWmVaeDecoderTile>& tiles,
                                          const SanaWmStage1Latents& latents,
                                          const SanaWmRuntimeConfig& config,
                                          const char* label) {
    const int32_t tile_latent_min_frames =
        std::max(1, config.vae_tile_sample_min_num_frames / config.vae_time_stride);
    const int32_t tile_latent_stride_frames =
        std::max(1, config.vae_tile_sample_stride_num_frames / config.vae_time_stride);
    const int32_t blend_frames = std::max(
        0, config.vae_tile_sample_min_num_frames - config.vae_tile_sample_stride_num_frames);
    const int32_t output_frames = decoded_frames_for_latent_frames(latents.frames, config.vae_time_stride);

    SanaWmDecodedCthw out;
    out.frames = output_frames;
    out.height = latents.height * config.vae_spatial_stride;
    out.width = latents.width * config.vae_spatial_stride;
    out.values.resize(static_cast<std::size_t>(3) * static_cast<std::size_t>(out.frames) *
                          static_cast<std::size_t>(out.height) *
                          static_cast<std::size_t>(out.width),
                      0.0F);

    std::vector<SanaWmDecodedCthw> temporal_tiles;
    if (config.vae_use_framewise_decoding && latents.frames > tile_latent_min_frames) {
        for (int32_t t0 = 0; t0 < latents.frames; t0 += tile_latent_stride_frames) {
            const int32_t tile_frames =
                std::min(tile_latent_min_frames + 1, latents.frames - t0);
            if (t0 > 0 && tile_frames <= 1)
                continue;
            auto decoded =
                decode_native_vae_spatial_tiled(tiles, latents, config, t0, tile_frames, label);
            if (t0 > 0)
                decoded = drop_last_frame(std::move(decoded));
            temporal_tiles.push_back(std::move(decoded));
        }
    } else {
        temporal_tiles.push_back(
            decode_native_vae_spatial_tiled(tiles, latents, config, 0, latents.frames, label));
    }

    int32_t dst_t = 0;
    for (std::size_t i = 0; i < temporal_tiles.size(); ++i) {
        auto& tile = temporal_tiles[i];
        int32_t crop_frames = 0;
        if (i == 0) {
            crop_frames = std::min(config.vae_tile_sample_stride_num_frames + 1, tile.frames);
        } else {
            blend_temporal(temporal_tiles[i - 1U], tile, blend_frames);
            crop_frames = std::min(config.vae_tile_sample_stride_num_frames, tile.frames);
        }
        for (int32_t c = 0; c < 3; ++c)
            for (int32_t t = 0; t < crop_frames && dst_t + t < out.frames; ++t)
                for (int32_t y = 0; y < out.height; ++y)
                    for (int32_t x = 0; x < out.width; ++x)
                        out.values[vae_video_index(c, dst_t + t, y, x, out.frames, out.height,
                                                   out.width)] =
                            tile.values[vae_video_index(c, t, y, x, tile.frames, tile.height,
                                                        tile.width)];
        dst_t += crop_frames;
    }
    return out;
}

ImageResult image_result_from_stage1_cthw(const SanaWmDecodedCthw& raw,
                                          const SanaWmRuntimeConfig& config) {
    if (raw.frames < config.num_frames || raw.height < config.height || raw.width < config.width)
        throw std::runtime_error("SANA-WM native tiled VAE output is smaller than requested video");

    ImageResult result;
    result.channels = 3;
    result.height = config.height;
    result.width = config.width;
    result.num_frames = config.num_frames;
    const auto raw_count =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(config.num_frames) *
        static_cast<std::size_t>(config.height) * static_cast<std::size_t>(config.width);
    result.pixels.resize(raw_count, 0.0F);
    for (int32_t t = 0; t < config.num_frames; ++t) {
        for (int32_t y = 0; y < config.height; ++y) {
            for (int32_t x = 0; x < config.width; ++x) {
                for (int32_t c = 0; c < 3; ++c) {
                    const auto src =
                        vae_video_index(c, t, y, x, raw.frames, raw.height, raw.width);
                    const auto dst =
                        (((static_cast<std::size_t>(t) * static_cast<std::size_t>(config.height) +
                           static_cast<std::size_t>(y)) *
                              static_cast<std::size_t>(config.width) +
                          static_cast<std::size_t>(x)) *
                             3U +
                         static_cast<std::size_t>(c));
                    result.pixels[dst] =
                        std::max(0.0F, std::min(1.0F, raw.values[src] * 0.5F + 0.5F));
                }
            }
        }
    }
    return result;
}

ImageResult decode_native_sana_vae(ITrtModule& vae_decoder, const SanaWmStage1Latents& latents,
                                   const SanaWmRuntimeConfig& config) {
    auto raw = run_native_vae_decoder_cthw(
        vae_decoder, latents.values, latents.channels, latents.frames, latents.height,
        latents.width, config.vae_time_stride, config.vae_spatial_stride, "VAE decoder");
    return image_result_from_stage1_cthw(raw, config);
}

ImageResult decode_native_sana_vae_tiled(const std::vector<SanaWmVaeDecoderTile>& tiles,
                                         const SanaWmStage1Latents& latents,
                                         const SanaWmRuntimeConfig& config) {
    auto raw = decode_native_vae_tiled(tiles, latents, config, "VAE decoder");
    log_float_stats("VAE decoder tiled", raw.values);
    return image_result_from_stage1_cthw(raw, config);
}

float normalize_refiner_pixel(float value) {
    if (value > 1.0F)
        return std::max(0.0F, std::min(1.0F, value / 255.0F));
    return std::max(0.0F, std::min(1.0F, value));
}

float normalize_refiner_cthw_pixel(float value) {
    return std::max(0.0F, std::min(1.0F, value * 0.5F + 0.5F));
}

ImageResult image_result_from_refiner_cthw(const SanaWmDecodedCthw& raw,
                                           const SanaWmRuntimeConfig& config) {
    const int32_t output_frames = std::max(config.num_frames - 1, 1);
    if (raw.frames < output_frames || raw.height < config.height || raw.width < config.width)
        throw std::runtime_error("SANA-WM native tiled refiner VAE output is smaller than "
                                 "requested video");
    const bool drop_sink = raw.frames >= output_frames + 1;
    const int32_t src_frame_offset = drop_sink ? 1 : 0;
    const auto direct_count = static_cast<std::size_t>(output_frames) *
                              static_cast<std::size_t>(config.height) *
                              static_cast<std::size_t>(config.width) * 3U;

    ImageResult result;
    result.channels = 3;
    result.height = config.height;
    result.width = config.width;
    result.num_frames = output_frames;
    result.pixels.resize(direct_count, 0.0F);
    for (int32_t t = 0; t < output_frames; ++t)
        for (int32_t y = 0; y < config.height; ++y)
            for (int32_t x = 0; x < config.width; ++x)
                for (int32_t c = 0; c < 3; ++c) {
                    const auto src =
                        vae_video_index(c, src_frame_offset + t, y, x, raw.frames, raw.height,
                                        raw.width);
                    const auto dst =
                        (((static_cast<std::size_t>(t) *
                               static_cast<std::size_t>(config.height) +
                           static_cast<std::size_t>(y)) *
                              static_cast<std::size_t>(config.width) +
                          static_cast<std::size_t>(x)) *
                             3U +
                         static_cast<std::size_t>(c));
                    result.pixels[dst] = normalize_refiner_cthw_pixel(raw.values[src]);
                }
    return result;
}

std::size_t cthw_video_index(int32_t channel, int32_t frame, int32_t y, int32_t x, int32_t frames,
                             int32_t height, int32_t width) {
    return (((static_cast<std::size_t>(channel) * static_cast<std::size_t>(frames) +
              static_cast<std::size_t>(frame)) *
                 static_cast<std::size_t>(height) +
             static_cast<std::size_t>(y)) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x));
}

std::size_t hwc_video_index(int32_t frame, int32_t y, int32_t x, int32_t channel, int32_t height,
                            int32_t width) {
    return (((static_cast<std::size_t>(frame) * static_cast<std::size_t>(height) +
              static_cast<std::size_t>(y)) *
                 static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)) *
                3U +
            static_cast<std::size_t>(channel));
}

bool is_ncthw_refiner_shape(const std::vector<int64_t>& shape, int32_t height, int32_t width) {
    return shape.size() == 5U && shape[0] == 1 && shape[1] == 3 && shape[3] == height &&
           shape[4] == width;
}

bool is_cthw_refiner_shape(const std::vector<int64_t>& shape, int32_t height, int32_t width) {
    return shape.size() == 4U && shape[0] == 3 && shape[2] == height && shape[3] == width;
}

bool is_nthwc_refiner_shape(const std::vector<int64_t>& shape, int32_t height, int32_t width) {
    return shape.size() == 5U && shape[0] == 1 && shape[2] == height && shape[3] == width &&
           shape[4] == 3;
}

bool is_thwc_refiner_shape(const std::vector<int64_t>& shape, int32_t height, int32_t width) {
    return shape.size() == 4U && shape[1] == height && shape[2] == width && shape[3] == 3;
}

int32_t cthw_refiner_frame_count(const std::vector<int64_t>& shape, int32_t height, int32_t width) {
    if (is_ncthw_refiner_shape(shape, height, width))
        return static_cast<int32_t>(shape[2]);
    if (is_cthw_refiner_shape(shape, height, width))
        return static_cast<int32_t>(shape[1]);
    return 0;
}

int32_t hwc_refiner_frame_count(const std::vector<int64_t>& shape, int32_t height, int32_t width) {
    if (is_nthwc_refiner_shape(shape, height, width))
        return static_cast<int32_t>(shape[1]);
    if (is_thwc_refiner_shape(shape, height, width))
        return static_cast<int32_t>(shape[0]);
    return 0;
}

bool valid_refiner_frame_count(int32_t frames, int32_t output_frames) {
    if (frames <= 0)
        return false;
    if (frames != output_frames && frames != output_frames + 1)
        return false;
    return true;
}

bool refiner_raw_has_values(const std::vector<float>& raw, int32_t frames, int32_t height,
                            int32_t width) {
    if (raw.size() < static_cast<std::size_t>(3) * static_cast<std::size_t>(frames) *
                         static_cast<std::size_t>(height) * static_cast<std::size_t>(width))
        return false;
    return true;
}

void copy_cthw_refiner_pixels(const std::vector<float>& raw, int32_t frames, int32_t output_frames,
                              int32_t height, int32_t width, std::vector<float>& out) {
    const int32_t frame_offset = frames == output_frames + 1 ? 1 : 0;
    for (int32_t f = 0; f < output_frames; ++f) {
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const auto dst = hwc_video_index(f, y, x, 0, height, width);
                for (int32_t c = 0; c < 3; ++c) {
                    const auto src =
                        cthw_video_index(c, f + frame_offset, y, x, frames, height, width);
                    out[dst + static_cast<std::size_t>(c)] = normalize_refiner_cthw_pixel(raw[src]);
                }
            }
        }
    }
}

bool copy_cthw_refiner_frames(const std::vector<float>& raw, const std::vector<int64_t>& shape,
                              int32_t output_frames, int32_t height, int32_t width,
                              std::vector<float>& out) {
    const auto frames = cthw_refiner_frame_count(shape, height, width);
    if (!valid_refiner_frame_count(frames, output_frames))
        return false;
    if (!refiner_raw_has_values(raw, frames, height, width))
        return false;
    copy_cthw_refiner_pixels(raw, frames, output_frames, height, width, out);
    return true;
}

bool copy_hwc_refiner_frames(const std::vector<float>& raw, const std::vector<int64_t>& shape,
                             int32_t output_frames, int32_t height, int32_t width,
                             std::vector<float>& out) {
    const auto frames = hwc_refiner_frame_count(shape, height, width);
    if (!valid_refiner_frame_count(frames, output_frames))
        return false;
    const auto frame_stride =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * 3U;
    if (raw.size() < static_cast<std::size_t>(frames) * frame_stride)
        return false;

    const auto offset = frames == output_frames + 1 ? frame_stride : 0U;
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = normalize_refiner_pixel(raw[offset + i]);
    return true;
}

ImageResult decode_native_refiner_vae(ITrtModule& vae_decoder, const SanaWmStage1Latents& latents,
                                      const SanaWmRuntimeConfig& config) {
    if (!vae_decoder.ok())
        throw std::runtime_error("SANA-WM native refiner VAE decoder is not ready");
    const auto input_name =
        pick_input_name(vae_decoder, {"latents", "sample", "z"}, "refiner VAE decoder");
    std::vector<half_bits_t> latent16;
    TensorMap inputs;
    inputs[input_name] = make_model_tensor(
        latents.values, latent16, input_dtype_or(vae_decoder, input_name, DType::kBFloat16),
        {1, latents.channels, latents.frames, latents.height, latents.width});

    auto outputs = vae_decoder.forward(inputs);
    const auto it = find_output_tensor(outputs, {"output0", "video", "sample", "decoder_output"});
    if (it == outputs.end())
        throw std::runtime_error("SANA-WM native refiner VAE decoder output tensor not found");

    const int32_t output_frames = std::max(config.num_frames - 1, 1);
    const auto direct_count = static_cast<std::size_t>(output_frames) *
                              static_cast<std::size_t>(config.height) *
                              static_cast<std::size_t>(config.width) * 3U;
    const auto with_sink_count = static_cast<std::size_t>(config.num_frames) *
                                 static_cast<std::size_t>(config.height) *
                                 static_cast<std::size_t>(config.width) * 3U;
    auto raw = tensor_to_float_vector(
        it->second, it->second.numel() >= with_sink_count ? with_sink_count : direct_count,
        "refiner VAE decoder");
    const bool drop_sink = raw.size() >= with_sink_count;
    const auto frame_stride =
        static_cast<std::size_t>(config.height) * static_cast<std::size_t>(config.width) * 3U;

    ImageResult result;
    result.channels = 3;
    result.height = config.height;
    result.width = config.width;
    result.num_frames = output_frames;
    result.pixels.resize(direct_count, 0.0F);
    if (copy_cthw_refiner_frames(raw, it->second.shape, output_frames, config.height, config.width,
                                 result.pixels))
        return result;
    if (copy_hwc_refiner_frames(raw, it->second.shape, output_frames, config.height, config.width,
                                result.pixels))
        return result;
    const auto src_offset = drop_sink ? frame_stride : 0U;
    for (std::size_t i = 0; i < direct_count; ++i)
        result.pixels[i] = normalize_refiner_pixel(raw[src_offset + i]);
    return result;
}

ImageResult decode_native_refiner_vae_tiled(const std::vector<SanaWmVaeDecoderTile>& tiles,
                                            const SanaWmStage1Latents& latents,
                                            const SanaWmRuntimeConfig& config) {
    auto raw = decode_native_vae_tiled(tiles, latents, config, "refiner VAE decoder");
    log_float_stats("refiner VAE decoder tiled", raw.values);
    return image_result_from_refiner_cthw(raw, config);
}

ImageResult run_native_image_path(SanaWmNativeModules& modules,
                                  const std::shared_ptr<ITokenizer>& stage1_tokenizer,
                                  const std::shared_ptr<ITokenizer>& refiner_tokenizer,
                                  const SanaWmRuntimeConfig& config, const SanaWmRequest& request,
                                  const GenerateConfig& cfg, const std::string& prompt) {
    const std::string prompt_text = trim_ascii_whitespace(prompt);
    if (!cfg.no_refiner) {
        if (!modules.has_refiner() || !refiner_tokenizer) {
            throw std::runtime_error("SANA-WM native refiner execution requires refiner text, "
                                     "denoiser, VAE, and tokenizer. Pass --no_refiner only when "
                                     "intentionally running a stage1-only bundle.");
        }
    }
    auto latents =
        run_native_stage1_path(modules, stage1_tokenizer, config, request, cfg, prompt_text);
    if (!cfg.no_refiner) {
        auto refined = run_native_refiner(*modules.refiner_text_encoder,
                                          modules.refiner_text_connector.get(),
                                          *modules.refiner_denoiser, *refiner_tokenizer, latents,
                                          config, cfg, prompt_text);
        if (!modules.refiner_vae_decoder_tiles.empty())
            return decode_native_refiner_vae_tiled(modules.refiner_vae_decoder_tiles, refined,
                                                   config);
        if (!modules.refiner_vae_decoder) {
            throw std::runtime_error("SANA-WM native refiner execution requires a refiner VAE "
                                     "decoder module");
        }
        return decode_native_refiner_vae(*modules.refiner_vae_decoder, refined, config);
    }
    if (!modules.vae_decoder_tiles.empty())
        return decode_native_sana_vae_tiled(modules.vae_decoder_tiles, latents, config);
    if (!modules.vae_decoder) {
        throw std::runtime_error("SANA-WM native TensorRT execution requires a VAE decoder module");
    }
    return decode_native_sana_vae(*modules.vae_decoder, latents, config);
}

} // namespace

SanaWmRuntimeConfig parse_sana_wm_config(const std::string& config_json) {
    SanaWmRuntimeConfig cfg;
    cfg.hf_id = extract_json_string(config_json, "sana_wm_hf_id", cfg.hf_id);
    cfg.default_image = extract_json_string(config_json, "sana_wm_default_image", "");
    cfg.action = extract_json_string(config_json, "sana_wm_action", cfg.action);
    cfg.translation_speed =
        extract_json_float(config_json, "sana_wm_translation_speed", cfg.translation_speed);
    cfg.rotation_speed_deg =
        extract_json_float(config_json, "sana_wm_rotation_speed_deg", cfg.rotation_speed_deg);
    cfg.num_frames = extract_json_int(config_json, "video_num_frames", cfg.num_frames);
    cfg.height = extract_json_int(config_json, "video_height", cfg.height);
    cfg.width = extract_json_int(config_json, "video_width", cfg.width);
    cfg.fps = extract_json_int(config_json, "fps", cfg.fps);
    cfg.num_steps = extract_json_int(config_json, "num_inference_steps", cfg.num_steps);
    cfg.cfg_scale = extract_json_float(config_json, "guidance_scale", cfg.cfg_scale);
    cfg.flow_shift = extract_json_float(config_json, "flow_shift", cfg.flow_shift);
    cfg.seed = extract_json_int(config_json, "seed", cfg.seed);
    cfg.vae_latent_dim = extract_json_int(config_json, "vae_latent_dim", cfg.vae_latent_dim);
    cfg.vae_time_stride = extract_json_int(config_json, "vae_time_stride", cfg.vae_time_stride);
    cfg.vae_spatial_stride = extract_json_int(
        config_json, "vae_spatial_stride",
        extract_json_int(config_json, "vae_downsample_rate", cfg.vae_spatial_stride));
    cfg.vae_use_framewise_decoding =
        extract_json_bool(config_json, "vae_use_framewise_decoding", cfg.vae_use_framewise_decoding);
    cfg.vae_use_spatial_tiling =
        extract_json_bool(config_json, "vae_use_spatial_tiling", cfg.vae_use_spatial_tiling);
    cfg.vae_tile_sample_min_height = extract_json_int(
        config_json, "vae_tile_sample_min_height", cfg.vae_tile_sample_min_height);
    cfg.vae_tile_sample_min_width = extract_json_int(
        config_json, "vae_tile_sample_min_width", cfg.vae_tile_sample_min_width);
    cfg.vae_tile_sample_stride_height = extract_json_int(
        config_json, "vae_tile_sample_stride_height", cfg.vae_tile_sample_stride_height);
    cfg.vae_tile_sample_stride_width = extract_json_int(
        config_json, "vae_tile_sample_stride_width", cfg.vae_tile_sample_stride_width);
    cfg.vae_tile_sample_min_num_frames = extract_json_int(
        config_json, "vae_tile_sample_min_num_frames", cfg.vae_tile_sample_min_num_frames);
    cfg.vae_tile_sample_stride_num_frames = extract_json_int(
        config_json, "vae_tile_sample_stride_num_frames", cfg.vae_tile_sample_stride_num_frames);
    cfg.text_encoder_max_length =
        extract_json_int(config_json, "text_encoder_max_length", cfg.text_encoder_max_length);
    cfg.refiner_text_encoder_max_length =
        extract_json_int(config_json, "sana_wm_refiner_text_max_length",
                         cfg.refiner_text_encoder_max_length);
    cfg.text_encoder_dim =
        extract_json_int(config_json, "sana_wm_dit_text_embed_dim",
                         extract_json_int(config_json, "text_encoder_dim", cfg.text_encoder_dim));
    cfg.chi_prompt = extract_json_string(config_json, "sana_wm_chi_prompt", cfg.chi_prompt);
    cfg.default_intrinsics =
        extract_json_float_array(config_json, "sana_wm_default_intrinsics", 9U);
    return cfg;
}

std::string sana_wm_make_conditioning_prompt(const std::string& prompt,
                                             const std::string& chi_prompt) {
    if (chi_prompt.empty())
        return prompt;
    return chi_prompt + prompt;
}

std::vector<SanaWmPose> sana_wm_action_to_c2w(const std::string& action, float translation_speed,
                                              float rotation_speed_deg) {
    if (translation_speed < 0.0F)
        throw std::invalid_argument("SANA-WM translation speed must be non-negative");
    if (rotation_speed_deg < 0.0F)
        throw std::invalid_argument("SANA-WM rotation speed must be non-negative");

    const auto per_frame = parse_action_string(action);
    const float rotate_rad = rotation_speed_deg * kPi / 180.0F;
    const float pitch_limit_rad = kDefaultPitchLimitDeg * kPi / 180.0F;

    Mat3 r = identity3();
    Vec3 t{0.0F, 0.0F, 0.0F};
    float current_pitch = 0.0F;

    std::vector<SanaWmPose> poses;
    poses.reserve(per_frame.size() + 1);
    poses.push_back(make_pose(r, t));

    for (const auto& keys : per_frame) {
        float next_pitch = current_pitch;
        const float pitch_delta =
            limited_pitch_delta(keys, rotate_rad, current_pitch, pitch_limit_rad, next_pitch);
        const float yaw_delta = static_cast<float>(key_direction(keys, 'l', 'j')) * rotate_rad;
        const Mat3 r_new = matmul3(matmul3(rot_y(yaw_delta), r), rot_x(pitch_delta));
        const Vec3 move = camera_ground_motion(keys, r_new, translation_speed);

        r = r_new;
        current_pitch = next_pitch;
        t[0] += move[0];
        t[1] += move[1];
        t[2] += move[2];
        poses.push_back(make_pose(r, t));
    }

    return poses;
}

std::vector<SanaWmPose> sana_wm_row_major_c2w_to_poses(const std::vector<float>& c2w_values) {
    if (c2w_values.empty() || c2w_values.size() % 16U != 0U)
        throw std::invalid_argument("SANA-WM camera poses must be flat row-major [F,4,4]");

    const auto count = c2w_values.size() / 16U;
    std::vector<SanaWmPose> poses(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::copy_n(c2w_values.data() + i * 16U, 16U, poses[i].c2w.begin());
    }
    return poses;
}

std::vector<SanaWmIntrinsics> sana_wm_expand_intrinsics(const std::vector<float>& values,
                                                        int32_t num_frames) {
    if (num_frames <= 0)
        throw std::invalid_argument("SANA-WM intrinsics frame count must be positive");

    auto from_four = [](const float* v) -> SanaWmIntrinsics { return {v[0], v[1], v[2], v[3]}; };
    auto from_matrix = [](const float* v) -> SanaWmIntrinsics { return {v[0], v[4], v[2], v[5]}; };

    if (values.size() == 4U)
        return std::vector<SanaWmIntrinsics>(static_cast<std::size_t>(num_frames),
                                             from_four(values.data()));
    if (values.size() == 9U)
        return std::vector<SanaWmIntrinsics>(static_cast<std::size_t>(num_frames),
                                             from_matrix(values.data()));

    const auto frames = static_cast<std::size_t>(num_frames);
    if (values.size() == frames * 9U) {
        std::vector<SanaWmIntrinsics> out(frames);
        for (std::size_t i = 0; i < frames; ++i)
            out[i] = from_matrix(values.data() + i * 9U);
        return out;
    }

    throw std::invalid_argument(
        "SANA-WM intrinsics must be (fx,fy,cx,cy), row-major [3,3], or row-major [F,3,3]");
}

SanaWmResizeCropPlan sana_wm_make_resize_crop_plan(int32_t src_width, int32_t src_height,
                                                   int32_t target_height, int32_t target_width) {
    if (src_width <= 0 || src_height <= 0 || target_height <= 0 || target_width <= 0)
        throw std::invalid_argument("SANA-WM resize/crop dimensions must be positive");

    const double scale =
        std::max(static_cast<double>(target_height) / static_cast<double>(src_height),
                 static_cast<double>(target_width) / static_cast<double>(src_width));
    const int32_t resized_width =
        std::max(target_width, python_round_to_int(static_cast<double>(src_width) * scale));
    const int32_t resized_height =
        std::max(target_height, python_round_to_int(static_cast<double>(src_height) * scale));

    SanaWmResizeCropPlan plan;
    plan.src_width = src_width;
    plan.src_height = src_height;
    plan.resized_width = resized_width;
    plan.resized_height = resized_height;
    plan.crop_left = (resized_width - target_width) / 2;
    plan.crop_top = (resized_height - target_height) / 2;
    plan.target_width = target_width;
    plan.target_height = target_height;
    return plan;
}

SanaWmIntrinsics sana_wm_transform_intrinsics_for_crop(const SanaWmIntrinsics& intrinsics,
                                                       const SanaWmResizeCropPlan& plan) {
    if (plan.src_width <= 0 || plan.src_height <= 0)
        throw std::invalid_argument("SANA-WM intrinsics transform requires a valid crop plan");
    const float sx = static_cast<float>(plan.resized_width) / static_cast<float>(plan.src_width);
    const float sy = static_cast<float>(plan.resized_height) / static_cast<float>(plan.src_height);
    return {
        intrinsics.fx * sx,
        intrinsics.fy * sy,
        intrinsics.cx * sx - static_cast<float>(plan.crop_left),
        intrinsics.cy * sy - static_cast<float>(plan.crop_top),
    };
}

SanaWmPreprocessedImage sana_wm_resize_and_center_crop(const std::vector<float>& src_hwc,
                                                       int32_t src_width, int32_t src_height,
                                                       int32_t target_height,
                                                       int32_t target_width) {
    SanaWmPreprocessedImage out;
    out.plan = sana_wm_make_resize_crop_plan(src_width, src_height, target_height, target_width);

    const auto expected_src =
        static_cast<std::size_t>(src_width) * static_cast<std::size_t>(src_height) * 3U;
    if (src_hwc.size() != expected_src)
        throw std::invalid_argument("SANA-WM source image buffer size does not match dimensions");

    std::vector<float> resized;
    if (!resize_lanczos3_hwc(src_hwc, src_width, src_height, out.plan.resized_width,
                             out.plan.resized_height, resized)) {
        return out;
    }

    out.pixels_hwc.assign(static_cast<std::size_t>(target_width) *
                              static_cast<std::size_t>(target_height) * 3U,
                          0.0F);
    for (int32_t y = 0; y < target_height; ++y) {
        const int32_t src_y = out.plan.crop_top + y;
        const float* src_row =
            resized.data() +
            (static_cast<std::size_t>(src_y) * static_cast<std::size_t>(out.plan.resized_width) +
             static_cast<std::size_t>(out.plan.crop_left)) *
                3U;
        float* dst_row = out.pixels_hwc.data() +
                         static_cast<std::size_t>(y) * static_cast<std::size_t>(target_width) * 3U;
        std::copy_n(src_row, static_cast<std::size_t>(target_width) * 3U, dst_row);
    }
    out.ok = true;
    return out;
}

SanaWmVaeInputImage sana_wm_prepare_vae_input_image(const std::vector<float>& src_hwc,
                                                    int32_t src_width, int32_t src_height,
                                                    int32_t target_height, int32_t target_width) {
    SanaWmVaeInputImage out;
    auto cropped =
        sana_wm_resize_and_center_crop(src_hwc, src_width, src_height, target_height, target_width);
    out.plan = cropped.plan;
    out.height = target_height;
    out.width = target_width;
    if (!cropped.ok)
        return out;

    out.pixels_chw.assign(static_cast<std::size_t>(target_height) *
                              static_cast<std::size_t>(target_width) * 3U,
                          0.0F);
    for (int32_t y = 0; y < target_height; ++y) {
        for (int32_t x = 0; x < target_width; ++x) {
            const auto src = (static_cast<std::size_t>(y) * static_cast<std::size_t>(target_width) +
                              static_cast<std::size_t>(x)) *
                             3U;
            for (int32_t c = 0; c < 3; ++c) {
                const float pixel =
                    std::max(0.0F, std::min(1.0F, cropped.pixels_hwc[src + static_cast<std::size_t>(c)]));
                out.pixels_chw[chw_index(c, y, x, target_height, target_width)] =
                    pixel * 2.0F - 1.0F;
            }
        }
    }
    out.ok = true;
    return out;
}

SanaWmCameraConditions
sana_wm_prepare_camera_conditions(const std::vector<SanaWmPose>& c2w,
                                  const std::vector<SanaWmIntrinsics>& intrinsics,
                                  int32_t target_height, int32_t target_width,
                                  int32_t vae_time_stride, int32_t vae_spatial_stride) {
    validate_camera_condition_inputs(c2w, intrinsics, target_height, target_width, vae_time_stride,
                                     vae_spatial_stride);

    const int32_t num_frames = static_cast<int32_t>(c2w.size());
    const int32_t latent_h = target_height / vae_spatial_stride;
    const int32_t latent_w = target_width / vae_spatial_stride;
    if (latent_h <= 0 || latent_w <= 0)
        throw std::invalid_argument("SANA-WM latent camera dimensions must be positive");

    const int32_t latent_frames = (num_frames - 1) / vae_time_stride + 1;
    const auto poses = relative_poses_from_first(c2w);

    std::vector<SanaWmIntrinsics> latent_intrinsics;
    latent_intrinsics.reserve(c2w.size());
    for (std::size_t i = 0; i < c2w.size(); ++i) {
        latent_intrinsics.push_back(scale_intrinsics_to_latent(
            intrinsics_at(intrinsics, i), latent_h, latent_w, target_height, target_width));
    }

    SanaWmCameraConditions out;
    out.num_frames = num_frames;
    out.latent_frames = latent_frames;
    out.latent_height = latent_h;
    out.latent_width = latent_w;
    out.vae_time_stride = vae_time_stride;
    out.vae_spatial_stride = vae_spatial_stride;
    out.time_indices = camera_time_indices(num_frames, latent_frames, vae_time_stride);
    out.raymap_width = 20;
    out.chunk_plucker_channels = vae_time_stride * 6;

    out.raymap.assign(out.time_indices.size() * static_cast<std::size_t>(out.raymap_width), 0.0F);
    for (std::size_t row = 0; row < out.time_indices.size(); ++row) {
        const auto pose_idx = static_cast<std::size_t>(out.time_indices[row]);
        pack_raymap_row(out.raymap, row, poses[pose_idx], latent_intrinsics[pose_idx]);
    }

    const int32_t chunk_count = static_cast<int32_t>(out.time_indices.size());
    out.raymats.assign(static_cast<std::size_t>(chunk_count) *
                           static_cast<std::size_t>(latent_h) *
                           static_cast<std::size_t>(latent_w) * 16U,
                       0.0F);
    out.raymats_inv.assign(static_cast<std::size_t>(chunk_count) *
                               static_cast<std::size_t>(latent_h) *
                               static_cast<std::size_t>(latent_w) * 16U,
                           0.0F);
    for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
        const auto pose_idx = static_cast<std::size_t>(out.time_indices[chunk]);
        for (int32_t y = 0; y < latent_h; ++y) {
            for (int32_t x = 0; x < latent_w; ++x) {
                pack_ucpe_raymat(out.raymats, out.raymats_inv, chunk, y, x, poses[pose_idx],
                                 latent_intrinsics[pose_idx], latent_h, latent_w);
            }
        }
    }

    out.chunk_plucker.assign(static_cast<std::size_t>(out.chunk_plucker_channels) *
                                 static_cast<std::size_t>(chunk_count) *
                                 static_cast<std::size_t>(latent_h) *
                                 static_cast<std::size_t>(latent_w),
                             0.0F);
    for (int32_t chunk = 0; chunk < chunk_count; ++chunk) {
        pack_chunk_plucker(out.chunk_plucker, poses, latent_intrinsics, chunk,
                           out.time_indices[static_cast<std::size_t>(chunk)], vae_time_stride,
                           latent_h, latent_w, chunk_count);
    }
    return out;
}

SanaWmStage1Latents sana_wm_prepare_stage1_latents(const std::vector<float>& first_frame_chw,
                                                   const std::vector<float>& initial_latents_cthw,
                                                   int32_t channels, int32_t latent_frames,
                                                   int32_t latent_height, int32_t latent_width,
                                                   uint64_t seed) {
    const auto expected_total =
        stage1_latent_count(channels, latent_frames, latent_height, latent_width);
    std::vector<float> values;
    if (initial_latents_cthw.empty()) {
        values = sample_stage1_noise(expected_total, seed);
    } else {
        if (initial_latents_cthw.size() != expected_total) {
            throw std::invalid_argument("SANA-WM initial latent size does not match [C,T,H,W]");
        }
        values = initial_latents_cthw;
    }

    overwrite_first_latent_frame(values, first_frame_chw, channels, latent_frames, latent_height,
                                 latent_width);

    return {std::move(values), channels, latent_frames, latent_height, latent_width};
}

bool SanaWmNativeModules::has_any() const {
    return text_encoder || stage1_denoiser || vae_encoder || vae_decoder ||
           !vae_decoder_tiles.empty() || refiner_text_encoder || refiner_text_connector ||
           refiner_denoiser || refiner_vae_decoder || !refiner_vae_decoder_tiles.empty();
}

bool SanaWmNativeModules::has_stage1() const {
    return text_encoder && stage1_denoiser && vae_encoder;
}

bool SanaWmNativeModules::has_refiner() const {
    return refiner_text_encoder && refiner_denoiser &&
           (refiner_vae_decoder || !refiner_vae_decoder_tiles.empty());
}

SanaWmPipeline::SanaWmPipeline(SanaWmRuntimeConfig config, SanaWmNativeModules native_modules,
                               std::shared_ptr<ITokenizer> stage1_tokenizer,
                               std::shared_ptr<ITokenizer> refiner_tokenizer)
    : config_(std::move(config)), native_modules_(std::move(native_modules)),
      stage1_tokenizer_(std::move(stage1_tokenizer)),
      refiner_tokenizer_(refiner_tokenizer ? std::move(refiner_tokenizer) : stage1_tokenizer_) {}

ImageResult SanaWmPipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    SanaWmRuntimeConfig runtime_config = config_;
    if (cfg.fps > 0)
        runtime_config.fps = cfg.fps;
    if (cfg.flow_shift >= 0.0F)
        runtime_config.flow_shift = cfg.flow_shift;

    const auto request = resolve_request(runtime_config, cfg);
    if (request.image_path.empty())
        throw std::runtime_error("SANA-WM generation requires --image");
    if (prompt.empty())
        throw std::runtime_error("SANA-WM generation requires a non-empty prompt");
    if (native_modules_.has_any()) {
        validate_native_module_set(native_modules_);
        return run_native_image_path(native_modules_, stage1_tokenizer_, refiner_tokenizer_,
                                     runtime_config, request, cfg, prompt);
    }
    throw std::runtime_error("SANA-WM pure C++ execution requires native TensorRT plan sections");
}

} // namespace trtmc
