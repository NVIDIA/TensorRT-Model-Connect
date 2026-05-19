#include "runtime/models/sana_wm/pipeline.h"

#include "runtime/domains/diffusion/diffusion_scheduler_helpers.h"
#include "stb_image_resize2.h"
#include "trtmc/trtmc_io.hpp"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
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

std::string make_temp_root() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = std::filesystem::temp_directory_path() /
                ("trtmc_sana_wm_" + std::to_string(static_cast<long long>(getpid())) + "_" +
                 std::to_string(static_cast<long long>(now)));
    std::filesystem::create_directories(path);
    return path.string();
}

std::vector<std::filesystem::path> sorted_frame_paths(const std::filesystem::path& frames_dir) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::is_directory(frames_dir))
        return paths;
    for (const auto& entry : std::filesystem::directory_iterator(frames_dir)) {
        if (!entry.is_regular_file())
            continue;
        const auto name = entry.path().filename().string();
        if (name.rfind("frame_", 0) == 0 && entry.path().extension() == ".png")
            paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

std::string stderr_summary(const std::string& stderr_text, const std::vector<char>& stdout_data) {
    std::string stdout_text(stdout_data.begin(), stdout_data.end());
    std::ostringstream oss;
    if (!stderr_text.empty())
        oss << stderr_text;
    if (!stdout_text.empty()) {
        if (!stderr_text.empty())
            oss << "\n";
        oss << stdout_text;
    }
    auto out = oss.str();
    constexpr std::size_t kMax = 4000;
    if (out.size() > kMax)
        return out.substr(out.size() - kMax);
    return out;
}

struct SanaWmPaths {
    std::string temp_root;
    std::filesystem::path output_dir;
    std::filesystem::path frames_dir;
    std::filesystem::path meta_json;
};

SanaWmPaths make_invocation_paths() {
    SanaWmPaths paths;
    paths.temp_root = make_temp_root();
    paths.output_dir = std::filesystem::path(paths.temp_root) / "official_output";
    paths.frames_dir = std::filesystem::path(paths.temp_root) / "frames";
    paths.meta_json = std::filesystem::path(paths.temp_root) / "meta.json";
    std::filesystem::create_directories(paths.output_dir);
    std::filesystem::create_directories(paths.frames_dir);
    return paths;
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

std::vector<float> sample_stage1_noise(std::size_t count, uint64_t seed) {
    std::vector<float> values(count);
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (float& value : values)
        value = dist(gen);
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
        return out;
    }
    if (tensor.dtype == DType::kFloat16) {
        const auto* src = static_cast<const half_bits_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = fp16_to_fp32(src[i]);
        return out;
    }
    if (tensor.dtype == DType::kBFloat16) {
        const auto* src = static_cast<const half_bits_t*>(tensor.data);
        for (std::size_t i = 0; i < count; ++i)
            out[i] = bf16_to_fp32(src[i]);
        return out;
    }
    throw std::runtime_error("SANA-WM " + label + " output tensor has unsupported dtype");
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

struct SanaWmRequest {
    std::string python;
    std::string image_path;
    std::string action;
    float translation_speed;
    float rotation_speed_deg;
    int32_t num_frames;
};

SanaWmRequest resolve_request(const SanaWmRuntimeConfig& config, const GenerateConfig& cfg,
                              const std::string& hf_python) {
    return {
        hf_python.empty() ? "python3" : hf_python,
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

struct SanaWmRefinerText {
    std::vector<float> values;
    std::vector<int64_t> shape;
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

std::vector<float> stage1_velocity_from_model_output(const std::vector<float>& model_output,
                                                     std::size_t latent_count, float cfg_scale) {
    std::vector<float> velocity(latent_count, 0.0F);
    if (cfg_scale <= 1.0F) {
        if (model_output.size() != latent_count)
            throw std::runtime_error("SANA-WM Stage-1 denoiser output size mismatch");
        for (std::size_t i = 0; i < latent_count; ++i)
            velocity[i] = -model_output[i];
        return velocity;
    }
    if (model_output.size() != latent_count * 2U)
        throw std::runtime_error("SANA-WM Stage-1 CFG denoiser output size mismatch");
    for (std::size_t i = 0; i < latent_count; ++i) {
        const float uncond = model_output[i];
        const float cond = model_output[latent_count + i];
        velocity[i] = -(uncond + cfg_scale * (cond - uncond));
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
    std::vector<half_bits_t> input16;
    TensorMap inputs;
    inputs[input_name] = make_model_tensor(first_frame.pixels_chw, input16, input_dtype,
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

std::vector<int32_t> attention_mask_from_tokens(const std::vector<int32_t>& ids) {
    std::vector<int32_t> mask(ids.size(), 0);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != 0)
            mask[i] = 1;
    }
    return mask;
}

std::vector<float> run_native_text_encoder(ITrtModule& text_encoder,
                                           const std::vector<int32_t>& input_ids,
                                           const std::vector<int32_t>& attention_mask,
                                           int32_t text_dim, const char* label) {
    if (!text_encoder.ok())
        throw std::runtime_error(std::string("SANA-WM native ") + label +
                                 " text encoder is not ready");

    const int64_t seq_len = static_cast<int64_t>(input_ids.size());
    TensorMap inputs;
    inputs["input_ids"] =
        Tensor{const_cast<int32_t*>(input_ids.data()), {1, seq_len}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{const_cast<int32_t*>(attention_mask.data()), {1, seq_len}, DType::kInt32};

    auto outputs = text_encoder.forward(inputs);
    const auto it = find_output_tensor(
        outputs, {"last_hidden_state", "hidden_states", "text_embeddings", "output0"});
    if (it == outputs.end())
        throw std::runtime_error(std::string("SANA-WM native ") + label +
                                 " text encoder output tensor not found");
    const auto count =
        static_cast<std::size_t>(input_ids.size()) * static_cast<std::size_t>(text_dim);
    return tensor_to_float_vector(it->second, count, label);
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
    auto cond_full = run_native_text_encoder(text_encoder, cond_ids, cond_mask_full,
                                             config.text_encoder_dim, "cond");

    auto neg_ids = tokenize_fixed(tokenizer, negative_prompt, config.text_encoder_max_length);
    auto neg_mask = attention_mask_from_tokens(neg_ids);
    auto neg = run_native_text_encoder(text_encoder, neg_ids, neg_mask, config.text_encoder_dim,
                                       "negative");

    return {select_stage1_text_window(cond_full, cond_len, config.text_encoder_max_length,
                                      config.text_encoder_dim),
            std::move(neg),
            select_stage1_mask_window(cond_mask_full, config.text_encoder_max_length),
            std::move(neg_mask)};
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
                           const SanaWmRuntimeConfig& config, float timestep, float cfg_scale) {
    if (!denoiser.ok())
        throw std::runtime_error("SANA-WM native Stage-1 denoiser is not ready");

    const bool do_cfg = cfg_scale > 1.0F;
    const int32_t batch = do_cfg ? 2 : 1;
    auto latent_input = repeat_batch(latents.values, batch);
    auto text_input = stage1_text_input(text, do_cfg);
    auto mask_input = stage1_mask_input(text, do_cfg);
    auto raymap_input = repeat_batch(camera.raymap, batch);
    auto plucker_input = repeat_batch(camera.chunk_plucker, batch);
    auto timestep_input = stage1_frame_timestep(batch, latents.frames, timestep);

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
    return tensor_to_float_vector(it->second, count, "Stage-1 denoiser");
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
                                                       timestep, cfg_scale);
        auto velocity =
            stage1_velocity_from_model_output(model_output, latents.values.size(), cfg_scale);
        scheduler.step(velocity.data(), latents.values.data(), next.data(), latents.values.size(),
                       step);
        keep_stage1_anchor_frame(next, latents.values, latents.channels, latents.frames,
                                 latents.height, latents.width);
        latents.values.swap(next);
    }
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

SanaWmRefinerText run_native_refiner_text_encoder(ITrtModule& text_encoder,
                                                  const ITokenizer& tokenizer,
                                                  const std::string& prompt) {
    if (!text_encoder.ok())
        throw std::runtime_error("SANA-WM native refiner text encoder is not ready");
    constexpr int32_t kRefinerTextLength = 256;
    auto input_ids = tokenize_fixed(tokenizer, prompt, kRefinerTextLength);
    auto attention_mask = attention_mask_from_tokens(input_ids);
    TensorMap inputs;
    inputs["input_ids"] = Tensor{input_ids.data(), {1, kRefinerTextLength}, DType::kInt32};
    inputs["attention_mask"] =
        Tensor{attention_mask.data(), {1, kRefinerTextLength}, DType::kInt32};

    auto outputs = text_encoder.forward(inputs);
    const auto it =
        find_output_tensor(outputs, {"video_encoding", "encoder_hidden_states", "text_embeddings",
                                     "last_hidden_state", "output0"});
    if (it == outputs.end())
        throw std::runtime_error("SANA-WM native refiner text output tensor not found");
    auto shape = it->second.shape;
    if (shape.empty())
        shape = {1, static_cast<int64_t>(it->second.numel())};
    return {tensor_to_float_vector(it->second, it->second.numel(), "refiner text encoder"),
            std::move(shape)};
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
    const auto sigma_name =
        pick_input_name(denoiser, {"sigma", "timestep"}, "refiner denoiser sigma");

    std::vector<half_bits_t> latent16;
    std::vector<half_bits_t> clean16;
    std::vector<half_bits_t> positions16;
    std::vector<half_bits_t> text16;
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
        out[i] = sample[i] + velocity * dt;
    }
    return out;
}

SanaWmStage1Latents run_native_refiner(ITrtModule& text_encoder, ITrtModule& denoiser,
                                       const ITokenizer& tokenizer,
                                       const SanaWmStage1Latents& stage1,
                                       const SanaWmRuntimeConfig& config, const GenerateConfig& cfg,
                                       const std::string& prompt) {
    constexpr int32_t kSinkFrames = 1;
    if (stage1.frames <= kSinkFrames)
        throw std::runtime_error("SANA-WM native refiner requires more than one latent frame");
    const int32_t current_frames = stage1.frames - kSinkFrames;
    const auto text = run_native_refiner_text_encoder(text_encoder, tokenizer, prompt);
    auto sink = patchify_refiner_latents(stage1.values, stage1.channels, 0, kSinkFrames,
                                         stage1.frames, stage1.height, stage1.width);
    auto current_clean =
        patchify_refiner_latents(stage1.values, stage1.channels, kSinkFrames, current_frames,
                                 stage1.frames, stage1.height, stage1.width);
    auto noise = sample_stage1_noise(current_clean.size(),
                                     cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed)
                                                   : static_cast<uint64_t>(config.seed));
    std::vector<float> current(current_clean.size(), 0.0F);
    for (std::size_t i = 0; i < current.size(); ++i)
        current[i] = (1.0F - kRefinerSigmas[0]) * current_clean[i] + kRefinerSigmas[0] * noise[i];

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
        current = refiner_euler_step(current, pred, kRefinerSigmas[i], kRefinerSigmas[i + 1U]);
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
    return modules.refiner_text_encoder || modules.refiner_denoiser || modules.refiner_vae_decoder;
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
                                     "refiner module set: refiner text encoder, refiner denoiser, "
                                     "and refiner VAE decoder");
        }
        return;
    }
    if (!modules.vae_decoder) {
        throw std::runtime_error("SANA-WM native TensorRT execution requires a VAE decoder module "
                                 "when no native refiner is bundled");
    }
}

ImageResult decode_native_sana_vae(ITrtModule& vae_decoder, const SanaWmStage1Latents& latents,
                                   const SanaWmRuntimeConfig& config) {
    if (!vae_decoder.ok())
        throw std::runtime_error("SANA-WM native VAE decoder is not ready");

    const auto input_name = pick_input_name(vae_decoder, {"latents", "sample", "z"}, "VAE decoder");
    std::vector<half_bits_t> latent16;
    TensorMap inputs;
    inputs[input_name] = make_model_tensor(
        latents.values, latent16, input_dtype_or(vae_decoder, input_name, DType::kBFloat16),
        {1, latents.channels, latents.frames, latents.height, latents.width});

    auto outputs = vae_decoder.forward(inputs);
    const auto it = find_output_tensor(outputs, {"output0", "sample", "decoder_output", "video"});
    if (it == outputs.end())
        throw std::runtime_error("SANA-WM native VAE decoder output tensor not found");

    const auto raw_count =
        static_cast<std::size_t>(3) * static_cast<std::size_t>(config.num_frames) *
        static_cast<std::size_t>(config.height) * static_cast<std::size_t>(config.width);
    auto raw = tensor_to_float_vector(it->second, raw_count, "VAE decoder");

    ImageResult result;
    result.channels = 3;
    result.height = config.height;
    result.width = config.width;
    result.num_frames = config.num_frames;
    result.pixels.resize(raw_count, 0.0F);
    for (int32_t t = 0; t < config.num_frames; ++t) {
        for (int32_t y = 0; y < config.height; ++y) {
            for (int32_t x = 0; x < config.width; ++x) {
                for (int32_t c = 0; c < 3; ++c) {
                    const auto src = (((static_cast<std::size_t>(c) *
                                            static_cast<std::size_t>(config.num_frames) +
                                        static_cast<std::size_t>(t)) *
                                           static_cast<std::size_t>(config.height) +
                                       static_cast<std::size_t>(y)) *
                                          static_cast<std::size_t>(config.width) +
                                      static_cast<std::size_t>(x));
                    const auto dst =
                        (((static_cast<std::size_t>(t) * static_cast<std::size_t>(config.height) +
                           static_cast<std::size_t>(y)) *
                              static_cast<std::size_t>(config.width) +
                          static_cast<std::size_t>(x)) *
                             3U +
                         static_cast<std::size_t>(c));
                    result.pixels[dst] = std::max(0.0F, std::min(1.0F, raw[src] * 0.5F + 0.5F));
                }
            }
        }
    }
    return result;
}

float normalize_refiner_pixel(float value) {
    if (value > 1.0F)
        return std::max(0.0F, std::min(1.0F, value / 255.0F));
    return std::max(0.0F, std::min(1.0F, value));
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
    const auto src_offset = drop_sink ? frame_stride : 0U;
    for (std::size_t i = 0; i < direct_count; ++i)
        result.pixels[i] = normalize_refiner_pixel(raw[src_offset + i]);
    return result;
}

ImageResult run_native_image_path(SanaWmNativeModules& modules,
                                  const std::shared_ptr<ITokenizer>& tokenizer,
                                  const SanaWmRuntimeConfig& config, const SanaWmRequest& request,
                                  const GenerateConfig& cfg, const std::string& prompt) {
    auto latents = run_native_stage1_path(modules, tokenizer, config, request, cfg, prompt);
    if (has_any_refiner_module(modules)) {
        if (!modules.has_refiner() || !tokenizer) {
            throw std::runtime_error("SANA-WM native refiner execution requires refiner text, "
                                     "denoiser, VAE, and tokenizer");
        }
        auto refined = run_native_refiner(*modules.refiner_text_encoder, *modules.refiner_denoiser,
                                          *tokenizer, latents, config, cfg, prompt);
        return decode_native_refiner_vae(*modules.refiner_vae_decoder, refined, config);
    }
    if (!modules.vae_decoder) {
        throw std::runtime_error("SANA-WM native TensorRT execution requires a VAE decoder module");
    }
    return decode_native_sana_vae(*modules.vae_decoder, latents, config);
}

std::vector<std::string> build_bridge_argv(const SanaWmRuntimeConfig& config,
                                           const SanaWmRequest& request, const SanaWmPaths& paths,
                                           const std::string& prompt) {
    std::vector<std::string> argv = {
        request.python,
        "-m",
        "tensorrt_model_connect.sana_wm_bridge",
        "--hf-id",
        config.hf_id,
        "--image",
        request.image_path,
        "--prompt-text",
        prompt,
        "--action",
        request.action,
        "--translation-speed",
        std::to_string(request.translation_speed),
        "--rotation-speed-deg",
        std::to_string(request.rotation_speed_deg),
        "--num-frames",
        std::to_string(request.num_frames),
        "--output-dir",
        paths.output_dir.string(),
        "--frames-dir",
        paths.frames_dir.string(),
        "--meta-json",
        paths.meta_json.string(),
    };
    if (!config.script_path.empty()) {
        argv.push_back("--sana-script");
        argv.push_back(config.script_path);
    }
    if (config.require_official_script)
        argv.push_back("--no-diffusers-fallback");
    return argv;
}

void run_bridge_or_throw(ISubprocessRunner& runner, const std::vector<std::string>& argv) {
    std::vector<char> stdout_data;
    std::string stderr_data;
    const int rc = runner.run(argv, nullptr, 0, stdout_data, stderr_data);
    if (rc != 0) {
        throw std::runtime_error("SANA-WM Python bridge failed (rc=" + std::to_string(rc) +
                                 "): " + stderr_summary(stderr_data, stdout_data));
    }
}

ImageResult load_frame_result(const std::filesystem::path& frames_dir) {
    const auto frames = sorted_frame_paths(frames_dir);
    if (frames.empty()) {
        throw std::runtime_error("SANA-WM Python bridge produced no frame_*.png files under " +
                                 frames_dir.string());
    }

    ImageResult result;
    result.channels = 3;
    result.num_frames = static_cast<int32_t>(frames.size());
    for (const auto& frame_path : frames) {
        auto image = io::read_image(frame_path.string());
        if (image.empty())
            throw std::runtime_error("Failed to read SANA-WM frame: " + frame_path.string());
        if (result.height == 0) {
            result.height = image.height;
            result.width = image.width;
        } else if (result.height != image.height || result.width != image.width) {
            throw std::runtime_error("SANA-WM frame dimensions changed within one output");
        }
        result.pixels.insert(result.pixels.end(), image.pixels.begin(), image.pixels.end());
    }
    return result;
}

} // namespace

SanaWmRuntimeConfig parse_sana_wm_config(const std::string& config_json) {
    SanaWmRuntimeConfig cfg;
    cfg.hf_id = extract_json_string(config_json, "sana_wm_hf_id", cfg.hf_id);
    cfg.script_path = extract_json_string(config_json, "sana_wm_script", "");
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
    cfg.text_encoder_max_length =
        extract_json_int(config_json, "text_encoder_max_length", cfg.text_encoder_max_length);
    cfg.text_encoder_dim =
        extract_json_int(config_json, "sana_wm_dit_text_embed_dim",
                         extract_json_int(config_json, "text_encoder_dim", cfg.text_encoder_dim));
    cfg.chi_prompt = extract_json_string(config_json, "sana_wm_chi_prompt", cfg.chi_prompt);
    cfg.default_intrinsics =
        extract_json_float_array(config_json, "sana_wm_default_intrinsics", 9U);
    cfg.require_official_script =
        extract_json_int(config_json, "sana_wm_require_official_script", 0) != 0;
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

    std::vector<float> resized(static_cast<std::size_t>(out.plan.resized_width) *
                               static_cast<std::size_t>(out.plan.resized_height) * 3U);
    void* resize_result = stbir_resize(
        src_hwc.data(), src_width, src_height, static_cast<int>(src_width * 3 * sizeof(float)),
        resized.data(), out.plan.resized_width, out.plan.resized_height,
        static_cast<int>(out.plan.resized_width * 3 * sizeof(float)), STBIR_RGB, STBIR_TYPE_FLOAT,
        STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT);
    if (resize_result == nullptr)
        return out;

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
                out.pixels_chw[chw_index(c, y, x, target_height, target_width)] =
                    cropped.pixels_hwc[src + static_cast<std::size_t>(c)] * 2.0F - 1.0F;
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
    return text_encoder || stage1_denoiser || vae_encoder || vae_decoder || refiner_text_encoder ||
           refiner_denoiser || refiner_vae_decoder;
}

bool SanaWmNativeModules::has_stage1() const {
    return text_encoder && stage1_denoiser && vae_encoder && vae_decoder;
}

bool SanaWmNativeModules::has_refiner() const {
    return refiner_text_encoder && refiner_denoiser && refiner_vae_decoder;
}

SanaWmPipeline::SanaWmPipeline(SanaWmRuntimeConfig config, std::string hf_python,
                               std::shared_ptr<ISubprocessRunner> subprocess_runner,
                               SanaWmNativeModules native_modules,
                               std::shared_ptr<ITokenizer> tokenizer)
    : config_(std::move(config)), hf_python_(std::move(hf_python)),
      subprocess_runner_(std::move(subprocess_runner)), native_modules_(std::move(native_modules)),
      tokenizer_(std::move(tokenizer)) {
    if (!subprocess_runner_)
        subprocess_runner_ = CreateDefaultSubprocessRunner();
}

ImageResult SanaWmPipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    const auto request = resolve_request(config_, cfg, hf_python_);
    if (request.image_path.empty())
        throw std::runtime_error("SANA-WM generation requires --image");
    if (prompt.empty())
        throw std::runtime_error("SANA-WM generation requires a non-empty prompt");
    if (native_modules_.has_any()) {
        validate_native_module_set(native_modules_);
        return run_native_image_path(native_modules_, tokenizer_, config_, request, cfg, prompt);
    }

    const auto paths = make_invocation_paths();
    run_bridge_or_throw(*subprocess_runner_, build_bridge_argv(config_, request, paths, prompt));
    auto result = load_frame_result(paths.frames_dir);
    std::error_code ec;
    std::filesystem::remove_all(paths.temp_root, ec);
    return result;
}

} // namespace trtmc
