#include "runtime/models/sana_wm/pipeline.h"

#include "trtmc/trtmc_io.hpp"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
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

std::vector<std::vector<char>> parse_action_string(const std::string& action) {
    std::string cleaned;
    cleaned.reserve(action.size());
    for (unsigned char ch : action) {
        if (!std::isspace(ch))
            cleaned.push_back(static_cast<char>(ch));
    }
    if (cleaned.empty())
        throw std::invalid_argument("SANA-WM action string is empty");

    std::vector<std::vector<char>> per_frame;
    std::size_t start = 0;
    while (start <= cleaned.size()) {
        const std::size_t end = cleaned.find(',', start);
        const std::string segment =
            cleaned.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (segment.empty())
            throw std::invalid_argument("Invalid empty SANA-WM action segment");

        const std::size_t dash = segment.rfind('-');
        if (dash == std::string::npos || dash == 0 || dash + 1 >= segment.size())
            throw std::invalid_argument("Invalid SANA-WM action segment: " + segment);

        const std::string keys_part = segment.substr(0, dash);
        const std::string duration_text = segment.substr(dash + 1);
        int duration = 0;
        for (char ch : duration_text) {
            if (!std::isdigit(static_cast<unsigned char>(ch)))
                throw std::invalid_argument("Invalid SANA-WM action duration: " + segment);
            duration = duration * 10 + (ch - '0');
        }
        if (duration <= 0)
            throw std::invalid_argument("SANA-WM action duration must be positive: " + segment);

        std::string keys_lower;
        keys_lower.reserve(keys_part.size());
        for (unsigned char ch : keys_part)
            keys_lower.push_back(static_cast<char>(std::tolower(ch)));

        std::vector<char> keys;
        if (keys_lower != "none") {
            std::set<char> unique;
            for (char key : keys_lower) {
                const bool allowed = key == 'w' || key == 'a' || key == 's' || key == 'd' ||
                                     key == 'i' || key == 'j' || key == 'k' || key == 'l';
                if (!allowed)
                    throw std::invalid_argument("Unknown SANA-WM action key in segment: " +
                                                segment);
                unique.insert(key);
            }
            keys.assign(unique.begin(), unique.end());
        }

        for (int i = 0; i < duration; ++i)
            per_frame.push_back(keys);

        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return per_frame;
}

bool has_key(const std::vector<char>& keys, char key) {
    return std::find(keys.begin(), keys.end(), key) != keys.end();
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
    cfg.require_official_script =
        extract_json_int(config_json, "sana_wm_require_official_script", 0) != 0;
    return cfg;
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
        float pitch_delta =
            (has_key(keys, 'i') ? rotate_rad : 0.0F) - (has_key(keys, 'k') ? rotate_rad : 0.0F);
        const float new_pitch = current_pitch + pitch_delta;
        if (new_pitch < -pitch_limit_rad || new_pitch > pitch_limit_rad) {
            pitch_delta = 0.0F;
        } else {
            current_pitch = new_pitch;
        }

        const float yaw_delta =
            (has_key(keys, 'l') ? rotate_rad : 0.0F) - (has_key(keys, 'j') ? rotate_rad : 0.0F);
        const Mat3 r_new = matmul3(matmul3(rot_y(yaw_delta), r), rot_x(pitch_delta));

        Vec3 forward = column(r_new, 2);
        Vec3 right = column(r_new, 0);
        normalize_horizontal(forward);
        normalize_horizontal(right);

        Vec3 move{0.0F, 0.0F, 0.0F};
        if (has_key(keys, 'w')) {
            move[0] += forward[0] * translation_speed;
            move[2] += forward[2] * translation_speed;
        }
        if (has_key(keys, 's')) {
            move[0] -= forward[0] * translation_speed;
            move[2] -= forward[2] * translation_speed;
        }
        if (has_key(keys, 'd')) {
            move[0] += right[0] * translation_speed;
            move[2] += right[2] * translation_speed;
        }
        if (has_key(keys, 'a')) {
            move[0] -= right[0] * translation_speed;
            move[2] -= right[2] * translation_speed;
        }

        r = r_new;
        t[0] += move[0];
        t[1] += move[1];
        t[2] += move[2];
        poses.push_back(make_pose(r, t));
    }

    return poses;
}

SanaWmPipeline::SanaWmPipeline(SanaWmRuntimeConfig config, std::string hf_python,
                               std::shared_ptr<ISubprocessRunner> subprocess_runner)
    : config_(std::move(config)), hf_python_(std::move(hf_python)),
      subprocess_runner_(std::move(subprocess_runner)) {
    if (!subprocess_runner_)
        subprocess_runner_ = CreateDefaultSubprocessRunner();
}

ImageResult SanaWmPipeline::generate_image(const std::string& prompt, const GenerateConfig& cfg) {
    const auto request = resolve_request(config_, cfg, hf_python_);
    if (request.image_path.empty())
        throw std::runtime_error("SANA-WM generation requires --image");
    if (prompt.empty())
        throw std::runtime_error("SANA-WM generation requires a non-empty prompt");

    const auto paths = make_invocation_paths();
    run_bridge_or_throw(*subprocess_runner_, build_bridge_argv(config_, request, paths, prompt));
    auto result = load_frame_result(paths.frames_dir);
    std::error_code ec;
    std::filesystem::remove_all(paths.temp_root, ec);
    return result;
}

} // namespace trtmc
