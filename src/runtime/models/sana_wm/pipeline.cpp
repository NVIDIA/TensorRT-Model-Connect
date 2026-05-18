#include "runtime/models/sana_wm/pipeline.h"

#include "trtmc/trtmc_io.hpp"
#include "utils/json_helpers.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

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
