// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SANAWM-CPP-01
// Architecture:   ARCH-RUNTIME-001
// Unit Design:    UD-SANAWM-01
// Intent:         SANA-WM C++ runtime forwards official action-control contract
// Preconditions:  Bundle config requests strict official-script execution
// Postconditions: Bridge argv includes action/speed/frame flags and strict mode
// =============================================================================

#include "../../src/runtime/models/sana_wm/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

bool contains_arg(const std::vector<std::string>& argv, const std::string& arg) {
    return std::find(argv.begin(), argv.end(), arg) != argv.end();
}

std::string value_after(const std::vector<std::string>& argv, const std::string& flag) {
    auto it = std::find(argv.begin(), argv.end(), flag);
    if (it == argv.end() || ++it == argv.end())
        return "";
    return *it;
}

class FakeSubprocessRunner final : public trtmc::ISubprocessRunner {
  public:
    std::vector<std::string> last_argv;
    int call_count{0};

    int run(const std::vector<std::string>& argv, const void*, std::size_t,
            std::vector<char>& out_stdout, std::string& out_stderr) override {
        ++call_count;
        last_argv = argv;
        out_stdout.clear();
        out_stderr.clear();

        if (value_after(argv, "--frames-dir").empty()) {
            out_stderr = "missing --frames-dir";
            return 2;
        }
        return 0;
    }
};

void test_bridge_command_forwards_strict_sana_wm_contract() {
    trtmc::SanaWmRuntimeConfig cfg;
    cfg.hf_id = "Efficient-Large-Model/SANA-WM_bidirectional";
    cfg.action = "bundle-action";
    cfg.translation_speed = 0.01F;
    cfg.rotation_speed_deg = 0.5F;
    cfg.num_frames = 99;
    cfg.require_official_script = true;

    auto runner = std::make_shared<FakeSubprocessRunner>();
    trtmc::SanaWmPipeline pipeline(cfg, "/usr/bin/python3", runner);

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.image_path = "asset/sana_wm/demo_0.png";
    gen_cfg.camera_action = "w-80,jw-40,w-40,lw-60,w-100";
    gen_cfg.translation_speed = 0.055F;
    gen_cfg.rotation_speed_deg = 1.2F;
    gen_cfg.num_frames = 321;

    bool missing_frames_reported = false;
    try {
        (void)pipeline.generate_image("drive forward", gen_cfg);
    } catch (const std::runtime_error& exc) {
        missing_frames_reported =
            std::string(exc.what()).find("produced no frame_*.png") != std::string::npos;
    }

    check(runner->call_count == 1, "sana wm: subprocess invoked once");
    check(missing_frames_reported, "sana wm: requires bridge to materialize frames");
    check(contains_arg(runner->last_argv, "-m"), "sana wm: python module mode");
    check(contains_arg(runner->last_argv, "tensorrt_model_connect.sana_wm_bridge"),
          "sana wm: bridge module");
    check(value_after(runner->last_argv, "--hf-id") ==
              "Efficient-Large-Model/SANA-WM_bidirectional",
          "sana wm: hf id forwarded");
    check(value_after(runner->last_argv, "--image") == "asset/sana_wm/demo_0.png",
          "sana wm: image forwarded");
    check(value_after(runner->last_argv, "--prompt-text") == "drive forward",
          "sana wm: prompt forwarded");
    check(value_after(runner->last_argv, "--action") == "w-80,jw-40,w-40,lw-60,w-100",
          "sana wm: action forwarded");
    check(value_after(runner->last_argv, "--translation-speed").rfind("0.055", 0) == 0,
          "sana wm: translation speed forwarded");
    check(value_after(runner->last_argv, "--rotation-speed-deg").rfind("1.200", 0) == 0,
          "sana wm: rotation speed forwarded");
    check(value_after(runner->last_argv, "--num-frames") == "321",
          "sana wm: frame count forwarded");
    check(contains_arg(runner->last_argv, "--no-diffusers-fallback"),
          "sana wm: strict official runtime required");
}

} // namespace

int main() {
    test_bridge_command_forwards_strict_sana_wm_contract();
    return failures == 0 ? 0 : 1;
}
