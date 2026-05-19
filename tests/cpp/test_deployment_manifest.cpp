#include "bundle/bundle_view.h"
#include "runtime/deployment/artifact_store.h"
#include "runtime/deployment/deployment_manifest.h"
#include "runtime/deployment/runtime_provider.h"
#include "test_helpers.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::filesystem::path make_temp_dir() {
    char pattern[] = "/tmp/trtmc_deployment_test_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr)
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    return std::filesystem::path(dir);
}

static void write_u64_le(std::ofstream& out, uint64_t value) {
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i)
        bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xFF);
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

static void write_bundle(const std::string& path, const std::string& header,
                         const std::vector<std::vector<char>>& sections) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    write_u64_le(out, header.size());
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& section : sections)
        out.write(section.data(), static_cast<std::streamsize>(section.size()));
}

static void test_manifest_parsing_and_variant_remap() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "deployment.trtfb").string();

    const std::string manifest = R"({
  "schema_version": 1,
  "target": {"platform": "gb300", "objective": "best_perf_memory"},
  "default_variant": "portable_default",
  "selected_variant": "ffi_attention",
  "variants": [
    {
      "id": "portable_default",
      "scope": "runtime",
      "provider": "native_trt",
      "runtime_strategy": "decoder_kv_cache",
      "fallback": true,
      "artifacts": [{"name": "engine_plan", "kind": "bundle_section", "section": "engine_plan"}]
    },
    {
      "id": "ffi_attention",
      "scope": "kernel",
      "provider": "tvm_ffi",
      "runtime_strategy": "decoder_kv_cache",
      "compatibility": {"platform": ["gb300"], "gpu_arch": ["sm100"]},
      "performance": {
        "target_id": "gb300",
        "variant_id": "ffi_attention",
        "provider": "tvm_ffi",
        "scope": "kernel",
        "throughput_tokens_per_s": 123.0,
        "peak_memory_mb": 456.0
      },
      "artifacts": [
        {"name": "engine_plan", "kind": "bundle_section", "section": "deployment/variants/ffi_attention/engine_plan"},
        {"name": "kernel_manifest.json", "kind": "bundle_section", "section": "deployment/variants/ffi_attention/kernel_manifest.json"}
      ]
    }
  ]
})";
    const std::string config = R"({"runtime_strategy": "decoder_kv_cache"})";
    const std::string header = R"({
  "model_id": "deployment-test",
  "runtime_strategy": "decoder_kv_cache",
  "sections": {
    "engine_plan": {"offset": 0, "size": 6},
    "deployment/variants/ffi_attention/engine_plan": {"offset": 6, "size": 3},
    "deployment/variants/ffi_attention/kernel_manifest.json": {"offset": 9, "size": 14},
    "deployment_manifest.json": {"offset": 23, "size": )" +
                               std::to_string(manifest.size()) + R"(},
    "config.json": {"offset": )" +
                               std::to_string(23 + manifest.size()) + R"(, "size": )" +
                               std::to_string(config.size()) + R"(}
  }
})";
    write_bundle(path, header,
                 {{'n', 'a', 't', 'i', 'v', 'e'},
                  {'f', 'f', 'i'},
                  {'{', '"', 'k', 'e', 'r', 'n', 'e', 'l', 's', '"', ':', '[', ']', '}'},
                  std::vector<char>(manifest.begin(), manifest.end()),
                  std::vector<char>(config.begin(), config.end())});

    const auto bundle = trtmc::ReadBundleFile(path);
    const auto parsed = trtmc::deployment::read_manifest(bundle);
    check(parsed.has_value(), "deployment manifest present");
    check(parsed->target_platform == "gb300", "target platform parsed");
    const auto* selected = trtmc::deployment::choose_variant(*parsed, nullptr);
    check(selected != nullptr && selected->id == "ffi_attention", "selected variant chosen");
    check(selected != nullptr &&
              selected->compatibility_json.find("\"platform\"") != std::string::npos,
          "compatibility metadata parsed");
    check(selected != nullptr && selected->performance_json.find(
                                     "\"variant_id\":\"ffi_attention\"") != std::string::npos,
          "performance metadata parsed");
    const auto inspected = trtmc::deployment::inspect_text(*parsed);
    check(inspected.find("compatibility:") != std::string::npos, "inspect shows compatibility");
    check(inspected.find("performance:") != std::string::npos, "inspect shows performance");

    const auto remapped = trtmc::deployment::bundle_with_variant_artifacts(bundle, *selected);
    const auto* engine = trtmc::find_section(remapped, "engine_plan");
    check(engine != nullptr && *engine == std::vector<char>({'f', 'f', 'i'}),
          "selected engine remapped");
    const auto* kernel_manifest = trtmc::find_section(remapped, "kernel_manifest.json");
    check(kernel_manifest != nullptr, "kernel manifest remapped");

    trtmc_test::remove_all_safe(tmp);
}

static void test_artifact_store_materializes_directory() {
    trtmc::BundleFile bundle;
    bundle.sections.push_back({"providers/edgellm/engine_dir/config.json", {'{', '}'}});
    bundle.sections.push_back({"providers/edgellm/engine_dir/nested/tokenizer.json", {'[', ']'}});

    const auto tmp = make_temp_dir();
    trtmc::deployment::ArtifactStore store(bundle, "bundle.trtfb", tmp.string());
    const auto out_dir = store.materialize_directory("providers/edgellm/engine_dir/");
    check(std::filesystem::exists(out_dir / "config.json"), "engine config materialized");
    check(std::filesystem::exists(out_dir / "nested" / "tokenizer.json"),
          "nested tokenizer materialized");
    trtmc_test::remove_all_safe(tmp);
}

static void test_edge_llm_provider_materializes_engine_dir() {
    trtmc::BundleFile bundle;
    bundle.sections.push_back({"providers/edgellm/engine_dir/config.json", {'{', '}'}});
    bundle.sections.push_back(
        {"providers/edgellm/engine_dir/tokenizer/tokenizer.json", {'[', ']'}});

    trtmc::deployment::Variant variant;
    variant.id = "edge_llm";
    variant.scope = "runtime";
    variant.provider = "tensorrt-edge-llm";
    trtmc::deployment::Artifact artifact;
    artifact.name = "engine_dir";
    artifact.kind = "directory";
    artifact.section_prefix = "providers/edgellm/engine_dir/";
    variant.artifacts.push_back(std::move(artifact));

    const auto tmp = make_temp_dir();
    const auto cache = tmp / "runtime_cache";
    trtmc::deployment::ArtifactStore store(bundle, "edge.trtfb", cache.string());

    bool threw = false;
    try {
        (void)trtmc::deployment::load_runtime_provider(variant, store, "edge.trtfb");
    } catch (const std::exception&) {
        threw = true;
    }

    check(threw, "edge provider reaches provider boundary in test build");
    const auto materialized = cache / "providers_edgellm_engine_dir_";
    check(std::filesystem::exists(materialized / "config.json"),
          "edge provider materializes engine config");
    check(std::filesystem::exists(materialized / "tokenizer" / "tokenizer.json"),
          "edge provider materializes nested tokenizer");
    trtmc_test::remove_all_safe(tmp);
}

int main() {
    test_manifest_parsing_and_variant_remap();
    test_artifact_store_materializes_directory();
    test_edge_llm_provider_materializes_engine_dir();
    if (failures == 0) {
        std::cerr << "All deployment manifest tests passed.\n";
        return 0;
    }
    std::cerr << failures << " deployment manifest tests failed.\n";
    return 1;
}
