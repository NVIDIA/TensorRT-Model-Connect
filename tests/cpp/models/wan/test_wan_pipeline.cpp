#include "runtime/models/wan/pipeline.h"

#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_wan_construction() {
    trtmc::DiffusionConfig cfg;
    trtmc::PreprocessorWeights weights;

    trtmc::WanPipeline pipeline(nullptr, nullptr, nullptr, cfg, weights, nullptr, "test-wan");

    check(std::string(pipeline.pipeline_type()) == "WanPipeline", "WanPipeline pipeline_type");
    check(std::string(pipeline.model_id()) == "test-wan", "WanPipeline model_id");
}

} // namespace

int main() {
    test_wan_construction();
    if (failures > 0) {
        std::cerr << failures << " wan pipeline test(s) FAILED\n";
    }
    return failures;
}
