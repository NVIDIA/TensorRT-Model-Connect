#include "runtime/core/cuda_common.h"
#include "runtime/models/magpie/magpie_config.h"
#include "runtime/models/magpie/pipeline.h"

#include <cuda_runtime_api.h>
#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_magpie_validates_modules() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool threw = false;
    try {
        trtmc::MagpieTTSConfig cfg;
        trtmc::MagpiePipeline p(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, {},
                                {}, {}, {}, trtmc::CudaBuffer(0), trtmc::CudaBuffer(0), {}, {}, {},
                                {}, cfg, stream, nullptr, "x");
        check(false, "null magpie decoder should throw");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "magpie: null decoder throws");

    cudaStreamDestroy(stream);
}

} // namespace

int main() {
    test_magpie_validates_modules();
    if (failures > 0) {
        std::cerr << failures << " magpie pipeline test(s) FAILED\n";
    }
    return failures;
}
