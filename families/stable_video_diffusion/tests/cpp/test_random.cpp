/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The family's noise stream against torch.randn(n, generator=torch.Generator().manual_seed(seed)).
// Reference values: torch 2.12.0 on the portable (DEFAULT) ATen CPU
// path and diffusers 0.39.0, from the checkpoint's scheduler and feature-extractor configs.

#include "families/stable_video_diffusion/runtime/torch_random.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void check_near(double actual, double expected, double tolerance, const std::string& message) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
        std::cerr << "FAIL: " << message << ": " << actual << " vs " << expected << '\n';
        ++failures;
    }
}

} // namespace

namespace svd = trtmc::stable_video_diffusion;

struct Case {
    std::uint32_t seed;
    std::size_t count;
    double sum;
    std::array<float, 12> head;
    std::array<float, 12> tail;
};

const Case kCases[] = {
    {0U,
     16U,
     -1.9091140404343605,
     {
         -1.1258398294448853F,
         -1.152360200881958F,
         -0.2505785822868347F,
         -0.4338788390159607F,
         0.8487103581428528F,
         0.6920092105865479F,
         -0.31601276993751526F,
         -2.1152195930480957F,
         0.32227492332458496F,
         -1.2633347511291504F,
         0.34998318552970886F,
         0.30813390016555786F,
     },
     {
         0.8487103581428528F,
         0.6920092105865479F,
         -0.31601276993751526F,
         -2.1152195930480957F,
         0.32227492332458496F,
         -1.2633347511291504F,
         0.34998318552970886F,
         0.30813390016555786F,
         0.11984150856733322F,
         1.237657904624939F,
         1.1167771816253662F,
         -0.2472776472568512F,
     }},
    {42U,
     37U,
     3.3561750762164593,
     {
         1.9269150495529175F,
         1.4872841835021973F,
         0.9007171988487244F,
         -2.1055214405059814F,
         0.6784184575080872F,
         -1.2345449924468994F,
         -0.043067481368780136F,
         -1.6046669483184814F,
         -0.7521361708641052F,
         1.6487228870391846F,
         -0.3924786448478699F,
         -1.403606653213501F,
     },
     {
         0.3211030960083008F,
         1.573600172996521F,
         -0.8454675078392029F,
         1.312307596206665F,
         2.1227850914001465F,
         -1.234653353691101F,
         -0.4879138767719269F,
         -1.4180599451065063F,
         0.8962682485580444F,
         0.04990537837147713F,
         2.2667179107666016F,
         1.1789507865905762F,
     }},
    {4294967295U,
     100U,
     -11.626139671774581,
     {
         -2.664297580718994F,
         -1.0910485982894897F,
         0.45144736766815186F,
         0.22836072742938995F,
         1.8986179828643799F,
         1.5642088651657104F,
         -1.2553997039794922F,
         0.7936442494392395F,
         -1.7514946460723877F,
         -0.7096747756004333F,
         1.218266248703003F,
         0.22224649786949158F,
     },
     {
         -0.5805537700653076F,
         -0.14519621431827545F,
         -1.019573450088501F,
         -0.3549883961677551F,
         1.1518298387527466F,
         0.20130053162574768F,
         0.1395149976015091F,
         -0.7070898413658142F,
         -0.2631515562534332F,
         1.8811429738998413F,
         -0.8279935717582703F,
         1.6752885580062866F,
     }},
    {7U,
     4099U,
     85.79866611897887,
     {
         -0.8201345205307007F,
         0.39563092589378357F,
         0.8989079594612122F,
         -1.3884040117263794F,
         -0.16699601709842682F,
         0.2851499617099762F,
         -0.641091525554657F,
         -0.8936551213264465F,
         0.9265430569648743F,
         -0.535512387752533F,
         -1.1597206592559814F,
         -0.46015679836273193F,
     },
     {
         -0.33320707082748413F,
         2.1331968307495117F,
         0.13798147439956665F,
         -0.19215071201324463F,
         0.603180468082428F,
         -0.08482041209936142F,
         -1.0128511190414429F,
         0.41424793004989624F,
         -1.0668336153030396F,
         -0.7660371661186218F,
         -0.02779705449938774F,
         0.44065773487091064F,
     }},
};

const std::array<float, 33> kContinued = {
    1.9230599403381348F,  0.28794968128204346F,  0.9367964863777161F,  -2.4031355381011963F,
    0.8183425068855286F,  -0.6632770895957947F,  0.2652658224105835F,  -1.5660384893417358F,
    -1.8153051137924194F, -0.2066287398338318F,  -0.5259512066841125F, -1.6977323293685913F,
    -1.0364669561386108F, 0.6704617738723755F,   -0.1777045875787735F, 0.4313727617263794F,
    0.1730271428823471F,  0.6070178747177124F,   -0.8779260516166687F, -0.742591917514801F,
    -0.4973025321960449F, -1.998277187347412F,   -2.3635120391845703F, 0.25412511825561523F,
    0.977179229259491F,   -0.13675589859485626F, 1.8703440427780151F,  0.15049892663955688F,
    0.9184275269508362F,  0.4697847366333008F,   0.949497401714325F,   -2.3509089946746826F,
    -1.4640164375305176F,
};

int main() {
    // glibc and the reference may differ in the last bit of logf/cosf/sinf.
    constexpr double kTolerance = 2e-6;
    for (const auto& item : kCases) {
        std::vector<float> values(item.count);
        svd::TorchCpuGenerator(item.seed).normal(values.data(), values.size());
        const std::string name =
            "seed " + std::to_string(item.seed) + " count " + std::to_string(item.count);
        for (std::size_t index = 0; index < item.head.size(); ++index) {
            check_near(values[index], item.head[index], kTolerance,
                       name + " head " + std::to_string(index));
            check_near(values[item.count - item.tail.size() + index], item.tail[index], kTolerance,
                       name + " tail " + std::to_string(index));
        }
        double sum = 0.0;
        for (const float value : values)
            sum += value;
        check_near(sum, item.sum, 1e-4, name + " sum");
    }

    // A second draw continues the stream of the first.
    svd::TorchCpuGenerator generator(5U);
    std::vector<float> first(20);
    std::vector<float> second(kContinued.size());
    generator.normal(first.data(), first.size());
    generator.normal(second.data(), second.size());
    for (std::size_t index = 0; index < second.size(); ++index)
        check_near(second[index], kContinued[index], kTolerance,
                   "continued " + std::to_string(index));

    bool rejected = false;
    try {
        std::vector<float> small(15);
        svd::TorchCpuGenerator(1U).normal(small.data(), small.size());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "fewer than 16 values are rejected");

    if (failures != 0)
        return 1;
    std::cout << "stable_video_diffusion random: OK\n";
    return 0;
}
