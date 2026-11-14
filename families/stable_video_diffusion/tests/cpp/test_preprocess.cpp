/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The CLIP conditioning preprocessing against StableVideoDiffusionPipeline._encode_image
// (_resize_with_antialiasing and CLIPImageProcessor normalization) on synthetic images.
// Reference values: torch 2.12.0 on the portable (DEFAULT) ATen CPU
// path and diffusers 0.39.0, from the checkpoint's scheduler and feature-extractor configs.

#include "families/stable_video_diffusion/runtime/preprocess.h"

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

struct Pick {
    int channel;
    int y;
    int x;
    float value;
};

struct Preview {
    int height;
    int width;
    double sum;
    std::vector<Pick> picks;
};

const Preview kPreviews[] = {
    {576,
     1024,
     22760.872428072827,
     {
         {0, 0, 0, -1.5079392194747925F},     {0, 0, 3, 0.8586409091949463F},
         {0, 0, 100, 1.041088342666626F},     {0, 0, 222, 1.0024751424789429F},
         {0, 0, 223, 1.2209991216659546F},    {0, 1, 0, -1.294517159461975F},
         {0, 1, 3, 1.087233543395996F},       {0, 1, 100, 1.2036268711090088F},
         {0, 1, 222, 1.1968896389007568F},    {0, 1, 223, 0.19428576529026031F},
         {0, 57, 0, -1.1951093673706055F},    {0, 57, 3, 1.1624653339385986F},
         {0, 57, 100, 1.1484097242355347F},   {0, 57, 222, 1.18977952003479F},
         {0, 57, 223, -0.5849177241325378F},  {0, 111, 0, -1.0939126014709473F},
         {0, 111, 3, 0.7493886947631836F},    {0, 111, 100, 0.9220274686813354F},
         {0, 111, 222, 0.8862992525100708F},  {0, 111, 223, 1.4903984069824219F},
         {0, 200, 0, -0.16079379618167877F},  {0, 200, 3, -0.9386700987815857F},
         {0, 200, 100, -1.1477339267730713F}, {0, 200, 222, -1.1122099161148071F},
         {0, 200, 223, -0.691604733467102F},  {0, 223, 0, -1.472782850265503F},
         {0, 223, 3, 0.8937971591949463F},    {0, 223, 100, 1.0775425434112549F},
         {0, 223, 222, 1.0385475158691406F},  {0, 223, 223, 1.2060587406158447F},
         {1, 0, 0, -1.0245760679244995F},     {1, 0, 3, 1.3634676933288574F},
         {1, 0, 100, 1.2863943576812744F},    {1, 0, 222, 1.3467165231704712F},
         {1, 0, 223, -0.6404306292533875F},   {1, 1, 0, -0.8051697611808777F},
         {1, 1, 3, 1.075646996498108F},       {1, 1, 100, 0.8503929376602173F},
         {1, 1, 222, 0.7517654299736023F},    {1, 1, 223, -1.3437832593917847F},
         {1, 57, 0, -0.7029746174812317F},    {1, 57, 3, 1.0649734735488892F},
         {1, 57, 100, 0.5266313552856445F},   {1, 57, 222, 0.7118799686431885F},
         {1, 57, 223, -1.246688961982727F},   {1, 111, 0, -1.1368911266326904F},
         {1, 111, 3, 1.2675601243972778F},    {1, 111, 100, 1.3694089651107788F},
         {1, 111, 222, 1.2858167886734009F},  {1, 111, 223, -0.42680084705352783F},
         {1, 200, 0, 0.3603435754776001F},    {1, 200, 3, -0.9679586291313171F},
         {1, 200, 100, -0.8004838228225708F}, {1, 200, 222, -0.8437680602073669F},
         {1, 200, 223, -0.1853513866662979F}, {1, 223, 0, -0.9884338974952698F},
         {1, 223, 3, 1.3240829706192017F},    {1, 223, 100, 1.1162415742874146F},
         {1, 223, 222, 1.2134671211242676F},  {1, 223, 223, -1.1034423112869263F},
         {2, 0, 0, -0.3785018026828766F},     {2, 0, 3, 0.7857498526573181F},
         {2, 0, 100, 0.13558784127235413F},   {2, 0, 222, 0.33433493971824646F},
         {2, 0, 223, -0.8955550193786621F},   {2, 1, 0, -0.17061127722263336F},
         {2, 1, 3, 0.14463356137275696F},     {2, 1, 100, -0.43311750888824463F},
         {2, 1, 222, -0.28318890929222107F},  {2, 1, 223, -0.6876645088195801F},
         {2, 57, 0, -0.07378016412258148F},   {2, 57, 3, -0.3184584081172943F},
         {2, 57, 100, -0.7289374470710754F},  {2, 57, 222, -0.6395989656448364F},
         {2, 57, 223, -0.590833306312561F},   {2, 111, 0, -0.48492199182510376F},
         {2, 111, 3, 1.0995844602584839F},    {2, 111, 100, 0.5582247376441956F},
         {2, 111, 222, 0.7400091290473938F},  {2, 111, 223, -1.0114344358444214F},
         {2, 200, 0, 0.9337286353111267F},    {2, 200, 3, -0.33026447892189026F},
         {2, 200, 100, -0.1627509444952011F}, {2, 200, 222, -0.19736233353614807F},
         {2, 200, 223, 0.41667526960372925F}, {2, 223, 0, -0.34425655007362366F},
         {2, 223, 3, 0.819995105266571F},     {2, 223, 100, 0.16983309388160706F},
         {2, 223, 222, 0.3685801923274994F},  {2, 223, 223, -0.8613098859786987F},
     }},
    {96,
     300,
     23029.0121881635,
     {
         {0, 0, 0, -1.7922625541687012F},      {0, 0, 3, -1.0269029140472412F},
         {0, 0, 100, 1.6906497478485107F},     {0, 0, 222, -0.2724967300891876F},
         {0, 0, 223, -0.011254520155489445F},  {0, 1, 0, -1.757637858390808F},
         {0, 1, 3, -0.9922778606414795F},      {0, 1, 100, 1.6188533306121826F},
         {0, 1, 222, -0.23787158727645874F},   {0, 1, 223, 0.02337072603404522F},
         {0, 57, 0, 0.6936473846435547F},      {0, 57, 3, 1.4593595266342163F},
         {0, 57, 100, 0.4958947002887726F},    {0, 57, 222, -1.4278419017791748F},
         {0, 57, 223, -1.1895490884780884F},   {0, 111, 0, -0.619803249835968F},
         {0, 111, 3, 0.14555667340755463F},    {0, 111, 100, -0.8175563812255859F},
         {0, 111, 222, 0.8999617695808411F},   {0, 111, 223, 1.161204218864441F},
         {0, 200, 0, -0.4090789556503296F},    {0, 200, 3, 0.35628119111061096F},
         {0, 200, 100, -0.6068321466445923F},  {0, 200, 222, 1.1106863021850586F},
         {0, 200, 223, 1.3719286918640137F},   {0, 223, 0, 0.5872810482978821F},
         {0, 223, 3, 1.3526415824890137F},     {0, 223, 100, 0.3895280957221985F},
         {0, 223, 222, -1.7675546407699585F},  {0, 223, 223, -1.2959160804748535F},
         {1, 0, 0, -1.3168718814849854F},      {1, 0, 3, -0.5300512313842773F},
         {1, 0, 100, -1.5201704502105713F},    {1, 0, 222, 0.24550825357437134F},
         {1, 0, 223, 0.5140759348869324F},     {1, 1, 0, -1.281275987625122F},
         {1, 1, 3, -0.4944552183151245F},      {1, 1, 100, -1.4845746755599976F},
         {1, 1, 222, 0.2811043858528137F},     {1, 1, 223, 0.5496721267700195F},
         {1, 57, 0, 1.2387433052062988F},      {1, 57, 3, 0.9807430505752563F},
         {1, 57, 100, 1.0354455709457397F},    {1, 57, 222, -0.9658254981040955F},
         {1, 57, 223, -0.6972584128379822F},   {1, 111, 0, -0.11153679341077805F},
         {1, 111, 3, 0.6752840280532837F},     {1, 111, 100, -0.31483498215675354F},
         {1, 111, 222, 1.450842261314392F},    {1, 111, 223, 1.7194104194641113F},
         {1, 200, 0, 0.10509634017944336F},    {1, 200, 3, 0.8919172883033752F},
         {1, 200, 100, -0.09820184111595154F}, {1, 200, 222, 1.7418581247329712F},
         {1, 200, 223, 1.2663530111312866F},   {1, 223, 0, 1.1293944120407104F},
         {1, 223, 3, 1.8514693975448608F},     {1, 223, 100, 0.9260964393615723F},
         {1, 223, 222, -1.0751750469207764F},  {1, 223, 223, -0.8066077828407288F},
         {2, 0, 0, -0.655456006526947F},       {2, 0, 3, 0.0900675430893898F},
         {2, 0, 100, -0.8480839133262634F},    {2, 0, 222, 0.8249205946922302F},
         {2, 0, 223, 1.0793921947479248F},     {2, 1, 0, -0.6217283606529236F},
         {2, 1, 3, 0.1237952709197998F},       {2, 1, 100, -0.8143564462661743F},
         {2, 1, 222, 0.858648419380188F},      {2, 1, 223, 1.113120198249817F},
         {2, 57, 0, 1.7660244703292847F},      {2, 57, 3, -1.0576887130737305F},
         {2, 57, 100, 1.5727084875106812F},    {2, 57, 222, -0.3228349983692169F},
         {2, 57, 223, -0.06836386024951935F},  {2, 111, 0, 0.486615389585495F},
         {2, 111, 3, 1.232139229774475F},      {2, 111, 100, 0.2939876914024353F},
         {2, 111, 222, 1.6173774003982544F},   {2, 111, 223, -1.347772479057312F},
         {2, 200, 0, 0.6918783783912659F},     {2, 200, 3, 1.4374022483825684F},
         {2, 200, 100, 0.49925076961517334F},  {2, 200, 222, -0.4857847988605499F},
         {2, 200, 223, -1.1425098180770874F},  {2, 223, 0, 1.6624149084091187F},
         {2, 223, 3, -1.1612989902496338F},    {2, 223, 100, 1.4697874784469604F},
         {2, 223, 222, -0.426444947719574F},   {2, 223, 223, -0.17197367548942566F},
     }},
};

// The preprocessor_config.json of the checkpoint.
const std::array<float, 3> kMean{0.48145466F, 0.4578275F, 0.40821073F};
const std::array<float, 3> kStd{0.26862954F, 0.26130258F, 0.27577711F};

std::vector<std::uint8_t> synthetic(int height, int width) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(height) * width * 3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < 3; ++c)
                bytes[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                    static_cast<std::uint8_t>((y * 7 + x * 13 + c * 29) % 251);
    return bytes;
}

int main() {
    for (const auto& preview : kPreviews) {
        const auto bytes = synthetic(preview.height, preview.width);
        trtmc::internal::ImageView view;
        view.data = bytes.data();
        view.byte_size = bytes.size();
        view.height = static_cast<std::uint32_t>(preview.height);
        view.width = static_cast<std::uint32_t>(preview.width);
        view.channels = 3;
        view.format = trtmc::internal::ImageFormat::UInt8;
        const auto pixels = svd::initial_image_pixels(view, preview.height, preview.width);
        const auto clip = svd::clip_pixels(pixels, preview.height, preview.width, 224, kMean, kStd);
        const std::string name =
            std::to_string(preview.height) + "x" + std::to_string(preview.width);
        check(clip.size() == 3U * 224U * 224U, name + " output size");
        for (const auto& pick : preview.picks) {
            const float value =
                clip[(static_cast<std::size_t>(pick.channel) * 224 + pick.y) * 224 + pick.x];
            check_near(value, pick.value, 2e-5,
                       name + " pixel " + std::to_string(pick.channel) + "," +
                           std::to_string(pick.y) + "," + std::to_string(pick.x));
        }
        double sum = 0.0;
        for (const float value : clip)
            sum += value;
        check_near(sum, preview.sum, 0.05, name + " sum");
    }

    // The VAE image is CHW in [-1, 1].
    const std::vector<float> hwc{0.0F, 0.5F, 1.0F, 0.25F, 0.75F, 0.5F};
    const auto chw = svd::vae_image(hwc, 1, 2);
    check(chw == std::vector<float>({-1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 0.0F}), "VAE image layout");

    // Only the bundle's size is accepted.
    const auto bytes = synthetic(8, 8);
    trtmc::internal::ImageView view;
    view.data = bytes.data();
    view.byte_size = bytes.size();
    view.height = 8;
    view.width = 8;
    view.channels = 3;
    view.format = trtmc::internal::ImageFormat::UInt8;
    bool rejected = false;
    try {
        (void)svd::initial_image_pixels(view, 8, 16);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "an image of another size is rejected");
    const std::vector<float> outside{0.0F, 0.5F, 1.5F};
    view.data = outside.data();
    view.byte_size = outside.size() * sizeof(float);
    view.height = 1;
    view.width = 1;
    view.format = trtmc::internal::ImageFormat::Float32;
    rejected = false;
    try {
        (void)svd::initial_image_pixels(view, 1, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "float pixels outside [0, 1] are rejected");

    if (failures != 0)
        return 1;
    std::cout << "stable_video_diffusion preprocess: OK\n";
    return 0;
}
