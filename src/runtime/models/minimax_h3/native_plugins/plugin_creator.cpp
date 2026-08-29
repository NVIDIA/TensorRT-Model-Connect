/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "audio_encoder_plugin.h"
#include "layer_norm_plugin.h"
#include "linear_plugin.h"
#include "patch_embed_plugin.h"
#include "vision_attention_plugin.h"

#include <NvInferRuntime.h>
#include <array>
#include <cstdint>
#include <new>

#ifndef TRTMC_MINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY
#define TRTMC_MINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY "unversioned"
#endif

namespace trtmc::minimax_h3 {

template <typename Plugin>
class FixedPluginCreator final : public nvinfer1::IPluginCreatorV3One {
  public:
    FixedPluginCreator() noexcept {
        fields_.nbFields = 0;
        fields_.fields = nullptr;
    }

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const*,
                                      nvinfer1::PluginFieldCollection const* fields,
                                      nvinfer1::TensorRTPhase) noexcept override {
        if (fields == nullptr)
            return nullptr;
        auto* plugin = new (std::nothrow) Plugin(*fields);
        if (plugin != nullptr && !plugin->isValid()) {
            delete plugin;
            return nullptr;
        }
        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return Plugin::kPLUGIN_NAME;
    }
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return Plugin::kPLUGIN_VERSION;
    }
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return namespace_.data();
    }

  private:
    nvinfer1::PluginFieldCollection fields_{};
    std::array<char, 1> namespace_{{'\0'}};
};

using MiniMaxH3VisionAttentionCreator = FixedPluginCreator<MiniMaxH3VisionAttentionPlugin>;
using MiniMaxH3AudioEncoderCreator = FixedPluginCreator<MiniMaxH3AudioEncoderPlugin>;
using MiniMaxH3LayerNormCreator = FixedPluginCreator<MiniMaxH3LayerNormPlugin>;
using MiniMaxH3LinearCreator = FixedPluginCreator<MiniMaxH3LinearPlugin>;
using MiniMaxH3PatchEmbedCreator = FixedPluginCreator<MiniMaxH3PatchEmbedPlugin>;

namespace {

MiniMaxH3VisionAttentionCreator plugin_creator_minimax_h3_vision_attention{};
MiniMaxH3AudioEncoderCreator plugin_creator_minimax_h3_audio_encoder{};
MiniMaxH3LayerNormCreator plugin_creator_minimax_h3_layer_norm{};
MiniMaxH3LinearCreator plugin_creator_minimax_h3_linear{};
MiniMaxH3PatchEmbedCreator plugin_creator_minimax_h3_patch_embed{};
const bool plugin_registrar_minimax_h3_vision_attention =
    ::getPluginRegistry()->registerCreator(plugin_creator_minimax_h3_vision_attention, "");
const bool plugin_registrar_minimax_h3_audio_encoder =
    ::getPluginRegistry()->registerCreator(plugin_creator_minimax_h3_audio_encoder, "");
const bool plugin_registrar_minimax_h3_layer_norm =
    ::getPluginRegistry()->registerCreator(plugin_creator_minimax_h3_layer_norm, "");
const bool plugin_registrar_minimax_h3_linear =
    ::getPluginRegistry()->registerCreator(plugin_creator_minimax_h3_linear, "");
const bool plugin_registrar_minimax_h3_patch_embed =
    ::getPluginRegistry()->registerCreator(plugin_creator_minimax_h3_patch_embed, "");

} // namespace

bool native_plugin_registry_matches() noexcept {
    auto* registry = ::getPluginRegistry();
    return registry != nullptr && plugin_registrar_minimax_h3_vision_attention &&
           plugin_registrar_minimax_h3_audio_encoder && plugin_registrar_minimax_h3_layer_norm &&
           plugin_registrar_minimax_h3_linear && plugin_registrar_minimax_h3_patch_embed &&
           registry->getCreator(MiniMaxH3VisionAttentionPlugin::kPLUGIN_NAME,
                                MiniMaxH3VisionAttentionPlugin::kPLUGIN_VERSION,
                                "") == &plugin_creator_minimax_h3_vision_attention &&
           registry->getCreator(MiniMaxH3AudioEncoderPlugin::kPLUGIN_NAME,
                                MiniMaxH3AudioEncoderPlugin::kPLUGIN_VERSION,
                                "") == &plugin_creator_minimax_h3_audio_encoder &&
           registry->getCreator(MiniMaxH3LayerNormPlugin::kPLUGIN_NAME,
                                MiniMaxH3LayerNormPlugin::kPLUGIN_VERSION,
                                "") == &plugin_creator_minimax_h3_layer_norm &&
           registry->getCreator(MiniMaxH3LinearPlugin::kPLUGIN_NAME,
                                MiniMaxH3LinearPlugin::kPLUGIN_VERSION,
                                "") == &plugin_creator_minimax_h3_linear &&
           registry->getCreator(MiniMaxH3PatchEmbedPlugin::kPLUGIN_NAME,
                                MiniMaxH3PatchEmbedPlugin::kPLUGIN_VERSION,
                                "") == &plugin_creator_minimax_h3_patch_embed;
}

} // namespace trtmc::minimax_h3

extern "C" __attribute__((visibility("default"))) char const*
trtmc_minimax_h3_native_plugin_identity() noexcept {
    return "trtmc.minimax_h3.native_plugin:aten-ops:1";
}

extern "C" __attribute__((visibility("default"))) std::uint32_t
trtmc_minimax_h3_native_plugin_abi_version() noexcept {
    return 1U;
}

extern "C" __attribute__((visibility("default"))) char const*
trtmc_minimax_h3_native_plugin_build_identity() noexcept {
    return TRTMC_MINIMAX_H3_NATIVE_PLUGIN_BUILD_IDENTITY;
}

extern "C" __attribute__((visibility("default"))) bool
trtmc_minimax_h3_native_plugin_registry_matches() noexcept {
    return trtmc::minimax_h3::native_plugin_registry_matches();
}
