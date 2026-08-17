/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_image_network.h"

#include "runtime/models/sam2/sam2_engine_contract.h"

#include <NvInferVersion.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace trtmc::sam2::native {
namespace {

static_assert(kHieraSmallBlockCount ==
              static_cast<int32_t>(trtmc::sam2::kImageAttentionMetadata.size()));

std::string blockPrefix(int32_t index) {
    return "image_encoder.trunk.blocks." + std::to_string(index);
}

std::string layerName(int32_t index, std::string_view suffix) {
    return "hiera.block." + std::to_string(index) + "." + std::string(suffix);
}

std::string_view attentionMetadata(int32_t index) {
    if (index < 0 ||
        static_cast<std::size_t>(index) >= trtmc::sam2::kImageAttentionMetadata.size()) {
        throw NetworkBuildError("invalid Hiera attention metadata index");
    }
    return trtmc::sam2::kImageAttentionMetadata[static_cast<std::size_t>(index)];
}

bool sameDimensions(const nvinfer1::Dims& actual, const std::array<int32_t, 4>& expected) {
    if (actual.nbDims != 4)
        return false;
    for (int32_t i = 0; i < actual.nbDims; ++i) {
        if (actual.d[i] != expected[static_cast<std::size_t>(i)])
            return false;
    }
    return true;
}

std::string dimensionsString(const nvinfer1::Dims& dimensions) {
    std::ostringstream stream;
    stream << '[';
    for (int32_t i = 0; i < dimensions.nbDims; ++i) {
        if (i != 0)
            stream << ',';
        stream << dimensions.d[i];
    }
    stream << ']';
    return stream.str();
}

} // namespace

Sam2ImageNetworkBuilder::Sam2ImageNetworkBuilder(nvinfer1::INetworkDefinition& network,
                                                 const CheckpointReader& checkpoint)
    : network_(network), checkpoint_(checkpoint), layers_(network, checkpoint) {}

void Sam2ImageNetworkBuilder::validateNetworkMode() const {
#if NV_TENSORRT_MAJOR == 10
    if (!network_.getFlag(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED))
        throw NetworkBuildError("SAM2 requires a strongly typed TensorRT network");
#elif NV_TENSORRT_MAJOR < 10
#error "The native SAM2 builder requires TensorRT 10 or newer"
#endif
    if (network_.getNbInputs() != 0 || network_.getNbOutputs() != 0 ||
        network_.getNbLayers() != 0) {
        throw NetworkBuildError("SAM2 image construction requires an empty network");
    }
    if (checkpoint_.tensorCount() != kDeliveredCheckpointTensorCount) {
        throw NetworkBuildError("checkpoint inventory mismatch: expected " +
                                std::to_string(kDeliveredCheckpointTensorCount) +
                                " tensors, received " + std::to_string(checkpoint_.tensorCount()));
    }
}

nvinfer1::ITensor& Sam2ImageNetworkBuilder::buildAttention(nvinfer1::ITensor& input, int32_t index,
                                                           int32_t output_channels, int32_t heads,
                                                           bool query_pool) {
    const nvinfer1::Dims input_dims = input.getDimensions();
    if (input_dims.nbDims != 4 || input_dims.d[3] <= 0 || output_channels % heads != 0)
        throw NetworkBuildError("invalid attention input at Hiera block " + std::to_string(index));
    const int32_t batch = input_dims.d[0];
    const int32_t height = input_dims.d[1];
    const int32_t width = input_dims.d[2];
    const int32_t input_channels = input_dims.d[3];
    const int32_t head_channels = output_channels / heads;
    const int32_t tokens = height * width;
    const std::string weights = blockPrefix(index) + ".attn";
    const std::string name = layerName(index, "attention");

    nvinfer1::ITensor& qkv = layers_.linearAutocastBf16(input, weights + ".qkv", input_channels,
                                                        3 * output_channels, name + ".qkv");
    nvinfer1::ITensor& grouped =
        layers_.shuffle(qkv, {batch, tokens, 3, heads, head_channels}, name + ".group");
    nvinfer1::ITensor& q_rank5 =
        layers_.slice(grouped, {0, 0, 0, 0, 0}, {batch, tokens, 1, heads, head_channels},
                      {1, 1, 1, 1, 1}, nvinfer1::SampleMode::kSTRICT_BOUNDS, name + ".q_slice");
    nvinfer1::ITensor& k_rank5 =
        layers_.slice(grouped, {0, 0, 1, 0, 0}, {batch, tokens, 1, heads, head_channels},
                      {1, 1, 1, 1, 1}, nvinfer1::SampleMode::kSTRICT_BOUNDS, name + ".k_slice");
    nvinfer1::ITensor& v_rank5 =
        layers_.slice(grouped, {0, 0, 2, 0, 0}, {batch, tokens, 1, heads, head_channels},
                      {1, 1, 1, 1, 1}, nvinfer1::SampleMode::kSTRICT_BOUNDS, name + ".v_slice");
    nvinfer1::ITensor& q0 =
        layers_.shuffle(q_rank5, {batch, tokens, heads, head_channels}, name + ".q");
    nvinfer1::ITensor& k =
        layers_.shuffle(k_rank5, {batch, tokens, heads, head_channels}, name + ".k");
    nvinfer1::ITensor& v =
        layers_.shuffle(v_rank5, {batch, tokens, heads, head_channels}, name + ".v");

    int32_t query_height = height;
    int32_t query_width = width;
    nvinfer1::ITensor* q = &q0;
    if (query_pool) {
        nvinfer1::ITensor& q_image =
            layers_.shuffle(q0, {batch, height, width, output_channels}, name + ".q_image");
        nvinfer1::ITensor& q_pooled = layers_.maxPoolNhwc(q_image, 2, 2, name + ".q_pool");
        query_height /= 2;
        query_width /= 2;
        q = &layers_.shuffle(q_pooled, {batch, query_height * query_width, heads, head_channels},
                             name + ".q_pooled_tokens");
    }

    nvinfer1::ITensor& qt = layers_.transpose(*q, {0, 2, 1, 3}, name + ".q_heads");
    nvinfer1::ITensor& kt = layers_.transpose(k, {0, 2, 1, 3}, name + ".k_heads");
    nvinfer1::ITensor& vt = layers_.transpose(v, {0, 2, 1, 3}, name + ".v_heads");
    if (head_channels != 96 || qt.getType() != nvinfer1::DataType::kBF16 ||
        kt.getType() != nvinfer1::DataType::kBF16 || vt.getType() != nvinfer1::DataType::kBF16) {
        throw NetworkBuildError("SAM2 IAttention requires BF16 Q/K/V with head dimension 96 at " +
                                name);
    }

    // IAttention computes unscaled BMM1. Pre-scale Q in BF16 by 1/sqrt(D),
    // which algebraically preserves the requested score scale. TensorRT 11.1
    // permits either single-sided or symmetric pre-scaling for fused MHA; the
    // single-sided form minimizes BF16 temperature error versus the prior
    // post-BMM scale. Equivalence is semantic, not bitwise.
    const float q_pre_scale = 1.0F / std::sqrt(static_cast<float>(head_channels));
    nvinfer1::ITensor& q_scale =
        layers_.scalar(q_pre_scale, 4, nvinfer1::DataType::kBF16, name + ".q_scale");
    nvinfer1::ITensor& q_scaled =
        layers_.elementWise(qt, q_scale, nvinfer1::ElementWiseOperation::kPROD, name + ".q_scaled");
    nvinfer1::IAttention* attention =
        network_.addAttentionV2(q_scaled, kt, vt, nvinfer1::AttentionNormalizationOp::kSOFTMAX,
                                nvinfer1::CausalMaskKind::kNONE);
    if (attention == nullptr)
        throw NetworkBuildError("TensorRT rejected IAttentionV2 at " + name);
    const std::string_view metadata = attentionMetadata(index);
    if (!attention->setMetadata(metadata.data()) || !attention->setName(name.c_str()) ||
        !attention->setQueryForm(nvinfer1::AttentionIOForm::kPADDED_BHND) ||
        !attention->setKeyValueForm(nvinfer1::AttentionIOForm::kPADDED_BHND) ||
        !attention->setDecomposable(false)) {
        throw NetworkBuildError("TensorRT rejected the IAttentionV2 configuration at " + name);
    }
    const char* retained_metadata = attention->getMetadata();
    if (retained_metadata == nullptr || std::string_view(retained_metadata) != metadata ||
        attention->getNormalizationOperation() != nvinfer1::AttentionNormalizationOp::kSOFTMAX ||
        attention->getCausalKind() != nvinfer1::CausalMaskKind::kNONE ||
        attention->getQueryForm() != nvinfer1::AttentionIOForm::kPADDED_BHND ||
        attention->getKeyValueForm() != nvinfer1::AttentionIOForm::kPADDED_BHND ||
        attention->getDecomposable() || attention->getMask() != nullptr ||
        attention->getNbInputs() != 3 || attention->getNbOutputs() != 1 ||
        attention->getInput(0) != &q_scaled || attention->getInput(1) != &kt ||
        attention->getInput(2) != &vt) {
        throw NetworkBuildError("TensorRT did not retain the exact IAttentionV2 contract at " +
                                name);
    }
    nvinfer1::ITensor* attention_output = attention->getOutput(0);
    const std::array<int32_t, 4> expected_attention_shape = {
        batch, heads, query_height * query_width, head_channels};
    if (attention_output == nullptr || attention_output->getType() != q_scaled.getType() ||
        !sameDimensions(attention_output->getDimensions(), expected_attention_shape)) {
        throw NetworkBuildError("TensorRT IAttentionV2 output contract drifted at " + name);
    }
    nvinfer1::ITensor& attended = *attention_output;
    nvinfer1::ITensor& token_major =
        layers_.transpose(attended, {0, 2, 1, 3}, name + ".token_major");
    nvinfer1::ITensor& image = layers_.shuffle(
        token_major, {batch, query_height, query_width, output_channels}, name + ".image");
    return layers_.linearAutocastBf16(image, weights + ".proj", output_channels, output_channels,
                                      name + ".projection");
}

nvinfer1::ITensor& Sam2ImageNetworkBuilder::buildHieraBlock(nvinfer1::ITensor& input, int32_t index,
                                                            const HieraBlockContract& contract) {
    const std::string weights = blockPrefix(index);
    const std::string name = layerName(index, "block");
    nvinfer1::ITensor& normalized = layers_.layerNormFp32(
        input, weights + ".norm1", contract.input_channels, 1.0e-6F, name + ".norm1");

    nvinfer1::ITensor* shortcut = &input;
    if (contract.input_channels != contract.output_channels) {
        nvinfer1::ITensor& projected =
            layers_.linearAutocastBf16(normalized, weights + ".proj", contract.input_channels,
                                       contract.output_channels, name + ".shortcut_projection");
        shortcut = &layers_.maxPoolNhwc(projected, 2, 2, name + ".shortcut_pool");
    }

    nvinfer1::ITensor* attended = nullptr;
    if (contract.window_size > 0) {
        const WindowTensor windows = layers_.windowPartition(
            normalized, contract.input_height, contract.input_height, contract.input_channels,
            contract.window_size, name + ".partition");
        nvinfer1::ITensor& window_attention = buildAttention(
            *windows.tensor, index, contract.output_channels, contract.heads, contract.query_pool);
        const int32_t output_height =
            contract.query_pool ? contract.input_height / 2 : contract.input_height;
        const int32_t output_window =
            contract.query_pool ? contract.window_size / 2 : contract.window_size;
        attended = &layers_.windowUnpartition(windows, window_attention, output_height,
                                              output_height, contract.output_channels,
                                              output_window, name + ".unpartition");
    } else {
        attended = &buildAttention(normalized, index, contract.output_channels, contract.heads,
                                   contract.query_pool);
    }

    nvinfer1::ITensor& residual_attention =
        layers_.cast(*attended, shortcut->getType(), name + ".attention_residual_cast");
    nvinfer1::ITensor& after_attention =
        layers_.elementWise(*shortcut, residual_attention, nvinfer1::ElementWiseOperation::kSUM,
                            name + ".attention_residual");
    nvinfer1::ITensor& mlp_input = layers_.layerNormFp32(
        after_attention, weights + ".norm2", contract.output_channels, 1.0e-6F, name + ".norm2");
    nvinfer1::ITensor& mlp_hidden =
        layers_.linearAutocastBf16(mlp_input, weights + ".mlp.layers.0", contract.output_channels,
                                   4 * contract.output_channels, name + ".mlp.fc1");
    nvinfer1::ITensor& activated = layers_.gelu(mlp_hidden, name + ".mlp.gelu");
    nvinfer1::ITensor& mlp_output = layers_.linearAutocastBf16(
        activated, weights + ".mlp.layers.1", 4 * contract.output_channels,
        contract.output_channels, name + ".mlp.fc2");
    nvinfer1::ITensor& residual_mlp =
        layers_.cast(mlp_output, after_attention.getType(), name + ".mlp_residual_cast");
    return layers_.elementWise(after_attention, residual_mlp, nvinfer1::ElementWiseOperation::kSUM,
                               name + ".mlp_residual");
}

std::array<nvinfer1::ITensor*, 4>
Sam2ImageNetworkBuilder::buildHiera(nvinfer1::ITensor& pixel_values) {
    nvinfer1::ITensor& input_bf16 =
        layers_.cast(pixel_values, nvinfer1::DataType::kBF16, "hiera.input_bf16");
    nvinfer1::ITensor& patch_nchw = layers_.convolution(
        input_bf16, "image_encoder.trunk.patch_embed.proj.weight",
        "image_encoder.trunk.patch_embed.proj.bias", 3, 96, 7, 4, 3, 1, "hiera.patch_embed");
    nvinfer1::ITensor& patch_nhwc =
        layers_.transpose(patch_nchw, {0, 2, 3, 1}, "hiera.patch_to_nhwc");
    nvinfer1::ITensor& patch_fp32 =
        layers_.cast(patch_nhwc, nvinfer1::DataType::kFLOAT, "hiera.patch_fp32");

    nvinfer1::ITensor& background = layers_.constant("image_encoder.trunk.pos_embed", {1, 96, 7, 7},
                                                     {1, 96, 7, 7}, "hiera.position.background");
    nvinfer1::ITensor& resized_background =
        layers_.resizeNchw(background, 256, 256, nvinfer1::InterpolationMode::kCUBIC,
                           nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL,
                           "hiera.position.background_resize", -0.75F);
    nvinfer1::ITensor& window =
        layers_.constant("image_encoder.trunk.pos_embed_window", {1, 96, 8, 8}, {1, 96, 8, 8},
                         "hiera.position.window");
    nvinfer1::ITensor& tiled_window =
        layers_.slice(window, {0, 0, 0, 0}, {1, 96, 256, 256}, {1, 1, 1, 1},
                      nvinfer1::SampleMode::kWRAP, "hiera.position.window_tile");
    nvinfer1::ITensor& position_nchw =
        layers_.elementWise(resized_background, tiled_window, nvinfer1::ElementWiseOperation::kSUM,
                            "hiera.position.sum");
    nvinfer1::ITensor& position_nhwc =
        layers_.transpose(position_nchw, {0, 2, 3, 1}, "hiera.position.to_nhwc");
    nvinfer1::ITensor* current =
        &layers_.elementWise(patch_fp32, position_nhwc, nvinfer1::ElementWiseOperation::kSUM,
                             "hiera.patch_plus_position");

    std::array<nvinfer1::ITensor*, 4> outputs{};
    std::size_t stage = 0;
    for (int32_t index = 0; index < kHieraSmallBlockCount; ++index) {
        current =
            &buildHieraBlock(*current, index, kHieraSmallBlocks[static_cast<std::size_t>(index)]);
        if (index == 0 || index == 2 || index == 13 || index == 15) {
            const std::string stage_name = "hiera.stage." + std::to_string(stage) + ".to_nchw";
            outputs[stage] = &layers_.transpose(*current, {0, 3, 1, 2}, stage_name);
            ++stage;
        }
    }
    return outputs;
}

std::array<nvinfer1::ITensor*, 4>
Sam2ImageNetworkBuilder::buildFpn(const std::array<nvinfer1::ITensor*, 4>& trunk_features) {
    constexpr std::array<int32_t, 4> channels = {96, 192, 384, 768};
    std::array<nvinfer1::ITensor*, 4> fpn{};
    for (int32_t level = 3; level >= 0; --level) {
        const int32_t convolution_index = 3 - level;
        const std::string module =
            "image_encoder.neck.convs." + std::to_string(convolution_index) + ".conv";
        const std::string name = "fpn.lateral." + std::to_string(level);
        nvinfer1::ITensor& bf16_input =
            layers_.cast(*trunk_features[static_cast<std::size_t>(level)],
                         nvinfer1::DataType::kBF16, name + ".input_bf16");
        fpn[static_cast<std::size_t>(level)] =
            &layers_.convolution(bf16_input, module + ".weight", module + ".bias",
                                 channels[static_cast<std::size_t>(level)], 256, 1, 1, 0, 1, name);
    }

    nvinfer1::ITensor& low_fp32 =
        layers_.cast(*fpn[3], nvinfer1::DataType::kFLOAT, "fpn.top_down.low_fp32");
    nvinfer1::ITensor& upsampled = layers_.resizeNchw(
        low_fp32, 64, 64, nvinfer1::InterpolationMode::kNEAREST,
        nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC, "fpn.top_down.upsample");
    nvinfer1::ITensor& lateral_fp32 =
        layers_.cast(*fpn[2], nvinfer1::DataType::kFLOAT, "fpn.top_down.lateral_fp32");
    fpn[2] = &layers_.elementWise(lateral_fp32, upsampled, nvinfer1::ElementWiseOperation::kSUM,
                                  "fpn.top_down.fusion");
    return fpn;
}

void Sam2ImageNetworkBuilder::buildBboxHead(const std::array<nvinfer1::ITensor*, 4>& fpn,
                                            std::array<nvinfer1::ITensor*, 3>& classification,
                                            std::array<nvinfer1::ITensor*, 3>& regression) {
    for (int32_t level = 0; level < 3; ++level) {
        nvinfer1::ITensor* cls = fpn[static_cast<std::size_t>(level + 1)];
        nvinfer1::ITensor* reg = cls;
        for (int32_t stack = 0; stack < 2; ++stack) {
            const std::string cls_module = "image_encoder.bbox_head.cls_convs." +
                                           std::to_string(level) + "." + std::to_string(stack);
            const std::string reg_module = "image_encoder.bbox_head.reg_convs." +
                                           std::to_string(level) + "." + std::to_string(stack);
            cls = &layers_.convolutionBatchNormSilu(*cls, cls_module, 256, 256, 3, 1, 1, 1, 1.0e-5F,
                                                    "bbox.level." + std::to_string(level) +
                                                        ".cls." + std::to_string(stack));
            reg = &layers_.convolutionBatchNormSilu(*reg, reg_module, 256, 256, 3, 1, 1, 1, 1.0e-5F,
                                                    "bbox.level." + std::to_string(level) +
                                                        ".reg." + std::to_string(stack));
        }
        const std::string cls_output = "image_encoder.bbox_head.rtm_cls." + std::to_string(level);
        const std::string reg_output = "image_encoder.bbox_head.rtm_reg." + std::to_string(level);
        classification[static_cast<std::size_t>(level)] =
            &layers_.convolution(*cls, cls_output + ".weight", cls_output + ".bias", 256, 2, 1, 1,
                                 0, 1, "bbox.level." + std::to_string(level) + ".classification");
        regression[static_cast<std::size_t>(level)] =
            &layers_.convolution(*reg, reg_output + ".weight", reg_output + ".bias", 256, 4, 1, 1,
                                 0, 1, "bbox.level." + std::to_string(level) + ".regression");
    }
}

void Sam2ImageNetworkBuilder::markCheckedOutput(nvinfer1::ITensor& tensor,
                                                const StaticTensorContract& contract) {
    if (tensor.getType() != contract.type ||
        !sameDimensions(tensor.getDimensions(), contract.dimensions)) {
        throw NetworkBuildError("output contract mismatch for " + std::string(contract.name) +
                                ": actual shape " + dimensionsString(tensor.getDimensions()));
    }
    tensor.setName(contract.name.data());
    network_.markOutput(tensor);
}

Sam2ImageNetworkOutputs Sam2ImageNetworkBuilder::build() {
    if (built_)
        throw NetworkBuildError("SAM2 image network builder is single-use");
    validateNetworkMode();
    built_ = true;

    nvinfer1::ITensor* pixel_values =
        network_.addInput(kImageInputContract.name.data(), kImageInputContract.type,
                          nvinfer1::Dims{4, {1, 3, kSam2ImageSize, kSam2ImageSize}});
    if (pixel_values == nullptr)
        throw NetworkBuildError("TensorRT rejected the SAM2 image input");
    if (!sameDimensions(pixel_values->getDimensions(), kImageInputContract.dimensions) ||
        pixel_values->getType() != kImageInputContract.type) {
        throw NetworkBuildError("SAM2 image input contract was not preserved");
    }

    const auto trunk = buildHiera(*pixel_values);
    const auto fpn = buildFpn(trunk);
    Sam2ImageNetworkOutputs result;
    result.pixel_values = pixel_values;
    result.tracker_fpn = {fpn[0], fpn[1], fpn[2]};
    buildBboxHead(fpn, result.bbox_classification, result.bbox_regression);

    for (std::size_t i = 0; i < result.tracker_fpn.size(); ++i)
        markCheckedOutput(*result.tracker_fpn[i], kTrackerFpnContracts[i]);
    for (std::size_t i = 0; i < result.bbox_classification.size(); ++i)
        markCheckedOutput(*result.bbox_classification[i], kBboxMapContracts[i]);
    for (std::size_t i = 0; i < result.bbox_regression.size(); ++i)
        markCheckedOutput(*result.bbox_regression[i], kBboxMapContracts[i + 3]);

    result.checkpoint_tensor_count = checkpoint_.tensorCount();
    result.referenced_tensor_count = layers_.referencedTensorCount();
    if (result.referenced_tensor_count != kImageNetworkReferencedTensorCount) {
        throw NetworkBuildError(
            "image graph referenced " + std::to_string(result.referenced_tensor_count) +
            " checkpoint tensors; expected " + std::to_string(kImageNetworkReferencedTensorCount));
    }
    result.unreferenced_tensor_count =
        result.checkpoint_tensor_count - result.referenced_tensor_count;
    result.added_layer_count = network_.getNbLayers();
    for (int32_t index = 0; index < network_.getNbLayers(); ++index) {
        const nvinfer1::ILayer* layer = network_.getLayer(index);
        if (layer == nullptr)
            throw NetworkBuildError("SAM2 image graph contains a null TensorRT layer");
        switch (layer->getType()) {
        case nvinfer1::LayerType::kCONVOLUTION:
            ++result.convolution_layer_count;
            break;
        case nvinfer1::LayerType::kACTIVATION:
            ++result.activation_layer_count;
            break;
        case nvinfer1::LayerType::kPOOLING:
            ++result.pooling_layer_count;
            break;
        case nvinfer1::LayerType::kELEMENTWISE:
            ++result.element_wise_layer_count;
            break;
        case nvinfer1::LayerType::kSHUFFLE:
            ++result.shuffle_layer_count;
            break;
        case nvinfer1::LayerType::kCONSTANT:
            ++result.constant_layer_count;
            break;
        case nvinfer1::LayerType::kSLICE:
            ++result.slice_layer_count;
            break;
        case nvinfer1::LayerType::kRESIZE:
            ++result.resize_layer_count;
            break;
        case nvinfer1::LayerType::kNORMALIZATION:
            ++result.normalization_layer_count;
            break;
        case nvinfer1::LayerType::kCAST:
            ++result.cast_layer_count;
            break;
        case nvinfer1::LayerType::kMATRIX_MULTIPLY:
            ++result.matrix_multiply_layer_count;
            break;
        case nvinfer1::LayerType::kSOFTMAX:
            ++result.softmax_layer_count;
            break;
        case nvinfer1::LayerType::kPLUGIN_V3:
            ++result.plugin_v3_layer_count;
            break;
        case nvinfer1::LayerType::kATTENTION_INPUT:
            ++result.attention_input_layer_count;
            break;
        case nvinfer1::LayerType::kATTENTION_OUTPUT:
            ++result.attention_output_layer_count;
            break;
        default:
            break;
        }
    }
    if (result.added_layer_count != kImageNetworkLayerCount ||
        result.convolution_layer_count != kImageNetworkConvolutionLayerCount ||
        result.activation_layer_count != kImageNetworkActivationLayerCount ||
        result.pooling_layer_count != kImageNetworkPoolingLayerCount ||
        result.element_wise_layer_count != kImageNetworkElementWiseLayerCount ||
        result.shuffle_layer_count != kImageNetworkShuffleLayerCount ||
        result.constant_layer_count != kImageNetworkConstantLayerCount ||
        result.slice_layer_count != kImageNetworkSliceLayerCount ||
        result.resize_layer_count != kImageNetworkResizeLayerCount ||
        result.normalization_layer_count != kImageNetworkNormalizationLayerCount ||
        result.cast_layer_count != kImageNetworkCastLayerCount ||
        result.matrix_multiply_layer_count != kImageNetworkMatrixMultiplyLayerCount ||
        result.softmax_layer_count != kImageNetworkSoftmaxLayerCount ||
        result.plugin_v3_layer_count != kImageNetworkPluginV3LayerCount ||
        result.attention_input_layer_count != kImageNetworkAttentionInputLayerCount ||
        result.attention_output_layer_count != kImageNetworkAttentionOutputLayerCount) {
        throw NetworkBuildError(
            "SAM2 image graph layer inventory drifted: actual total=" +
            std::to_string(result.added_layer_count) +
            " convolution=" + std::to_string(result.convolution_layer_count) +
            " activation=" + std::to_string(result.activation_layer_count) +
            " pooling=" + std::to_string(result.pooling_layer_count) +
            " element_wise=" + std::to_string(result.element_wise_layer_count) +
            " shuffle=" + std::to_string(result.shuffle_layer_count) +
            " constant=" + std::to_string(result.constant_layer_count) +
            " slice=" + std::to_string(result.slice_layer_count) +
            " resize=" + std::to_string(result.resize_layer_count) +
            " normalization=" + std::to_string(result.normalization_layer_count) +
            " cast=" + std::to_string(result.cast_layer_count) +
            " matrix_multiply=" + std::to_string(result.matrix_multiply_layer_count) +
            " softmax=" + std::to_string(result.softmax_layer_count) +
            " plugin_v3=" + std::to_string(result.plugin_v3_layer_count) +
            " attention_input=" + std::to_string(result.attention_input_layer_count) +
            " attention_output=" + std::to_string(result.attention_output_layer_count) +
            "; expected total=" + std::to_string(kImageNetworkLayerCount) +
            " convolution=" + std::to_string(kImageNetworkConvolutionLayerCount) +
            " activation=" + std::to_string(kImageNetworkActivationLayerCount) +
            " pooling=" + std::to_string(kImageNetworkPoolingLayerCount) +
            " element_wise=" + std::to_string(kImageNetworkElementWiseLayerCount) +
            " shuffle=" + std::to_string(kImageNetworkShuffleLayerCount) +
            " constant=" + std::to_string(kImageNetworkConstantLayerCount) +
            " slice=" + std::to_string(kImageNetworkSliceLayerCount) +
            " resize=" + std::to_string(kImageNetworkResizeLayerCount) +
            " normalization=" + std::to_string(kImageNetworkNormalizationLayerCount) +
            " cast=" + std::to_string(kImageNetworkCastLayerCount) +
            " matrix_multiply=" + std::to_string(kImageNetworkMatrixMultiplyLayerCount) +
            " softmax=" + std::to_string(kImageNetworkSoftmaxLayerCount) +
            " plugin_v3=" + std::to_string(kImageNetworkPluginV3LayerCount) +
            " attention_input=" + std::to_string(kImageNetworkAttentionInputLayerCount) +
            " attention_output=" + std::to_string(kImageNetworkAttentionOutputLayerCount));
    }
    return result;
}

} // namespace trtmc::sam2::native
