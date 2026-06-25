// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-04
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-SEG-01
// Intent:         SAM prompt seam: point sparse prompt building, embedding encoding, multimask
// selection Preconditions:  Point coordinates and embedding data available Postconditions: Sparse
// prompts padded correctly, missing data returns zeros, multimask selection correct
// =============================================================================

#include "runtime/models/sam/sam_output_selection.h"
#include "runtime/models/sam/sam_postprocess_seam.h"
#include "runtime/models/sam/sam_prompt_seam.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, const char* test_name) {
    constexpr float kTolerance = 1e-5F;
    if (std::fabs(actual - expected) > kTolerance) {
        std::cerr << "FAIL: " << test_name << " actual=" << actual << " expected=" << expected
                  << '\n';
        ++g_failures;
    }
}

void test_build_sam_point_sparse_prompt_happy_path() {
    const std::vector<float> shared_image_pe = {1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> point_embed_fg = {0.1F, 0.2F, 0.3F, 0.4F};
    const std::vector<float> point_embed_bg = {9.0F, 9.0F, 9.0F, 9.0F};
    const std::vector<float> not_a_point_embed = {9.0F, 8.0F, 7.0F, 6.0F};

    const auto sparse =
        trtmc::build_sam_point_sparse_prompt(0.5F, 0.5F, true, 9, 9, 10, 4, shared_image_pe,
                                             point_embed_fg, point_embed_bg, not_a_point_embed);

    check(sparse.size() == 8, "happy path: sparse size");
    if (sparse.size() != 8) {
        return;
    }

    check_close(sparse[0], 0.1F, "happy path: point value[0]");
    check_close(sparse[1], 0.2F, "happy path: point value[1]");
    check_close(sparse[2], 1.3F, "happy path: point value[2]");
    check_close(sparse[3], 1.4F, "happy path: point value[3]");
    check_close(sparse[4], 9.0F, "happy path: pad value[0]");
    check_close(sparse[5], 8.0F, "happy path: pad value[1]");
    check_close(sparse[6], 7.0F, "happy path: pad value[2]");
    check_close(sparse[7], 6.0F, "happy path: pad value[3]");
}

void test_encode_sam_point_embedding_missing_data_returns_zeroes() {
    const std::vector<float> shared_image_pe = {1.0F, 2.0F, 3.0F};
    const std::vector<float> point_embed_fg = {0.1F, 0.2F, 0.3F, 0.4F};
    const std::vector<float> point_embed_bg = {5.0F, 6.0F, 7.0F};

    const auto embedding = trtmc::encode_sam_point_embedding(
        2.0F, 3.0F, false, 10, 4, shared_image_pe, point_embed_fg, point_embed_bg);

    check(embedding.size() == 4, "missing data: embedding size");
    if (embedding.size() != 4) {
        return;
    }

    check_close(embedding[0], 0.0F, "missing data: value[0]");
    check_close(embedding[1], 0.0F, "missing data: value[1]");
    check_close(embedding[2], 0.0F, "missing data: value[2]");
    check_close(embedding[3], 0.0F, "missing data: value[3]");
}

void test_build_sam_point_sparse_prompt_short_padding_embedding() {
    const std::vector<float> shared_image_pe = {1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> point_embed_fg = {8.0F, 8.0F, 8.0F, 8.0F};
    const std::vector<float> point_embed_bg = {0.5F, 0.5F, 0.5F, 0.5F};
    const std::vector<float> not_a_point_embed = {1.0F, 2.0F, 3.0F};

    const auto sparse =
        trtmc::build_sam_point_sparse_prompt(0.5F, 0.5F, false, 9, 9, 10, 4, shared_image_pe,
                                             point_embed_fg, point_embed_bg, not_a_point_embed);

    check(sparse.size() == 8, "short padding: sparse size");
    if (sparse.size() != 8) {
        return;
    }

    check_close(sparse[0], 0.5F, "short padding: point value[0]");
    check_close(sparse[1], 0.5F, "short padding: point value[1]");
    check_close(sparse[2], 1.5F, "short padding: point value[2]");
    check_close(sparse[3], 1.5F, "short padding: point value[3]");
    check_close(sparse[4], 0.0F, "short padding: pad value[0]");
    check_close(sparse[5], 0.0F, "short padding: pad value[1]");
    check_close(sparse[6], 0.0F, "short padding: pad value[2]");
    check_close(sparse[7], 0.0F, "short padding: pad value[3]");
}

void test_select_sam_multimask_outputs_keeps_trailing_masks() {
    trtmc::SamResult result;
    result.mask_height = 1;
    result.mask_width = 2;
    result.num_masks = 4;
    result.masks = {
        10.0F, 11.0F, 20.0F, 21.0F, 30.0F, 31.0F, 40.0F, 41.0F,
    };
    result.iou_scores = {0.1F, 0.2F, 0.3F, 0.4F};

    const auto trimmed = trtmc::select_sam_multimask_outputs(std::move(result), 3);

    check(trimmed.num_masks == 3, "multimask: keeps requested count");
    check(trimmed.masks == std::vector<float>({20.0F, 21.0F, 30.0F, 31.0F, 40.0F, 41.0F}),
          "multimask: keeps trailing mask values");
    check(trimmed.iou_scores == std::vector<float>({0.2F, 0.3F, 0.4F}),
          "multimask: keeps trailing mask scores");
}

void test_select_sam_multimask_outputs_keeps_original_for_invalid_requests() {
    trtmc::SamResult result;
    result.mask_height = 1;
    result.mask_width = 2;
    result.num_masks = 2;
    result.masks = {1.0F, 2.0F, 3.0F, 4.0F};
    result.iou_scores = {0.1F, 0.2F};

    const auto keep_all = trtmc::select_sam_multimask_outputs(result, 2);
    check(keep_all.num_masks == 2, "multimask invalid: keeps original when count already fits");
    check(keep_all.masks == result.masks, "multimask invalid: preserves masks");

    const auto invalid_shape = trtmc::select_sam_multimask_outputs(
        trtmc::SamResult{result.masks, result.iou_scores, 2, 0, 2}, 1);
    check(invalid_shape.num_masks == 2,
          "multimask invalid: keeps original when mask shape invalid");
}

void test_sample_and_resize_sam_mask_helpers_handle_clamp_identity_and_invalid_inputs() {
    const std::vector<float> mask = {
        1.0F,
        2.0F,
        3.0F,
        4.0F,
    };

    check_close(trtmc::sample_mask_bilinear(mask, 2, 2, -5.0F, -5.0F), 1.0F,
                "sample bilinear: clamps to top-left");
    check_close(trtmc::sample_mask_bilinear(mask, 2, 2, 5.0F, 5.0F), 4.0F,
                "sample bilinear: clamps to bottom-right");
    check_close(trtmc::sample_mask_bilinear(mask, 2, 2, 0.5F, 0.5F), 2.5F,
                "sample bilinear: averages center point");
    check_close(trtmc::sample_mask_bilinear({}, 2, 2, 0.0F, 0.0F), 0.0F,
                "sample bilinear: returns zero for empty mask");

    check(trtmc::resize_mask_bilinear(mask, 2, 2, 2, 2) == mask,
          "resize bilinear: identity resize preserves mask");
    check(trtmc::resize_mask_bilinear(mask, 2, 2, 0, 2).empty(),
          "resize bilinear: invalid destination returns empty mask");
}

void test_postprocess_sam_result_crops_padding_region() {
    trtmc::SamResult result;
    result.num_masks = 1;
    result.mask_width = 4;
    result.mask_height = 4;
    result.masks = {
        0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F,
        0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F,
    };

    const auto processed = trtmc::postprocess_sam_result(std::move(result),
                                                         /*image_size=*/4,
                                                         /*rescaled_w=*/2,
                                                         /*rescaled_h=*/4,
                                                         /*original_w=*/2,
                                                         /*original_h=*/4);

    check(processed.mask_width == 2, "postprocess: width updated to original");
    check(processed.mask_height == 4, "postprocess: height updated to original");
    check(processed.masks == std::vector<float>(8, 0.0F),
          "postprocess: right-padding region is cropped away");
}

void test_postprocess_sam_result_preserves_invalid_requests_and_incomplete_payload() {
    trtmc::SamResult invalid_request;
    invalid_request.num_masks = 1;
    invalid_request.mask_width = 2;
    invalid_request.mask_height = 2;
    invalid_request.masks = {1.0F, 2.0F, 3.0F, 4.0F};

    const auto unchanged_invalid = trtmc::postprocess_sam_result(invalid_request,
                                                                 /*image_size=*/0,
                                                                 /*rescaled_w=*/2,
                                                                 /*rescaled_h=*/2,
                                                                 /*original_w=*/2,
                                                                 /*original_h=*/2);
    check(unchanged_invalid.masks == invalid_request.masks,
          "postprocess invalid: returns original result");

    trtmc::SamResult incomplete_payload = invalid_request;
    incomplete_payload.masks = {1.0F, 2.0F, 3.0F};
    const auto unchanged_incomplete = trtmc::postprocess_sam_result(incomplete_payload,
                                                                    /*image_size=*/2,
                                                                    /*rescaled_w=*/2,
                                                                    /*rescaled_h=*/2,
                                                                    /*original_w=*/2,
                                                                    /*original_h=*/2);
    check(unchanged_incomplete.masks == incomplete_payload.masks,
          "postprocess incomplete: returns original result");
}

} // namespace

int main() {
    test_build_sam_point_sparse_prompt_happy_path();
    test_encode_sam_point_embedding_missing_data_returns_zeroes();
    test_build_sam_point_sparse_prompt_short_padding_embedding();
    test_select_sam_multimask_outputs_keeps_trailing_masks();
    test_select_sam_multimask_outputs_keeps_original_for_invalid_requests();
    test_sample_and_resize_sam_mask_helpers_handle_clamp_identity_and_invalid_inputs();
    test_postprocess_sam_result_crops_padding_region();
    test_postprocess_sam_result_preserves_invalid_requests_and_incomplete_payload();

    if (g_failures != 0) {
        std::cerr << g_failures << " SAM prompt seam test(s) failed" << '\n';
        return 1;
    }

    std::cout << "SAM prompt seam tests passed" << '\n';
    return 0;
}
