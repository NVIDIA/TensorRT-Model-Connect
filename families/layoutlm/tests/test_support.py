# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free identity and default Task of LayoutLM document question answering."""

from families.layoutlm.support import describe
from tensorrt_model_connect.model_support import ModelMetadata


def _metadata(model_type: str, *architectures: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type, "architectures": list(architectures)},
                         model_index={})


def test_the_question_answering_head_answers_about_document_images():
    support = describe(_metadata("layoutlm", "LayoutLMForQuestionAnswering"))
    assert support is not None
    assert support.tasks == ("images_text_to_text",)
    assert support.default_task == "images_text_to_text"
    assert support.default_precision == "fp32"


def test_the_other_layoutlm_heads_are_not_claimed():
    for architecture in ("LayoutLMForMaskedLM", "LayoutLMForSequenceClassification",
                         "LayoutLMForTokenClassification", "LayoutLMModel"):
        assert describe(_metadata("layoutlm", architecture)) is None
    assert describe(_metadata("layoutlm")) is None


def test_later_layoutlm_generations_are_not_claimed():
    for model_type, architecture in (("layoutlmv2", "LayoutLMv2ForQuestionAnswering"),
                                     ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
                                     ("roberta", "RobertaForQuestionAnswering")):
        assert describe(_metadata(model_type, architecture)) is None
