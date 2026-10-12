# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from families.patchtsmixer.tests.benchmark.reference import prediction_model_class


def test_reference_adds_transformers_tied_weights_compatibility_field() -> None:
    class PredictionModel:
        pass

    assert not hasattr(PredictionModel, "all_tied_weights_keys")

    selected = prediction_model_class(
        SimpleNamespace(PatchTSMixerForPrediction=PredictionModel)
    )

    assert selected is PredictionModel
    assert selected.all_tied_weights_keys == {}
