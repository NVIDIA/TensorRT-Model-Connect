# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2's native pipeline for ``predict_structure``: the family's seeded eager reference (reference.py, the
pinned ``boltz`` checkpoint under bf16 autocast) on the request's features padded to the bundle profile, so both
sides draw the same diffusion noise. The document is the bundle's embedded request or a prepared one
(tests/aiperf_requests.py); reported as the masked atom coordinates, per-token pLDDT and confidence score."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Mapping

from families.boltz2.model_config import CHECKPOINT, MOLS
from families.boltz2.tests.aiperf_requests import PACKAGE, processed_directory


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from families.boltz2.reference import load_reference_model

        self.host = host
        self.model = load_reference_model(PACKAGE / CHECKPOINT)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        from families.boltz2.reference import predict_reference
        from families.boltz2.request_preparation import load_profile_features

        host = self.host
        document = host.load_bytes(str(host.required(request, "document_path")))
        with tempfile.TemporaryDirectory() as scratch:
            processed = processed_directory(document, Path(scratch))
            batch = load_profile_features(processed, PACKAGE / MOLS)
        prediction, model_ms = host.timed(lambda: predict_reference(self.model, batch))

        def values(name: str) -> Any:
            return prediction[name].detach().float().cpu().numpy().reshape(-1)

        coords = prediction["coords"].detach().float().cpu().numpy().reshape(-1, 3)[values("masks").astype(bool)]
        plddt = values("plddt")[values("token_masks").astype(bool)]
        return host.invocation({"structures": 1, "atoms": int(coords.shape[0]), "coords": coords.reshape(-1).tolist(),
                                "confidence": {"confidence_score": float(values("confidence_score")[0]),
                                               "plddt": plddt.tolist()}}, model_ms)
