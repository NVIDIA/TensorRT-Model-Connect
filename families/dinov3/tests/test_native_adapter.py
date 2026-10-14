# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from families.dinov3.reference import adapter  # noqa: E402
from trtmc_perf_serving.backends.reference import ReferenceBackend, common  # noqa: E402


def test_native_adapter_returns_complete_public_task_features(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    hidden = torch.arange(72, dtype=torch.float32).reshape(1, 9, 8)
    pooled = hidden[:, 0, :]
    calls = []

    class Model:
        config = SimpleNamespace(num_register_tokens=4)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            assert inputs["pixel_values"].dtype == torch.float16
            return SimpleNamespace(last_hidden_state=hidden, pooler_output=pooled)

    def load_model(model, **kwargs):
        calls.append((model, kwargs))
        return Model()

    processor = lambda **kwargs: {"pixel_values": torch.ones(1, 3, 2, 2)}  # noqa: E731
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: processor
    )
    monkeypatch.setattr(common, "synchronize", lambda: None)
    spec = common.ReferenceSpec(
        "extract_features",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        revision="114c1379950215c8b35dfcd4e90a5c251dde0d32",
        precision="fp16",
        device="cpu",
    )
    backend = object.__new__(ReferenceBackend)
    backend._adapter = adapter.Adapter(spec, common.NativeHost())
    path = tmp_path / "input.png"
    Image.new("RGB", (2, 2)).save(path)
    result = backend.invoke({"image_path": str(path)}, tmp_path / "features").observation
    assert calls[0][1]["revision"] == spec.revision
    assert result["last_hidden_state_shape"] == [1, 9, 8]
    assert result["pooler_output_shape"] == [1, 8]
    assert np.array_equal(np.asarray(result["last_hidden_state"]).reshape(1, 9, 8), hidden.numpy())
    assert result["pooler_output"] == pooled.flatten().tolist()
    assert result["feature_elements"] == 80 and result["processed_images"] == 1
    assert result["num_register_tokens"] == 4
    assert "_pending_values" not in result
    assert not list(tmp_path.glob("*.npy"))
