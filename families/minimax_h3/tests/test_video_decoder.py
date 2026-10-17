# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from families.minimax_h3 import vae_builder as vae


@pytest.mark.parametrize("batch", [0, -1, 2, 29, True, 1.0, "1", None])
def test_video_decoder_rejects_invalid_tile_batch_before_cuda(batch):
    with pytest.raises(ValueError, match="batch_size must be 1 or 28"):
        vae.build_vae_tile_decoder_engine({}, batch_size=batch)


@pytest.mark.parametrize("batch", [1, 28])
def test_video_decoder_static_batch_threads_through_typed_input(monkeypatch, batch):
    class CapturedInput(Exception):
        pass

    class Network:
        def add_input(self, name, dtype, shape):
            assert name == "latent_tiles" and dtype == vae.trt.float32
            assert shape == (batch, 24, 7, 16, 16)
            raise CapturedInput

    class Config:
        def clear_flag(self, flag):
            assert flag == vae.trt.BuilderFlag.TF32

        def set_memory_pool_limit(self, _pool, value):
            self.limit = value

        def get_memory_pool_limit(self, _pool):
            return self.limit

    class Builder:
        def create_network(self, flags):
            assert flags == 1 << int(vae.trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            return Network()

        def create_builder_config(self):
            return Config()

    monkeypatch.setattr(vae.trt, "Builder", lambda _logger: Builder())
    with pytest.raises(CapturedInput):
        vae.build_vae_tile_decoder_engine({}, batch_size=batch, workspace_bytes=1 << 30)
