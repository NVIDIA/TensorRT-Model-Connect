# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from trtmc_perf_serving.latents import Checkpoint, Replay, canonical_shape, noise, read_checkpoint

SD_VAE = {"block_out_channels": [128, 256, 512, 512]}


def test_canonical_shapes_follow_each_pipelines_own_latent_draw():
    image = {"height": 1024, "width": 768}
    assert canonical_shape("FluxPipeline", {"in_channels": 64}, SD_VAE, image) == (1, 16, 128, 96)
    assert canonical_shape("Flux2Pipeline", {"in_channels": 128}, SD_VAE, image) == (1, 128, 64, 48)
    assert canonical_shape("PixArtSigmaPipeline", {"in_channels": 4}, SD_VAE, image) == (1, 4, 128, 96)
    assert canonical_shape("QwenImagePipeline", {"in_channels": 64}, {"temperal_downsample": [False, True, True]},
                           image) == (1, 1, 16, 128, 96)
    assert canonical_shape("ZImagePipeline", {"in_channels": 16}, SD_VAE, image) == (1, 16, 128, 96)
    video = {"height": 480, "width": 832, "num_frames": 17}
    assert canonical_shape("WanPipeline", {"in_channels": 16}, {}, video) == (1, 16, 5, 60, 104)
    assert canonical_shape("StableDiffusionPipeline", {"in_channels": 4}, SD_VAE, image) is None
    assert canonical_shape("FluxPipeline", {"in_channels": 64}, SD_VAE, {"height": 0}) is None


def test_replay_writes_the_same_seeded_noise_and_drops_unknown_layouts(tmp_path):
    snapshot = tmp_path / "snapshot"
    for name, config in (("model_index.json", {"_class_name": "WanPipeline"}), ("transformer/config.json",
                         {"in_channels": 16}), ("vae/config.json", {"scale_factor_temporal": 4})):
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_text(json.dumps(config))
    replay = Replay(lambda: read_checkpoint(snapshot))
    request = {"prompt": "cat", "latent_seed": 7, "height": 64, "width": 32, "num_frames": 5}
    replayed, done = replay(request, tmp_path / "a")
    assert done and "latent_seed" not in replayed and replayed["prompt"] == "cat"
    written = np.fromfile(replayed["initial_latents_path"], dtype=np.float32)
    assert written.size == 16 * 2 * 8 * 4 and np.array_equal(written, noise((1, 16, 2, 8, 4), 7).reshape(-1))
    again, _ = replay(request, tmp_path / "b")
    assert np.array_equal(np.fromfile(again["initial_latents_path"], dtype=np.float32), written)
    unknown = Replay(lambda: Checkpoint("StableDiffusionPipeline", {"in_channels": 4}, SD_VAE))
    assert unknown(request, tmp_path / "c") == ({key: value for key, value in request.items() if key != "latent_seed"},
                                                False)
