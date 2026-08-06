# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import os

ENGINE = "/tmp/flux2_dit_fp8_bf16_clean.engine"
DONOR = "/tmp/flux2_exp18.bundle"
OUTPUT = "/tmp/flux2_fp8_bf16.bundle"

with open(DONOR, "rb") as f:
    f.read(8)
    jl = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(jl))
    ds = 16 + jl

sd = {}
keep = ["text_encoder_0_plan", "vae_decoder_plan", "preprocessor_weights",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
for nm in keep:
    if nm in hdr["sections"]:
        s = hdr["sections"][nm]
        with open(DONOR, "rb") as f:
            f.seek(ds + s["offset"])
            sd[nm] = f.read(s["size"])

with open(ENGINE, "rb") as f:
    sd["denoiser_plan"] = f.read()
print(f"Denoiser: {len(sd['denoiser_plan']) // (1024**3)} GB")

cs = hdr["sections"]["config.json"]
with open(DONOR, "rb") as f:
    f.seek(ds + cs["offset"])
    cfg = json.loads(f.read(cs["size"]))
cfg["onnx_denoiser"] = 0
cfg["dit_baked_embeddings"] = 0
sd["config.json"] = json.dumps(cfg).encode()

nh = {}
for k in hdr:
    if k != "sections":
        nh[k] = hdr[k]
nh["sections"] = {}
off = 0
order = ["text_encoder_0_plan", "denoiser_plan", "vae_decoder_plan",
         "preprocessor_weights", "config.json", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json"]
for nm in order:
    if nm in sd:
        nh["sections"][nm] = {"offset": off, "size": len(sd[nm])}
        off += len(sd[nm])

hj = json.dumps(nh).encode()
with open(OUTPUT, "wb") as f:
    f.write(b"BUNDLE\x01\x00")
    f.write(struct.pack("<Q", len(hj)))
    f.write(hj)
    for nm in order:
        if nm in sd:
            f.write(sd[nm])

print(f"Bundle: {os.path.getsize(OUTPUT) / (1024**3):.1f} GB -> {OUTPUT}")
