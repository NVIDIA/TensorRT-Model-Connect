#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate image using FP8+BF16 TRT engine + HF text encoder/VAE.

This engine has I/O: hidden_states, encoder_hidden_states, temb, rotary_cos, rotary_sin -> output
Preprocessing (timestep MLP, RoPE, x_embedder, context_embedder) is baked into the engine.
We need to compute temb and RoPE externally.
"""
import time
import math
import glob
import numpy as np
import torch

from tensorrt_model_connect import trt_compat

trt = trt_compat.get_trt()

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

ENGINE_PATH = "/tmp/flux2_dit_fp8_bf16_v2.engine"
OUTPUT_PATH = "/workspace/tensorrt-model-connect/flux2_fp8_bf16_result.png"
PROMPT = "A photo of a cat sitting on a windowsill at sunset"
NUM_STEPS = 28
GUIDANCE = 3.5
H, W = 1024, 1024
SEED = 42

# ============================================================
# 1. Load engine
# ============================================================
print("Loading TRT engine...", flush=True)
with open(ENGINE_PATH, "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
ctx = engine.create_execution_context()
stream = cudart.cudaStreamCreate()[1]

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = engine.get_tensor_dtype(name)
    mode = "IN" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUT"
    print(f"  [{mode}] {name}: {shape} {dtype}", flush=True)

gpu_bufs = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = tuple(max(1, s) for s in engine.get_tensor_shape(name))
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    d_ptr = cudart.cudaMalloc(nbytes)[1]
    ctx.set_tensor_address(name, d_ptr)
    gpu_bufs[name] = (d_ptr, shape, dtype, nbytes)

def upload(name, arr):
    d_ptr, shape, dtype, nbytes = gpu_bufs[name]
    h = np.ascontiguousarray(arr, dtype=dtype)
    cudart.cudaMemcpy(d_ptr, h.ctypes.data, nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

def download(name):
    d_ptr, shape, dtype, nbytes = gpu_bufs[name]
    h = np.zeros(shape, dtype=dtype)
    cudart.cudaMemcpy(h.ctypes.data, d_ptr, nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    return h

def run_engine():
    ctx.execute_async_v3(stream)
    cudart.cudaStreamSynchronize(stream)

# ============================================================
# 2. Text encoding (Mistral on CPU)
# ============================================================
print("Loading Mistral text encoder on CPU...", flush=True)
te_path = "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/text_encoder"
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
system_msg = ("You are an AI that reasons about image descriptions. "
              "You give structured responses focusing on object relationships, object\n"
              "attribution and actions without speculation.")
chat_prompt = f"<s>[SYSTEM_PROMPT]{system_msg}[/SYSTEM_PROMPT][INST]{PROMPT}[/INST]"
tokens = tokenizer(chat_prompt, return_tensors="pt", padding="max_length",
                   max_length=512, truncation=True)

t0 = time.time()
full_model = Mistral3ForConditionalGeneration.from_pretrained(
    te_path, torch_dtype=torch.bfloat16, device_map="cpu")
lang_model = full_model.model
lang_model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

with torch.no_grad():
    output = lang_model(tokens.input_ids, output_hidden_states=True)
extract_layers = [10, 20, 30]
encoder_hidden = torch.cat([output.hidden_states[l] for l in extract_layers], dim=-1)
encoder_hidden_np = encoder_hidden.float().numpy()  # [1, 512, 15360]
print(f"Text encoding: {encoder_hidden_np.shape}", flush=True)
del full_model, lang_model

# ============================================================
# 3. Prepare inputs
# ============================================================
h_lat, w_lat = H // 8, W // 8
h_packed, w_packed = h_lat // 2, w_lat // 2
num_img = h_packed * w_packed  # 4096
packed_ch = 32 * 4  # 128
text_seq = 512
total_seq = text_seq + num_img  # 4608
dim = 6144
head_dim = 128

# The engine expects encoder_hidden_states as [512, 6144].
# But our text encoder gives [1, 512, 15360].
# The context_embedder linear (15360 -> 6144) is baked into the engine.
# So we need to pass the raw [512, 15360] to the engine.
# Check engine shape to decide:
enc_shape = gpu_bufs["encoder_hidden_states"][1]
print(f"Engine expects encoder_hidden_states: {enc_shape}")
if enc_shape[-1] == 6144:
    # Engine has context_embedder baked - need to apply it ourselves or
    # the engine expects pre-projected embeddings
    # Actually the builder bakes context_embedder, so input should be raw 15360 dim...
    # But the engine shape says 6144. Let me check.
    # The engine was built with encoder_hidden_states input at [512, 6144].
    # This means context_embedder is baked (projects 15360->6144 internally?
    # No - the engine INPUT is [512, 6144], meaning we need to project externally.
    # The x_embedder and context_embedder are typically done in preprocessing.
    print("Need to project encoder_hidden_states from 15360->6144 externally")
    # Load context_embedder weights
    from safetensors import safe_open
    tf_dir = "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/transformer"
    readers = []
    for f in sorted(glob.glob(f"{tf_dir}/*.safetensors")):
        readers.append(safe_open(f, framework="numpy"))

    # Find context_embedder weight
    ctx_w = None
    ctx_b = None
    for r in readers:
        for k in r.keys():
            if "context_embedder.weight" in k:
                ctx_w = r.get_tensor(k)
            if "context_embedder.bias" in k:
                ctx_b = r.get_tensor(k)

    if ctx_w is not None:
        print(f"context_embedder weight: {ctx_w.shape}")
        enc_proj = encoder_hidden_np[0] @ ctx_w.T  # [512, 6144]
        if ctx_b is not None:
            enc_proj += ctx_b
        upload("encoder_hidden_states", enc_proj)
    else:
        print("WARNING: context_embedder not found, using raw embeddings")
        upload("encoder_hidden_states", encoder_hidden_np[0, :, :6144])
elif enc_shape[-1] == 15360:
    upload("encoder_hidden_states", encoder_hidden_np[0])

# Compute RoPE
print("Computing RoPE...", flush=True)
# FLUX.2-dev 4D RoPE: theta=10000, axes=(32,32,32,32)
axes_dims = [32, 32, 32, 32]
theta = 10000.0

# img_ids: [4096, 4] positions
img_ids = np.zeros((num_img, 4), dtype=np.float32)
for h in range(h_packed):
    for w in range(w_packed):
        idx = h * w_packed + w
        img_ids[idx, 1] = float(h)
        img_ids[idx, 2] = float(w)

# txt_ids: [512, 4] (all zeros)
txt_ids = np.zeros((text_seq, 4), dtype=np.float32)

# Concatenate [txt, img]
all_ids = np.concatenate([txt_ids, img_ids], axis=0)  # [4608, 4]

# Compute freqs for each axis
cos_vals = []
sin_vals = []
for ax, ax_dim in enumerate(axes_dims):
    positions = all_ids[:, ax]  # [4608]
    freqs = 1.0 / (theta ** (np.arange(0, ax_dim, 2, dtype=np.float64) / ax_dim))
    angles = np.outer(positions, freqs)  # [4608, ax_dim/2]
    cos_vals.append(np.cos(angles))
    sin_vals.append(np.sin(angles))

rotary_cos = np.concatenate(cos_vals, axis=-1).astype(np.float32)  # [4608, 64]
rotary_sin = np.concatenate(sin_vals, axis=-1).astype(np.float32)  # [4608, 64]
# Tile to head_dim=128: each freq appears twice (for x, y rotation pairs)
rotary_cos = np.repeat(rotary_cos, 2, axis=-1)  # [4608, 128]
rotary_sin = np.repeat(rotary_sin, 2, axis=-1)  # [4608, 128]
print(f"RoPE: cos={rotary_cos.shape}, sin={rotary_sin.shape}")
upload("rotary_cos", rotary_cos)
upload("rotary_sin", rotary_sin)

# ============================================================
# 4. Denoising loop
# ============================================================
np.random.seed(SEED)
latents = np.random.randn(packed_ch * h_packed * w_packed).astype(np.float32)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # noqa: E402
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "black-forest-labs/FLUX.2-dev", subfolder="scheduler")
image_seq_len = num_img
mu = 0.5 + (1.15 - 0.5) * (image_seq_len / 4096)
mu = math.log(math.e ** mu)
scheduler.set_timesteps(NUM_STEPS, mu=mu)

# Compute temb for each timestep
# temb = timestep_embedding(t) passed through timestep MLP
# But our engine expects pre-computed temb [6144]
# The timestep MLP is baked into the engine... or is it?
# Check: engine input "temb" shape
temb_shape = gpu_bufs["temb"][1]
print(f"Engine expects temb: {temb_shape}")

# Load timestep MLP weights for external computation
from safetensors import safe_open  # noqa: E402
tf_dir = "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/transformer"
readers = []
for f in sorted(glob.glob(f"{tf_dir}/*.safetensors")):
    readers.append(safe_open(f, framework="numpy"))

# Find time_text_embed weights
tte = {}
for r in readers:
    for k in r.keys():
        if "time_text_embed" in k:
            tte[k] = r.get_tensor(k)

def sinusoidal_embedding(t, dim=256, max_period=10000.0):
    half = dim // 2
    freqs = np.exp(-np.log(max_period) * np.arange(half, dtype=np.float64) / half)
    args = t * freqs
    return np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)

def compute_temb(timestep_val, guidance_val=GUIDANCE):
    # timestep embedding
    t_emb = sinusoidal_embedding(timestep_val, dim=256)
    # guidance embedding
    g_emb = sinusoidal_embedding(guidance_val, dim=256)

    # timestep MLP: timestep_embedder (Linear 256->6144 + SiLU + Linear 6144->6144)
    w1 = tte.get("time_text_embed.timestep_embedder.linear_1.weight")
    b1 = tte.get("time_text_embed.timestep_embedder.linear_1.bias")
    w2 = tte.get("time_text_embed.timestep_embedder.linear_2.weight")
    b2 = tte.get("time_text_embed.timestep_embedder.linear_2.bias")

    h = t_emb @ w1.T + b1
    h = h * (1.0 / (1.0 + np.exp(-h)))  # SiLU
    temb = h @ w2.T + b2

    # guidance MLP
    gw1 = tte.get("time_text_embed.guidance_embedder.linear_1.weight")
    gb1 = tte.get("time_text_embed.guidance_embedder.linear_1.bias")
    gw2 = tte.get("time_text_embed.guidance_embedder.linear_2.weight")
    gb2 = tte.get("time_text_embed.guidance_embedder.linear_2.bias")

    gh = g_emb @ gw1.T + gb1
    gh = gh * (1.0 / (1.0 + np.exp(-gh)))  # SiLU
    gemb = gh @ gw2.T + gb2

    return (temb + gemb).astype(np.float32)

print(f"Starting {NUM_STEPS}-step denoising loop...", flush=True)
t_start = time.time()

for i, t in enumerate(scheduler.timesteps):
    lat_chw = latents.reshape(packed_ch, h_packed, w_packed)
    hidden = lat_chw.transpose(1, 2, 0).reshape(num_img, packed_ch)
    upload("hidden_states", hidden)

    temb = compute_temb(t.item() / 1000.0)
    upload("temb", temb)

    run_engine()

    out = download("output")
    noise_pred = out  # [4096, 128]
    velocity = noise_pred.reshape(h_packed, w_packed, packed_ch).transpose(2, 0, 1).flatten()

    dt = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
    latents = latents + dt.item() * velocity

    if i < 3 or i >= NUM_STEPS - 2 or i % 7 == 0:
        print(f"  Step {i+1}/{NUM_STEPS} t={t.item():.1f} "
              f"vel_std={velocity.std():.4f} lat_std={latents.std():.4f}", flush=True)

denoise_time = time.time() - t_start
print(f"Denoising: {denoise_time:.1f}s ({denoise_time/NUM_STEPS:.3f}s/step)", flush=True)

# ============================================================
# 5. VAE decode
# ============================================================
print("Decoding VAE on CPU...", flush=True)
t0 = time.time()

z_dim = 32
lat_packed = latents.reshape(packed_ch, h_packed, w_packed)
lat_full = np.zeros((z_dim, h_lat, w_lat), dtype=np.float32)
for c in range(z_dim):
    for dy in range(2):
        for dx in range(2):
            src_ch = c * 4 + dy * 2 + dx
            for py in range(h_packed):
                for px in range(w_packed):
                    lat_full[c, py*2+dy, px*2+dx] = lat_packed[src_ch, py, px]

# BN denorm
from safetensors import safe_open  # noqa: E402
vae_dir = "/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/26afe3a78bb242c0a8bb181dcc8937bb16e5c66c/vae"
for f in glob.glob(f"{vae_dir}/*.safetensors"):
    with safe_open(f, framework="numpy") as sf:
        if "bn.running_mean" in sf.keys():
            bn_mean = sf.get_tensor("bn.running_mean")
            bn_var = sf.get_tensor("bn.running_var")
            break

for c in range(min(len(bn_mean), z_dim)):
    s = np.sqrt(bn_var[c] + 1e-4)
    lat_full[c] = lat_full[c] * s + bn_mean[c]

lat_tensor = torch.from_numpy(lat_full).unsqueeze(0).to(torch.bfloat16)
from diffusers import AutoencoderKL  # noqa: E402
vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=torch.bfloat16).to("cpu")
with torch.no_grad():
    image = vae.decode(lat_tensor).sample[0]

image = (image.float().clamp(-1, 1) + 1) / 2 * 255
image = image.permute(1, 2, 0).byte().numpy()
from PIL import Image  # noqa: E402
Image.fromarray(image).save(OUTPUT_PATH)
print(f"VAE decode: {time.time()-t0:.1f}s", flush=True)
print(f"SAVED: {OUTPUT_PATH}", flush=True)
