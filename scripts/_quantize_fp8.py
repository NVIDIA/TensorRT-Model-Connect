#!/usr/bin/env python3
"""FP8 quantize FLUX.2-dev DiT transformer using ModelOpt on CPU.

Bypasses CUBLAS issues by running calibration entirely on CPU.
Only quantizes the transformer backbone (not text encoder or VAE).
"""
import os
import re
import time

import torch
from tensorrt_model_connect.fp8_calibrate import FP8_MHA_CONFIG

print("Loading ModelOpt...", flush=True)
import modelopt.torch.quantization as mtq  # noqa: E402
import modelopt.torch.opt as mto  # noqa: E402

# FLUX.2-dev layer exclusion (keep these in BF16)
def filter_func_flux_dev(name):
    pattern = re.compile(
        r"(proj_out.*|.*(time_text_embed|context_embedder|x_embedder"
        r"|norm_out|time_guidance_embed|stream_modulation).*)"
    )
    return pattern.match(name) is not None

# Load FLUX.2 transformer on CPU
print("Loading FLUX.2-dev transformer on CPU...", flush=True)
t0 = time.time()
from diffusers import Flux2Pipeline  # noqa: E402

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
)
transformer = pipe.transformer.to("cpu")
transformer.eval()
del pipe  # free memory
print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

# Prepare calibration inputs (diverse timesteps for good amax coverage)
def make_flux2_inputs(timestep_val=500.0):
    """Create FLUX.2-dev transformer inputs for calibration."""
    # FLUX.2-dev: hidden_states [B, 4096, 128], encoder [B, 512, 15360]
    return {
        "hidden_states": torch.randn(1, 4096, 128, dtype=torch.bfloat16, device="cpu"),
        "encoder_hidden_states": torch.randn(1, 512, 15360, dtype=torch.bfloat16, device="cpu"),
        "timestep": torch.tensor([timestep_val / 1000.0], dtype=torch.float32, device="cpu"),
        "guidance": torch.tensor([3.5], dtype=torch.float32, device="cpu"),
        "txt_ids": torch.zeros(512, 4, dtype=torch.bfloat16, device="cpu"),
        "img_ids": torch.zeros(4096, 4, dtype=torch.bfloat16, device="cpu"),
    }

# Calibration forward loop
CALIB_STEPS = 8   # timestep diversity
CALIB_SAMPLES = 4  # number of random inputs per timestep

def calibration_loop(model):
    """Run diverse calibration to capture activation ranges."""
    timesteps = torch.linspace(50, 950, CALIB_STEPS)
    total = CALIB_STEPS * CALIB_SAMPLES
    done = 0
    for t in timesteps:
        for _ in range(CALIB_SAMPLES):
            inputs = make_flux2_inputs(t.item())
            with torch.no_grad():
                model(**inputs)
            done += 1
            if done % 4 == 0:
                print(f"  Calibration: {done}/{total} "
                      f"(t={t.item():.0f})", flush=True)

# Apply FP8 quantization
print(f"Quantizing to FP8 (calibration: {CALIB_STEPS}×{CALIB_SAMPLES}={CALIB_STEPS*CALIB_SAMPLES} passes)...", flush=True)
print("This runs on CPU — expect ~2-5 min per calibration pass...", flush=True)
t0 = time.time()

transformer = mtq.quantize(
    transformer,
    config=FP8_MHA_CONFIG,
    forward_loop=calibration_loop,
)

# Disable quantization on excluded layers
mtq.disable_quantizer(transformer, filter_func_flux_dev)

print(f"Quantization done in {time.time()-t0:.0f}s", flush=True)
mtq.print_quant_summary(transformer)

# Save quantized checkpoint
save_path = "/tmp/flux2_fp8_quantized.pt"
mto.save(transformer, save_path)
print(f"Saved: {save_path}", flush=True)

# Export to ONNX
print("Exporting FP8 ONNX...", flush=True)
t0 = time.time()
dummy = make_flux2_inputs()

onnx_dir = "/tmp/flux2_fp8_onnx"
os.makedirs(onnx_dir, exist_ok=True)

torch.onnx.export(
    transformer,
    args=(),
    kwargs=dummy,
    f=f"{onnx_dir}/flux2_dit_fp8.onnx",
    opset_version=20,
    input_names=list(dummy.keys()),
    output_names=["output"],
    do_constant_folding=True,
)
print(f"ONNX export done in {time.time()-t0:.0f}s", flush=True)
print(f"Files: {os.listdir(onnx_dir)}", flush=True)
