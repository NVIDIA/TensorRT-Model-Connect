# SPDX-License-Identifier: Apache-2.0
"""Score velocity files in a steps dir: python compare_velocity.py DIR  (pairs: ref fp32/bf16 vs trt cp1/cp2)."""
import os, sys, glob, numpy as np

def m(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return (float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())), float(np.linalg.norm(a-b)/np.linalg.norm(a)), float(np.abs(a-b).max()))

root = sys.argv[1]
for sd in sorted(glob.glob(os.path.join(root, "step*")), key=lambda s: int(os.path.basename(s)[4:])):
    files = {os.path.basename(f)[:-4]: np.load(f) for f in glob.glob(os.path.join(sd, "*.npz"))}
    names = sorted(files)
    print(f"== {os.path.basename(sd)}  files: {names}")
    pairs = [("ref_fp32", "ref_bf16"), ("ref_fp32", "ref_bf16_fold"), ("ref_bf16", "ref_bf16_fold"),
             ("ref_fp32", "trt_cp1"), ("ref_bf16", "trt_cp1"), ("ref_fp32", "trt_cp2"), ("ref_bf16", "trt_cp2"), ("trt_cp1", "trt_cp2")]
    for x, y in pairs:
        if x in files and y in files:
            for k, lab in (("video_velocity", "video"), ("audio_velocity", "audio")):
                c, r, mx = m(files[x][k], files[y][k])
                print(f"  {x:>9} vs {y:<13} {lab}: cos={c:.6f} relL2={r:.4g} maxabs={mx:.3g}")
