#!/usr/bin/env python3
"""Inject FP8 Q/DQ nodes into the working BF16 ONNX at the proto level.

Uses ONNX proto manipulation (not graphsurgeon) to handle FP8 types.
Reads the per-layer calibrated scales and inserts proper FP8 E4M3 Q/DQ.
"""
import json
import os
import onnx
from onnx import TensorProto
import onnx_graphsurgeon as gs
import numpy as np

ONNX_INPUT = "/tmp/flux2_dit_onnx/flux2_dit.onnx"
ONNX_OUTPUT = "/tmp/flux2_fp8_onnx_proto/flux2_dit_fp8.onnx"
MAPPING_PATH = "/tmp/flux2_fp8_onnx_mapping.json"

os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)

# Load per-layer calibrated scales
with open(MAPPING_PATH) as f:
    per_layer = json.load(f)
print(f"Loaded {len(per_layer)} per-layer scale mappings", flush=True)

# Load ONNX model via graphsurgeon for easy graph manipulation
print("Loading ONNX model...", flush=True)
graph = gs.import_onnx(onnx.load(ONNX_INPUT))

matmuls = [n for n in graph.nodes if n.op == "MatMul"]
print(f"Total MatMul nodes: {len(matmuls)}", flush=True)

# For each calibrated MatMul, insert Q → DQ on both inputs
# Using INT8 proxy dtype in graphsurgeon (TRT ONNX parser reads zero_point type)
# We'll fix the zero_point type to FP8 in the final ONNX proto
qdq_count = 0
fp8_zp_names = []  # Track which zero_points need FP8 type fix

for node in matmuls:
    if node.name not in per_layer:
        continue

    entry = per_layer[node.name]
    inp_scale = entry.get("input_scale")
    wt_scale = entry.get("weight_scale")
    if inp_scale is None or wt_scale is None:
        continue

    for inp_idx, scale_val in [(0, inp_scale), (1, wt_scale)]:
        inp_tensor = node.inputs[inp_idx]
        prefix = f"{node.name}_{'A' if inp_idx == 0 else 'B'}"

        # Scale constant (FP32 scalar; TensorRT expects a scalar here, not [1])
        scale = gs.Constant(
            name=f"{prefix}_scale",
            values=np.array(scale_val, dtype=np.float32))  # scalar, no brackets

        # Zero-point as uint8 (placeholder — will be fixed to FP8 in proto)
        zp_name = f"{prefix}_zp"
        zp = gs.Constant(
            name=zp_name,
            values=np.array([0], dtype=np.uint8))
        fp8_zp_names.append(zp_name)

        # Q output (int8 proxy for graphsurgeon — TRT infers FP8 from zp type)
        q_out = gs.Variable(name=f"{prefix}_q", dtype=np.int8)

        # DQ output
        dq_out = gs.Variable(name=f"{prefix}_dq", dtype=np.float32)

        # QuantizeLinear node
        q_node = gs.Node(
            op="QuantizeLinear",
            name=f"{prefix}_Q",
            inputs=[inp_tensor, scale, zp],
            outputs=[q_out],
            attrs={"saturate": 1})

        # DequantizeLinear node
        dq_node = gs.Node(
            op="DequantizeLinear",
            name=f"{prefix}_DQ",
            inputs=[q_out, scale, zp],
            outputs=[dq_out])

        graph.nodes.extend([q_node, dq_node])

        # Rewire MatMul input
        node.inputs[inp_idx] = dq_out
        qdq_count += 1

print(f"Inserted {qdq_count} Q/DQ pairs, {len(fp8_zp_names)} zero_points to fix", flush=True)

# Clean up and export
graph.cleanup().toposort()

# Update opset to 21 (needed for FP8 QuantizeLinear support)
model = gs.export_onnx(graph)
for opset in model.opset_import:
    if opset.domain == "" or opset.domain == "ai.onnx":
        opset.version = 21

# Fix zero_point types: change uint8 → FLOAT8E4M3FN
# Walk the model's graph initializers and fix the zero_point tensors
zp_name_set = set(fp8_zp_names)
fixed = 0
for initializer in model.graph.initializer:
    if initializer.name in zp_name_set:
        # Replace with FP8 E4M3 zero_point
        initializer.data_type = TensorProto.FLOAT8E4M3FN
        initializer.raw_data = bytes([0])  # FP8 E4M3 value 0
        initializer.dims[:] = []  # scalar, same shape as scale
        # Clear any numpy data
        while len(initializer.float_data) > 0:
            initializer.float_data.pop()
        while len(initializer.int32_data) > 0:
            initializer.int32_data.pop()
        fixed += 1

print(f"Fixed {fixed} zero_points to FP8 E4M3FN type", flush=True)

# Also fix the Q output tensor types in value_info
for vi in model.graph.value_info:
    if vi.name.endswith("_q"):
        vi.type.tensor_type.elem_type = TensorProto.FLOAT8E4M3FN

# Save with external data
print(f"Saving {ONNX_OUTPUT}...", flush=True)
onnx.save(model, ONNX_OUTPUT,
          save_as_external_data=True,
          all_tensors_to_one_file=True,
          location=os.path.basename(ONNX_OUTPUT) + ".data",
          size_threshold=1024)

print(f"Output: {os.path.getsize(ONNX_OUTPUT)/(1024**2):.1f} MB (model)")
data_path = ONNX_OUTPUT + ".data"
if os.path.exists(data_path):
    print(f"  + {os.path.getsize(data_path)/(1024**3):.1f} GB (data)")
print("DONE", flush=True)
