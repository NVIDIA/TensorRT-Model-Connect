# SPDX-License-Identifier: Apache-2.0
"""Score public HSTU via the native Triton Model Connect backend."""

import argparse
from pathlib import Path
import numpy as np
import tritonclient.http as http
from tritonclient.utils import np_to_triton_dtype
from families.hstu.config import Config
from families.hstu.reference import example_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="hstu")
    p.add_argument("--url", default="localhost:8000")
    p.add_argument("--batch", type=int, default=1)
    a = p.parse_args()
    with http.InferenceServerClient(a.url) as client:
        tensors = []
        for name, value in example_inputs(Config.load(a.checkpoint), a.batch).items():
            value = np.ascontiguousarray(value)
            tensor = http.InferInput(name, value.shape, np_to_triton_dtype(value.dtype))
            tensor.set_data_from_numpy(value)
            tensors.append(tensor)
        response = client.infer(a.model, tensors)
        print("picked_indices:", response.as_numpy("picked_indices"))
        print("scores:", response.as_numpy("scores"))


if __name__ == "__main__":
    main()
