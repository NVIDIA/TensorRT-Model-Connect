# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from importlib.metadata import version
from pathlib import Path

import torch
from openfold3.projects.of3_all_atom import model as model_module
from openfold3.projects.of3_all_atom.model import OpenFold3


if version("openfold3") != "0.5.0":
    raise RuntimeError("OpenFold3 build profile requires openfold3==0.5.0")
if not torch.cuda.is_available():
    raise RuntimeError("OpenFold3 build profile requires CUDA")
if OpenFold3.__name__ != "OpenFold3":
    raise RuntimeError("OpenFold3 model class is unavailable")
model_sha256 = hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest()
if model_sha256 != "1cd02947e24bcea5d30dba98122e6a38e707f88ebe1d4c4f2aa16f4b177f7e0c":
    raise RuntimeError("OpenFold3 model source differs from pinned v0.5.0 revision")
print(f"openfold3={version('openfold3')} torch={torch.__version__}")
